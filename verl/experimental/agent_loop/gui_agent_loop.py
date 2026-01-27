# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
GUI Agent Loop for mobile device interaction.

This agent loop implements the multi-turn GUI interaction workflow following
the Qwen3-VL MobileAgent approach. It handles:
1. Screenshot-based visual understanding
2. Action prediction (click, swipe, type, etc.)
3. Multi-turn interaction with the mobile environment
4. Task completion assessment
"""

import asyncio
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.mobile_use_tool import (
    MobileAction,
    MobileUseTool,
    SimulatedMobileEnvironment,
    TerminateStatus,
    get_mobile_use_tool_schema,
)
from verl.tools.schemas import ToolResponse
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class GUIAgentState(Enum):
    """State machine states for GUI Agent."""

    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_ACTION = "processing_action"
    TERMINATED = "terminated"


class GUIAgentData:
    """Encapsulates all state variables for the GUI Agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        metrics: dict[str, Any],
        request_id: str,
        instruction: str,
        screenshots: list[Image.Image],
        expected_actions: Optional[list[dict]] = None,
        screen_width: int = 999,
        screen_height: int = 999,
    ):
        self.messages = messages
        self.image_data = image_data or []
        self.metrics = metrics
        self.request_id = request_id
        self.instruction = instruction
        self.screenshots = screenshots
        self.expected_actions = expected_actions or []
        self.screen_width = screen_width
        self.screen_height = screen_height

        # State variables for token tracking
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.routed_experts: Optional[Any] = None

        # Turn tracking
        self.user_turns = 0
        self.assistant_turns = 0

        # Action history
        self.action_history: list[dict] = []
        self.action_rewards: list[float] = []

        # Current action being processed
        self.current_action: Optional[FunctionCall] = None

        # Environment state
        self.environment: Optional[SimulatedMobileEnvironment] = None
        self.is_terminated = False
        self.terminate_status: Optional[TerminateStatus] = None

        # Extra fields for reward computation
        self.extra_fields: dict[str, Any] = {}


@register("gui_agent")
class GUIAgentLoop(AgentLoopBase):
    """
    GUI Agent Loop for mobile device interaction.

    This loop handles multi-turn interactions with a mobile device,
    processing screenshots and predicting actions following the
    Qwen3-VL MobileAgent approach.
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        dataset_config: Any,
        **kwargs,
    ):
        super().__init__(
            trainer_config,
            server_manager,
            tokenizer,
            processor,
            dataset_cls,
            dataset_config,
            **kwargs,
        )

        config = trainer_config.config

        # GUI Agent specific configurations
        gui_config = config.actor_rollout_ref.rollout.get("gui_agent", {})
        self.max_turns = gui_config.get("max_turns", 20)
        self.screen_width = gui_config.get("screen_width", 999)
        self.screen_height = gui_config.get("screen_height", 999)
        self.return_screenshot = gui_config.get("return_screenshot", True)

        # Multi-turn configuration
        multi_turn_config = config.actor_rollout_ref.rollout.get("multi_turn", {})
        self.max_assistant_turns = multi_turn_config.get("max_assistant_turns", 20)
        self.max_user_turns = multi_turn_config.get("max_user_turns", 20)

        # Length limits
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length

        # Tool parser for extracting function calls
        tool_parser_format = multi_turn_config.get("format", "qwen")
        self.tool_parser = ToolParser.get_tool_parser(tool_parser_format, self.tokenizer)

        # Tool schema
        self.tool_schema = get_mobile_use_tool_schema(self.screen_width, self.screen_height)
        self.tool_schemas = [self.tool_schema.model_dump(exclude_unset=True, exclude_none=True)]

    def _build_system_prompt(self) -> str:
        """Build the system prompt for GUI Agent."""
        return f"""

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{json.dumps(self.tool_schema.model_dump(exclude_unset=True, exclude_none=True), indent=2)}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

# Response format

Response format for every step:
1) Thought: one concise sentence explaining the next move (no multi-step reasoning).
2) Action: a short imperative describing what to do in the UI.
3) A single <tool_call>...</tool_call> block containing only the JSON: {{"name": <function-name>, "arguments": <args-json-object>}}.

Rules:
- Output exactly in the order: Thought, Action, <tool_call>.
- Be brief: one sentence for Thought, one for Action.
- Do not output anything else outside those three parts.
- If finishing, use action=terminate in the tool call."""

    def _build_task_history(self, action_history: list[dict]) -> str:
        """Build task progress history string."""
        if not action_history:
            return ""

        history_items = []
        for idx, action in enumerate(action_history):
            action_type = action.get("action", "unknown")
            if action_type == "click":
                coord = action.get("coordinate", [0, 0])
                history_items.append(f"Step {idx + 1}: Clicked at ({coord[0]}, {coord[1]})")
            elif action_type == "swipe":
                coord1 = action.get("coordinate", [0, 0])
                coord2 = action.get("coordinate2", [0, 0])
                history_items.append(f"Step {idx + 1}: Swiped from ({coord1[0]}, {coord1[1]}) to ({coord2[0]}, {coord2[1]})")
            elif action_type == "type":
                text = action.get("text", "")
                history_items.append(f"Step {idx + 1}: Typed '{text}'")
            elif action_type == "system_button":
                button = action.get("button", "")
                history_items.append(f"Step {idx + 1}: Pressed {button} button")
            elif action_type == "wait":
                time = action.get("time", 1.0)
                history_items.append(f"Step {idx + 1}: Waited {time} seconds")
            elif action_type == "long_press":
                coord = action.get("coordinate", [0, 0])
                time = action.get("time", 1.0)
                history_items.append(f"Step {idx + 1}: Long pressed at ({coord[0]}, {coord[1]}) for {time}s")
            else:
                history_items.append(f"Step {idx + 1}: {action_type}")

        return "; ".join(history_items) + "; "

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run the GUI Agent loop."""
        # Initialize agent data
        messages = list(kwargs.get("raw_prompt", []))
        instruction = kwargs.get("instruction", "")

        # Extract screenshots from data
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images", [])

        # Get expected actions for training
        extra_info = kwargs.get("extra_info", {})
        expected_actions = extra_info.get("expected_actions", [])
        screenshots = extra_info.get("screenshots", images)

        metrics = {}
        request_id = uuid4().hex

        # Create agent data
        agent_data = GUIAgentData(
            messages=messages,
            image_data=images,
            metrics=metrics,
            request_id=request_id,
            instruction=instruction,
            screenshots=screenshots if isinstance(screenshots, list) else [screenshots],
            expected_actions=expected_actions,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
        )

        # Initialize simulated environment
        agent_data.environment = SimulatedMobileEnvironment(
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            screenshots=agent_data.screenshots,
            expected_actions=agent_data.expected_actions,
        )

        # State machine loop
        state = GUIAgentState.PENDING
        while state != GUIAgentState.TERMINATED:
            if state == GUIAgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == GUIAgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == GUIAgentState.PROCESSING_ACTION:
                state = await self._handle_processing_action_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = GUIAgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]

        multi_modal_output = {}
        if agent_data.image_data:
            multi_modal_output["images"] = agent_data.image_data

        # Calculate final reward
        final_reward = self._calculate_final_reward(agent_data)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            response_logprobs=agent_data.response_logprobs[: self.response_length] if agent_data.response_logprobs else None,
            routed_experts=agent_data.routed_experts,
            multi_modal_data=multi_modal_output,
            reward_score=final_reward,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=AgentLoopMetrics(**agent_data.metrics),
            extra_fields={
                "action_history": agent_data.action_history,
                "action_rewards": agent_data.action_rewards,
                "terminate_status": agent_data.terminate_status.value if agent_data.terminate_status else None,
            },
        )

        return output

    async def _handle_pending_state(
        self,
        agent_data: GUIAgentData,
        sampling_params: dict[str, Any],
    ) -> GUIAgentState:
        """Handle the pending state: build initial prompt."""
        # Build system message with tool schema
        system_content = self._build_system_prompt()

        # Build user query with task progress
        task_history = self._build_task_history(agent_data.action_history)
        user_query = f"The user query: {agent_data.instruction}.\n"
        if task_history:
            user_query += f"Task progress (You have done the following operation on the current device): {task_history}\n"

        # Build messages
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_content}]},
        ]

        # Add user message with screenshot
        user_content = [{"type": "text", "text": user_query}]
        if agent_data.image_data:
            user_content.insert(0, {"type": "image"})

        messages.append({"role": "user", "content": user_content})

        agent_data.messages = messages

        # Apply chat template
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self.tool_schemas,
            images=agent_data.image_data,
            videos=None,
        )

        agent_data.prompt_ids = prompt_ids

        return GUIAgentState.GENERATING

    async def _handle_generating_state(
        self,
        agent_data: GUIAgentData,
        sampling_params: dict[str, Any],
    ) -> GUIAgentState:
        """Handle the generating state: generate model response."""
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=None,
            )

        # Update metrics
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)

        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if len(agent_data.response_mask) >= self.response_length:
            return GUIAgentState.TERMINATED
        if agent_data.assistant_turns >= self.max_assistant_turns:
            return GUIAgentState.TERMINATED
        if agent_data.user_turns >= self.max_user_turns:
            return GUIAgentState.TERMINATED

        # Extract tool calls
        _, tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        if tool_calls:
            agent_data.current_action = tool_calls[0]  # Process one action at a time
            return GUIAgentState.PROCESSING_ACTION
        else:
            # No tool call found, terminate
            return GUIAgentState.TERMINATED

    async def _handle_processing_action_state(self, agent_data: GUIAgentData) -> GUIAgentState:
        """Handle the processing action state: execute the action."""
        if agent_data.current_action is None:
            return GUIAgentState.TERMINATED

        with simple_timer("tool_calls", agent_data.metrics):
            # Parse action parameters
            try:
                action_params = json.loads(agent_data.current_action.arguments)
            except json.JSONDecodeError:
                action_params = {}

            action_type = action_params.get("action", "")

            # Execute action on environment
            env = agent_data.environment
            result_text = ""
            new_screenshot = None
            step_reward = 0.0

            try:
                if action_type == MobileAction.CLICK.value:
                    coord = action_params.get("coordinate", [0, 0])
                    result_text, new_screenshot = await env.click(int(coord[0]), int(coord[1]))
                    step_reward = env.get_step_reward()

                elif action_type == MobileAction.LONG_PRESS.value:
                    coord = action_params.get("coordinate", [0, 0])
                    duration = action_params.get("time", 1.0)
                    result_text, new_screenshot = await env.long_press(int(coord[0]), int(coord[1]), float(duration))
                    step_reward = env.get_step_reward()

                elif action_type == MobileAction.SWIPE.value:
                    coord1 = action_params.get("coordinate", [0, 0])
                    coord2 = action_params.get("coordinate2", [0, 0])
                    result_text, new_screenshot = await env.swipe(
                        int(coord1[0]), int(coord1[1]),
                        int(coord2[0]), int(coord2[1])
                    )
                    step_reward = env.get_step_reward()

                elif action_type == MobileAction.TYPE.value:
                    text = action_params.get("text", "")
                    result_text, new_screenshot = await env.type_text(text)
                    step_reward = env.get_step_reward()

                elif action_type == MobileAction.SYSTEM_BUTTON.value:
                    from verl.tools.mobile_use_tool import SystemButton
                    button_str = action_params.get("button", "Back")
                    button = SystemButton(button_str)
                    result_text, new_screenshot = await env.press_system_button(button)
                    step_reward = env.get_step_reward()

                elif action_type == MobileAction.WAIT.value:
                    duration = action_params.get("time", 1.0)
                    result_text, new_screenshot = await env.wait(float(duration))

                elif action_type == MobileAction.TERMINATE.value:
                    status_str = action_params.get("status", "success")
                    agent_data.terminate_status = TerminateStatus(status_str)
                    agent_data.is_terminated = True
                    env.terminate(agent_data.terminate_status)
                    step_reward = 1.0 if agent_data.terminate_status == TerminateStatus.SUCCESS else 0.0

                elif action_type == MobileAction.ANSWER.value:
                    answer = action_params.get("text", "")
                    result_text = f"Answer: {answer}"
                    agent_data.extra_fields["answer"] = answer

            except Exception as e:
                logger.error(f"Error executing action: {e}")
                result_text = f"Error: {str(e)}"

            # Record action
            agent_data.action_history.append(action_params)
            agent_data.action_rewards.append(step_reward)

            # Check if terminated
            if agent_data.is_terminated:
                return GUIAgentState.TERMINATED

            # Update images with new screenshot
            if new_screenshot is not None and self.return_screenshot:
                agent_data.image_data.append(new_screenshot)

            # Build tool response message
            tool_response_content = [{"type": "text", "text": result_text}]
            if new_screenshot is not None and self.return_screenshot:
                tool_response_content.insert(0, {"type": "image"})

            tool_message = {"role": "tool", "content": tool_response_content}
            agent_data.messages.append(tool_message)

            # Tokenize tool response
            response_ids = await self.apply_chat_template(
                [tool_message],
                images=[new_screenshot] if new_screenshot is not None else None,
                videos=None,
                remove_system_prompt=True,
            )

            # Check if adding response would exceed limit
            if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
                return GUIAgentState.TERMINATED

            # Update state
            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [0] * len(response_ids)  # Tool responses are masked
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(response_ids)

            agent_data.user_turns += 1
            agent_data.current_action = None

            return GUIAgentState.GENERATING

    def _calculate_final_reward(self, agent_data: GUIAgentData) -> float:
        """Calculate the final reward for the trajectory."""
        # If terminated successfully, give high reward
        if agent_data.terminate_status == TerminateStatus.SUCCESS:
            return 1.0

        # If terminated with failure, give low reward
        if agent_data.terminate_status == TerminateStatus.FAILURE:
            return 0.0

        # Calculate based on action rewards
        if agent_data.action_rewards:
            # Average of step rewards with bonus for more correct actions
            avg_reward = sum(agent_data.action_rewards) / len(agent_data.action_rewards)
            # Bonus for completing more steps correctly
            correct_actions = sum(1 for r in agent_data.action_rewards if r > 0.5)
            completion_bonus = correct_actions / max(len(agent_data.expected_actions), 1) if agent_data.expected_actions else 0
            return 0.5 * avg_reward + 0.5 * completion_bonus

        return 0.0
