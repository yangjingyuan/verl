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

This implementation uses an interaction format instead of tool_use format:
- Response format: Thought: <thinking> + Action: <action_type>(<params>)
- Example: Action: click(500, 300)
"""

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

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
from verl.tools.mobile_use_tool import (
    MobileAction,
    SimulatedMobileEnvironment,
    TerminateStatus,
)
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class ParsedAction:
    """Parsed action from interaction format response."""

    action_type: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result = {"action": self.action_type}
        result.update(self.params)
        return result


def parse_interaction_action(response_text: str) -> Optional[ParsedAction]:
    """
    Parse action from interaction format response.

    Supported formats:
    - Action: click(x, y)
    - Action: long_press(x, y, time)
    - Action: swipe(x1, y1, x2, y2)
    - Action: type("text")
    - Action: answer("text")
    - Action: system_button(button_name)
    - Action: wait(seconds)
    - Action: terminate(status)

    Args:
        response_text: The model's response string.

    Returns:
        ParsedAction object or None if parsing fails.
    """
    # Find Action: line
    action_pattern = r"Action:\s*(\w+)\s*\(([^)]*)\)"
    match = re.search(action_pattern, response_text, re.IGNORECASE)

    if not match:
        return None

    action_type = match.group(1).lower()
    params_str = match.group(2).strip()

    try:
        if action_type == "click":
            # click(x, y)
            coords = _parse_coordinates(params_str)
            if len(coords) >= 2:
                return ParsedAction(action_type="click", params={"coordinate": [coords[0], coords[1]]})

        elif action_type == "long_press":
            # long_press(x, y, time) or long_press(x, y)
            parts = _parse_params(params_str)
            if len(parts) >= 2:
                coord = [int(float(parts[0])), int(float(parts[1]))]
                time = float(parts[2]) if len(parts) > 2 else 1.0
                return ParsedAction(action_type="long_press", params={"coordinate": coord, "time": time})

        elif action_type == "swipe":
            # swipe(x1, y1, x2, y2)
            coords = _parse_coordinates(params_str)
            if len(coords) >= 4:
                return ParsedAction(
                    action_type="swipe",
                    params={"coordinate": [coords[0], coords[1]], "coordinate2": [coords[2], coords[3]]},
                )

        elif action_type == "type":
            # type("text") or type(text)
            text = _parse_text_param(params_str)
            return ParsedAction(action_type="type", params={"text": text})

        elif action_type == "answer":
            # answer("text") or answer(text)
            text = _parse_text_param(params_str)
            return ParsedAction(action_type="answer", params={"text": text})

        elif action_type == "system_button":
            # system_button(Back) or system_button("Back")
            button = _parse_text_param(params_str)
            return ParsedAction(action_type="system_button", params={"button": button})

        elif action_type == "wait":
            # wait(seconds)
            parts = _parse_params(params_str)
            time = float(parts[0]) if parts else 1.0
            return ParsedAction(action_type="wait", params={"time": time})

        elif action_type == "terminate":
            # terminate(success) or terminate(failure)
            status = _parse_text_param(params_str).lower()
            if status not in ["success", "failure"]:
                status = "success"
            return ParsedAction(action_type="terminate", params={"status": status})

    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse action params: {e}")
        return None

    return None


def _parse_coordinates(params_str: str) -> list[int]:
    """Parse coordinate values from parameter string."""
    # Remove quotes and brackets
    clean = params_str.replace('"', "").replace("'", "").replace("[", "").replace("]", "")
    # Split by comma and convert to integers
    parts = [p.strip() for p in clean.split(",") if p.strip()]
    return [int(float(p)) for p in parts]


def _parse_params(params_str: str) -> list[str]:
    """Parse comma-separated parameters."""
    clean = params_str.replace('"', "").replace("'", "").replace("[", "").replace("]", "")
    return [p.strip() for p in clean.split(",") if p.strip()]


def _parse_text_param(params_str: str) -> str:
    """Parse text parameter, handling quotes."""
    # Try to extract quoted string first
    quote_match = re.search(r'["\']([^"\']*)["\']', params_str)
    if quote_match:
        return quote_match.group(1)
    # Otherwise return stripped string
    return params_str.strip().strip('"').strip("'")


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
        self.current_action: Optional[ParsedAction] = None

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

    def _build_system_prompt(self) -> str:
        """Build the system prompt for GUI Agent using interaction format."""
        return f"""You are a GUI agent that interacts with a mobile device touchscreen.

# Screen Information
- Screen resolution: {self.screen_width}x{self.screen_height}
- Coordinates: (x, y) where x is pixels from left edge, y is pixels from top edge

# Available Actions
- click(x, y): Click at position (x, y)
- long_press(x, y, time): Long press at (x, y) for specified seconds (default: 1.0)
- swipe(x1, y1, x2, y2): Swipe from (x1, y1) to (x2, y2)
- type("text"): Input text into the current input field
- answer("text"): Output the answer to the user's question
- system_button(button): Press system button (Back, Home, Menu, Enter)
- wait(seconds): Wait for specified seconds
- terminate(status): End task with status (success or failure)

# Response Format
For every step, respond in exactly this format:
Thought: <one concise sentence explaining your reasoning>
Action: <action_name>(<parameters>)

# Examples
Thought: I need to tap the search icon to open search.
Action: click(450, 120)

Thought: I need to scroll down to see more content.
Action: swipe(500, 700, 500, 200)

Thought: I need to enter the search query.
Action: type("weather forecast")

Thought: The task is complete, I found the information.
Action: terminate(success)

# Rules
- Always output Thought first, then Action on the next line
- Be concise: one sentence for Thought
- Click the center of UI elements, not their edges
- Wait when needed for UI to respond
- Use terminate(success) when task is done, terminate(failure) if stuck"""

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
        # Build system message
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

        # Apply chat template (no tools parameter for interaction format)
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=None,
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

        # Decode response and extract action using interaction format
        response_text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        parsed_action = parse_interaction_action(response_text)

        if parsed_action:
            agent_data.current_action = parsed_action
            return GUIAgentState.PROCESSING_ACTION
        else:
            # No valid action found, terminate
            return GUIAgentState.TERMINATED

    async def _handle_processing_action_state(self, agent_data: GUIAgentData) -> GUIAgentState:
        """Handle the processing action state: execute the action."""
        if agent_data.current_action is None:
            return GUIAgentState.TERMINATED

        with simple_timer("action_execution", agent_data.metrics):
            # Get action parameters from ParsedAction
            action_params = agent_data.current_action.to_dict()
            action_type = agent_data.current_action.action_type

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

            # Build observation message (using user role for interaction format)
            observation_content = [{"type": "text", "text": f"Observation: {result_text}"}]
            if new_screenshot is not None and self.return_screenshot:
                observation_content.insert(0, {"type": "image"})

            observation_message = {"role": "user", "content": observation_content}
            agent_data.messages.append(observation_message)

            # Tokenize observation message
            response_ids = await self.apply_chat_template(
                [observation_message],
                images=[new_screenshot] if new_screenshot is not None else None,
                videos=None,
                remove_system_prompt=True,
            )

            # Check if adding response would exceed limit
            if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
                return GUIAgentState.TERMINATED

            # Update state
            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [0] * len(response_ids)  # Observation responses are masked
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
