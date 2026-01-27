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
Tool Agent Loop for multi-turn agent interactions.

This agent loop implements multi-turn interactions with tools, supporting:
1. Standard tool calling with JSON format (hermes, gpt-oss)
2. GUI/Mobile interaction format (interaction) for GUI agents

The loop uses a state machine to manage the interaction flow:
PENDING -> GENERATING -> PROCESSING_TOOLS -> GENERATING -> ... -> TERMINATED
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
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def build_gui_agent_system_prompt(screen_width: int = 999, screen_height: int = 999) -> str:
    """Build the system prompt for GUI Agent using interaction format."""
    return f"""You are a GUI agent that interacts with a mobile device touchscreen.

# Screen Information
- Screen resolution: {screen_width}x{screen_height}
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


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.routed_experts: Optional[Any] = None
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        # GUI Agent specific state
        self.is_terminated = False
        self.terminate_status: Optional[str] = None  # "success" or "failure"
        self.action_history: list[dict] = []

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    """
    Tool Agent Loop for multi-turn agent interactions.

    Supports two modes:
    1. Standard tool calling (format: hermes, gpt-oss) - uses <tool_call> XML format
    2. GUI/Mobile interaction (format: interaction) - uses Action: action_name(params) format

    For GUI Agent mode, set format="interaction" in multi_turn config.
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        config = trainer_config.config

        # Initialize tools from config file
        self.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        self.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        self.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        self.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format
        self.tool_parser = ToolParser.get_tool_parser(self.tool_parser_name, self.tokenizer)

        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length

        # GUI Agent specific configurations
        self.is_gui_agent_mode = self.tool_parser_name == "interaction"
        gui_config = config.actor_rollout_ref.rollout.get("gui_agent", {})
        self.screen_width = gui_config.get("screen_width", 999)
        self.screen_height = gui_config.get("screen_height", 999)
        self.return_screenshot = gui_config.get("return_screenshot", True)

        # Initialize interactions from config file
        self.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # GUI Agent: get extra info for screenshots and expected actions
        extra_info = kwargs.get("extra_info", {})
        if self.is_gui_agent_mode:
            # For GUI Agent, pass screenshots and expected_actions to tools_kwargs
            screenshots = extra_info.get("screenshots", images if images else [])
            expected_actions = extra_info.get("expected_actions", [])
            if "mobile_use" not in tools_kwargs:
                tools_kwargs["mobile_use"] = {}
            tools_kwargs["mobile_use"]["create_kwargs"] = {
                "screenshots": screenshots if isinstance(screenshots, list) else [screenshots],
                "expected_actions": expected_actions,
            }

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = extra_info.get("interaction_kwargs", {})
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)

        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        # Store instruction for GUI Agent mode
        if self.is_gui_agent_mode:
            agent_data.extra_fields["instruction"] = kwargs.get("instruction", "")

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_output = {}
        if agent_data.image_data is not None:
            multi_modal_output["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_output["videos"] = agent_data.video_data

        # Calculate final reward for GUI Agent mode
        reward_score = None
        if self.is_gui_agent_mode:
            reward_score = self._calculate_gui_agent_reward(agent_data)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            routed_experts=agent_data.routed_experts,
            multi_modal_data=multi_modal_output,
            reward_score=reward_score,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=AgentLoopMetrics(**agent_data.metrics),
            extra_fields={},
        )

        # Update extra fields
        output.extra_fields.update({
            "turn_scores": agent_data.turn_scores,
            "tool_rewards": agent_data.tool_rewards,
        })

        # GUI Agent specific extra fields
        if self.is_gui_agent_mode:
            output.extra_fields.update({
                "action_history": agent_data.action_history,
                "terminate_status": agent_data.terminate_status,
            })

        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        if self.is_gui_agent_mode:
            # For GUI Agent, build system prompt and user query
            system_content = build_gui_agent_system_prompt(self.screen_width, self.screen_height)

            # Build user query with task progress
            instruction = agent_data.extra_fields.get("instruction", "")
            task_history = self._build_task_history(agent_data.action_history)
            user_query = f"The user query: {instruction}.\n"
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

            # Apply chat template without tools for interaction format
            prompt_ids = await self.apply_chat_template(
                agent_data.messages,
                tools=None,
                images=agent_data.image_data,
                videos=agent_data.video_data,
            )
        else:
            # Standard tool calling mode
            prompt_ids = await self.apply_chat_template(
                agent_data.messages,
                tools=self.tool_schemas,
                images=agent_data.image_data,
                videos=agent_data.video_data,
            )

        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    def _build_task_history(self, action_history: list[dict]) -> str:
        """Build task progress history string for GUI Agent."""
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

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
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
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        for tool_response, tool_reward, tool_extra in responses:
            # GUI Agent: track action history and check for termination
            if self.is_gui_agent_mode:
                action_info = tool_extra.get("action", None)
                if action_info:
                    # Get the arguments from the tool call
                    for tool_call in agent_data.tool_calls:
                        try:
                            args = json.loads(tool_call.arguments)
                            agent_data.action_history.append(args)
                            # Check for terminate action
                            if args.get("action") == "terminate":
                                agent_data.is_terminated = True
                                agent_data.terminate_status = args.get("status", "success")
                            break
                        except json.JSONDecodeError:
                            pass

            # Create message from tool response
            if self.is_gui_agent_mode:
                # GUI Agent: use "user" role with "Observation:" prefix for interaction format
                observation_text = f"Observation: {tool_response.text}" if tool_response.text else "Observation: Action executed."
                if tool_response.image or tool_response.video:
                    if not getattr(self.processor, "image_processor", None):
                        raise ValueError(
                            "Multimedia data can only be processed by `processor`, but the processor is None. "
                            "This error is often caused if you are using a LLM model but your tool returns multimodal "
                            "data. Please use a VLM as the base model."
                        )
                    content = []
                    if tool_response.image:
                        content.append({"type": "image"})
                    content.append({"type": "text", "text": observation_text})
                    message = {"role": "user", "content": content}
                else:
                    message = {"role": "user", "content": observation_text}
            elif tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        # GUI Agent: check if terminated
        if self.is_gui_agent_mode and agent_data.is_terminated:
            return AgentState.TERMINATED

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            response_ids = await self.apply_chat_template(
                add_messages,
                images=new_images_this_turn,  # Using local variable
                videos=None,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def _initialize_interactions(self, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        return interaction_map

    def _calculate_gui_agent_reward(self, agent_data: AgentData) -> float:
        """Calculate the final reward for GUI Agent trajectory."""
        # If terminated successfully, give high reward
        if agent_data.terminate_status == "success":
            return 1.0

        # If terminated with failure, give low reward
        if agent_data.terminate_status == "failure":
            return 0.0

        # Calculate based on tool rewards (action step rewards)
        if agent_data.tool_rewards:
            # Average of step rewards
            avg_reward = sum(agent_data.tool_rewards) / len(agent_data.tool_rewards)
            # Bonus for completing more steps correctly
            correct_actions = sum(1 for r in agent_data.tool_rewards if r > 0.5)
            num_expected = len(agent_data.tools_kwargs.get("mobile_use", {}).get("create_kwargs", {}).get("expected_actions", []))
            completion_bonus = correct_actions / max(num_expected, 1) if num_expected > 0 else 0
            return 0.5 * avg_reward + 0.5 * completion_bonus

        return 0.0
