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
Mobile Use Tool for GUI Agent.

This tool implements the mobile device interaction interface following
the Qwen3-VL MobileAgent approach. It supports actions like click, swipe,
type, long_press, etc.
"""

import asyncio
import logging
import os
from enum import Enum
from typing import Any, Optional

from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MobileAction(str, Enum):
    """Supported mobile actions."""

    CLICK = "click"
    LONG_PRESS = "long_press"
    SWIPE = "swipe"
    TYPE = "type"
    ANSWER = "answer"
    SYSTEM_BUTTON = "system_button"
    WAIT = "wait"
    TERMINATE = "terminate"


class SystemButton(str, Enum):
    """System buttons for mobile device."""

    BACK = "Back"
    HOME = "Home"
    MENU = "Menu"
    ENTER = "Enter"


class TerminateStatus(str, Enum):
    """Task termination status."""

    SUCCESS = "success"
    FAILURE = "failure"


# Default screen resolution (can be configured per instance)
DEFAULT_SCREEN_WIDTH = 999
DEFAULT_SCREEN_HEIGHT = 999


def get_mobile_use_tool_schema(screen_width: int = DEFAULT_SCREEN_WIDTH, screen_height: int = DEFAULT_SCREEN_HEIGHT):
    """Get the OpenAI function tool schema for mobile_use."""
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="mobile_use",
            description=(
                f"Use a touchscreen to interact with a mobile device, and take screenshots.\n"
                f"* This is an interface to a mobile device with touchscreen. You can perform actions "
                f"like clicking, typing, swiping, etc.\n"
                f"* Some applications may take time to start or process actions, so you may need to "
                f"wait and take successive screenshots to see the results of your actions.\n"
                f"* The screen's resolution is {screen_width}x{screen_height}.\n"
                f"* Make sure to click any buttons, links, icons, etc with the cursor tip in the "
                f"center of the element. Don't click boxes on their edges unless asked."
            ),
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "action": OpenAIFunctionPropertySchema(
                        type="string",
                        description=(
                            "The action to perform. The available actions are:\n"
                            "* `click`: Click the point on the screen with coordinate (x, y).\n"
                            "* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\n"
                            "* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\n"
                            "* `type`: Input the specified text into the activated input box.\n"
                            "* `answer`: Output the answer.\n"
                            "* `system_button`: Press the system button.\n"
                            "* `wait`: Wait specified seconds for the change to happen.\n"
                            "* `terminate`: Terminate the current task and report its completion status."
                        ),
                        enum=["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"],
                    ),
                    "coordinate": OpenAIFunctionPropertySchema(
                        type="array",
                        description=(
                            "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) "
                            "coordinates to move the mouse to. Required only by `action=click`, "
                            "`action=long_press`, and `action=swipe`."
                        ),
                    ),
                    "coordinate2": OpenAIFunctionPropertySchema(
                        type="array",
                        description=(
                            "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) "
                            "coordinates to move the mouse to. Required only by `action=swipe`."
                        ),
                    ),
                    "text": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Required only by `action=type` and `action=answer`.",
                    ),
                    "time": OpenAIFunctionPropertySchema(
                        type="number",
                        description="The seconds to wait. Required only by `action=long_press` and `action=wait`.",
                    ),
                    "button": OpenAIFunctionPropertySchema(
                        type="string",
                        description=(
                            "Back means returning to the previous interface, Home means returning to the desktop, "
                            "Menu means opening the application background menu, and Enter means pressing the enter. "
                            "Required only by `action=system_button`"
                        ),
                        enum=["Back", "Home", "Menu", "Enter"],
                    ),
                    "status": OpenAIFunctionPropertySchema(
                        type="string",
                        description="The status of the task. Required only by `action=terminate`.",
                        enum=["success", "failure"],
                    ),
                },
                required=["action"],
            ),
        ),
    )


class MobileUseEnvironment:
    """
    Abstract mobile environment interface.

    This class provides the interface for interacting with a mobile device.
    Subclass this to implement actual device interaction (e.g., Android ADB,
    iOS, emulator, or simulation).
    """

    def __init__(
        self,
        screen_width: int = DEFAULT_SCREEN_WIDTH,
        screen_height: int = DEFAULT_SCREEN_HEIGHT,
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self._terminated = False
        self._terminate_status: Optional[TerminateStatus] = None

    async def click(self, x: int, y: int) -> tuple[str, Optional[Image.Image]]:
        """Click at position (x, y)."""
        raise NotImplementedError("Subclass must implement click()")

    async def long_press(self, x: int, y: int, duration: float = 1.0) -> tuple[str, Optional[Image.Image]]:
        """Long press at position (x, y) for duration seconds."""
        raise NotImplementedError("Subclass must implement long_press()")

    async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5) -> tuple[str, Optional[Image.Image]]:
        """Swipe from (x1, y1) to (x2, y2)."""
        raise NotImplementedError("Subclass must implement swipe()")

    async def type_text(self, text: str) -> tuple[str, Optional[Image.Image]]:
        """Type text into the current input field."""
        raise NotImplementedError("Subclass must implement type_text()")

    async def press_system_button(self, button: SystemButton) -> tuple[str, Optional[Image.Image]]:
        """Press a system button (Back, Home, Menu, Enter)."""
        raise NotImplementedError("Subclass must implement press_system_button()")

    async def wait(self, seconds: float) -> tuple[str, Optional[Image.Image]]:
        """Wait for specified seconds."""
        await asyncio.sleep(seconds)
        screenshot = await self.get_screenshot()
        return f"Waited for {seconds} seconds.", screenshot

    async def get_screenshot(self) -> Optional[Image.Image]:
        """Get the current screenshot."""
        raise NotImplementedError("Subclass must implement get_screenshot()")

    def terminate(self, status: TerminateStatus) -> str:
        """Terminate the task with the given status."""
        self._terminated = True
        self._terminate_status = status
        return f"Task terminated with status: {status.value}"

    @property
    def is_terminated(self) -> bool:
        return self._terminated

    @property
    def terminate_status(self) -> Optional[TerminateStatus]:
        return self._terminate_status


class SimulatedMobileEnvironment(MobileUseEnvironment):
    """
    Simulated mobile environment for training.

    This environment simulates mobile device interactions without actual device.
    It can be used for:
    1. Training with pre-recorded screenshot sequences
    2. Evaluating model predictions against expected actions
    """

    def __init__(
        self,
        screen_width: int = DEFAULT_SCREEN_WIDTH,
        screen_height: int = DEFAULT_SCREEN_HEIGHT,
        screenshots: Optional[list[Image.Image]] = None,
        expected_actions: Optional[list[dict]] = None,
    ):
        super().__init__(screen_width, screen_height)
        self.screenshots = screenshots or []
        self.expected_actions = expected_actions or []
        self.current_step = 0
        self.action_history: list[dict] = []

    async def click(self, x: int, y: int) -> tuple[str, Optional[Image.Image]]:
        action = {"action": "click", "coordinate": [x, y]}
        self.action_history.append(action)
        self.current_step += 1
        screenshot = await self.get_screenshot()
        return f"Clicked at ({x}, {y}).", screenshot

    async def long_press(self, x: int, y: int, duration: float = 1.0) -> tuple[str, Optional[Image.Image]]:
        action = {"action": "long_press", "coordinate": [x, y], "time": duration}
        self.action_history.append(action)
        self.current_step += 1
        screenshot = await self.get_screenshot()
        return f"Long pressed at ({x}, {y}) for {duration} seconds.", screenshot

    async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5) -> tuple[str, Optional[Image.Image]]:
        action = {"action": "swipe", "coordinate": [x1, y1], "coordinate2": [x2, y2]}
        self.action_history.append(action)
        self.current_step += 1
        screenshot = await self.get_screenshot()
        return f"Swiped from ({x1}, {y1}) to ({x2}, {y2}).", screenshot

    async def type_text(self, text: str) -> tuple[str, Optional[Image.Image]]:
        action = {"action": "type", "text": text}
        self.action_history.append(action)
        self.current_step += 1
        screenshot = await self.get_screenshot()
        return f"Typed: {text}", screenshot

    async def press_system_button(self, button: SystemButton) -> tuple[str, Optional[Image.Image]]:
        action = {"action": "system_button", "button": button.value}
        self.action_history.append(action)
        self.current_step += 1
        screenshot = await self.get_screenshot()
        return f"Pressed {button.value} button.", screenshot

    async def get_screenshot(self) -> Optional[Image.Image]:
        """Return the screenshot for the current step."""
        if self.screenshots and self.current_step < len(self.screenshots):
            return self.screenshots[self.current_step]
        elif self.screenshots:
            return self.screenshots[-1]  # Return last screenshot if we've exceeded
        return None

    def get_step_reward(self) -> float:
        """
        Calculate step reward based on action history vs expected actions.

        Returns a reward between 0 and 1 based on action matching.
        """
        if not self.expected_actions or self.current_step == 0:
            return 0.0

        step_idx = self.current_step - 1
        if step_idx >= len(self.expected_actions):
            return 0.0

        actual = self.action_history[step_idx] if step_idx < len(self.action_history) else None
        expected = self.expected_actions[step_idx]

        if actual is None:
            return 0.0

        # Check if action type matches
        if actual.get("action") != expected.get("action"):
            return 0.0

        # For coordinate-based actions, check proximity
        if actual.get("action") in ["click", "long_press", "swipe"]:
            expected_coord = expected.get("coordinate", [0, 0])
            actual_coord = actual.get("coordinate", [0, 0])
            distance = ((expected_coord[0] - actual_coord[0]) ** 2 +
                       (expected_coord[1] - actual_coord[1]) ** 2) ** 0.5
            # Normalize by screen diagonal
            max_distance = (self.screen_width ** 2 + self.screen_height ** 2) ** 0.5
            return max(0.0, 1.0 - distance / max_distance)

        # For type action, check text similarity
        if actual.get("action") == "type":
            expected_text = expected.get("text", "")
            actual_text = actual.get("text", "")
            if expected_text == actual_text:
                return 1.0
            # Simple character-level similarity
            common = sum(1 for a, b in zip(actual_text, expected_text) if a == b)
            max_len = max(len(expected_text), len(actual_text), 1)
            return common / max_len

        return 1.0  # Other actions match by type


class MobileUseTool(BaseTool):
    """
    Mobile Use Tool for GUI Agent interactions.

    This tool handles mobile device interactions following the Qwen3-VL
    MobileAgent approach.
    """

    # Class-level instance storage for environment instances
    _instances: dict[str, MobileUseEnvironment] = {}

    def __init__(
        self,
        config: dict,
        tool_schema: Optional[OpenAIFunctionToolSchema] = None,
    ):
        self.screen_width = config.get("screen_width", DEFAULT_SCREEN_WIDTH)
        self.screen_height = config.get("screen_height", DEFAULT_SCREEN_HEIGHT)
        self.return_screenshot = config.get("return_screenshot", True)

        if tool_schema is None:
            tool_schema = get_mobile_use_tool_schema(self.screen_width, self.screen_height)

        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return get_mobile_use_tool_schema(self.screen_width, self.screen_height)

    async def create(
        self,
        instance_id: Optional[str] = None,
        create_kwargs: Optional[dict] = None,
    ) -> tuple[str, ToolResponse]:
        """Create a mobile environment instance."""
        from uuid import uuid4

        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = create_kwargs or {}

        # Get screenshots and expected actions from create_kwargs
        screenshots = create_kwargs.get("screenshots", [])
        expected_actions = create_kwargs.get("expected_actions", [])

        # Create simulated environment
        env = SimulatedMobileEnvironment(
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            screenshots=screenshots,
            expected_actions=expected_actions,
        )

        self._instances[instance_id] = env

        # Get initial screenshot
        initial_screenshot = await env.get_screenshot()

        return instance_id, ToolResponse(
            text="Mobile environment initialized.",
            image=[initial_screenshot] if initial_screenshot else None,
        )

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        """Execute a mobile action."""
        env = self._instances.get(instance_id)
        if env is None:
            return ToolResponse(text="Error: Environment not initialized."), 0.0, {}

        action = parameters.get("action")
        if not action:
            return ToolResponse(text="Error: No action specified."), 0.0, {}

        try:
            action_enum = MobileAction(action)
        except ValueError:
            return ToolResponse(text=f"Error: Unknown action '{action}'."), 0.0, {}

        result_text = ""
        screenshot = None
        reward = 0.0

        try:
            if action_enum == MobileAction.CLICK:
                coord = parameters.get("coordinate", [0, 0])
                if len(coord) < 2:
                    return ToolResponse(text="Error: coordinate requires [x, y]."), 0.0, {}
                result_text, screenshot = await env.click(int(coord[0]), int(coord[1]))

            elif action_enum == MobileAction.LONG_PRESS:
                coord = parameters.get("coordinate", [0, 0])
                duration = parameters.get("time", 1.0)
                if len(coord) < 2:
                    return ToolResponse(text="Error: coordinate requires [x, y]."), 0.0, {}
                result_text, screenshot = await env.long_press(int(coord[0]), int(coord[1]), float(duration))

            elif action_enum == MobileAction.SWIPE:
                coord1 = parameters.get("coordinate", [0, 0])
                coord2 = parameters.get("coordinate2", [0, 0])
                if len(coord1) < 2 or len(coord2) < 2:
                    return ToolResponse(text="Error: swipe requires coordinate and coordinate2."), 0.0, {}
                result_text, screenshot = await env.swipe(
                    int(coord1[0]), int(coord1[1]),
                    int(coord2[0]), int(coord2[1])
                )

            elif action_enum == MobileAction.TYPE:
                text = parameters.get("text", "")
                result_text, screenshot = await env.type_text(text)

            elif action_enum == MobileAction.ANSWER:
                text = parameters.get("text", "")
                result_text = f"Answer: {text}"

            elif action_enum == MobileAction.SYSTEM_BUTTON:
                button_str = parameters.get("button", "Back")
                try:
                    button = SystemButton(button_str)
                except ValueError:
                    return ToolResponse(text=f"Error: Unknown button '{button_str}'."), 0.0, {}
                result_text, screenshot = await env.press_system_button(button)

            elif action_enum == MobileAction.WAIT:
                duration = parameters.get("time", 1.0)
                result_text, screenshot = await env.wait(float(duration))

            elif action_enum == MobileAction.TERMINATE:
                status_str = parameters.get("status", "success")
                try:
                    status = TerminateStatus(status_str)
                except ValueError:
                    return ToolResponse(text=f"Error: Unknown status '{status_str}'."), 0.0, {}
                result_text = env.terminate(status)
                # Give reward based on final status
                reward = 1.0 if status == TerminateStatus.SUCCESS else 0.0

            # Calculate step reward for non-terminal actions
            if isinstance(env, SimulatedMobileEnvironment) and action_enum != MobileAction.TERMINATE:
                reward = env.get_step_reward()

        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            return ToolResponse(text=f"Error executing action: {str(e)}"), 0.0, {}

        # Build response
        response_kwargs = {"text": result_text}
        if self.return_screenshot and screenshot is not None:
            response_kwargs["image"] = [screenshot]

        return ToolResponse(**response_kwargs), reward, {"action": action}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the final reward for the trajectory."""
        env = self._instances.get(instance_id)
        if env is None:
            return 0.0

        # If terminated successfully, return 1.0
        if env.is_terminated and env.terminate_status == TerminateStatus.SUCCESS:
            return 1.0

        # For simulated environment, calculate based on action matching
        if isinstance(env, SimulatedMobileEnvironment):
            if not env.expected_actions:
                return 0.0

            # Calculate average reward over all steps
            total_reward = 0.0
            for i in range(len(env.action_history)):
                if i < len(env.expected_actions):
                    env.current_step = i + 1
                    total_reward += env.get_step_reward()

            return total_reward / max(len(env.expected_actions), 1)

        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the environment instance."""
        if instance_id in self._instances:
            del self._instances[instance_id]
