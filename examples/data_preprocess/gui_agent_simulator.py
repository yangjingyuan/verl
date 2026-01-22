# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Generate synthetic GUI agent training data in parquet format.
"""

import argparse
import json
import os
import random

import datasets

# System prompt for GUI agent with mobile_use tool
SYSTEM_PROMPT = """

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `answer`: Output the answer.\\n* `system_button`: Press the system button.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Thought: one concise sentence explaining the next move (no multi-step reasoning).
2) Action: a short imperative describing what to do in the UI.
3) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Thought, Action, <tool_call>.
- Be brief: one sentence for Thought, one for Action.
- Do not output anything else outside those three parts.
- If finishing, use action=terminate in the tool call."""

# 50 specific GUI operation tasks
GUI_TASKS = [
    # Search and navigation
    "Search for 'Machine Learning' on YouTube and open the first video",
    "Find 'New York weather' on Google and check today's forecast",
    "Search for 'pizza delivery near me' on Maps and open the top result",
    "Look up 'Python tutorial' on X and go to the first post",
    "Search for 'smartphone reviews' on Amazon and view the top rated product",
    # App opening and navigation
    "Open Gmail and navigate to the Promotions folder",
    "Launch WhatsApp and go to the settings page",
    "Open Instagram and navigate to your profile page",
    "Start the Camera app and switch to video mode",
    "Open Settings and go to the Wi-Fi configuration page",
    # Messaging and communication
    "Send a message 'Meeting at 3pm today' to John on WhatsApp",
    "Compose an email to alice@example.com with subject 'Project Update'",
    "Reply 'Thanks!' to the latest message from Mom",
    "Create a new group chat named 'Team Discussion' on Telegram",
    "Send a voice message to Sarah on Messenger",
    # Media and content
    "Take a photo and share it via Instagram",
    "Record a 10-second video and save it to gallery",
    "Play the song 'Imagine' on Spotify",
    "Pause the current video and adjust volume to 50%",
    "Take a screenshot and send it via email",
    # System operations
    "Turn on Airplane mode from settings",
    "Adjust screen brightness to maximum",
    "Enable Do Not Disturb mode for 2 hours",
    "Connect to Wi-Fi network 'Home-5G'",
    "Check battery usage statistics",
    # Social media
    "Like the latest post on Instagram feed",
    "Share the current article on Facebook",
    "Retweet the top tweet with a comment",
    "Post a status update 'Beautiful day!' on Facebook",
    "Follow the account @TechNews on X",
    # E-commerce
    "Add 'Wireless Mouse' to Amazon shopping cart",
    "Check order status for recent purchase",
    "Apply coupon code 'SAVE20' at checkout",
    "Compare prices for 'Running Shoes' across different sellers",
    "Write a 5-star review for recently purchased item",
    # Productivity
    "Create a new calendar event for tomorrow at 2pm",
    "Set a reminder 'Call dentist' for 10am",
    "Add 'Buy groceries' to the Notes app",
    "Mark the task 'Finish report' as completed in ToDo list",
    "Schedule an alarm for 7am tomorrow",
    # File management
    "Download the PDF file from the current webpage",
    "Share the current location via Messages",
    "Save the current article to reading list",
    "Create a new folder named 'Vacation Photos' in Files",
    "Move the selected photo to 'Favorites' album",
    # Advanced interactions
    "Scroll down to the bottom of the page and load more content",
    "Swipe left to delete the email notification",
    "Long press on the app icon to show quick actions",
    "Pinch to zoom in on the current map view",
    "Double tap to like the Instagram photo",
]


def make_map_fn(split):
    def process_fn(example, idx):
        task = example["task"]
        data = {
            "data_source": "gui_agent_simulator",
            "agent_name": "tool_agent",
            "prompt": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": task,
                },
            ],
            "ability": "gui_interaction",
            "reward_model": {"style": "rule", "ground_truth": task},
            "extra_info": {
                "split": split,
                "index": idx,
                "task": task,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "mobile_use": {
                        "create_kwargs": {"task": task},
                        # "execute_kwargs": {},
                        # "calc_reward_kwargs": {},
                        # "release_kwargs": {},
                    },
                },
                "interaction_kwargs": {
                    "query": task,
                },
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--local_save_dir", default="~/data/gui_agent", help="Save directory")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    random.seed(args.seed)
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    # Generate samples by randomly selecting from tasks
    train_samples = [{"task": random.choice(GUI_TASKS)} for _ in range(args.num_samples)]
    test_samples = [{"task": random.choice(GUI_TASKS)} for _ in range(args.num_samples // 10)]

    train_dataset = datasets.Dataset.from_list(train_samples)
    test_dataset = datasets.Dataset.from_list(test_samples)

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    # Save example for reference
    with open(os.path.join(local_save_dir, "train_example.json"), "w") as f:
        json.dump(train_dataset[0], f, indent=2)

    print(f"Saved {len(train_dataset)} train samples and {len(test_dataset)} test samples to {local_save_dir}")
    print(f"Example: {train_dataset[0]}")

    if args.hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs

        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
