#!/usr/bin/env python3
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
Prepare GUI Agent training data in the format required by VERL.

This script converts GUI Agent datasets (like AITW, MobileAgent-Bench, etc.)
to the parquet format required for VERL training.

Dataset Format:
--------------
Each sample should have:
- instruction: The task instruction (e.g., "Search for Musk in X and go to his homepage")
- screenshots: List of screenshot images for each step
- actions: List of action dicts (e.g., {"action": "click", "coordinate": [x, y]})

Output Format (parquet):
-----------------------
- raw_prompt: List of messages following the chat format
- extra_info: Dict containing screenshots, expected_actions, instruction
- images: List of PIL images (initial screenshot)
- ground_truth: JSON string of ground truth actions for reward computation

Usage:
------
python prepare_gui_agent_data.py \
    --input_dir /path/to/gui_agent_data \
    --output_dir /path/to/output \
    --screen_width 999 \
    --screen_height 999
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm


def load_image(image_path: str) -> Image.Image:
    """Load an image from file."""
    return Image.open(image_path).convert("RGB")


def encode_image_base64(image: Image.Image) -> str:
    """Encode image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def resize_image(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Resize image to target dimensions while maintaining aspect ratio."""
    # Calculate scaling factor
    width_ratio = target_width / image.width
    height_ratio = target_height / image.height
    ratio = min(width_ratio, height_ratio)

    new_width = int(image.width * ratio)
    new_height = int(image.height * ratio)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def rescale_coordinates(
    coord: list[int],
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int,
) -> list[int]:
    """Rescale coordinates from original to target dimensions."""
    x_scale = target_width / original_width
    y_scale = target_height / original_height
    return [int(coord[0] * x_scale), int(coord[1] * y_scale)]


def build_chat_messages(instruction: str, initial_screenshot: Image.Image) -> list[dict]:
    """Build chat messages for GUI Agent."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Task: {instruction}"},
            ],
        }
    ]
    return messages


def process_sample(
    sample: dict[str, Any],
    screen_width: int,
    screen_height: int,
    image_dir: str = None,
) -> dict[str, Any]:
    """
    Process a single sample into VERL format.

    Args:
        sample: Raw sample dict with instruction, screenshots, actions.
        screen_width: Target screen width for coordinate normalization.
        screen_height: Target screen height for coordinate normalization.
        image_dir: Base directory for image paths.

    Returns:
        Processed sample dict in VERL format.
    """
    instruction = sample.get("instruction", "")

    # Load screenshots
    screenshots = []
    screenshot_paths = sample.get("screenshots", sample.get("images", []))

    if isinstance(screenshot_paths, str):
        screenshot_paths = [screenshot_paths]

    for path in screenshot_paths:
        if image_dir:
            path = os.path.join(image_dir, path)
        if os.path.exists(path):
            img = load_image(path)
            # Get original dimensions for coordinate rescaling
            orig_width, orig_height = img.width, img.height
            # Resize to target dimensions
            img = resize_image(img, screen_width, screen_height)
            screenshots.append(img)

    # Process actions - rescale coordinates
    actions = sample.get("actions", [])
    processed_actions = []

    for action in actions:
        processed_action = action.copy()

        # Rescale coordinates if present
        if "coordinate" in action:
            processed_action["coordinate"] = rescale_coordinates(
                action["coordinate"],
                orig_width,
                orig_height,
                screen_width,
                screen_height,
            )
        if "coordinate2" in action:
            processed_action["coordinate2"] = rescale_coordinates(
                action["coordinate2"],
                orig_width,
                orig_height,
                screen_width,
                screen_height,
            )

        processed_actions.append(processed_action)

    # Build chat messages
    initial_screenshot = screenshots[0] if screenshots else None
    raw_prompt = build_chat_messages(instruction, initial_screenshot)

    # Build extra_info for agent loop
    extra_info = {
        "instruction": instruction,
        "expected_actions": processed_actions,
        "screen_width": screen_width,
        "screen_height": screen_height,
    }

    # Build ground_truth for reward computation (first action)
    ground_truth = json.dumps(processed_actions[0]) if processed_actions else "{}"

    return {
        "raw_prompt": raw_prompt,
        "extra_info": extra_info,
        "images": [initial_screenshot] if initial_screenshot else [],
        "ground_truth": ground_truth,
        "instruction": instruction,
    }


def load_aitw_dataset(data_dir: str) -> list[dict]:
    """
    Load AITW (Android in the Wild) dataset.

    Expected format:
    - Each sample is a directory containing:
      - metadata.json: {"instruction": "...", "actions": [...]}
      - screenshots/: Directory with step_0.png, step_1.png, etc.
    """
    samples = []
    data_path = Path(data_dir)

    for sample_dir in tqdm(list(data_path.iterdir()), desc="Loading AITW"):
        if not sample_dir.is_dir():
            continue

        metadata_path = sample_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Get screenshots
        screenshots_dir = sample_dir / "screenshots"
        if screenshots_dir.exists():
            screenshot_files = sorted(screenshots_dir.glob("*.png"))
            screenshots = [str(f) for f in screenshot_files]
        else:
            screenshots = []

        samples.append({
            "instruction": metadata.get("instruction", ""),
            "screenshots": screenshots,
            "actions": metadata.get("actions", []),
        })

    return samples


def load_jsonl_dataset(data_file: str) -> list[dict]:
    """
    Load dataset from JSONL file.

    Expected format (each line):
    {"instruction": "...", "screenshots": ["path1.png", ...], "actions": [...]}
    """
    samples = []

    with open(data_file, "r") as f:
        for line in tqdm(f, desc="Loading JSONL"):
            if line.strip():
                samples.append(json.loads(line))

    return samples


def create_demo_dataset(output_dir: str, num_samples: int = 100) -> list[dict]:
    """
    Create a demo dataset for testing.

    This creates synthetic samples for testing the training pipeline.
    """
    import random

    samples = []

    instructions = [
        "Open the Settings app and enable dark mode",
        "Search for 'Python tutorial' in the browser",
        "Send a message 'Hello' to the first contact",
        "Open the camera app and take a photo",
        "Navigate to the home screen",
        "Open the app drawer and find the calculator",
        "Scroll down and find the WiFi settings",
        "Type 'Hello World' in the search box",
    ]

    action_templates = [
        {"action": "click", "coordinate": [500, 500]},
        {"action": "type", "text": "search query"},
        {"action": "swipe", "coordinate": [500, 800], "coordinate2": [500, 200]},
        {"action": "system_button", "button": "Home"},
        {"action": "long_press", "coordinate": [500, 500], "time": 1.0},
        {"action": "terminate", "status": "success"},
    ]

    for i in range(num_samples):
        instruction = random.choice(instructions)

        # Generate random actions
        num_actions = random.randint(2, 5)
        actions = []
        for _ in range(num_actions - 1):
            action = random.choice(action_templates[:-1]).copy()
            if "coordinate" in action:
                action["coordinate"] = [
                    random.randint(100, 900),
                    random.randint(100, 900),
                ]
            if "coordinate2" in action:
                action["coordinate2"] = [
                    random.randint(100, 900),
                    random.randint(100, 900),
                ]
            if "text" in action:
                action["text"] = f"text_{i}_{len(actions)}"
            actions.append(action)

        # Add terminate action
        actions.append({"action": "terminate", "status": "success"})

        # Create dummy screenshot paths
        screenshots = [f"demo_{i}_step_{j}.png" for j in range(len(actions))]

        samples.append({
            "instruction": instruction,
            "screenshots": screenshots,
            "actions": actions,
        })

    # Save demo data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "demo_data.jsonl", "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Created demo dataset with {num_samples} samples at {output_path / 'demo_data.jsonl'}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare GUI Agent training data")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory containing raw GUI Agent data",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input JSONL file containing GUI Agent data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed parquet files",
    )
    parser.add_argument(
        "--screen_width",
        type=int,
        default=999,
        help="Target screen width (default: 999)",
    )
    parser.add_argument(
        "--screen_height",
        type=int,
        default=999,
        help="Target screen height (default: 999)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of data for training (default: 0.9)",
    )
    parser.add_argument(
        "--create_demo",
        action="store_true",
        help="Create a demo dataset for testing",
    )
    parser.add_argument(
        "--demo_samples",
        type=int,
        default=100,
        help="Number of demo samples to create (default: 100)",
    )

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.create_demo:
        samples = create_demo_dataset(args.output_dir, args.demo_samples)
        image_dir = None
    elif args.input_file:
        samples = load_jsonl_dataset(args.input_file)
        image_dir = os.path.dirname(args.input_file)
    elif args.input_dir:
        samples = load_aitw_dataset(args.input_dir)
        image_dir = args.input_dir
    else:
        print("Error: Must specify --input_dir, --input_file, or --create_demo")
        return

    print(f"Loaded {len(samples)} samples")

    # Process samples
    processed_samples = []
    for sample in tqdm(samples, desc="Processing samples"):
        try:
            processed = process_sample(
                sample,
                args.screen_width,
                args.screen_height,
                image_dir,
            )
            processed_samples.append(processed)
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

    print(f"Successfully processed {len(processed_samples)} samples")

    # Split into train/test
    import random
    random.shuffle(processed_samples)

    split_idx = int(len(processed_samples) * args.train_ratio)
    train_samples = processed_samples[:split_idx]
    test_samples = processed_samples[split_idx:]

    print(f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")

    # Convert to DataFrame and save as parquet
    def samples_to_dataframe(samples: list[dict]) -> pd.DataFrame:
        """Convert samples to DataFrame, serializing complex types."""
        records = []
        for sample in samples:
            record = {
                "raw_prompt": sample["raw_prompt"],
                "extra_info": sample["extra_info"],
                "ground_truth": sample["ground_truth"],
                "instruction": sample["instruction"],
            }
            # Handle images separately (store as list of bytes)
            if sample.get("images"):
                image_bytes_list = []
                for img in sample["images"]:
                    if img is not None:
                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        image_bytes_list.append(buffer.getvalue())
                record["images"] = image_bytes_list
            else:
                record["images"] = []

            records.append(record)

        return pd.DataFrame(records)

    if train_samples:
        train_df = samples_to_dataframe(train_samples)
        train_path = output_path / "train.parquet"
        train_df.to_parquet(train_path)
        print(f"Saved training data to {train_path}")

    if test_samples:
        test_df = samples_to_dataframe(test_samples)
        test_path = output_path / "test.parquet"
        test_df.to_parquet(test_path)
        print(f"Saved test data to {test_path}")


if __name__ == "__main__":
    main()
