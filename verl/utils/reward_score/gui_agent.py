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
Reward functions for GUI Agent training.

This module provides reward computation functions for evaluating
GUI Agent trajectories, including:
1. Action accuracy reward
2. Coordinate proximity reward
3. Task completion reward
4. Format compliance reward
"""

import json
import re
from typing import Any, Optional


def extract_tool_call(response_str: str) -> Optional[dict]:
    """
    Extract tool call from response string.

    Args:
        response_str: The model's response string.

    Returns:
        Parsed tool call dict or None if not found.
    """
    # Try to extract tool_call from XML tags
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    match = re.search(pattern, response_str, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

    return None


def action_type_reward(predicted_action: str, ground_truth_action: str) -> float:
    """
    Compute reward based on action type matching.

    Args:
        predicted_action: The predicted action type.
        ground_truth_action: The ground truth action type.

    Returns:
        1.0 if actions match, 0.0 otherwise.
    """
    return 1.0 if predicted_action == ground_truth_action else 0.0


def coordinate_reward(
    predicted_coord: list[int],
    ground_truth_coord: list[int],
    screen_width: int = 999,
    screen_height: int = 999,
    threshold: float = 0.1,
) -> float:
    """
    Compute reward based on coordinate proximity.

    Uses normalized distance with exponential decay for smooth reward.

    Args:
        predicted_coord: Predicted [x, y] coordinates.
        ground_truth_coord: Ground truth [x, y] coordinates.
        screen_width: Screen width for normalization.
        screen_height: Screen height for normalization.
        threshold: Distance threshold (as fraction of screen diagonal) for full reward.

    Returns:
        Reward between 0 and 1 based on coordinate accuracy.
    """
    if len(predicted_coord) < 2 or len(ground_truth_coord) < 2:
        return 0.0

    # Calculate Euclidean distance
    dx = predicted_coord[0] - ground_truth_coord[0]
    dy = predicted_coord[1] - ground_truth_coord[1]
    distance = (dx**2 + dy**2) ** 0.5

    # Normalize by screen diagonal
    diagonal = (screen_width**2 + screen_height**2) ** 0.5
    normalized_distance = distance / diagonal

    # Full reward if within threshold
    if normalized_distance <= threshold:
        return 1.0

    # Exponential decay for larger distances
    return max(0.0, 1.0 - (normalized_distance - threshold) / (1.0 - threshold))


def text_similarity_reward(predicted_text: str, ground_truth_text: str) -> float:
    """
    Compute reward based on text similarity.

    Args:
        predicted_text: The predicted text.
        ground_truth_text: The ground truth text.

    Returns:
        Similarity score between 0 and 1.
    """
    if not ground_truth_text and not predicted_text:
        return 1.0
    if not ground_truth_text or not predicted_text:
        return 0.0

    # Exact match
    if predicted_text == ground_truth_text:
        return 1.0

    # Case-insensitive match
    if predicted_text.lower() == ground_truth_text.lower():
        return 0.9

    # Character-level similarity (Levenshtein-like)
    max_len = max(len(predicted_text), len(ground_truth_text))
    common = sum(1 for a, b in zip(predicted_text, ground_truth_text) if a == b)

    return common / max_len


def format_reward(response_str: str) -> float:
    """
    Compute reward based on response format compliance.

    Expected format:
    1) Thought: <thought>
    2) Action: <action>
    3) <tool_call>...</tool_call>

    Args:
        response_str: The model's response string.

    Returns:
        Format compliance score between 0 and 1.
    """
    score = 0.0

    # Check for Thought section
    if re.search(r"Thought:", response_str, re.IGNORECASE):
        score += 0.25

    # Check for Action section
    if re.search(r"Action:", response_str, re.IGNORECASE):
        score += 0.25

    # Check for tool_call tags
    if re.search(r"<tool_call>", response_str) and re.search(r"</tool_call>", response_str):
        score += 0.25

    # Check for valid JSON in tool_call
    tool_call = extract_tool_call(response_str)
    if tool_call is not None:
        score += 0.25

    return score


def compute_action_reward(
    predicted: dict[str, Any],
    ground_truth: dict[str, Any],
    screen_width: int = 999,
    screen_height: int = 999,
) -> float:
    """
    Compute reward for a single action prediction.

    Args:
        predicted: Predicted action dict with 'action', 'coordinate', etc.
        ground_truth: Ground truth action dict.
        screen_width: Screen width for coordinate normalization.
        screen_height: Screen height for coordinate normalization.

    Returns:
        Action reward between 0 and 1.
    """
    predicted_action = predicted.get("action", "")
    ground_truth_action = ground_truth.get("action", "")

    # Action type must match for any reward
    if predicted_action != ground_truth_action:
        return 0.0

    # For coordinate-based actions
    if predicted_action in ["click", "long_press", "swipe"]:
        pred_coord = predicted.get("coordinate", [0, 0])
        gt_coord = ground_truth.get("coordinate", [0, 0])

        coord_reward = coordinate_reward(
            pred_coord, gt_coord, screen_width, screen_height
        )

        # For swipe, also check second coordinate
        if predicted_action == "swipe":
            pred_coord2 = predicted.get("coordinate2", [0, 0])
            gt_coord2 = ground_truth.get("coordinate2", [0, 0])
            coord2_reward = coordinate_reward(
                pred_coord2, gt_coord2, screen_width, screen_height
            )
            return 0.5 * coord_reward + 0.5 * coord2_reward

        return coord_reward

    # For type action
    if predicted_action == "type":
        pred_text = predicted.get("text", "")
        gt_text = ground_truth.get("text", "")
        return text_similarity_reward(pred_text, gt_text)

    # For system_button action
    if predicted_action == "system_button":
        pred_button = predicted.get("button", "")
        gt_button = ground_truth.get("button", "")
        return 1.0 if pred_button == gt_button else 0.0

    # For terminate action
    if predicted_action == "terminate":
        pred_status = predicted.get("status", "")
        gt_status = ground_truth.get("status", "")
        return 1.0 if pred_status == gt_status else 0.0

    # For wait action
    if predicted_action == "wait":
        # Wait time doesn't need to match exactly
        return 1.0

    # For answer action
    if predicted_action == "answer":
        pred_text = predicted.get("text", "")
        gt_text = ground_truth.get("text", "")
        return text_similarity_reward(pred_text, gt_text)

    return 1.0  # Other matching actions


def compute_score(
    predict_str: str,
    ground_truth: dict[str, Any],
    screen_width: int = 999,
    screen_height: int = 999,
    format_weight: float = 0.2,
    action_weight: float = 0.8,
) -> float:
    """
    Compute the overall reward score for a GUI Agent response.

    Args:
        predict_str: The model's response string.
        ground_truth: Ground truth dict containing 'action' and other fields.
        screen_width: Screen width for coordinate normalization.
        screen_height: Screen height for coordinate normalization.
        format_weight: Weight for format compliance (default: 0.2).
        action_weight: Weight for action accuracy (default: 0.8).

    Returns:
        Overall reward score between 0 and 1.
    """
    # Compute format reward
    fmt_reward = format_reward(predict_str)

    # Extract predicted action
    tool_call = extract_tool_call(predict_str)
    if tool_call is None:
        # No valid tool call, only format reward
        return format_weight * fmt_reward

    # Get predicted arguments
    predicted = tool_call.get("arguments", {})

    # Compute action reward
    act_reward = compute_action_reward(
        predicted, ground_truth, screen_width, screen_height
    )

    return format_weight * fmt_reward + action_weight * act_reward


def compute_trajectory_score(
    responses: list[str],
    ground_truth_actions: list[dict[str, Any]],
    screen_width: int = 999,
    screen_height: int = 999,
    completion_bonus: float = 0.5,
) -> float:
    """
    Compute the reward score for a full trajectory.

    Args:
        responses: List of model response strings.
        ground_truth_actions: List of ground truth action dicts.
        screen_width: Screen width for coordinate normalization.
        screen_height: Screen height for coordinate normalization.
        completion_bonus: Bonus reward for completing all steps correctly.

    Returns:
        Trajectory reward score.
    """
    if not ground_truth_actions:
        return 0.0

    total_score = 0.0
    num_correct = 0

    for i, response in enumerate(responses):
        if i >= len(ground_truth_actions):
            break

        score = compute_score(
            response,
            ground_truth_actions[i],
            screen_width,
            screen_height,
        )
        total_score += score

        if score > 0.5:
            num_correct += 1

    # Average score
    avg_score = total_score / len(ground_truth_actions)

    # Completion bonus if all actions are correct
    if num_correct == len(ground_truth_actions):
        avg_score += completion_bonus

    return min(1.0, avg_score)


# Alias for compatibility with verl reward system
def default_compute_score(predict_str: str, ground_truth: Any, **kwargs) -> float:
    """Default compute_score function for verl integration."""
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return 0.0

    return compute_score(
        predict_str,
        ground_truth,
        screen_width=kwargs.get("screen_width", 999),
        screen_height=kwargs.get("screen_height", 999),
    )
