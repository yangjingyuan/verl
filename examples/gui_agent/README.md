# GUI Agent Training with GRPO

This example demonstrates how to train a GUI Agent for mobile device interaction using VERL's GRPO (Group Relative Policy Optimization) algorithm.

## Overview

The GUI Agent is based on the Qwen3-VL MobileAgent approach, which enables vision-language models to interact with mobile device screens through:

- **Visual Understanding**: Processing screenshots to understand UI elements
- **Action Prediction**: Predicting appropriate actions (click, swipe, type, etc.)
- **Multi-turn Interaction**: Handling sequential interactions with the device

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GUI Agent Training                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Dataset    │───▶│  Agent Loop  │───▶│   Reward     │  │
│  │  (Parquet)   │    │ (GUI Agent)  │    │  Function    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                    │          │
│         ▼                   ▼                    ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Screenshots │    │  vLLM/SGLang │    │    GRPO      │  │
│  │  + Actions   │    │   Rollout    │    │   Update     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                             │                    │          │
│                             ▼                    ▼          │
│                      ┌──────────────┐    ┌──────────────┐  │
│                      │  Qwen3-VL-8B │    │    FSDP      │  │
│                      │    Model     │◀───│   Training   │  │
│                      └──────────────┘    └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Mobile Use Tool (`verl/tools/mobile_use_tool.py`)

Implements the mobile device interaction interface:

- **Actions**: click, long_press, swipe, type, answer, system_button, wait, terminate
- **Coordinate System**: Normalized to screen dimensions (default 999x999)
- **Screenshot Handling**: Returns screenshots after actions for visual feedback

### 2. GUI Agent Loop (`verl/experimental/agent_loop/gui_agent_loop.py`)

Handles the multi-turn interaction workflow:

- State machine: PENDING → GENERATING → PROCESSING_ACTION → TERMINATED
- Tool call extraction and execution
- Response masking for GRPO training

### 3. Reward Function (`verl/utils/reward_score/gui_agent.py`)

Computes rewards based on:

- Action type accuracy
- Coordinate proximity (for click/swipe actions)
- Text similarity (for type actions)
- Format compliance (Thought → Action → tool_call)
- Task completion bonus

## Data Format

Training data should be in parquet format with the following columns:

```python
{
    "raw_prompt": [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Task: Search for Musk in X..."}
            ]
        }
    ],
    "extra_info": {
        "instruction": "Search for Musk in X...",
        "expected_actions": [
            {"action": "click", "coordinate": [500, 300]},
            {"action": "type", "text": "Musk"},
            {"action": "terminate", "status": "success"}
        ],
        "screen_width": 999,
        "screen_height": 999
    },
    "images": [<PIL.Image>],
    "ground_truth": "{\"action\": \"click\", \"coordinate\": [500, 300]}"
}
```

## Quick Start

### 1. Prepare Data

```bash
# Create demo dataset for testing
python examples/gui_agent/prepare_gui_agent_data.py \
    --output_dir $HOME/data/gui_agent \
    --create_demo \
    --demo_samples 1000

# Or convert your own dataset
python examples/gui_agent/prepare_gui_agent_data.py \
    --input_file /path/to/your/data.jsonl \
    --output_dir $HOME/data/gui_agent \
    --screen_width 999 \
    --screen_height 999
```

### 2. Train on NPU (Ascend)

```bash
# Set data directory
export DATA_DIR=$HOME/data/gui_agent
export MODEL_PATH=Qwen/Qwen3-VL-8B

# Run training
bash examples/gui_agent/run_qwen3_vl_8b_gui_agent_npu.sh
```

### 3. Train on GPU (NVIDIA)

```bash
# Set data directory
export DATA_DIR=$HOME/data/gui_agent
export MODEL_PATH=Qwen/Qwen3-VL-8B

# Run training
bash examples/gui_agent/run_qwen3_vl_8b_gui_agent_gpu.sh
```

## Configuration

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data.train_batch_size` | Number of prompts per batch | 256 (NPU) / 512 (GPU) |
| `actor_rollout_ref.rollout.n` | Samples per prompt (GRPO group size) | 5 |
| `actor_rollout_ref.actor.kl_loss_coef` | KL regularization coefficient | 0.01 |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | vLLM tensor parallelism | 4 (NPU) / 2 (GPU) |
| `gui_agent.max_turns` | Maximum interaction turns | 20 |
| `gui_agent.screen_width/height` | Screen dimensions | 999 |

### GRPO-specific Settings

```bash
# Enable GRPO
algorithm.adv_estimator=grpo

# Group sampling
actor_rollout_ref.rollout.n=5

# KL loss (instead of reward KL)
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01
algorithm.use_kl_in_reward=False
```

## Response Format

The model should generate responses in the following format:

```
Thought: I need to click on the search icon to open the search bar.
Action: Click on the search icon at the top right.
<tool_call>
{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [850, 50]}}
</tool_call>
```

## Datasets

Compatible datasets for GUI Agent training:

- **AITW** (Android in the Wild): General mobile interaction tasks
- **MobileAgent-Bench**: Mobile agent benchmark dataset
- **ScreenSpot**: UI element grounding dataset
- **Rico**: Mobile UI screenshots with semantic annotations

## Monitoring

Training metrics are logged to console and optionally to WandB:

- `actor/policy_loss`: PPO/GRPO policy loss
- `actor/kl_loss`: KL divergence loss
- `reward/mean`: Average reward across batch
- `agent_loop/generate_sequences/mean`: Generation time
- `agent_loop/tool_calls/mean`: Tool execution time

## Troubleshooting

### Out of Memory

- Reduce `ppo_micro_batch_size_per_gpu`
- Enable `fsdp_config.param_offload=True`
- Reduce `data.max_response_length`

### Slow Training

- Increase `tensor_model_parallel_size` for faster inference
- Enable `enable_chunked_prefill=True` for better batching
- Use `enforce_eager=False` for GPU (CUDA graphs)

### Poor Convergence

- Adjust `kl_loss_coef` (try 0.001 - 0.1)
- Increase `rollout.n` for better advantage estimation
- Check reward function is providing meaningful gradients

## Citation

If you use this GUI Agent implementation, please cite:

```bibtex
@article{qwen3vl,
  title={Qwen3-VL Technical Report},
  author={Qwen Team},
  year={2025}
}

@article{grpo,
  title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author={DeepSeek-AI},
  journal={arXiv preprint arXiv:2402.03300},
  year={2024}
}
```

## License

Apache License 2.0
