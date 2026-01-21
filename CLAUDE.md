# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**verl** (Volcano Engine Reinforcement Learning) is a flexible, efficient, and production-ready RL training library for large language models. It implements the HybridFlow architecture with a hybrid-controller programming model for complex post-training dataflows (PPO, GRPO, etc.).

Key technologies: PyTorch, Ray, FSDP/Megatron-LM (training), vLLM/SGLang (inference), Hydra (configuration).

## Development Commands

### Installation

```bash
# Development installation with vLLM backend
pip install -e .[test,vllm]

# Development installation with SGLang backend
pip install -e .[test,sglang]

# For full dependency setup, see: https://verl.readthedocs.io/en/latest/start/install.html
```

### Code Quality

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run linting on staged changes
pre-commit run

# Run linting on all files
pre-commit run --all-files

# Run specific hook (e.g., ruff)
pre-commit run --all-files --show-diff-on-failure --color=always ruff
```

### Testing

```bash
# Run tests matching pattern (GPU tests)
pytest tests/path/to/test_file.py

# Run CPU-only tests (files ending with _on_cpu.py)
pytest tests/**/test_*_on_cpu.py

# Run specific test
pytest tests/trainer/test_ppo.py::test_function_name -v
```

Test categories:
- `tests/trainer/` - Trainer functionality tests
- `tests/models/` - Model implementation tests
- `tests/special_distributed/` - Multi-GPU tests
- `tests/special_e2e/` - End-to-end training tests
- `tests/special_sanity/` - Quick sanity checks
- `tests/*_on_cpu.py` - CPU-only tests

### Running Training

```bash
# PPO training example
python3 -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    data.train_batch_size=256

# GRPO training example
bash examples/grpo_trainer/run_qwen3-8b.sh

# See examples/ directory for more training scripts
```

### Documentation

```bash
# Build documentation
cd docs
pip install -r requirements-docs.txt
make clean
make html

# Preview locally
python -m http.server -d _build/html/
# Open http://localhost:8000
```

## Architecture

### Hybrid Controller Programming Model

verl uses a **single-controller, multiple-workers** architecture with Ray:

**Core Components:**
- **Controller (Driver)**: Orchestrates the training loop, dispatches tasks to workers, computes advantages
- **Worker Groups**: Distributed workers implementing specific roles:
  - `ActorRolloutRefWorker` - Policy model for rollout and training
  - `CriticWorker` - Value function estimation (PPO only)
  - `RolloutWorker` - Optimized inference engine (vLLM/SGLang)
  - `RewardModelWorker` - Reward computation (optional)

**Data Flow:**
1. Controller loads prompts → Rollout workers generate sequences
2. Actor computes log probs → Critic computes values
3. Controller computes advantages using GAE/GRPO
4. Actor/Critic update via mini-batch gradient descent

### Key Abstractions

**DataProto** (`verl/protocol.py`):
- Core data container for passing data between controller and workers
- Wraps `TensorDict` for efficient tensor operations and serialization
- Supports indexing, slicing, chunking, concatenation
- Handles both tensor and non-tensor data with metadata

**Worker Groups** (`verl/single_controller/`):
- `WorkerGroup` base class manages multiple distributed workers
- `RayWorkerGroup` implements Ray-specific execution
- Decorator-based dispatch system (`@register(dispatch_mode=...)`)
- Dispatch modes: `ONE_TO_ALL`, `DP_COMPUTE_PROTO`, `ALL_TO_ALL`

**Training Backends** (`verl/workers/`):
- **FSDP**: PyTorch native distributed training (fully sharded data parallel)
- **FSDP2**: Improved FSDP with better memory/throughput (torch 2.5+)
- **Megatron-LM**: Tensor/pipeline parallelism for large models

**Inference Engines** (`verl/workers/rollout/`):
- **vLLM**: High-throughput inference with paged attention
- **SGLang**: Structured generation, multi-turn, tool calling
- **TensorRT-LLM**: Production-optimized compiled inference

### Directory Structure

```
verl/
├── trainer/           # Training orchestration (PPO, GRPO algorithms)
│   ├── main_ppo.py   # PPO entry point
│   ├── ppo/          # PPO-specific implementations
│   └── config/       # Hydra configs for all trainers
├── workers/          # Worker implementations
│   ├── fsdp_workers.py      # FSDP backend
│   ├── megatron_workers.py  # Megatron backend
│   ├── rollout/             # Inference engines (vllm, sglang, etc.)
│   ├── actor/               # Actor model workers
│   ├── critic/              # Critic model workers
│   └── reward_manager/      # Reward computation
├── single_controller/  # Controller and worker group base classes
├── models/            # Model implementations (HF, Megatron)
├── utils/             # Utilities (datasets, metrics, logging)
├── protocol.py        # DataProto definition
└── experimental/      # Experimental features (async, off-policy)

examples/
├── ppo_trainer/       # PPO training scripts
├── grpo_trainer/      # GRPO training scripts
├── sft/               # Supervised fine-tuning
└── sglang_multiturn/  # Multi-turn examples

tests/                 # Test suite
```

### Configuration System

verl uses **Hydra** for hierarchical configuration management:

**Config hierarchy** (`verl/trainer/config/`):
```
ppo_trainer/          # PPO configs
├── default.yaml      # Base config
├── algorithm/        # Algorithm-specific (GAE, GRPO, etc.)
├── actor_rollout_ref/  # Actor and rollout configs
├── critic/           # Critic configs
├── data/             # Data pipeline configs
└── reward_model/     # Reward model configs
```

**Override syntax:**
```bash
# Command-line overrides
python -m verl.trainer.main_ppo \
    data.train_batch_size=256 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.name=vllm
```

### Training Loop Flow (PPO/GRPO)

```python
for epoch in epochs:
    for batch in dataloader:
        # 1. ROLLOUT: Generate sequences with inference engine
        sequences = rollout_wg.generate_sequences(prompts)

        # 2. COMPUTE: Get log probs, values, rewards
        old_log_probs = actor_wg.compute_log_prob(sequences)
        values = critic_wg.compute_values(sequences)  # PPO only
        rewards = compute_rewards(sequences)

        # 3. ADVANTAGES: Compute on controller
        advantages = compute_gae_advantage(rewards, values, gamma, lam)

        # 4. UPDATE: Train actor and critic
        for epoch in ppo_epochs:
            actor_wg.update_policy(sequences, advantages)
            critic_wg.update_policy(sequences, returns)
```

### Important Patterns

**Worker Dispatch Pattern:**
```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def update_policy(self, data: DataProto) -> DataProto:
    """
    Automatically chunks data by DP ranks, executes on workers,
    and concatenates results back to controller.
    """
    pass
```

**DataProto Usage:**
```python
# Create from tensors
data = DataProto.from_dict(
    tensors={'input_ids': tensor, 'attention_mask': mask},
    non_tensors={'prompts': ['text1', 'text2']},
    meta_info={'batch_size': 2}
)

# Index and slice
item = data[0]           # Returns DataProtoItem
chunk = data[10:20]      # Returns DataProto
selected = data[[1,3,5]] # Returns DataProto

# Split for distributed processing
chunks = data.chunk(num_chunks=8)
```

**3D Hybrid Engine Pattern:**
- Actor model parameters are shared between training and rollout
- During rollout: optimized inference (vLLM/SGLang)
- During training: distributed training (FSDP/Megatron)
- Efficient resharding eliminates memory redundancy

## Common Tasks

### Adding a New Model

**For FSDP backend:**
1. Add model config to `verl/trainer/config/actor_rollout_ref/fsdp_models/`
2. Implement tokenizer in `verl/utils/model.py` if custom
3. Update `verl/models/registry.py` if custom architecture
4. See: https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html

**For Megatron backend:**
1. Add model to `verl/models/mcore/`
2. Register in model registry
3. See: https://verl.readthedocs.io/en/latest/advance/megatron_extension.html

### Adding a New RL Algorithm

1. Implement advantage estimator in `verl/trainer/ppo/core_algos.py`
2. Add config to `verl/trainer/config/algorithm/`
3. Update trainer loop in `verl/trainer/main_ppo.py` if needed
4. Add example script to `examples/`

### Adding a Reward Function

For rule-based rewards:
```python
# In your training script or verl/utils/reward_score/
def my_reward_fn(data: DataProto) -> DataProto:
    # Extract responses
    responses = data.non_tensor_batch['responses']

    # Compute scores (e.g., correctness checking)
    scores = [check_correctness(r) for r in responses]

    # Return as DataProto with token-level rewards
    return DataProto.from_dict(
        tensors={'token_level_scores': torch.tensor(scores)}
    )
```

For model-based rewards, implement a reward model worker in `verl/workers/reward_manager/`.

### Debugging Distributed Training

- Set `trainer.logger=console` for immediate console output
- Use `NCCL_DEBUG=INFO` for NCCL debugging
- Check Ray dashboard: `ray start --head --dashboard-host=0.0.0.0`
- Add breakpoints in worker methods (runs on worker processes)
- Use `DataProto.save_to_disk()` to inspect intermediate data

## Performance Optimization

- **Sequence Packing**: Set `data.packed_seq=True` for variable-length efficiency
- **FSDP2**: Use `strategy=fsdp2` for better throughput (torch 2.5+)
- **Gradient Checkpointing**: Enable with `enable_gradient_checkpointing=True`
- **CPU Offload**: Use `fsdp_config.offload_policy=True` for memory savings
- **Tensor Parallelism**: Increase `rollout.tensor_model_parallel_size` for large models
- **Micro-batch sizes**: Tune `ppo_micro_batch_size_per_gpu` to avoid OOM
- See: https://verl.readthedocs.io/en/latest/perf/perf_tuning.html

## Key Dependencies

- PyTorch >= 2.0 (FSDP2 requires >= 2.5)
- Ray >= 2.41.0 (distributed execution)
- tensordict >= 0.8.0, <= 0.10.0 (data containers)
- vLLM >= 0.8.5, <= 0.12.0 (avoid 0.7.x due to bugs)
- SGLang == 0.5.6 (if using SGLang)
- Hydra (configuration)
- transformers, accelerate, peft (model utilities)

## Additional Resources

- Documentation: https://verl.readthedocs.io/
- Paper: https://arxiv.org/abs/2409.19256 (HybridFlow)
- Examples: `examples/` directory
- Recipes: https://github.com/verl-project/verl-recipe (submodule)
- Community: Slack, GitHub Discussions
