#!/bin/bash
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

# =============================================================================
# GUI Agent Training Script with GRPO on NPU
# =============================================================================
# This script trains a GUI Agent based on Qwen3-VL-8B using GRPO algorithm.
#
# Features:
# - Training: FSDP (Fully Sharded Data Parallel)
# - Inference: vLLM for efficient rollout
# - Environment: NPU (Ascend)
# - Algorithm: GRPO (Group Relative Policy Optimization)
# - Model: Qwen3-VL-8B
#
# Usage:
#   bash run_qwen3_vl_8b_gui_agent_npu.sh [ENGINE]
#
# Arguments:
#   ENGINE: Rollout engine, default is 'vllm'
#
# Environment Variables:
#   DATA_DIR: Directory containing training data (default: $HOME/data/gui_agent)
#   MODEL_PATH: Path to Qwen3-VL-8B model (default: Qwen/Qwen3-VL-8B)
#   OUTPUT_DIR: Output directory for checkpoints (default: ./outputs/gui_agent)
# =============================================================================

set -x

# Engine configuration (vllm or sglang)
ENGINE=${1:-vllm}

# Data paths
DATA_DIR=${DATA_DIR:-$HOME/data/gui_agent}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-8B}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/gui_agent}

# NPU-specific optimizations
# Some models are optimized by vllm ascend. While in some cases, e.g. RLHF training,
# the optimized model may not be suitable. Set this value to 0 to disable.
export USE_OPTIMIZED_MODEL=0

# Create output directory
mkdir -p $OUTPUT_DIR

# =============================================================================
# Training Configuration
# =============================================================================
#
# Key GRPO Parameters:
# - algorithm.adv_estimator=grpo: Use GRPO instead of GAE
# - actor_rollout_ref.rollout.n=5: Number of samples per prompt (group sampling)
# - actor_rollout_ref.actor.use_kl_loss=True: Use KL loss instead of reward KL
# - actor_rollout_ref.actor.kl_loss_coef=0.01: KL regularization coefficient
# - algorithm.use_kl_in_reward=False: GRPO doesn't use KL in reward
#
# GUI Agent Specific:
# - data.image_key=images: Key for image data in dataset
# - actor_rollout_ref.rollout.multi_turn.*: Multi-turn interaction settings
# - actor_rollout_ref.rollout.gui_agent.*: GUI Agent specific settings
# =============================================================================

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    `# ==================== Data Configuration ====================` \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    \
    `# ==================== Model Configuration ====================` \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    `# ==================== Actor (FSDP Training) Configuration ====================` \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    \
    `# ==================== FSDP Configuration ====================` \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    `# ==================== Rollout (vLLM Inference) Configuration ====================` \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    \
    `# ==================== GUI Agent Configuration ====================` \
    +actor_rollout_ref.rollout.gui_agent.max_turns=20 \
    +actor_rollout_ref.rollout.gui_agent.screen_width=999 \
    +actor_rollout_ref.rollout.gui_agent.screen_height=999 \
    +actor_rollout_ref.rollout.gui_agent.return_screenshot=True \
    \
    `# ==================== Multi-turn Configuration ====================` \
    +actor_rollout_ref.rollout.multi_turn.max_user_turns=20 \
    +actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 \
    +actor_rollout_ref.rollout.multi_turn.format=qwen \
    \
    `# ==================== Agent Loop Configuration ====================` \
    +actor_rollout_ref.rollout.agent.default_agent_loop=gui_agent \
    +actor_rollout_ref.rollout.agent.num_workers=8 \
    \
    `# ==================== Reference Model Configuration ====================` \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    `# ==================== Algorithm Configuration ====================` \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    \
    `# ==================== Reward Configuration ====================` \
    +custom_reward_function.path=verl/utils/reward_score/gui_agent.py \
    +custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.screen_width=999 \
    +custom_reward_function.reward_kwargs.screen_height=999 \
    \
    `# ==================== Trainer Configuration ====================` \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_gui_agent' \
    trainer.experiment_name='qwen3_vl_8b_gui_agent_grpo' \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=30 \
    trainer.default_local_dir=$OUTPUT_DIR \
    \
    "$@"
