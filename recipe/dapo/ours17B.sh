#!/usr/bin/env bash
set -xeuo pipefail

project_name='RLVR-final'
exp_name='Qwen3-1.7B-S2D-Credit-3Anchor'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((2048))
max_response_length=$((10240))
enable_overlong_buffer=True
overlong_buffer_len=$((2048))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=128
gen_prompt_bsz=$((train_prompt_bsz * 3))
train_prompt_mini_bsz=8
n_resp_per_prompt=8

credit_train_mc_branches=True
credit_branch_groups_per_prompt=2
credit_num_samples=4

loss_mode=vanilla

# Sparse counterfactual step credit
credit_use_probe=True # False: Sparse-MC ablation; keep MC/branches and centering, skip the probe.
step_split_min_tokens=96
step_split_max_tokens=512
credit_entropy_top_ratio=0.2
credit_anchors_per_group=3
credit_sampling_temperature=1.0
credit_uniform_mix=0.1
credit_mc_value_loss_weight=0.6
credit_start_value_loss_weight=0.2
credit_terminal_value_loss_weight=0.2
credit_advantage_coef=0.3
credit_warmup_ratio=0.05
credit_head_hidden_dim=512
credit_head_lr=1e-3

# Ray
RAY_ADDRESS="http://127.0.0.1:8265"
WORKING_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_ENV="${WORKING_DIR}/recipe/dapo/runtime_env.yaml"

# Paths
MODEL_PATH=${MODEL_PATH:-"/data2/zzn/models/Qwen/Qwen3-1.7B-Base"}
CKPTS_DIR=${CKPTS_DIR:-"/data3/zzn/models/verl_ckpts/S2D-Credit-3Anchor/Qwen3-1.7B-Base"}
TRAIN_FILE=${TRAIN_FILE:-"/data2/zzn/dataset/RLVR/rlvr-math.parquet"}
TEST_FILE=${TEST_FILE:-"/data2/zzn/dataset/RLVR/aime-2024.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
offload=True
gen_tp=1

ray job submit --address="${RAY_ADDRESS}" --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.step_split.enabled=True \
    algorithm.step_split.min_step_tokens=${step_split_min_tokens} \
    algorithm.step_split.max_step_tokens=${step_split_max_tokens} \
    algorithm.sparse_counterfactual_credit.enabled=True \
    algorithm.sparse_counterfactual_credit.use_probe=${credit_use_probe} \
    algorithm.sparse_counterfactual_credit.entropy_top_ratio=${credit_entropy_top_ratio} \
    algorithm.sparse_counterfactual_credit.anchors_per_group=${credit_anchors_per_group} \
    algorithm.sparse_counterfactual_credit.sampling_temperature=${credit_sampling_temperature} \
    algorithm.sparse_counterfactual_credit.uniform_mix=${credit_uniform_mix} \
    algorithm.sparse_counterfactual_credit.num_samples=${credit_num_samples} \
    algorithm.sparse_counterfactual_credit.train_mc_branches=${credit_train_mc_branches} \
    algorithm.sparse_counterfactual_credit.branch_groups_per_prompt=${credit_branch_groups_per_prompt} \
    algorithm.sparse_counterfactual_credit.correctness_key=acc \
    algorithm.sparse_counterfactual_credit.mc_value_loss_weight=${credit_mc_value_loss_weight} \
    algorithm.sparse_counterfactual_credit.start_value_loss_weight=${credit_start_value_loss_weight} \
    algorithm.sparse_counterfactual_credit.terminal_value_loss_weight=${credit_terminal_value_loss_weight} \
    algorithm.sparse_counterfactual_credit.advantage_coef=${credit_advantage_coef} \
    algorithm.sparse_counterfactual_credit.warmup_ratio=${credit_warmup_ratio} \
    algorithm.sparse_counterfactual_credit.selection_seed=42 \
    algorithm.sparse_counterfactual_credit.temperature=${temperature} \
    algorithm.sparse_counterfactual_credit.top_p=${top_p} \
    algorithm.sparse_counterfactual_credit.top_k=${top_k} \
    algorithm.sparse_counterfactual_credit.repetition_penalty=1.0 \
    algorithm.sparse_counterfactual_credit.max_new_tokens=null \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.use_fused_kernels=False \
    actor_rollout_ref.actor.use_prefix_grouper=False \
    actor_rollout_ref.actor.counterfactual_credit_head.enabled=${credit_use_probe} \
    actor_rollout_ref.actor.counterfactual_credit_head.hidden_dim=${credit_head_hidden_dim} \
    actor_rollout_ref.actor.counterfactual_credit_head.lr=${credit_head_lr} \
    actor_rollout_ref.actor.counterfactual_credit_head.weight_decay=0.0 \
    actor_rollout_ref.actor.counterfactual_credit_head.save_checkpoint=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=3 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.checkpoint_engine.backend=naive \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
    reward_model.reward_manager=dapo \
    reward_model.num_workers=4 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.total_epochs=2 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto
