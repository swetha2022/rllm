#!/usr/bin/env bash

set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# Edit these if needed
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Choose the model (HF repo id or absolute local path)
MODEL_ID=${MODEL_ID:-"Qwen/Qwen3-4B"}

# Find the directory where rllm package is located
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")

python3 -m examples.math_tool.train_math_with_tool \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.val_batch_size=500 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=${MODEL_ID} \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_mini_batch=True \
    actor_rollout_ref.actor.ppo_num_mini_batches=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.grad_norm_threshold=10 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.mask_truncated_samples=False \
    algorithm.clip_advantages=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name='4b-math-tool-dynamic' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    agent.max_steps=2 \
    agent.async_engine=True \
    agent.use_stepwise_advantage=False \
    agent.stepwise_advantage_mode="mc_return" \
    trainer.total_epochs=100 \
    +env.dynamic_goals.enable=true \
    +env.dynamic_goals.threshold=50 \
    +env.dynamic_goals.interval=100 \
    +env.dynamic_goals.style=deepscaler_math \
    +env.dynamic_goals.per_call_count=8 \
    +env.dynamic_goals.problem_char_limit=2000 \
    +env.dynamic_goals.target_pass_low=0.3 \
    +env.dynamic_goals.target_pass_high=0.7


# python3 -m examples.math_tool.train_math_with_tool \
#     algorithm.adv_estimator=grpo \
#     data.train_batch_size=16 \
#     data.val_batch_size=250 \
#     data.max_prompt_length=2048 \
#     data.max_response_length=8192 \
#     actor_rollout_ref.model.path=${MODEL_ID} \
#     actor_rollout_ref.hybrid_engine=True \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
#     actor_rollout_ref.actor.ppo_mini_batch_size=16 \
#     actor_rollout_ref.actor.use_dynamic_mini_batch=True \
#     actor_rollout_ref.actor.ppo_num_mini_batches=1 \
#     actor_rollout_ref.actor.use_dynamic_bsz=True \
#     actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
#     actor_rollout_ref.actor.use_kl_loss=False \
#     actor_rollout_ref.actor.clip_ratio_high=0.28 \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
#     actor_rollout_ref.actor.grad_norm_threshold=10 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=True \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.mode="async" \
#     actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler \
#     actor_rollout_ref.rollout.enforce_eager=False \
#     actor_rollout_ref.rollout.temperature=0.6 \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
#     actor_rollout_ref.rollout.n=8 \
#     actor_rollout_ref.rollout.val_kwargs.n=1 \
#     actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
#     actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
#     actor_rollout_ref.actor.entropy_coeff=0 \
#     algorithm.kl_ctrl.kl_coef=0.001 \
#     algorithm.mask_truncated_samples=False \
#     algorithm.clip_advantages=False \
#     trainer.critic_warmup=0 \
#     trainer.logger=['console','wandb'] \
#     trainer.project_name='rllm-agent' \
#     trainer.experiment_name='4b-math-tool' \
#     trainer.val_before_train=True \
#     trainer.n_gpus_per_node=8 \
#     trainer.nnodes=1 \
#     trainer.save_freq=100 \
#     trainer.test_freq=20 \
#     trainer.default_hdfs_dir=null \
#     agent.max_steps=2 \
#     agent.async_engine=True \
#     agent.use_stepwise_advantage=False \
#     agent.stepwise_advantage_mode="mc_return" \
#     trainer.total_epochs=100 \
#     +env.dynamic_goals.enable=true \
#     +env.dynamic_goals.threshold=200 \
#     +env.dynamic_goals.interval=50 \
#     +env.dynamic_goals.style=deepscaler_math \
#     +env.dynamic_goals.per_call_count=8 \
#     +env.dynamic_goals.problem_char_limit=2000 \
#     +env.dynamic_goals.target_pass_low=0.2 \
#     +env.dynamic_goals.target_pass_high=0.8