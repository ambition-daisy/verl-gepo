set -x

deepscaler_train_path=/mnt/hdfs/anglv_ssd_hldy/gepo/verl/dataset/deepscaler.parquet
dapo_train_path=/mnt/hdfs/anglv_ssd_hldy/gepo/verl/dataset/dapo_train.parquet
aime25_path=/mnt/hdfs/anglv_ssd_hldy/gepo/verl/dataset/aime25.parquet
aime24_path=/mnt/hdfs/anglv_ssd_hldy/gepo/verl/dataset/aime24.parquet

model_path=/mnt/hdfs/anglv_ssd_hldy/hf_ckpts/Qwen3-30B-A3B

project_name=qwen3-30b-rl
experiment_name=qwen3-3b-grpo

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_temperature=0.0

clip_ratio_low=0.2
clip_ratio_high=0.2

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- \
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=[$deepscaler_train_path] \
        data.val_files=[$aime25_path,$aime24_path] \
        data.train_batch_size=256 \
        data.max_prompt_length=1024 \
        data.max_response_length=15348 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=$model_path \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
        actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.max_num_batched_tokens=16372 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.temperature=${temperature} \
        actor_rollout_ref.rollout.top_p=${top_p} \
        actor_rollout_ref.rollout.top_k=${top_k} \
        actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name=$project_name \
        trainer.experiment_name=$experiment_name \
        trainer.val_before_train=False \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=4 \
        trainer.save_freq=20 \
        trainer.test_freq=5 \
        trainer.total_training_steps=200 \
        trainer.total_epochs=15 $@