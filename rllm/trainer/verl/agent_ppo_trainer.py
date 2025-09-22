import asyncio
import json
import math
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from pprint import pprint
from queue import Queue
from threading import Thread
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from openai import OpenAI

from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    RayWorkerGroup,
    ResourcePoolManager,
    Role,
    WorkerType,
    _timer,
    compute_advantage,
    compute_data_metrics,
    compute_response_mask,
    compute_timing_metrics,
    reduce_metrics,
)

# DeepCoder-specific base prompt for GPT-4o dynamic problem generation
DEEPCODER_BASE_PROMPT = """
Generate ONE competitive programming problem in the exact format shown below. The problem should be appropriate for the model's current ability level.

**Problem Requirements:**
- Competitive programming style (like CodeForces/LeetCode)
- Solvable in Python
- Include 3-5 test cases with input/output
- Specify difficulty: Easy, Medium, or Hard
- Include algorithm type/category

**Output Format (exactly this structure):**

<problem>
[Write a clear, concise problem statement here. Include:
- Problem description
- Input format specification
- Output format specification
- Constraints
- Examples with input/output]
</problem>

<tests>
[
  {"input": "first test input", "output": "expected output", "testtype": "stdin_stdout"},
  {"input": "second test input", "output": "expected output", "testtype": "stdin_stdout"},
  {"input": "third test input", "output": "expected output", "testtype": "stdin_stdout"}
]
</tests>

<metadata>
{
  "difficulty": "Easy|Medium|Hard",
  "type": "Algorithm category (e.g., Array, String, Graph, DP, etc.)",
  "func_name": null
}
</metadata>

---

Model's recent output (for context on current ability):

"""


class AgentPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        env_class=None,
        agent_class=None,
        env_args=None,
        agent_args=None,
    ):
        super().__init__(config=config, tokenizer=tokenizer, role_worker_mapping=role_worker_mapping, resource_pool_manager=resource_pool_manager, ray_worker_group_cls=ray_worker_group_cls, reward_fn=reward_fn, val_reward_fn=val_reward_fn)
        self.env_class = env_class
        self.agent_class = agent_class
        self.env_args = env_args or {}
        self.agent_args = agent_args or {}

        # Dynamic generation configuration with type validation
        self.dynamic_generation_threshold = int(self.config.trainer.get("dynamic_generation_threshold", 0))
        self.dynamic_generation_frequency = int(self.config.trainer.get("dynamic_generation_frequency", 10))
        self.problems_per_generation = int(self.config.trainer.get("problems_per_generation", 8))  # Reduced since we generate one at a time
        self.use_dynamic_generation = self.dynamic_generation_threshold > 0
        
        # Initialize OpenAI client for dynamic problem generation
        self.openai_client = None
        if self.use_dynamic_generation:
            api_key = self.config.trainer.get("openai_api_key")
            if api_key:
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    # Test API key with a simple call
                    self._validate_openai_api_key()
                    print(f"Dynamic generation enabled: threshold={self.dynamic_generation_threshold}, frequency={self.dynamic_generation_frequency}")
                except Exception as e:
                    print(f"Warning: Failed to initialize OpenAI client: {e}")
                    self.openai_client = None
                    self.use_dynamic_generation = False
            else:
                print("Warning: Dynamic generation enabled but no OpenAI API key provided")
                self.use_dynamic_generation = False
        
        # Track dynamic generation state
        self.dynamic_generation_active = False
        self.last_generation_step = 0
        self.generated_problems_count = 0

        if self.config.agent.use_stepwise_advantage:
            print("Using step-level advantage, max_prompt_length and max_response_length will be applied step-wise")
        else:
            print("Using trajectory-level advantage, max_prompt_length and max_response_length will be applied episode-wise")

    def init_workers(self):
        super().init_workers()

        # Initialize additional agent class
        # Number of agents is set to be 0 initially
        if self.hybrid_engine:
            agent_rollout_wg = self.actor_rollout_wg
        else:
            agent_rollout_wg = self.rollout_wg

        if self.config.actor_rollout_ref.rollout.mode == "async":
            rollout_engine = self.async_rollout_manager
        else:
            rollout_engine = agent_rollout_wg

        self.agent_execution_engine = AsyncAgentExecutionEngine(
            rollout_engine=rollout_engine,
            config=self.config,
            engine_name="verl",
            tokenizer=self.tokenizer,
            model_path=self.config.actor_rollout_ref.model.path,
            max_steps=self.config.agent.max_steps,
            max_response_length=self.config.data.max_response_length,
            max_prompt_length=self.config.data.max_prompt_length,
            agent_class=self.agent_class,
            agent_args=self.agent_args,
            env_class=self.env_class,
            env_args=self.env_args,
            enforce_max_prompt_length=self.config.agent.use_stepwise_advantage,
            trajectory_timeout=self.config.agent.trajectory_timeout,
            overlong_filter=self.config.agent.overlong_filter,
            **self.config.agent.get("engine_args", {}),
        )

    def init_envs_and_agents(self, batch):
        """
        Initialize environment depending on env_class with the necessary extra_info, also set uid of the batch.
        """
        env_args = batch.non_tensor_batch["extra_info"].tolist()

        full_agent_args = dict(self.config.agent.get("agent_args", {})) | self.agent_args
        base_env_args = dict(self.config.env.get("env_args", {})) | self.env_args

        def _create_env(i):
            if isinstance(env_args[i], str):
                env_args[i] = json.loads(env_args[i])
            return i, self.env_class.from_dict({**env_args[i], **base_env_args})

        def _create_agent(i):
            return i, self.agent_class(**full_agent_args)

        # Create environments in parallel while preserving order
        envs = [None] * len(env_args)
        with ThreadPoolExecutor(max_workers=64) as executor:
            env_futures = [executor.submit(_create_env, i) for i in range(len(env_args))]
            for future in as_completed(env_futures):
                idx, env = future.result()
                envs[idx] = env

        # Create agents in parallel while preserving order
        agents = [None] * len(envs)
        with ThreadPoolExecutor(max_workers=64) as executor:
            agent_futures = [executor.submit(_create_agent, i) for i in range(len(envs))]
            for future in as_completed(agent_futures):
                idx, agent = future.result()
                agents[idx] = agent
        self.agent_execution_engine.update_envs_and_agents(envs, agents)
        return envs

    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        import time

        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate agent: {time.time() - start_time}")
        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            
            # Check if we should switch to dynamic generation
            if (self.use_dynamic_generation and 
                not self.dynamic_generation_active and 
                self.global_steps >= self.dynamic_generation_threshold):
                print(f"Switching to dynamic generation at step {self.global_steps}")
                self.dynamic_generation_active = True
                self.last_generation_step = self.global_steps
            
            # Check if we should generate new problems
            if (self.dynamic_generation_active and 
                self.global_steps - self.last_generation_step >= self.dynamic_generation_frequency):
                print(f"Generating new problems at step {self.global_steps}")
                self._generate_and_update_dataloader()
                self.last_generation_step = self.global_steps
            
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                
                # Store batch for potential dynamic generation
                if self.use_dynamic_generation:
                    self.last_batch = batch
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                metrics = {}
                timing_raw = {}

                batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
                batch.meta_info = {
                    "agent_rollout": True,  # no need to generate multiple ones since environment is repeated already
                }

                with _timer("step", timing_raw):
                    self.init_envs_and_agents(batch)

                    if self.config.agent.use_stepwise_advantage:
                        final_gen_batch_output = self.generate_agent_steps(timing_raw=timing_raw, meta_info=batch.meta_info, uids=batch.non_tensor_batch["uid"])
                        repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                        # need to repeat to make shape match
                        batch = batch.repeat_by_counts(repeat_counts, interleave=True)
                        final_gen_batch_output.meta_info.pop("repeat_counts", None)  # no longer needed after this
                        # batch needs to be padded to divisor of world size, we will pad with everything masked out
                        batch = batch.union(final_gen_batch_output)
                        batch = self._pad_dataproto_to_world_size(batch=batch)
                    else:
                        final_gen_batch_output, generate_metrics = self.generate_agent_trajectory(timing_raw=timing_raw, meta_info=batch.meta_info)
                        batch = batch.union(final_gen_batch_output)
                        metrics.update(generate_metrics)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # reward tensor for env-based trajectory data can be obtained by processing the trajectories
                        if "token_level_scores" not in batch.batch:
                            reward_tensor = self.reward_fn(batch)
                            batch.batch["token_level_scores"] = reward_tensor
                        else:
                            reward_tensor = batch.batch["token_level_scores"]  # filled in by environment collected trajectory transformation

                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch["uid"]
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

                            # Check if all rewards are <= 0 or all are 1 >= for this uid
                            if (uid_rewards <= 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards >= 1).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1

                        # Log to metrics
                        metrics["batch/solve_none"] = solve_none
                        metrics["batch/solve_all"] = solve_all
                        metrics["batch/solve_partial"] = len(unique_uids) - solve_none - solve_all

                        if self.config.trainer.rejection_sample:
                            # log the actual complete training rewards before rejection sampling
                            token_level_rewards = None  # for metrics calculation
                            if self.config.agent.use_stepwise_advantage:
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                non_pad_steps = batch.select_idxs(non_pad_step_indices)
                                is_last_step = non_pad_steps.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)
                                token_level_rewards = last_step_batch.batch["token_level_scores"]
                            else:
                                token_level_rewards = batch.batch["token_level_scores"]
                            full_sequence_score = token_level_rewards.sum(-1)
                            metrics["critic/full-score/mean"] = torch.mean(full_sequence_score).detach().item()
                            metrics["critic/full-score/max"] = torch.max(full_sequence_score).detach().item()
                            metrics["critic/full-score/min"] = torch.min(full_sequence_score).detach().item()

                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                continue

                            # Filter batch to keep only valid samples
                            batch = batch[valid_mask]

                            if self.config.agent.use_stepwise_advantage and self.config.agent.stepwise_advantage_mode == "broadcast":
                                # batch now only contains steps with valid uids
                                # filter out padding steps
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps

                                # need to make sure both number of last steps (number of uids) and number of total steps in the batch (batch size after processing) are all multiples of world size
                                # separate out last step and intermediate steps
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                not_last_step_indices = np.where(is_last_step == False)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)  # This batch only has valid last steps
                                non_last_step_batch = batch.select_idxs(not_last_step_indices)

                                # filter last_step_batch to make sure its multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (
                                    last_step_batch.batch["input_ids"].shape[0]  # 1 per trajectory
                                    // num_trainer_replicas
                                ) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    continue

                                size_mask = torch.zeros(last_step_batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                last_step_batch = last_step_batch[size_mask]  # filtered last steps

                                # now we go through all the non_last_step_batch and keep everything that has same idxs that exists in the filtered last steps
                                valid_last_step_idxs = last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_idxs = non_last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_mask = np.isin(non_last_step_idxs, valid_last_step_idxs)
                                non_last_step_batch = non_last_step_batch[non_last_step_mask]

                                # concatenate then pad
                                batch = DataProto.concat([last_step_batch, non_last_step_batch])
                                batch = self._pad_dataproto_to_world_size(batch)
                            else:
                                # Round down to the nearest multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (batch.batch["input_ids"].shape[0] // num_trainer_replicas) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    continue

                                size_mask = torch.zeros(batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                batch = batch[size_mask]

                        # recompute old_log_probs
                        with _timer("old_log_prob", timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer("ref", timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                        kl_ctrl=self.kl_ctrl,
                        #                                        kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if self.config.agent.use_stepwise_advantage:
                            if self.config.agent.stepwise_advantage_mode == "mc_return":
                                batch.batch["token_level_rewards"] = batch.batch["mc_returns"]
                                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]

                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps
                            elif self.config.agent.stepwise_advantage_mode == "broadcast":
                                # In case of step-wise advantage broadcast, we would split out the final steps, then merge again
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                last_step_indices = np.where(is_last_step == True)[0]
                                other_step_indices = np.where(is_last_step == False)[0]
                                other_step_batch = batch.select_idxs(other_step_indices)
                                batch = batch.select_idxs(last_step_indices)  # This batch only has last steps
                            else:
                                raise ValueError(f"Stepwise advantage mode {self.config.agent.stepwise_advantage_mode} not supported")

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            mask_truncated_samples=self.config.algorithm.mask_truncated_samples,
                            clip_advantages=self.config.algorithm.clip_advantages,
                        )

                        if self.config.agent.use_stepwise_advantage and self.config.agent.stepwise_advantage_mode == "broadcast":
                            # remove the padded last steps
                            # Merging the separated out steps using the advantage from last steps
                            self._stepwise_advantage_broadcast(batch, other_step_batch=other_step_batch)
                            # batch = batch.merge(other_step_batch)
                            batch = DataProto.concat([batch, other_step_batch])

                    batch = self._pad_dataproto_to_world_size(batch=batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

    def _validate_agent(self):
        rewards_lst = []
        data_source_lst = []
        uid_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.pop(["input_ids", "attention_mask", "position_ids"])  # these are not needed for environment based interaction
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
                "agent_rollout": True,
            }
            self.init_envs_and_agents(test_batch)

            if self.config.agent.use_stepwise_advantage:
                test_output_gen_batch = self.generate_agent_steps(meta_info=test_batch.meta_info, uids=test_batch.non_tensor_batch["uid"])
                # for validation, we only need the last step
                is_last_step = test_output_gen_batch.non_tensor_batch["is_last_step"]
                last_step_indices = np.where(is_last_step == True)[0]
                test_output_gen_batch = test_output_gen_batch.select_idxs(last_step_indices)  # This batch only has last steps
            else:
                test_output_gen_batch, _ = self.generate_agent_trajectory(meta_info=test_batch.meta_info)

            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor = test_batch.batch["token_level_scores"]

            rewards_lst.append(reward_tensor.sum(-1).cpu())
            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            uid_lst.append(test_batch.non_tensor_batch["uid"])

        reward_tensor = torch.cat(rewards_lst, dim=0)  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}

        # to group for pass@k
        uid_tensor = np.concatenate(uid_lst, axis=0)
        data_source_uid_pass_rates = {}  # data source to {uid: pass or not}

        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]

            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

            # pass@k
            if data_source not in data_source_uid_pass_rates:
                data_source_uid_pass_rates[data_source] = {}

            uid = uid_tensor[i]
            if uid not in data_source_uid_pass_rates[data_source]:
                data_source_uid_pass_rates[data_source][uid] = 0  # default to not pass
            # take highest score
            data_source_uid_pass_rates[data_source][uid] = max(data_source_uid_pass_rates[data_source][uid], reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            # clip rewards to be between 0 and 1
            rewards_array = np.array(rewards)
            rewards_array = np.clip(rewards_array, 0, 1)
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards_array)

        for data_source, pass_rates in data_source_uid_pass_rates.items():
            pass_k_lst = []
            for uid, pass_score in pass_rates.items():
                pass_k_lst.append(pass_score >= 1)  # assuming 1 means passed
            metric_dict[f"val/test_score/pass@k/{data_source}"] = np.mean(pass_k_lst)

        return metric_dict

    def generate_agent_trajectory(self, timing_raw=None, meta_info=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards

        Args:
            envs: The environments in which the agent interacts.
            agents: The agents to use for interation.
            timing_raw: Dictionary to store timing information for profiling.
            meta_info (optional): Metadata for veRL generation.

        Returns:
            DataProto: Representation of the agent's trajectories.
            Dict[str:float]: Metrics for the generation process.
        """
        if timing_raw is None:
            timing_raw = {}
        with _timer("collect_trajectory", timing_raw):
            trajectories = []
            if self.config.agent.async_engine:
                gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Token")
                for _, trajectory in enumerate(gen_seq_generator):
                    trajectories.append(trajectory)
            else:
                # generate_trajectories returns list of trajectories.
                trajectories = self.agent_execution_engine.generate_trajectories(timing_raw=timing_raw, mode="Token", meta_info=meta_info)
        # Sort trajectories by their idx, to ensure they are in order.
        trajectories.sort(key=lambda x: x["idx"])

        with _timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output, metrics = self._transform_agent_trajectories(trajectories)
        return final_gen_batch_output, metrics

    def generate_agent_steps(self, timing_raw=None, meta_info=None, uids=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards.

        Returns:
            DataProto: Representation of the last step of agent's trajectories.
            Dict[str:List[DataProto]]: Index of the trajectory to the rest of the steps from the trajectory.
        """
        if timing_raw is None:
            timing_raw = {}
        if uids is None:
            uids = []
        with _timer("collect_trajectory", timing_raw):
            steps = []
            if self.config.agent.async_engine:
                gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Step")
                for _, trajectory in enumerate(gen_seq_generator):
                    steps.append(trajectory)
            else:
                # generate_trajectories returns list of trajectories.
                steps = self.agent_execution_engine.generate_trajectories(timing_raw=timing_raw, mode="Step", meta_info=meta_info)
        # Sort trajectories by their idx, to ensure they are in order.
        steps.sort(key=lambda x: x["idx"])

        with _timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output = self._transform_agent_steps(steps, uids=uids)
        return final_gen_batch_output

    def _transform_agent_trajectories(self, trajectories: list[dict]):
        """
        Helper function to transform a list of trajectories into tokenized DataProto format.

        Args:
            trajectories (list of dict): List of trajectories to process.

        Returns:
            DataProto: A structured dataset containing input tokens, masks, and rewards.
        """
        from verl.utils.torch_functional import pad_sequence_to_length

        all_initial_tokens_list = []
        all_response_tokens_list = []
        all_masks_list = []
        traj_scores = []
        chat_completions = []
        traj_metrics = []
        metrics = {}

        for traj in trajectories:
            prompt_tokens = traj["prompt_tokens"]
            response_tokens = traj["response_tokens"]
            # test if trajectory is empty
            assert prompt_tokens.numel() != 0 and response_tokens.numel() != 0, f"Both prompt {prompt_tokens.numel()} and response {response_tokens.numel()} of trajectory shouldn't be empty. Please check make sure environment is working and the config"
            all_initial_tokens_list.append(prompt_tokens)
            all_response_tokens_list.append(response_tokens)
            all_masks_list.append(traj["response_masks"])
            traj_scores.append(traj["trajectory_reward"])
            chat_completions.append(traj["chat_completions"])
            traj_metrics.append(traj["metrics"])

        # Flatten traj_metrics into a dict of lists
        traj_metrics = {k: [d[k] for d in traj_metrics] for k in traj_metrics[0]}
        # Aggregate metrics (mean, min, max)
        for k, v_list in traj_metrics.items():
            v_list = [v for v in v_list if v is not None and v >= 0]
            if not v_list:
                continue
            v_list = np.array(v_list)
            metrics.update(
                {
                    f"traj/{k}_mean": v_list.mean(),
                    f"traj/{k}_min": v_list.min(),
                    f"traj/{k}_max": v_list.max(),
                }
            )

        # Save chat completions to a file
        save_dir = os.path.join(self.config.trainer.default_local_dir, "chat_completions")
        os.makedirs(save_dir, exist_ok=True)
        # Save it into a jsonl files (self.global_steps)
        with open(os.path.join(save_dir, f"{self.global_steps}.jsonl"), "w") as f:
            for chat_completion in chat_completions:
                f.write(json.dumps(chat_completion) + "\n")

        # reverse the list and create tensors, pad, then flip to achieve left padding
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_initial_tokens_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])

        prompts_batch = pad_sequence_to_length(prompts_batch, self.config.data.max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)

        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_response_tokens_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        max_response_length = self.config.data.max_response_length
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)

        traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
        traj_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False)

        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)

        attention_mask = torch.where(trajectory_batch != self.tokenizer.pad_token_id, 1, 0)

        # Compute position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        prompt_length = prompts_batch.shape[1]
        valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)

        for i, traj_score in enumerate(traj_scores):
            last_valid_idx = valid_response_length_sequences[i] - 1
            if last_valid_idx >= 0 and last_valid_idx < score_batch.shape[1]:
                score_batch[i, last_valid_idx] = traj_score

        tensor_batch = {
            "input_ids": trajectory_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "traj_mask": traj_mask,
        }

        self.visualize_trajectory(DataProto.from_dict(tensors=tensor_batch))

        return DataProto.from_dict(tensors=tensor_batch), metrics

    def visualize_trajectory(self, tensor_batch, sample_idx=0, max_samples=1, mask_key="traj_mask"):
        """
        Visualize the trajectory from tensor_batch by detokenizing prompts and responses,
        and highlighting the masked parts with color.

        Args:
            tensor_batch: The tensor batch containing trajectory data
            sample_idx: Starting index of samples to visualize
            max_samples: Maximum number of samples to visualize
        """
        from rllm.misc import colorful_print

        # Get the relevant tensors
        prompts = tensor_batch.batch["prompts"]
        responses = tensor_batch.batch["responses"]
        traj_mask = tensor_batch.batch[mask_key]
        token_level_scores = tensor_batch.batch["token_level_scores"]

        batch_size = prompts.shape[0]
        end_idx = min(sample_idx + max_samples, batch_size)

        for i in range(sample_idx, end_idx):
            colorful_print(f"\n===== Sample {i} =====", fg="cyan", bold=True)

            # Detokenize prompt
            prompt_tokens = prompts[i]
            prompt_mask = prompt_tokens != self.tokenizer.pad_token_id
            valid_prompt_tokens = prompt_tokens[prompt_mask]
            prompt_text = self.tokenizer.decode(valid_prompt_tokens)

            colorful_print("Prompt:", fg="green", bold=True)
            colorful_print(f"{prompt_text}\n", fg="green")

            # Detokenize response with color highlighting for masked tokens
            response_tokens = responses[i]
            response_mask = traj_mask[i]

            # Get non-padding tokens
            valid_indices = response_tokens != self.tokenizer.pad_token_id
            valid_response_tokens = response_tokens[valid_indices]
            valid_response_mask = response_mask[valid_indices]

            # Then show token-by-token with masking
            colorful_print("Response with masking:", fg="yellow", bold=True)

            for j, (token, mask) in enumerate(zip(valid_response_tokens, valid_response_mask, strict=False)):
                token_text = self.tokenizer.decode(token)

                # Check if this token has a reward
                has_reward = token_level_scores[i, j] != 0

                # Apply different colors based on mask and rewards
                if mask == 0:
                    # Masked token (not used in training)
                    colorful_print(token_text, fg="red", end="")
                elif has_reward:
                    # Token with reward
                    colorful_print(token_text, bg="green", end="")

                    reward_info = ""
                    if has_reward:
                        reward_info += f" R:{token_level_scores[i, j].item():.2f}"

                    colorful_print(reward_info, fg="magenta", end="")
                else:
                    # Normal token used in training
                    colorful_print(token_text, fg="blue", end="")

            print()  # New line after all tokens

            # Print reward summary
            total_reward = token_level_scores[i].sum().item()
            colorful_print("Rewards:", fg="green", bold=True)
            print(f" Trajectory Reward={total_reward:.2f}")

    def generate_agent_trajectories_async(self, timing_raw=None, meta_info=None, mode="Token"):
        """
        Generates agent trajectories asynchronously using the agent execution engine.

        This method runs the asynchronous `trajectory_generator` in a
        separate thread and yields the results synchronously through a queue.
        This allows the main training loop (which might be synchronous) to consume
        asynchronously generated trajectories.

        Args:
            timing_raw (dict, optional): Dictionary to store timing information. Defaults to {}.
            meta_info (dict, optional): Additional metadata for the generation process. Defaults to None.

        Yields:
            Any: Items generated by the `trajectory_generator`, typically
                 representing parts or results of agent trajectories in token format.
        """
        if timing_raw is None:
            timing_raw = {}
        queue = Queue()

        def runner():
            async def consume():
                async for item in self.agent_execution_engine.trajectory_generator(timing_raw=timing_raw, mode=mode, meta_info=meta_info):
                    queue.put(item)
                queue.put(None)  # sentinel to signal done

            asyncio.run(consume())

        Thread(target=runner, daemon=True).start()
        while True:
            item = queue.get()
            if item is None:
                break
            yield item

    def _transform_agent_steps(self, steps: list[dict], uids: np.ndarray):
        from verl.utils.torch_functional import pad_sequence_to_length

        all_prompts_list = []
        all_responses_list = []

        step_numbers = []  # number of steps of each episode, 0 indexed
        all_steps_idx_list = []
        all_steps_is_last_step_list = []
        all_steps_step_num = []  # total number of steps the trajectory this step belongs to have
        all_steps_step_ids = []
        training_rewards = []
        all_mc_returns = []  # Monte Carlo returns for each episode
        # the last step will have reward assigned and be used for advantage calculation

        for episode in steps:
            episode_steps = episode["steps"]
            idx = episode["idx"]
            training_reward = episode["trajectory_reward"]
            mc_returns = episode["mc_returns"]

            all_prompts_list.extend([torch.tensor(self.tokenizer.encode(s["prompt"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])
            all_responses_list.extend([torch.tensor(self.tokenizer.encode(s["response"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])

            step_numbers.append(len(episode_steps) - 1)
            training_rewards.append(training_reward)
            all_mc_returns.extend(mc_returns)

            all_steps_idx_list.extend([idx for _ in range(len(episode_steps))])
            all_steps_is_last_step_list.extend([False for _ in range(len(episode_steps))])
            all_steps_is_last_step_list[-1] = True

            all_steps_step_num.extend([len(episode_steps) for _ in range(len(episode_steps))])
            all_steps_step_ids.extend([f"{uids[idx]}_step{i}" for i in range(len(episode_steps))])

        # Convert all steps into token tensors
        # reverse the list and create tensors, pad, then flip to achieve left padding
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_prompts_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])

        prompts_batch = pad_sequence_to_length(prompts_batch, self.config.data.max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)

        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_responses_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        max_response_length = self.config.data.max_response_length
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)

        complete_step_batch = torch.concat([prompts_batch, response_batch], dim=1)
        attention_mask = torch.where(complete_step_batch != self.tokenizer.pad_token_id, 1, 0)
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # same as regular repsonse_mask, padded tensors will have this zeroed out
        traj_mask = torch.where(response_batch != self.tokenizer.pad_token_id, 1, 0)

        # Place all rewards to last response token of the last_step response
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        mc_return_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        prompt_length = prompts_batch.shape[1]
        valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)

        # reward is given for last token of every step for logging purposes, but only last steps will be used to calculate advantage
        step_index = 0
        for i, traj_score in enumerate(training_rewards):
            step_num = step_numbers[i] + 1  # since step_numbers is 0 indexed
            for _ in range(step_num):
                last_valid_idx = valid_response_length_sequences[step_index] - 1
                if last_valid_idx >= 0 and last_valid_idx < score_batch.shape[1]:
                    score_batch[step_index, last_valid_idx] = traj_score
                    mc_return_batch[step_index, last_valid_idx] = all_mc_returns[step_index]
                step_index += 1
        assert step_index == score_batch.shape[0], f"Number of total steps used should equal to batch size, but got {step_index} and {score_batch.shape[0]}"

        tensor_batch = {
            "input_ids": complete_step_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "mc_returns": mc_return_batch,
            "traj_mask": traj_mask,
        }

        batch_id = str(uuid.uuid4())
        non_tensor_batch = {
            "idxs": np.array(all_steps_idx_list),
            "step_nums": np.array(all_steps_step_num),
            "is_last_step": np.array(all_steps_is_last_step_list),
            "is_pad_step": np.array([False for _ in range(len(all_steps_idx_list))]),
            "batch_id": np.array([batch_id for _ in range(len(all_steps_idx_list))]),  # in case need to differentiate which iteration the step is coming from
            "step_ids": np.array(all_steps_step_ids),
        }

        meta_info = {"repeat_counts": [x + 1 for x in step_numbers]}

        result = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch, meta_info=meta_info)

        # Find indices of last steps for visualization
        last_step_indices = [i for i, is_last in enumerate(non_tensor_batch["is_last_step"]) if is_last]
        if last_step_indices:
            sample_indices = np.random.choice(last_step_indices, size=min(2, len(last_step_indices)), replace=False)
            for idx in sample_indices:
                self.visualize_trajectory(result, sample_idx=idx, max_samples=1)
        return result

    def _stepwise_advantage_broadcast(self, last_step_batch, other_step_batch):
        """
        Broadcast the advantage from last_step_batch to all other steps.
        """

        # NOTE: Currently takes the average of advantages. For GRPO, advantage and returns is uniform for each token so this makes no difference.
        # NOTE: For simplicity, assumes advantage and return is the same, which also holds for GRPO variants
        if "response_mask" not in other_step_batch.batch.keys():
            other_step_batch.batch["response_mask"] = compute_response_mask(other_step_batch)
        if "response_mask" not in last_step_batch.batch.keys():
            last_step_batch.batch["response_mask"] = compute_response_mask(last_step_batch)
        src_indices = last_step_batch.non_tensor_batch["idxs"]
        src_total_steps = last_step_batch.non_tensor_batch["step_nums"]
        tgt_indices = other_step_batch.non_tensor_batch["idxs"]
        src_advantages = last_step_batch.batch["advantages"]
        src_mask = last_step_batch.batch["response_mask"]
        tgt_mask = other_step_batch.batch["response_mask"]

        # Build idx -> scalar advantage
        idx_to_scalar_adv = {}
        for i, idx in enumerate(src_indices):
            mask = src_mask[i].bool()
            scalar = src_advantages[i][mask].mean()

            if self.config.agent.normalize_step_advantage:
                # normalize the advantage against number of steps
                scalar = scalar / src_total_steps[i]
                # reassign the normalized advantage to last_step_batch as well
                last_step_batch.batch["advantages"][i][mask] = scalar

            idx_to_scalar_adv[int(idx)] = scalar

        # Create new tensor for other_step_batch with per-token assignment
        scalar_rows = torch.stack([torch.full_like(tgt_mask[i], fill_value=idx_to_scalar_adv[int(idx)], dtype=torch.float32) for i, idx in enumerate(tgt_indices)])  # shape: (N2, T)

        # Apply the response mask of the target batch
        final_advantage = scalar_rows * tgt_mask

        # Assignment
        other_step_batch.batch["advantages"] = final_advantage
        other_step_batch.batch["returns"] = final_advantage

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        for i in range(pad_size):
            idx = original_batch_size + i
            batch.non_tensor_batch["is_last_step"][idx] = False
            batch.non_tensor_batch["is_pad_step"][idx] = True

        return batch

    def _extract_sample_output(self, batch: DataProto) -> str:
        """Extract a sample output from the batch for use in ChatGPT prompt."""
        try:
            # Get the first example from the batch
            response_batch = batch.batch['responses'][0]  # First response
            
            # Decode just the response part
            sample_text = self.tokenizer.decode(response_batch, skip_special_tokens=True)
            
            # Extract the part after "Assistant:" if present
            if "Assistant:" in sample_text:
                sample_text = "Assistant:" + sample_text.split("Assistant:", 1)[1]
            
            # Truncate if too long to avoid token limits
            if len(sample_text) > 2000:
                sample_text = sample_text[:1000] + "\n...(truncated)...\n" + sample_text[-1000:]
            
            return sample_text.strip()
        except Exception as e:
            print(f"Error extracting sample output: {e}")
            return "No sample available"

    def _generate_new_problems_with_chatgpt(self, sample_output: str) -> List[Dict[str, Any]]:
        """Generate new DeepCoder problems using ChatGPT based on the sample output."""
        try:
            if self.openai_client is None:
                print("OpenAI client not initialized. Cannot generate new problems.")
                return []
            
            print(f"Generating {self.problems_per_generation} problems one at a time...")
            all_problems = []
            successful_generations = 0
            failed_generations = 0
            
            for i in range(self.problems_per_generation):
                retry_count = 0
                problem_generated = False
                
                while retry_count < 3 and not problem_generated:
                    try:
                        # Construct the full prompt for single problem generation
                        full_prompt = DEEPCODER_BASE_PROMPT + "\n" + sample_output
                        
                        response = self.openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that generates competitive programming problems."},
                                {"role": "user", "content": full_prompt}
                            ],
                            max_completion_tokens=8000,
                            temperature=0.7
                        )
                        
                        gpt4_output = response.choices[0].message.content
                        
                        # Extract single problem from the GPT-4 output
                        problem = self._extract_single_problem_from_gpt4_output(gpt4_output)
                        
                        if problem:
                            all_problems.append(problem)
                            successful_generations += 1
                            problem_generated = True
                            print(f"Generated problem {i+1}/{self.problems_per_generation}")
                        else:
                            retry_count += 1
                            print(f"Failed to generate problem {i+1}, retry {retry_count}/3")
                        
                    except Exception as e:
                        if "rate_limit" in str(e).lower() or "429" in str(e):
                            if self._handle_api_rate_limit(retry_count):
                                retry_count += 1
                                continue
                            else:
                                retry_count += 1  # Ensure retry_count is incremented
                                break
                        else:
                            retry_count += 1
                            print(f"Error generating problem {i+1}, retry {retry_count}/3: {e}")
                
                if not problem_generated:
                    failed_generations += 1
                    print(f"Failed to generate problem {i+1} after all retries")
                
                # Small delay to avoid rate limiting
                import time
                time.sleep(1.0)  # Increased delay for better rate limiting
            
            print(f"Generation complete: {successful_generations} successful, {failed_generations} failed")
            
            # Validate all problems
            valid_problems = self._validate_generated_problems(all_problems)
            
            # If we don't have enough valid problems, try to fallback
            if len(valid_problems) < self.problems_per_generation // 2:
                print(f"Warning: Only {len(valid_problems)} valid problems generated, expected at least {self.problems_per_generation // 2}")
                if len(valid_problems) == 0:
                    self._fallback_to_original_dataloader()
            
            return valid_problems
            
        except Exception as e:
            print(f"Error in problem generation pipeline: {e}")
            return []

    def _extract_single_problem_from_gpt4_output(self, gpt4_output: str) -> Dict[str, Any]:
        """Extract a single problem from GPT-4 output and convert to DeepCoder format."""
        try:
            # Extract problem statement
            if "<problem>" not in gpt4_output or "</problem>" not in gpt4_output:
                print("No problem section found in GPT-4 output")
                return None
            
            problem_text = gpt4_output.split("<problem>")[1].split("</problem>")[0].strip()
            
            # Extract tests
            if "<tests>" not in gpt4_output or "</tests>" not in gpt4_output:
                print("No tests section found in GPT-4 output")
                return None
            
            tests_section = gpt4_output.split("<tests>")[1].split("</tests>")[0].strip()
            try:
                tests = json.loads(tests_section)
            except json.JSONDecodeError as e:
                print(f"Failed to parse tests JSON: {e}")
                return None
            
            # Extract metadata
            if "<metadata>" not in gpt4_output or "</metadata>" not in gpt4_output:
                print("No metadata section found in GPT-4 output")
                return None
            
            metadata_section = gpt4_output.split("<metadata>")[1].split("</metadata>")[0].strip()
            try:
                metadata = json.loads(metadata_section)
            except json.JSONDecodeError as e:
                print(f"Failed to parse metadata JSON: {e}")
                return None
            
            # Ensure tests are in the correct format
            for test in tests:
                if "testtype" not in test:
                    test["testtype"] = "stdin_stdout"
                if "metadata" not in test:
                    test["metadata"] = {"func_name": None}
            
            # Format the problem with LiveCodeBench system prompt
            try:
                from rllm.data.utils import fetch_live_code_bench_system_prompt
                formatted_question = fetch_live_code_bench_system_prompt(problem_text, None)
            except ImportError:
                print("Warning: Could not import fetch_live_code_bench_system_prompt, using raw problem text")
                formatted_question = problem_text
            
            deepcoder_problem = {
                "question": formatted_question,
                "ground_truth": json.dumps(tests),
                "data_source": "dynamic_generated",
                "uid": f"dynamic_{self.generated_problems_count}",
                "index": self.generated_problems_count,
                "starter_code": "",
                "metadata": json.dumps({
                    "difficulty": metadata.get("difficulty", "Medium"),
                    "type": metadata.get("type", "Algorithm"),
                    "generated_by": "gpt4o",
                    "generation_step": self.global_steps
                })
            }
            
            self.generated_problems_count += 1
            return deepcoder_problem
            
        except Exception as e:
            print(f"Error extracting single problem from GPT-4 output: {e}")
            return None

    def _extract_problems_from_gpt4_output(self, gpt4_output: str) -> List[Dict[str, Any]]:
        """Legacy method for extracting multiple problems (kept for compatibility)."""
        # This method is now deprecated in favor of single problem generation
        single_problem = self._extract_single_problem_from_gpt4_output(gpt4_output)
        return [single_problem] if single_problem else []

    def _create_dynamic_dataloader(self, problems: List[Dict[str, Any]]):
        """Create a new dataloader from ChatGPT generated problems."""
        try:
            if not problems:
                print("No problems provided for dynamic dataloader")
                return None
            
            # Convert problems to the format expected by the dataloader
            from datasets import Dataset
            from rllm.data.dataset import DatasetRegistry
            
            # Create dataset from problems
            dataset = Dataset.from_list(problems)
            
            # Register as a temporary dataset
            dataset_name = f"dynamic_deepcoder_{self.global_steps}"
            DatasetRegistry.register_dataset(dataset_name, dataset, "train")
            
            # Create new dataloader
            from verl.data import DataLoader
            from verl.data.data_utils import get_data_collator
            
            data_collator = get_data_collator(
                tokenizer=self.tokenizer,
                max_prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
                template_type=self.config.data.get("template_type", "base"),
                return_raw_chat=self.config.data.get("return_raw_chat", False),
            )
            
            new_dataloader = DataLoader(
                dataset=dataset,
                batch_size=self.config.data.train_batch_size,
                shuffle=True,
                collate_fn=data_collator,
                num_workers=0,
            )
            
            print(f"Created dynamic dataloader with {len(problems)} problems")
            return new_dataloader
            
        except Exception as e:
            print(f"Error creating dynamic dataloader: {e}")
            return None

    def _generate_and_update_dataloader(self):
        """Generate new problems and update the training dataloader."""
        try:
            # Extract a sample from the current batch if available
            sample_output = "No sample available"
            if hasattr(self, 'last_batch') and self.last_batch is not None:
                sample_output = self._extract_sample_output(self.last_batch)
            
            # Generate new problems
            new_problems = self._generate_new_problems_with_chatgpt(sample_output)
            
            if new_problems:
                # Log detailed statistics
                self._log_generation_stats(new_problems)
                
                # Save problems for analysis
                self._save_generated_problems(new_problems, self.global_steps)
                
                # Create new dataloader
                new_dataloader = self._create_dynamic_dataloader(new_problems)
                
                if new_dataloader is not None:
                    # Update the training dataloader
                    self.train_dataloader = new_dataloader
                    print(f"Updated training dataloader with {len(new_problems)} new problems")
                    
                    # Log comprehensive metrics
                    metrics = {
                        "dynamic_generation/problems_generated": len(new_problems),
                        "dynamic_generation/total_generated": self.generated_problems_count,
                        "dynamic_generation/generation_step": self.global_steps,
                        "dynamic_generation/active": self.dynamic_generation_active,
                        "dynamic_generation/dataloader_size": len(new_dataloader.dataset) if hasattr(new_dataloader, 'dataset') else 0
                    }
                    
                    # Add problem difficulty distribution
                    difficulty_counts = {}
                    problem_types = {}
                    for problem in new_problems:
                        try:
                            metadata = json.loads(problem.get("metadata", "{}"))
                            difficulty = metadata.get("difficulty", "Unknown")
                            problem_type = metadata.get("type", "Unknown")
                            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
                            problem_types[problem_type] = problem_types.get(problem_type, 0) + 1
                        except:
                            pass
                    
                    for difficulty, count in difficulty_counts.items():
                        metrics[f"dynamic_generation/difficulty/{difficulty}"] = count
                    for ptype, count in problem_types.items():
                        metrics[f"dynamic_generation/type/{ptype}"] = count
                    
                    # Log metrics
                    if hasattr(self, 'logger'):
                        self.logger.log(metrics, step=self.global_steps)
                    else:
                        print(f"Dynamic generation metrics: {metrics}")
                else:
                    print("Failed to create new dataloader, keeping current one")
            else:
                print("No new problems generated, keeping current dataloader")
                
        except Exception as e:
            print(f"Error in dynamic generation: {e}")
            print("Continuing with current dataloader")

    def _validate_generated_problems(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate generated problems and filter out invalid ones."""
        valid_problems = []
        
        for problem in problems:
            try:
                # Check required fields
                required_fields = ["question", "ground_truth", "data_source", "uid"]
                if not all(field in problem for field in required_fields):
                    print(f"Skipping problem missing required fields: {problem.get('uid', 'unknown')}")
                    continue
                
                # Validate ground_truth is valid JSON
                try:
                    tests = json.loads(problem["ground_truth"])
                    if not isinstance(tests, list) or len(tests) == 0:
                        print(f"Skipping problem with invalid tests: {problem['uid']}")
                        continue
                except json.JSONDecodeError as e:
                    print(f"Skipping problem with invalid JSON in ground_truth: {problem['uid']}, error: {e}")
                    continue
                
                # Check that tests have required fields
                for test in tests:
                    if not all(field in test for field in ["input", "output", "testtype"]):
                        print(f"Skipping problem with invalid test format: {problem['uid']}")
                        break
                else:
                    valid_problems.append(problem)
                    
            except Exception as e:
                print(f"Error validating problem {problem.get('uid', 'unknown')}: {e}")
                continue
        
        print(f"Validated {len(valid_problems)}/{len(problems)} problems")
        return valid_problems

    def _fallback_to_original_dataloader(self):
        """Fallback to original dataloader if dynamic generation fails."""
        try:
            print("Attempting to fallback to original dataloader...")
            # This would need to be implemented based on how the original dataloader is stored
            # For now, we'll just disable dynamic generation
            self.dynamic_generation_active = False
            print("Disabled dynamic generation due to errors")
        except Exception as e:
            print(f"Error in fallback: {e}")

    def _should_generate_new_problems(self) -> bool:
        """Check if we should generate new problems based on various conditions."""
        if not self.use_dynamic_generation or not self.dynamic_generation_active:
            return False
        
        # Check frequency
        if self.global_steps - self.last_generation_step < self.dynamic_generation_frequency:
            return False
        
        # Check if we have enough problems in current dataloader
        try:
            current_size = len(self.train_dataloader.dataset) if hasattr(self.train_dataloader, 'dataset') else 0
            if current_size < self.problems_per_generation:
                return True
        except:
            pass
        
        return True

    def _save_generated_problems(self, problems: List[Dict[str, Any]], step: int):
        """Save generated problems to file for analysis."""
        try:
            save_dir = os.path.join(self.config.trainer.default_local_dir, "generated_problems")
            os.makedirs(save_dir, exist_ok=True)
            
            # Use timestamp to avoid race conditions
            import time
            timestamp = int(time.time() * 1000)  # milliseconds
            filename = os.path.join(save_dir, f"problems_step_{step}_{timestamp}.json")
            
            with open(filename, 'w') as f:
                json.dump(problems, f, indent=2)
            
            print(f"Saved {len(problems)} generated problems to {filename}")
        except Exception as e:
            print(f"Error saving generated problems: {e}")

    def _log_generation_stats(self, problems: List[Dict[str, Any]]):
        """Log detailed statistics about generated problems."""
        if not problems:
            return
        
        stats = {
            "total_problems": len(problems),
            "avg_tests_per_problem": 0,
            "difficulty_distribution": {},
            "type_distribution": {},
            "avg_question_length": 0
        }
        
        total_tests = 0
        total_question_length = 0
        
        for problem in problems:
            # Count tests
            try:
                tests = json.loads(problem.get("ground_truth", "[]"))
                total_tests += len(tests)
            except:
                pass
            
            # Question length
            total_question_length += len(problem.get("question", ""))
            
            # Difficulty and type
            try:
                metadata = json.loads(problem.get("metadata", "{}"))
                difficulty = metadata.get("difficulty", "Unknown")
                problem_type = metadata.get("type", "Unknown")
                
                stats["difficulty_distribution"][difficulty] = stats["difficulty_distribution"].get(difficulty, 0) + 1
                stats["type_distribution"][problem_type] = stats["type_distribution"].get(problem_type, 0) + 1
            except:
                pass
        
        if problems:
            stats["avg_tests_per_problem"] = total_tests / len(problems)
            stats["avg_question_length"] = total_question_length / len(problems)
        
        print(f"Generation stats: {stats}")
        return stats

    def _handle_api_rate_limit(self, retry_count: int = 0, max_retries: int = 3) -> bool:
        """Handle API rate limiting with exponential backoff."""
        if retry_count >= max_retries:
            print(f"Max retries ({max_retries}) exceeded for API rate limiting")
            return False
        
        import time
        wait_time = (2 ** retry_count) * 5  # Exponential backoff: 5s, 10s, 20s
        print(f"Rate limited. Waiting {wait_time} seconds before retry {retry_count + 1}/{max_retries}")
        time.sleep(wait_time)
        return True

    def _validate_openai_api_key(self):
        """Validate OpenAI API key with a simple test call."""
        try:
            # Make a minimal test call to validate the API key
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Test"}],
                max_completion_tokens=1
            )
            print("OpenAI API key validated successfully")
        except Exception as e:
            raise Exception(f"OpenAI API key validation failed: {e}")
