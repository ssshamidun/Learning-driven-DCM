# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
#
# Rollout buffer for on-policy RL with teacher fields.
# 面向 on-policy 强化学习的采样缓冲，包含教师监督相关字段。

import torch
import numpy as np
from typing import Optional


class RolloutStorage:
    # === Per-step cache before commit | 单步转移缓存（提交前的暂存） ===
    class Transition:
        def __init__(self):
            # Core obs/action/value/logp | 基础观测/动作/价值/对数概率
            self.observations = None
            self.next_observations = None
            self.critic_obs = None
            self.observation_history = None
            self.commands = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None  # (actor_state, critic_state) or None

            # Teacher & stability fields | 教师与稳定性字段
            self.teacher_actions = None
            self.stab_xi_meas = None
            self.stab_xi_corr = None
            self.stab_delta_xi = None

        def clear(self):
            # Re-init to drop refs | 重新初始化以清理引用
            self.__init__()

    # === Buffer init | 缓冲区初始化 ===
    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape,
        all_obs_shape,
        obs_history_shape,
        commands_shape,
        actions_shape,
        device: str = "cpu",
    ):
        self.device = device
        self.obs_shape = obs_shape
        self.actions_shape = actions_shape

        # --- Core tensors | 核心张量 ---
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.next_observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)

        if all_obs_shape[0] is not None:
            self.critic_obs = torch.zeros(num_transitions_per_env, num_envs, *all_obs_shape, device=self.device)
        else:
            self.critic_obs = None

        self.observation_history = torch.zeros(num_transitions_per_env, num_envs, *obs_history_shape, device=self.device)
        self.commands = torch.zeros(num_transitions_per_env, num_envs, *commands_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # --- PPO extras | PPO 额外数据 ---
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # --- Teacher/stability tensors | 教师/稳定性张量 ---
        self.teacher_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.stab_xi_meas = torch.zeros(num_transitions_per_env, num_envs, 2, device=self.device)
        self.stab_xi_corr = torch.zeros(num_transitions_per_env, num_envs, 2, device=self.device)
        self.stab_delta_xi = torch.zeros(num_transitions_per_env, num_envs, 2, device=self.device)

        # --- Meta ---
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0

        # RNN hidden states timeline (actor/critic) | RNN 隐状态时间轴（actor/critic）
        self.saved_hidden_states_a: Optional[list] = None
        self.saved_hidden_states_c: Optional[list] = None

    # === Commit one transition | 写入一步转移 ===
    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        # Core copies | 基础字段拷贝
        self.observations[self.step].copy_(transition.observations)
        self.next_observations[self.step].copy_(transition.next_observations)
        if self.critic_obs is not None:
            self.critic_obs[self.step].copy_(transition.critic_obs)
        self.observation_history[self.step].copy_(transition.observation_history)
        self.commands[self.step].copy_(transition.commands)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        # Teacher & stability | 教师与稳定性
        self.teacher_actions[self.step].copy_(transition.teacher_actions)
        self.stab_xi_meas[self.step].copy_(transition.stab_xi_meas)
        self.stab_xi_corr[self.step].copy_(transition.stab_xi_corr)
        self.stab_delta_xi[self.step].copy_(transition.stab_delta_xi)

        # RNN states | RNN 隐状态
        self._save_hidden_states(transition.hidden_states)

        self.step += 1

    # === Save hidden states timeline | 保存隐状态时间轴 ===
    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return

        # Make tuple shape consistent with LSTM format | 统一成LSTM元组格式
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # Lazy init storages | 延迟初始化存储
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device)
                for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device)
                for i in range(len(hid_c))
            ]

        # Copy current step | 拷贝本步隐状态
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    # === Reset write pointer | 重置写指针 ===
    def clear(self):
        self.step = 0

    # === GAE/TD(λ) returns | GAE/TD(λ) 回报计算 ===
    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Normalize advantages | 归一化优势
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    # === Quick stats: avg length & reward | 简要统计：平均长度与回报 ===
    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    # === Mini-batch generator (policy) | 策略更新的小批量生成器 ===
    def mini_batch_generator(
        self,
        num_group: int,
        num_mini_batches: int,
        num_epochs: int = 8,
    ):
        # Grouped sampling by env-index [0..num_group-1]
        # 按环境索引 [0..num_group-1] 分组采样
        group_batch_size = num_group * self.num_transitions_per_env
        group_mini_batch_size = group_batch_size // num_mini_batches
        group_indices = torch.randperm(num_mini_batches * group_mini_batch_size, requires_grad=False, device=self.device)
        group_group_idx = torch.arange(0, num_group, device=self.device)

        # Flatten time/env within group | 组内展平时间与环境维
        group_observations   = self.observations[:, group_group_idx, :].flatten(0, 1)
        group_critic_obs     = self.critic_obs[:, group_group_idx, :].flatten(0, 1)
        group_obs_history    = self.observation_history[:, group_group_idx, :].flatten(0, 1)
        group_commands       = self.commands[:, group_group_idx, :].flatten(0, 1)
        group_actions        = self.actions[:, group_group_idx, :].flatten(0, 1)
        group_values         = self.values[:, group_group_idx, :].flatten(0, 1)
        group_returns        = self.returns[:, group_group_idx, :].flatten(0, 1)
        group_old_log_prob   = self.actions_log_prob[:, group_group_idx, :].flatten(0, 1)
        group_advantages     = self.advantages[:, group_group_idx, :].flatten(0, 1)
        group_old_mu         = self.mu[:, group_group_idx, :].flatten(0, 1)
        group_old_sigma      = self.sigma[:, group_group_idx, :].flatten(0, 1)

        # Teacher/stability | 教师/稳定性
        group_teacher_action = self.teacher_actions[:, group_group_idx, :].flatten(0, 1)
        group_stab_xi_meas   = self.stab_xi_meas[:, group_group_idx, :].flatten(0, 1)
        group_stab_xi_corr   = self.stab_xi_corr[:, group_group_idx, :].flatten(0, 1)
        group_stab_delta_xi  = self.stab_delta_xi[:, group_group_idx, :].flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * group_mini_batch_size
                end = (i + 1) * group_mini_batch_size
                idx = group_indices[start:end]

                # Assemble batch views | 组装批数据视图
                obs_batch                 = group_observations[idx]
                critic_obs_batch          = group_critic_obs[idx]
                obs_history_batch         = group_obs_history[idx]
                group_obs_history_batch   = group_obs_history[idx]   # kept for API parity | 保持接口一致
                group_commands_batch      = group_commands[idx]
                actions_batch             = group_actions[idx]
                target_values_batch       = group_values[idx]
                advantages_batch          = group_advantages[idx]
                returns_batch             = group_returns[idx]
                old_actions_log_prob_batch= group_old_log_prob[idx]
                old_mu_batch              = group_old_mu[idx]
                old_sigma_batch           = group_old_sigma[idx]

                teacher_action            = group_teacher_action[idx]
                stab_xi_meas              = group_stab_xi_meas[idx]
                stab_xi_corr              = group_stab_xi_corr[idx]
                stab_delta_xi             = group_stab_delta_xi[idx]

                yield (
                    obs_batch,
                    critic_obs_batch,
                    obs_history_batch,
                    group_obs_history_batch,
                    group_commands_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    teacher_action,
                    stab_xi_meas,
                    stab_xi_corr,
                    stab_delta_xi,
                )

    # === Mini-batch generator (encoder aux) | 编码器辅助训练小批量生成器 ===
    def encoder_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        critic_obs = self.critic_obs.flatten(0, 1) if self.critic_obs is not None else observations
        obs_history = self.observation_history.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                idx = indices[start:end]

                next_obs_batch = next_observations[idx]
                critic_obs_batch = critic_obs[idx]
                obs_history_batch = obs_history[idx]
                yield next_obs_batch, critic_obs_batch, obs_history_batch
