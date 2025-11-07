# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
#
# (Bilingual brief notes | 双语简注)
# PPO core with optional encoder head and teacher-supervised fields in storage.
# 带可选编码器与教师监督字段的 PPO 主体。

import torch
import torch.nn as nn
import torch.optim as optim

from .mlp_encoder import MLP_Encoder
from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage


class PPO:
    """
    Proximal Policy Optimization with encoder aux updates.
    结合编码器辅助更新的 PPO 实现。
    """
    actor_critic: ActorCritic
    encoder: MLP_Encoder

    def __init__(
        self,
        num_group,
        encoder,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        vae_beta=1.0,
        est_learning_rate=1.0e-3,
        ts_learning_rate=1.0e-4,
        critic_take_latent=False,
        early_stop=False,
        anneal_lr=False,
        device="cpu",
    ):
        # === Hyperparams & objects | 超参与组件 ===
        self.device = device
        self.num_group = num_group

        self.desired_kl = desired_kl
        self.early_stop = early_stop
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.vae_beta = vae_beta
        self.critic_take_latent = critic_take_latent

        self.encoder = encoder
        self.actor_critic = actor_critic.to(self.device)
        self.storage = None  # set by init_storage | 由 init_storage 设置

        # Optimizers | 优化器
        self.optimizer = optim.Adam([{"params": self.actor_critic.parameters()}], lr=learning_rate)
        if self.encoder.num_output_dim != 0:
            self.extra_optimizer = optim.Adam(self.encoder.parameters(), lr=est_learning_rate)
        else:
            self.extra_optimizer = None

        # One-step transition cache | 单步转移缓存
        self.transition = RolloutStorage.Transition()

        # PPO params | PPO 超参
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    # === Storage init | 轨迹存储初始化 ===
    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        obs_history_shape,
        commands_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_history_shape,
            commands_shape,
            action_shape,
            self.device,
        )

    # === Mode switches | 训练/测试模式切换 ===
    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    # === Policy act step (writes transition) | 策略前向，写入转移 ===
    def act(self, obs, obs_history, commands, critic_obs):
        # Critic input always concat commands | critic 输入拼接 commands
        critic_obs = torch.cat((critic_obs, commands), dim=-1)

        # Student encoder | 学生编码器
        encoder_out = self.encoder.encode(obs_history)

        # Actions | 动作
        self.transition.actions = self.actor_critic.act(
            torch.cat((encoder_out, obs, commands), dim=-1)
        ).detach()

        # Values | 价值评估
        if self.critic_take_latent:
            critic_obs = torch.cat((critic_obs, encoder_out), dim=-1)
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()

        # Log-prob & stats | 概率与统计
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        # Cache current obs before env.step | 先存当前观测
        self.transition.observations = obs
        self.transition.critic_obs = critic_obs
        self.transition.observation_history = obs_history
        self.transition.commands = commands
        return self.transition.actions

    # === Env step handler with teacher info | 环境步处理（含教师信息） ===
    def process_env_step_with_teacher(
        self,
        rewards,
        dones,
        infos,
        next_obs,
        teacher_actions,
        stab_info,
    ):
        """
        Same as process_env_step plus teacher/stability fields.
        等同原版 process_env_step，并多记录教师/稳定性字段。
        """
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Time-limit bootstrap | 时间截断回填
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        self.transition.next_observations = next_obs

        # Teacher & stability | 教师与稳定性信息
        self.transition.teacher_actions = teacher_actions.detach().clone()
        self.transition.stab_xi_meas = stab_info["xi_meas"].detach().clone()
        self.transition.stab_xi_corr = stab_info["xi_corr"].detach().clone()
        self.transition.stab_delta_xi = stab_info["delta_xi"].detach().clone()

        # Push to storage & reset | 写入存储并复位策略内部状态
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    # === GAE returns | 广义优势估计回报 ===
    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    # === PPO update loop | PPO 更新循环 ===
    def update(self):
        num_updates = 0
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_kl = 0.0

        # -------- Policy update (PPO) | 策略更新 --------
        generator = self.storage.mini_batch_generator(
            self.num_group, self.num_mini_batches, self.num_learning_epochs
        )
        for (
            obs_batch,
            critic_obs_batch,
            obs_history_batch, _,
            group_commands_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            teacher_action_batch,
            stab_xi_meas_batch,
            stab_xi_corr_batch,
            stab_delta_xi_batch,
        ) in generator:
            # Forward | 前向
            encoder_out_batch = self.encoder.encode(obs_history_batch)
            commands_batch = group_commands_batch
            self.actor_critic.act(
                torch.cat((encoder_out_batch, obs_batch, commands_batch), dim=-1)
            )

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL monitor (no grad) | KL 监控（不求导）
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

            # Adaptive LR / early stop | 自适应学习率 / 早停
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for g in self.optimizer.param_groups:
                        g["lr"] = self.learning_rate

            if self.desired_kl is not None and self.early_stop:
                if kl_mean > self.desired_kl * 1.5:
                    print("early stop, num_updates =", num_updates)
                    break

            # Surrogate loss | 策略替代损失
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss | 价值损失
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Entropy | 熵正则
            entropy_batch_mean = entropy_batch.mean()

            # Teacher imitation (L2) | 教师模仿损失（L2）
            imit_weight = 0.2#0.2  # could be cfg | 可放入配置
            imitation_loss = (actions_batch - teacher_action_batch).pow(2).mean()

            # Total loss | 总损失
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch_mean
                + imit_weight * imitation_loss
            )

            # Optional LR anneal | 可选学习率退火
            if self.anneal_lr:
                frac = 1.0 - num_updates / (self.num_learning_epochs * self.num_mini_batches)
                self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

            # Backprop | 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Accumulate logs | 统计
            num_updates += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl += kl_mean.item()

        # -------- History Encoder aux update | 编码器辅助更新 --------
        num_updates_extra = 0
        mean_extra_loss = 0.0
        if self.extra_optimizer is not None:
            generator = self.storage.encoder_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
            for (
                next_obs_batch,
                critic_obs_batch,
                obs_history_batch,
            ) in generator:
                # Forward | 前向
                if self.encoder.is_mlp_encoder:
                    self.encoder.encode(obs_history_batch)
                    encode_batch = self.encoder.get_encoder_out()

                # Example aux loss (align first 3 dims) | 示例辅助损失（对齐前三维）
                if self.encoder.is_mlp_encoder:
                    extra_loss = (encode_batch[:, 0:3] - critic_obs_batch[:, 0:3]).pow(2).mean()
                else:
                    extra_loss = torch.zeros_like(value_loss)

                # Step | 更新
                self.extra_optimizer.zero_grad()
                extra_loss.backward()
                self.extra_optimizer.step()

                num_updates_extra += 1
                mean_extra_loss += extra_loss.item()

        # === Averages | 均值统计 ===
        mean_value_loss /= max(1, num_updates)
        if num_updates_extra > 0:
            mean_extra_loss /= num_updates_extra  # fix: use num_updates_extra | 修正：用额外更新步数
        mean_surrogate_loss /= max(1, num_updates)
        mean_kl /= max(1, num_updates)

        # Clear storage | 清空存储
        self.storage.clear()

        return (mean_value_loss, mean_extra_loss, mean_surrogate_loss, mean_kl)
