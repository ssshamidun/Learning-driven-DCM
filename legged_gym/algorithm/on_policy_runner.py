# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: BSD-3-Clause

import time
import os
from collections import deque
import statistics
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

from .ppo import PPO
from .mlp_encoder import MLP_Encoder
from .actor_critic import ActorCritic
from legged_gym.envs.vec_env import VecEnv
from typing import Dict, Any, Tuple
import csv



class OnPolicyRunner:
    """
    On-policy training loop with CTS + DCM teacher. 
    结合并行PPO与DCM教师的on-policy训练器。
    - Creates encoder/actor-critic/algorithm and storages.
      初始化编码器、策略网络、算法与存储。
    - Integrates DCM preview + IK + offset net into rollout.
      将DCM预观+IK+偏置网络并入采样流程。
    """

    def __init__(self, env: VecEnv, train_cfg: Dict[str, Any], log_dir=None, device="cpu"):
        # === Parse configs | 解析配置 ===
        self.cfg = train_cfg["runner"]
        self.ecd_cfg = train_cfg[self.cfg["encoder_class_name"]]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # === Build encoder | 构建编码器 ===
        encoder = eval(self.cfg["encoder_class_name"])(**self.ecd_cfg).to(self.device)

        # === Compute critic obs dim | 计算critic观测维度 ===
        num_critic_obs = self.env.num_critic_obs + self.env.num_commands
        if self.alg_cfg["critic_take_latent"]:
            num_critic_obs += encoder.num_output_dim

        # === Build policy (Actor-Critic) | 构建策略网络 ===
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(
            self.env.num_obs + encoder.num_output_dim + self.env.num_commands,
            num_critic_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        # === Build algorithm (PPO or alike) | 构建算法（PPO等） ===
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg = alg_class(
            self.env.num_envs,
            encoder,
            actor_critic,
            device=self.device,
            **self.alg_cfg,
        )

        # === Storage init | 轨迹存储初始化 ===
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [num_critic_obs],
            [self.env.obs_history_length * self.env.num_obs],
            [self.env.num_commands],
            [self.env.num_actions],
        )

        # === Running norm placeholders | 观测归一化占位 ===
        self.obs_mean = torch.tensor(0, dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_std = torch.tensor(1, dtype=torch.float, device=self.device, requires_grad=False)

        # === Logging | 日志记录 ===
        self.log_dir = log_dir
        # === Offset CSV export knobs (lightweight) ===
        self.export_offset_csv = self.cfg.get("export_offset_csv", True)          # 是否启用导出
        self.export_offset_env_ids = self.cfg.get("export_offset_env_ids", [0,5,42])  # 抽样的 env 索引
        self.export_offset_every = self.cfg.get("export_offset_every", 50)        # 每隔多少个迭代导出
        self.export_offset_dir = os.path.join(self.log_dir or ".", "offset_exports")
        os.makedirs(self.export_offset_dir, exist_ok=True)
        self._export_csv_path = os.path.join(self.export_offset_dir, "fixed_vs_offset_samples.csv")
        self._export_csv_header_written = False

        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self._last_xi_meas = None
        self._last_com_vel = None
        self._last_V_fixed = None
        self._last_V_corr = None

        _ = self.env.reset()

        # === DCM teacher stack | DCM教师模块 ===
        from .dcm_teacher import DCMPredictor, OffsetNetwork, InverseKinematics, build_teacher_action

        # (A) DCM predictor | DCM预测器
        self.omega0 = np.sqrt(9.81 / 0.5)  # z_c=0.5m, ω₀≈4.43
        self.dcm_predictor = DCMPredictor(
            omega0=self.omega0,
            dt=self.env.sim_params.dt,
            preview_horizon=20,
            z_c=0.5,
        )

        # (B) Offset net | 稳定偏置网络
        # in_dim: dcm_err(2)+xi_meas(2)+contact(2)+cmd(2)+terrain(?)
        self.offset_net = OffsetNetwork(in_dim=8 + 9, hidden_dim=128, out_dim=2).to(self.device)
        self.offset_optimizer = torch.optim.Adam(self.offset_net.parameters(), lr=self.cfg.get("offset_lr", 3e-4))

        # (C) IK | 逆运动学
        self.ik_solver = InverseKinematics(l1=0.3, l2=0.32)

        # (D) Hook builder | 教师动作构建函数
        self.build_teacher_action = build_teacher_action

        # (E) Offset train knobs | 偏置训练超参
        self.offset_loss_coef = self.cfg.get("offset_loss_coef", 1.0)
        self.offset_max_grad_norm = self.cfg.get("offset_max_grad_norm", 1.0)

    # 在 on_policy_runner.py 中修改 _maybe_write_offset_samples 方法


    def _maybe_write_offset_samples(self, it, t, teacher_info):
        """
        导出物理稳定性评估指标，包括：
        1. 物理一致性残差（fixed vs corrected）
        2. CMP（质心力矩点）偏差
        3. 稳定域裕度
        4. 能量散逸率
        """
        if not self.export_offset_csv or (it % self.export_offset_every != 0):
            return
        if teacher_info is None:
            return

        import csv
        import os
        os.makedirs(self.export_offset_dir, exist_ok=True)

        # === 基础量 ===
        xi_ref   = teacher_info["xi_ref"].detach()           # [N, 2] 固定DCM参考
        xi_corr  = teacher_info["xi_corrected"].detach()     # [N, 2] 修正后DCM参考
        xi_meas  = teacher_info["xi_meas"].detach()          # [N, 2] 当前测量DCM
        delta_xi = teacher_info["delta_xi"].detach()         # [N, 2] 偏移量
        
        zmp_ref_fixed = teacher_info.get("zmp_ref_fixed", xi_ref).detach()  # [N, 2]
        zmp_ref_corr  = teacher_info.get("zmp_ref_corr", xi_corr).detach()   # [N, 2]
        
        # 获取物理常数
        omega0 = float(teacher_info.get("omega0", self.omega0))
        dt     = float(teacher_info.get("dt", self.env.sim_params.dt if hasattr(self.env, "sim_params") else self.env.dt))

        # === 计算 ξ̇ (DCM速度) ===
        if self._last_xi_meas is None or self._last_xi_meas.shape != xi_meas.shape:
            self._last_xi_meas = xi_meas.clone()
        xi_dot = (xi_meas - self._last_xi_meas) / max(dt, 1e-6)  # [N, 2]
        
        # Reset帧处理
        reset_mask = None
        if hasattr(self.env, "reset_buf"):
            reset_mask = (self.env.reset_buf > 0)
        elif hasattr(self.env, "done"):
            reset_mask = self.env.done
        if reset_mask is not None:
            xi_dot[reset_mask] = 0.0
        
        self._last_xi_meas = xi_meas.clone()

        # === 核心物理指标 ===
        
        # 1. 物理一致性残差: ε = ||ξ̇ - ω₀(ξ - r)||
        # Fixed版本（不加偏移）
        xi_dynamics_fixed = omega0 * (xi_meas - zmp_ref_fixed)  # [N, 2] 理论ξ̇
        physics_residual_fixed = torch.norm(xi_dot - xi_dynamics_fixed, dim=-1)  # [N]
        
        # Corrected版本（加偏移）
        xi_dynamics_corr = omega0 * (xi_meas - zmp_ref_corr)
        physics_residual_corr = torch.norm(xi_dot - xi_dynamics_corr, dim=-1)
        
        # 2. CMP偏差（质心力矩点与ZMP的距离，反映动力学误差）
        # CMP = c - c̈/ω₀² （假设测量的加速度）
        com_pos = self.env.base_position[:, :2].detach()  # [N, 2]
        com_vel = self.env.base_lin_vel[:, :2].detach()   # [N, 2]
        
        # 估计加速度（简单差分）
        # 估计加速度（简单差分）
        if self._last_com_vel is None:
            self._last_com_vel = com_vel.clone()
            com_acc = torch.zeros_like(com_vel)  # 第一次调用时加速度为0
        else:
            com_acc = (com_vel - self._last_com_vel) / max(dt, 1e-6)
            if reset_mask is not None:
                com_acc[reset_mask] = 0.0
            self._last_com_vel = com_vel.clone()
        if reset_mask is not None:
            com_acc[reset_mask] = 0.0
        self._last_com_vel = com_vel.clone()
        
        cmp_estimated = com_pos - com_acc / (omega0**2)  # [N, 2]
        cmp_error_fixed = torch.norm(cmp_estimated - zmp_ref_fixed, dim=-1)
        cmp_error_corr  = torch.norm(cmp_estimated - zmp_ref_corr, dim=-1)
        
        # 3. 稳定域裕度（DCM到支撑边界的距离）
        support_center = self.env.base_position[:, :2].detach()
        stability_radius = 0.15  # 可行走基的半径（米）
        
        # Fixed版本的裕度
        margin_fixed = stability_radius - torch.norm(xi_ref - support_center, dim=-1)
        # Corrected版本的裕度
        margin_corr = stability_radius - torch.norm(xi_corr - support_center, dim=-1)
        
        # 4. 能量散逸率（Lyapunov函数变化率）
        # V = 0.5 * ||ξ - r||²，理想情况下 dV/dt < 0
        V_fixed = 0.5 * torch.sum((xi_meas - zmp_ref_fixed)**2, dim=-1)
        V_corr  = 0.5 * torch.sum((xi_meas - zmp_ref_corr)**2, dim=-1)
        
        if self._last_V_fixed is None:
            self._last_V_fixed = V_fixed.clone()
            self._last_V_corr = V_corr.clone()
            dVdt_fixed = torch.zeros_like(V_fixed)  # 第一次调用时变化率为0
            dVdt_corr = torch.zeros_like(V_corr)
        else:
            dVdt_fixed = (V_fixed - self._last_V_fixed) / max(dt, 1e-6)
            dVdt_corr  = (V_corr - self._last_V_corr) / max(dt, 1e-6)
            
            if reset_mask is not None:
                dVdt_fixed[reset_mask] = 0.0
                dVdt_corr[reset_mask] = 0.0
            
            self._last_V_fixed = V_fixed.clone()
            self._last_V_corr = V_corr.clone()
        
        if reset_mask is not None:
            dVdt_fixed[reset_mask] = 0.0
            dVdt_corr[reset_mask] = 0.0
        
        self._last_V_fixed = V_fixed.clone()
        self._last_V_corr = V_corr.clone()
        
        # 5. 接触状态（用于分析步态影响）
        contact_forces = self.env.contact_forces[:, self.env.feet_indices, 2].detach()
        is_left_support = (contact_forces[:, 0] > contact_forces[:, 1]).float()
        both_contact = ((contact_forces[:, 0] > 1.0) & (contact_forces[:, 1] > 1.0)).float()
        
        # 6. 地形信息
        terrain_mean = getattr(self.env, "terrain_mean", torch.zeros(self.env.num_envs, device=self.device))
        terrain_std  = getattr(self.env, "terrain_std", torch.zeros(self.env.num_envs, device=self.device))
        terrain_lvl  = getattr(self.env, "terrain_level", torch.zeros(self.env.num_envs, device=self.device))
        
        # 7. 指令速度（用于分析不同速度下的表现）
        cmd_vel = self.env.commands[:, :2].detach()
        cmd_vel_norm = torch.norm(cmd_vel, dim=-1)

        # === 写入CSV ===
        csv_path = os.path.join(self.export_offset_dir, "physics_stability_metrics.csv")
        write_header = not os.path.exists(csv_path)
        
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            
            if write_header:
                header = [
                    "it", "t", "env_id",
                    # 基础DCM量
                    "xi_meas_x", "xi_meas_y", "xi_dot_x", "xi_dot_y",
                    "xi_ref_fixed_x", "xi_ref_fixed_y",
                    "xi_ref_corr_x", "xi_ref_corr_y",
                    "delta_xi_x", "delta_xi_y",
                    # 核心物理指标
                    "physics_residual_fixed", "physics_residual_corr",
                    "cmp_error_fixed", "cmp_error_corr",
                    "stability_margin_fixed", "stability_margin_corr",
                    "energy_dissipation_fixed", "energy_dissipation_corr",
                    # 辅助信息
                    "is_left_support", "both_feet_contact",
                    "cmd_vel_norm", "cmd_vel_x", "cmd_vel_y",
                    "terrain_mean", "terrain_std", "terrain_level",
                    "omega0", "dt",
                ]
                w.writerow(header)
            
            # 仅记录指定env_id（节流）
            env_ids = self.export_offset_env_ids
            for i in env_ids:
                if i < 0 or i >= xi_meas.shape[0]:
                    continue
                
                row = [
                    int(it), int(t), int(i),
                    # 基础DCM量
                    xi_meas[i,0].item(), xi_meas[i,1].item(),
                    xi_dot[i,0].item(), xi_dot[i,1].item(),
                    xi_ref[i,0].item(), xi_ref[i,1].item(),
                    xi_corr[i,0].item(), xi_corr[i,1].item(),
                    delta_xi[i,0].item(), delta_xi[i,1].item(),
                    # 核心物理指标
                    physics_residual_fixed[i].item(),
                    physics_residual_corr[i].item(),
                    cmp_error_fixed[i].item(),
                    cmp_error_corr[i].item(),
                    margin_fixed[i].item(),
                    margin_corr[i].item(),
                    dVdt_fixed[i].item(),
                    dVdt_corr[i].item(),
                    # 辅助信息
                    is_left_support[i].item(),
                    both_contact[i].item(),
                    cmd_vel_norm[i].item(),
                    cmd_vel[i,0].item(), cmd_vel[i,1].item(),
                    terrain_mean[i].item() if terrain_mean is not None else 0.0,
                    terrain_std[i].item() if terrain_std is not None else 0.0,
                    terrain_lvl[i].item() if terrain_lvl is not None else 0.0,
                    omega0, dt,
                ]
                w.writerow(row)


    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # === Writer init | 日志写入器初始化 ===
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
            if self.logger_type == "wandb":
                from ..utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        # === Randomize ep length (optional) | 随机化episode起点（可选） ===
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # === Get initial obs | 获取初始观测 ===
        obs, obs_history, commands, critic_obs = self.env.get_observations()
        obs, obs_history, commands, critic_obs = (
            obs.to(self.device),
            obs_history.to(self.device),
            commands.to(self.device),
            critic_obs.to(self.device),
        )

        # === Train mode switches | 训练模式开关 ===
        self.alg.actor_critic.train()
        self.offset_net.train()

        # === Rolling stats | 统计缓存 ===
        ep_infos = []
        rewbuffer, lenbuffer = deque(maxlen=100), deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        offset_loss_val = 0.0  # for logging

        # === Main training loop | 训练主循环 ===
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # ----- (I) Rollout collection | 轨迹采样 -----
            for _ in range(self.num_steps_per_env):
                # 1) Student action | 学生策略出动作
                with torch.no_grad():
                    actions = self.alg.act(obs, obs_history, commands, critic_obs)

                # 2) Teacher state | 教师状态
                teacher_state = self.env.get_teacher_state()

                # 3) Teacher action & DCM info | 教师动作与DCM信息
                a_ref_t, teacher_info = self.build_teacher_action(
                    self.dcm_predictor,
                    self.offset_net,
                    self.ik_solver,
                    teacher_state,
                )
                #print(a_ref_t[1,:])

                # 4) Push DCM buffers for loss | 推送DCM张量供环境计算偏置损失
                self.env.set_dcm_info(
                    teacher_info["xi_meas"].detach(),   # meas no-grad | 测量值不回传梯度
                    teacher_info["xi_corrected"],       # keep graph | 保留计算图
                    teacher_info["delta_xi"],           # keep graph | 保留计算图
                )

                # 5) Step env | 环境前进一步（loss中含delta_xi图）
                (
                    obs_next,
                    rewards_main,
                    loss_offset,    # [N], differentiable to offset_net | 可回传到偏置网
                    dones,
                    infos,
                    obs_history_next,
                    commands_next,
                    critic_obs_buf,
                ) = self.env.step(actions)
                self._maybe_write_offset_samples(it, self.env.episode_length_buf[0].item() if hasattr(self.env, "episode_length_buf") else 0, teacher_info)
                # 6) Offset update (per-step) | 偏置网络每步更新
                offset_loss = self.offset_loss_coef * loss_offset.mean()
                self.offset_optimizer.zero_grad(set_to_none=True)
                offset_loss.backward()
                if self.offset_max_grad_norm and self.offset_max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.offset_net.parameters(), self.offset_max_grad_norm)
                self.offset_optimizer.step()
                offset_loss_val = float(offset_loss.item())

                # 7) Write transition to PPO | 写入PPO缓存
                a_ref_t_device = a_ref_t.to(self.device)
                stab_info = {
                    "xi_ref": teacher_info["xi_ref"].detach(),
                    "xi_meas": teacher_info["xi_meas"].detach(),
                    "xi_corr": teacher_info["xi_corrected"].detach(),
                    "delta_xi": teacher_info["delta_xi"].detach(),
                    "dcm_error": teacher_info["dcm_error"].detach(),
                }
                self.alg.process_env_step_with_teacher(
                    rewards_main, dones, infos, obs_next, a_ref_t_device, stab_info
                )

                # 8) Rolling metrics | 采样阶段统计
                if self.log_dir is not None:
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    cur_reward_sum += rewards_main
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                # 9) advance obs | 推进观测
                obs, obs_history, commands, critic_obs = (
                    obs_next, obs_history_next, commands_next, critic_obs_buf
                )

            stop = time.time()
            collection_time = stop - start

            # ----- (II) Compute returns | 计算价值回报 -----
            start = stop
            critic_obs_ = torch.cat((critic_obs, commands), dim=-1)
            with torch.no_grad():
                if self.alg.critic_take_latent:
                    encoder_out = self.alg.encoder.encode(obs_history)
                    self.alg.compute_returns(torch.cat((critic_obs_, encoder_out), dim=-1))
                else:
                    self.alg.compute_returns(critic_obs_)
            stop = time.time()
            learn_prep_time = stop - start  # (kept implicit)

            # ----- (III) PPO update | 策略更新 -----
            start = stop
            (mean_value_loss, mean_extra_loss, mean_surrogate_loss, mean_kl) = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            # ----- (IV) Logging & ckpt | 日志与存档 -----
            if self.log_dir is not None and self.writer is not None:
                self.writer.add_scalar("Loss/offset", offset_loss_val, it)
                if hasattr(self.env, "xi_meas_buf"):
                    dcm_error_norm = torch.norm(self.env.xi_meas_buf - self.env.xi_corr_buf, dim=-1).mean()
                    self.writer.add_scalar("DCM/tracking_error", dcm_error_norm.item(), it)
                    delta_xi_norm = torch.norm(self.env.delta_xi_buf, dim=-1).mean()
                    self.writer.add_scalar("DCM/delta_xi_magnitude", delta_xi_norm.item(), it)

                self.log(
                    {
                        **locals(),
                        "it": it,
                        "num_learning_iterations": num_learning_iterations,
                        "ep_infos": ep_infos,
                        "rewbuffer": rewbuffer,
                        "lenbuffer": lenbuffer,
                        "collection_time": collection_time,
                        "learn_time": learn_time,
                        "mean_value_loss": mean_value_loss,
                        "mean_extra_loss": mean_extra_loss,
                        "mean_surrogate_loss": mean_surrogate_loss,
                        "mean_kl": mean_kl,
                        "offset_loss_val": offset_loss_val,
                    }
                )

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

        # === Wrap up | 收尾 ===
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs, width=80, pad=35):
        """
        Console & TB logging for training stats.
        训练过程的控制台与TensorBoard日志。
        """
        # --- Global perf | 全局性能 ---
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # --- Episode stats | 回合统计 ---

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))

                value = torch.mean(infotensor)
                if key == "group_terrain_level":
                    value = value * 1.5
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""


        # --- Scalar logs | 标量日志 ---
        mean_std = torch.exp(self.alg.actor_critic.logstd).mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/encoder", locs["mean_extra_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Policy/mean_kl", locs["mean_kl"], locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        title_str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{title_str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.4f}\n"""
                f"""{'Learning rate:':>{pad}} {self.alg.learning_rate:.4f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                f"""{'Offset loss (mean):':>{pad}} {locs.get('offset_loss_val', 0.0):.6f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{title_str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Offset loss (mean):':>{pad}} {locs.get('offset_loss_val', 0.0):.6f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        """ Save policy/encoder/optimizer/offset states. 保存策略/编码器/优化器/偏置网络状态 """
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "encoder_state_dict": self.alg.encoder.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "offset_net_state_dict": self.offset_net.state_dict(),
                "offset_optimizer_state_dict": self.offset_optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=False):
        """ Load checkpoints; optionally restore optimizers. 加载权重；可选恢复优化器 """
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.encoder.load_state_dict(loaded_dict["encoder_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

        if "offset_net_state_dict" in loaded_dict:
            self.offset_net.load_state_dict(loaded_dict["offset_net_state_dict"])
            print("✓ Loaded offset_net weights")
        if "offset_optimizer_state_dict" in loaded_dict and load_optimizer:
            self.offset_optimizer.load_state_dict(loaded_dict["offset_optimizer_state_dict"])
            print("✓ Loaded offset_optimizer state")

        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos", None)

    def get_inference_policy(self, device=None):
        """ Return policy inference fn. 返回推理用策略函数 """
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_inference_encoder(self, device=None):
        """ Return encoder inference fn. 返回推理用编码器函数 """
        self.alg.encoder.eval()
        if device is not None:
            self.alg.encoder.to(device)
        return self.alg.encoder.encode

    def get_actor_critic(self, device=None):
        """ Return actor-critic module. 返回Actor-Critic模块 """
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
