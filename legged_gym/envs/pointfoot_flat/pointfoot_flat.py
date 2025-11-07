# -*- coding: utf-8 -*-
"""
Optimized Point-Foot Flat Environment (BipedPF)

Key changes (2025-11-06):
- Dedup imports; add clear constants (HEIGHT_DIM).
- Robust terrain height handling: pad/truncate to HEIGHT_DIM, zeros if unavailable.
- DCM/offset stability loss: optional Vstab soft-constraint (paper Eq.(10)(11)) with safe defaults,
  while keeping body posture regularization term; both gated by cfg flags.
- Safer first-step behavior for offset loss (keeps autograd path alive).
- Clearer comments and docstrings; minor nits (naming/typos) fixed.
"""

import os
import math
import torch
import numpy as np

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from .pointfoot_flat_config import BipedCfgPF

# -------------------- Constants --------------------
HEIGHT_DIM = 9  # number of local height samples for teacher/offset input


class BipedPF(BaseTask):
    def __init__(self, cfg: BipedCfgPF, sim_params, physics_engine, sim_device, headless):
        """
        Parses the provided config file, creates sim/terrain/envs via BaseTask.__init__,
        and initializes PyTorch buffers used during training.
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None

        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2
        self.group_idx = torch.arange(0, self.cfg.env.num_envs, device=self.device)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()
        self._prepare_reward_function()

        # Runtime buffers for DCM/offset (runner-injected)
        # keep autograd for xi_corr/delta_xi, detach xi_meas for logging/comparison
        self.xi_meas_buf = torch.zeros(self.num_envs, 2, device=self.device)
        self.xi_corr_buf = torch.zeros(self.num_envs, 2, device=self.device)
        self.delta_xi_buf = torch.zeros(self.num_envs, 2, device=self.device)
        self.loss_buf_offset = torch.zeros(self.num_envs, device=self.device)
        self._xi_meas_ref = None
        self._xi_corr_ref = None
        self._delta_xi_ref = None
        self.init_done = True

    # -------------------- DCM info --------------------
    def set_dcm_info(self, xi_meas: torch.Tensor, xi_corr: torch.Tensor, delta_xi: torch.Tensor):
        """
        Set DCM tensors for the current step.
        - xi_meas: measured/estimated DCM (detached; no backprop needed)
        - xi_corr: corrected DCM reference (retain graph for offset_net grad)
        - delta_xi: offset (retain graph for offset_net grad)
        """
        self._xi_meas_ref = xi_meas.detach()
        self._xi_corr_ref = xi_corr              # keep gradient
        self._delta_xi_ref = delta_xi            # keep gradient

        # mirrored detached copies for logging only
        self.xi_meas_buf = self._xi_meas_ref.detach()
        self.xi_corr_buf = self._xi_corr_ref.detach()
        self.delta_xi_buf = self._delta_xi_ref.detach()

    # -------------------- Sim step --------------------
    def step(self, actions):
        """
        Apply actions, simulate, and return tuple:
        (obs, rewards_main, rewards_offset, dones, infos, obs_history, scaled_commands, critic_obs)
        """
        self._action_clip(actions)
        self.render()
        self.pre_physics_step()

        for _ in range(self.cfg.control.decimation):
            self.action_fifo = torch.cat((self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1)
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs, device=self.device), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()

        self.post_physics_step()

        # Offset-net stability term (separate head)
        self.loss_buf_offset = self._reward_stability_offset()

        # Clip obs for numerical stability
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        return (
            self.obs_buf,
            self.rew_buf,
            self.loss_buf_offset,
            self.reset_buf,
            self.extras,
            self.obs_history,
            self.commands[:, :3] * self.commands_scale,
            self.critic_obs_buf,
        )

    # -------------------- Commands --------------------
    def _resample_commands(self, env_ids):
        """Randomly select commands for given env indices."""
        if len(env_ids) == 0:
            return

        self.commands[env_ids, 0] = (
            self.command_ranges["lin_vel_x"][env_ids, 1] - self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges["lin_vel_x"][env_ids, 0]

        self.commands[env_ids, 1] = (
            self.command_ranges["lin_vel_y"][env_ids, 1] - self.command_ranges["lin_vel_y"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges["lin_vel_y"][env_ids, 0]

        self.commands[env_ids, 2] = (
            self.command_ranges["ang_vel_yaw"][env_ids, 1] - self.command_ranges["ang_vel_yaw"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges["ang_vel_yaw"][env_ids, 0]

        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        zero_command_idx = (
            (torch_rand_float(0, 1, (len(env_ids), 1), device=self.device) > self.cfg.commands.zero_command_prob)
            .squeeze(1)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.commands[zero_command_idx, :3] = 0

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat[zero_command_idx], self.forward_vec[zero_command_idx])
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[zero_command_idx, 3] = heading

    # -------------------- Control --------------------
    def _compute_torques(self, actions):
        """Compute torques from actions."""
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if control_type == "P":
            torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        return torch.clip(torques * self.torques_scale, -self.torque_limits, self.torque_limits)

    # -------------------- Noise --------------------
    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:12] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12:18] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[18:] = 0.0  # previous actions
        return noise_vec

    # -------------------- Reset --------------------
    def reset_idx(self, env_ids):
        """Reset selected environments."""
        if len(env_ids) == 0:
            return

        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            self.update_command_curriculum(time_out_env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history[env_ids] = 0

        obs_buf, _ = self.compute_group_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length)
        self.gait_indices[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        self.dof_pos_int[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0

        if self.cfg.terrain.curriculum:
            self.extras["episode"]["group_terrain_level"] = torch.mean(self.terrain_levels[self.group_idx].float())
            self.extras["episode"]["group_terrain_level_stair_up"] = torch.mean(self.terrain_levels[self.stair_up_idx].float())

        if self.cfg.terrain.curriculum and self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.smooth_slope_idx, 1].float()
            )

        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

    # -------------------- Observations --------------------
    def compute_group_observations(self):
        """
        Compose policy/critic observations for each env.
        NOTE: if you add new signals here, remember to update _get_noise_scale_vec() indices accordingly.
        """
        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,        # [3]
                self.projected_gravity,                              # [3]
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # [6]
                self.dof_vel * self.obs_scales.dof_vel,             # [6]
                self.actions,                                       # [6] (or num_actions)
                self.clock_inputs_sin.view(self.num_envs, 1),       # [1]
                self.clock_inputs_cos.view(self.num_envs, 1),       # [1]
                self.gaits,                                         # [num_gaits]
            ),
            dim=-1,
        )
        critic_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, self.obs_buf), dim=-1)
        return obs_buf, critic_obs_buf

    # -------------------- Teacher State (for DCM/Offset) --------------------
    def _pad_or_truncate_heights(self, heights: torch.Tensor, target_dim: int = HEIGHT_DIM) -> torch.Tensor:
        """Pad/truncate last dim to target_dim. Assumes shape [N, D]."""
        if heights is None:
            return torch.zeros(self.num_envs, target_dim, device=self.device)
        if heights.dim() == 1:
            heights = heights.unsqueeze(0)
        d = heights.shape[-1]
        if d == target_dim:
            return heights
        elif d > target_dim:
            return heights[..., :target_dim]
        else:
            pad = torch.zeros(heights.shape[:-1] + (target_dim - d,), device=self.device, dtype=heights.dtype)
            return torch.cat([heights, pad], dim=-1)

    def get_teacher_state(self):
        """
        Package privileged state for teacher (DCM predictor + offset net):
        - CoM ≈ base_position/lin_vel
        - command velocities
        - contact forces on feet
        - gait phase
        - optional terrain heights (padded/truncated to HEIGHT_DIM)
        - last actions (flattened if needed)
        - (optional) feet positions if available
        """
        com_pos = self.base_position.clone()
        com_vel = self.base_lin_vel.clone()
        command_vel = self.commands[:, :3].clone()
        contact_forces = self.contact_forces[:, self.feet_indices, 2].clone()  # [N, n_feet]

        gait_phase = ((self.gait_indices.float() if hasattr(self, "gait_indices") else self.episode_length_buf.float()) % 24) / 24.0

        # Robust terrain height acquisition
        terrain_heights = None
        if hasattr(self, "_get_heights"):
            try:
                terrain_heights = self._get_heights().clone()
            except TypeError:
                try:
                    terrain_heights = self._get_heights(self.base_position).clone()
                except Exception:
                    terrain_heights = None
        terrain_heights = self._pad_or_truncate_heights(terrain_heights, HEIGHT_DIM)

        last_actions = self.last_actions[:, :, 0] if self.last_actions.dim() == 3 else self.last_actions
        last_actions = last_actions.clone()

        teacher_state = {
            "com_pos": com_pos,
            "com_vel": com_vel,
            "command_vel": command_vel,
            "contact_forces": contact_forces,
            "gait_phase": gait_phase,  # [N]
            "last_actions": last_actions,
            "terrain_heights": terrain_heights,
        }

        if hasattr(self, "foot_positions"):
            teacher_state["feet_positions"] = self.foot_positions.clone()

        return teacher_state

    # -------------------- Reward functions --------------------
    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        """
        Gaussian-shaped height reward (smooth gradients).
        Equivalent (monotonic) shaping of squared error around target height.
        """
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)  # [N]
        err = base_height - self.cfg.rewards.base_height_target
        k = getattr(self.cfg.rewards, "base_height_gain", 150.0)  # positive, e.g., 50~300
        reward = torch.exp(-k * err * err)
        return reward

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square(self.dof_acc), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1)

    def _reward_action_smooth(self):
        return torch.sum(torch.square(self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1)

    def _reward_keep_balance(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_force"] > 0:
            for i in range(len(self.feet_indices)):
                reward += (1 - desired_contact[:, i]) * torch.exp(-foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma)
        else:
            for i in range(len(self.feet_indices)):
                reward += (1 - desired_contact[:, i]) * (1 - torch.exp(-foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma))
        return reward / len(self.feet_indices)

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocities, dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_vel"] > 0:
            for i in range(len(self.feet_indices)):
                reward += desired_contact[:, i] * torch.exp(-foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma)
        else:
            for i in range(len(self.feet_indices)):
                reward += desired_contact[:, i] * (1 - torch.exp(-foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma))
        return reward / len(self.feet_indices)

    def _reward_feet_distance(self):
        feet_distance = torch.norm(self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1)
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1)
        return reward

    def _reward_feet_regulation(self):
        feet_height = self.cfg.rewards.base_height_target * 0.001
        reward = torch.sum(torch.exp(-self.foot_heights / feet_height) * torch.square(torch.norm(self.foot_velocities[:, :, :2], dim=-1)), dim=1)
        return reward

    def _reward_collision(self):
        return torch.sum(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1.0, dim=1)

    def _reward_foot_landing_vel(self):
        z_vels = self.foot_velocities[:, :, 2]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        about_to_land = (self.foot_heights < self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward

    def _reward_stability_offset(self):
        """
        Stability cost for offset_net (per-step).
        Includes: DCM tracking error, offset magnitude, optional body posture regularizer,
        and optional Vstab soft-constraint (distance of xi_corr to support center threshold).
        Returns a tensor [N] with grad for offset_net.
        """
        # First step after reset: runner may not have set DCM info yet.
        if (self._xi_corr_ref is None) or (self._delta_xi_ref is None) or (self._xi_meas_ref is None):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32, requires_grad=True)

        # 1) core terms (grad flows to offset_net via xi_corr/delta_xi)
        track_err_sq = torch.sum((self._xi_meas_ref - self._xi_corr_ref) ** 2, dim=-1)  # [N]
        offset_mag_sq = torch.sum(self._delta_xi_ref ** 2, dim=-1)                      # [N]

        # 2) body posture regularization (no grad to state)
        use_body_reg = bool(getattr(self.cfg.rewards, "use_body_regularizer", True))
        if use_body_reg:
            ang_xy_sq = torch.sum(self.base_ang_vel[:, :2] ** 2, dim=-1)
            tilt_xy_sq = torch.sum(self.projected_gravity[:, :2] ** 2, dim=-1)
            body_unstable = 0.5 * ang_xy_sq + 0.5 * tilt_xy_sq
        else:
            body_unstable = torch.zeros_like(track_err_sq)

        # 3) Vstab soft-constraint (paper Eq.(10)(11)) — optional
        use_vstab = bool(getattr(self.cfg.rewards, "use_vstab", True))
        if use_vstab:
            # Approximate support center with CoM (point-foot support region is small)
            c_support = self.base_position[:, :2].detach()
            d_thresh = float(getattr(self.cfg.rewards, "vstab_radius_m", 0.30))  # meters
            stab_margin = torch.sum((self._xi_corr_ref - c_support) ** 2, dim=-1) - d_thresh ** 2
            stab_margin = torch.clamp(stab_margin, min=0.0)  # ReLU
        else:
            stab_margin = torch.zeros_like(track_err_sq)

        # 4) weights (safe defaults; can be tuned in cfg)
        w_track = float(getattr(self.cfg.rewards, "w_track", 1.0))
        w_mag   = float(getattr(self.cfg.rewards, "w_offset_mag", 0.1))
        w_body  = float(getattr(self.cfg.rewards, "w_body", 0.2)) if use_body_reg else 0.0
        w_stab  = float(getattr(self.cfg.rewards, "w_vstab", 0.2)) if use_vstab else 0.0

        stab_cost = w_track * track_err_sq + w_mag * offset_mag_sq + w_body * body_unstable + w_stab * stab_margin
        return stab_cost  # [N], has grad via xi_corr/delta_xi