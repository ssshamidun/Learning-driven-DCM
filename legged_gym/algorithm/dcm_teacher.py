import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
import math


class DCMPredictor:
    """
    Standard ICP-based DCM preview | 标准ICP/DCM预观
    ξ = c + ċ/ω0,  ξ̇ = ω0(ξ - r), r≈ZMP(支撑足)
    核心：用闭式“步伐方程”求下步落脚点 p*；用一致的离散公式推进 CoM。
    """
    def __init__(self, omega0: float, dt: float, preview_horizon: int = 20, z_c: float = 0.65):
        self.omega0 = omega0
        self.dt = dt
        self.preview_horizon = preview_horizon
        self.z_c = z_c

        # 步态参数（你原来的接口/数值保持）
        self.step_period = 24
        self.step_length = 0.13
        self.step_width = 0.05

        # 预瞄参考用的权重（可留，可不用）
        w = np.exp(-np.arange(self.preview_horizon) * 0.3)
        self.preview_weights = (w / w.sum()).astype(np.float32)

    def plan_reference(
        self,
        com_pos: torch.Tensor,
        com_vel: torch.Tensor,
        command_vel: torch.Tensor,
        contact_forces: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        标准ICP步伐方程 + CoM一致离散推进
        Args:
            com_pos: [N,3], com_vel: [N,3], command_vel: [N,3], contact_forces: [N,2]
        Returns:
            xi_ref: [N,2], c_ref: [N,3], foot_placement: [N,2,2] (L/R, x/y)
        """
        N = com_pos.shape[0]
        device = com_pos.device

        # (A) 当前 DCM (ξ_s)
        xi_s = torch.stack([
            com_pos[:, 0] + com_vel[:, 0] / self.omega0,
            com_pos[:, 1] + com_vel[:, 1] / self.omega0
        ], dim=-1)  # [N,2]

        # (B) 支撑判定（左>右被视为左支撑）
        is_left_support = contact_forces[:, 0] > contact_forces[:, 1]

        # 近似用 CoM 的水平位置当作“当前支撑足位置”（没有真实足端就用这个近似）
        support_pos = com_pos[:, :2].clone()  # [N,2]

        # (C) 步末 DCM 目标 ξ_e*: 朝指令速度方向的步末期望，并加横向偏置
        T = self.step_period * self.dt  # 单步时长（假设 step_period 是步数）
        T = max(float(T), 1e-3)
        xi_e_star = xi_s + command_vel[:, :2] * T
        side_bias = torch.where(
            is_left_support,
            torch.full((N,), -self.step_width, device=device),  # 下一步落右脚：偏向右侧
            torch.full((N,),  self.step_width, device=device)   # 下一步落左脚：偏向左侧
        )
        xi_e_star = xi_e_star + torch.stack([torch.zeros(N, device=device), side_bias], dim=-1)

        # (D) ICP 步伐方程： p* = (ξ_e* - e^{ω0 T} ξ_s) / (1 - e^{ω0 T})
        expwT = torch.exp(torch.clamp(torch.as_tensor(self.omega0 * T, device=device), max=8.0))  # clamped
        p_star = xi_e_star - expwT * (xi_s - support_pos)

        # (E) 软约束：把 p* 夹到可达盒内（相对当前支撑点的步长/步宽限制）
        rel = p_star - support_pos
        rel_x = rel[:, 0].clamp(-self.step_length, self.step_length)
        rel_y = rel[:, 1].clamp(-1.5 * self.step_width, 1.5 * self.step_width)
        p_star = support_pos + torch.stack([rel_x, rel_y], dim=-1)

        # (F) 组装左右脚落脚（形状不变）
        foot_placement = torch.zeros(N, 2, 2, device=device)  # [N, 2(L/R), 2(xy)]
        foot_placement[is_left_support, 0, :] = support_pos[is_left_support]  # 左支撑保持
        foot_placement[is_left_support, 1, :] = p_star[is_left_support]       # 右摆动至 p*
        foot_placement[~is_left_support, 1, :] = support_pos[~is_left_support]  # 右支撑保持
        foot_placement[~is_left_support, 0, :] = p_star[~is_left_support]       # 左摆动至 p*

        # (G) 参考 DCM：让 ξ_ref ≈ 下步 ZMP（简洁且一致）
        xi_ref = p_star  # [N,2]

        # (H) CoM 一致离散推进： c_{k+1} = ξ_k + e^{-ω0 Δt} (c_k - ξ_k)
        alpha = torch.exp(torch.clamp(torch.as_tensor(-self.omega0 * self.dt, device=device), min=-8.0))  # clamped
        c_ref = torch.zeros_like(com_pos)
        c_ref[:, :2] = xi_ref + alpha * (com_pos[:, :2] - xi_ref)
        c_ref[:, 2] = self.z_c

        return xi_ref, c_ref, foot_placement


class OffsetNetwork(nn.Module):
    """
    学习的稳定性偏置 Δξ（接口不变）
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, stab_obs: torch.Tensor) -> torch.Tensor:
        return self.net(stab_obs) * 0.05  # ±5cm 限幅


class InverseKinematics:
    """
    平面二连杆腿部 IK（髋俯仰+膝）
    """
    def __init__(self, l1: float = 0.3, l2: float = 0.32):
        self.l1 = l1
        self.l2 = l2

    def solve(
        self,
        hip_pos: torch.Tensor,
        com: torch.Tensor,
        foot_pos: torch.Tensor,
        dz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = foot_pos.device
        l1 = torch.tensor(self.l1, device=device)
        l2 = torch.tensor(self.l2, device=device)

        dx = foot_pos[..., 0] - com[:, 0:1]
        dy = dz
        dist = torch.sqrt(dx**2 + dy**2)

        max_reach = l1 + l2
        over = dist > max_reach
        scale = torch.where(over, max_reach / (dist + 1e-6), torch.ones_like(dist))
        dx, dy = dx * scale, dy * scale

        cos_k = (dx**2 + dy**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_k = torch.clamp(cos_k, -1.0, 1.0)
        knee = -torch.acos(cos_k)  # 向下弯曲

        k1 = l1 + l2 * torch.cos(knee)
        k2 = l2 * torch.sin(knee)
        hip = torch.atan2(dy, dx) - torch.atan2(k2, k1)
        return hip, knee


def build_teacher_action(
    dcm_predictor: DCMPredictor,
    offset_net: OffsetNetwork,
    ik_solver: InverseKinematics,
    teacher_state: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    教师流水线：DCM规划(标准) → Δξ修正 → 一致 CoM → IK → 关节参考
    接口不变
    """
    # === (1) 状态 ===
    com_pos = teacher_state["com_pos"]
    com_vel = teacher_state["com_vel"]
    command_vel = teacher_state["command_vel"]
    contact_forces = teacher_state["contact_forces"]
    gait_phase = teacher_state.get("gait_phase", torch.zeros(com_pos.shape[0], device=com_pos.device))
    height = teacher_state["terrain_heights"]

    if not isinstance(gait_phase, torch.Tensor):
        gait_phase = torch.full((com_pos.shape[0],), gait_phase, device=com_pos.device)
    elif gait_phase.dim() == 0:
        gait_phase = gait_phase.unsqueeze(0).expand(com_pos.shape[0])

    device = com_pos.device
    N = com_pos.shape[0]
    omega0 = dcm_predictor.omega0

    # === (2) DCM 参考（标准） ===
    xi_ref, c_ref, foot_placement = dcm_predictor.plan_reference(
        com_pos, com_vel, command_vel, contact_forces
    )

    # === (3) 测得 DCM 与稳定观测 ===
    xi_meas = torch.stack([
        com_pos[:, 0] + com_vel[:, 0] / omega0,
        com_pos[:, 1] + com_vel[:, 1] / omega0
    ], dim=-1)

    dcm_error = xi_ref - xi_meas
    contact_state = contact_forces / (contact_forces.sum(dim=-1, keepdim=True) + 1e-6)
    stab_obs = torch.cat([dcm_error, xi_meas, contact_state, command_vel[:, :2], height], dim=-1)
    stab_obs = stab_obs.to(next(offset_net.parameters()).device)

    # === (4) Δξ 学习修正 ===
    delta_xi = offset_net(stab_obs)
    xi_corrected = xi_ref + delta_xi  # [N,2]

    # === (5) CoM 一致离散推进（用修正后的 ξ）===
    alpha = math.exp(-omega0 * dcm_predictor.dt)  # 标量
    c_corrected = torch.zeros_like(com_pos)
    c_corrected[:, :2] = xi_corrected + alpha * (com_pos[:, :2] - xi_corrected)
    c_corrected[:, 2] = dcm_predictor.z_c

    # === (6) 摆动高度（与你原来保持一致）===
    is_left_support = contact_forces[:, 0] > contact_forces[:, 1]
    step_phase = gait_phase * 2 * np.pi
    swing_height = 0.1 * torch.sin(step_phase).clamp(min=0)

    z_support = -0.51
    dz_left = torch.where(
        is_left_support,
        torch.full((N,), z_support, device=device),
        torch.full((N,), z_support, device=device) + swing_height
    )
    dz_right = torch.where(
        is_left_support,
        torch.full((N,), z_support, device=device) + swing_height,
        torch.full((N,), z_support, device=device)
    )
    dz = torch.stack([dz_left, dz_right], dim=-1)

    # === (7) IK ===
    if "feet_positions" in teacher_state:
        feet_pos = teacher_state["feet_positions"]  # [N,2,3]
        hip_pos = feet_pos[:, :, [0, 2]]            # 用 x,z
    else:
        hip_pos = torch.zeros(N, 2, 2, device=device)

    hip_angles, knee_angles = ik_solver.solve(
        hip_pos, c_corrected[:, :2], foot_placement, dz
    )

    # === (8) 关节目标（接口保持）===
    a_ref = torch.zeros(N, 6, device=device)
    # 左腿
    a_ref[:, 0] = 0.0
    a_ref[:, 1] = (hip_angles[:, 0] + np.pi / 3) * 2
    a_ref[:, 2] = (-knee_angles[:, 0] - np.pi / 3) * 2
    # 右腿
    a_ref[:, 3] = 0.0
    a_ref[:, 4] = (-hip_angles[:, 1] - np.pi / 3) * 2
    a_ref[:, 5] = (knee_angles[:, 1] + np.pi / 3) * 2

    # === (9) 可选平滑 ===
    if "last_actions" in teacher_state:
        alpha_smooth = 0.2
        a_ref = alpha_smooth * a_ref + (1 - alpha_smooth) * teacher_state["last_actions"]

    # === (10) 调试信息 ===
    teacher_info = {
        "xi_ref": xi_ref,
        "xi_meas": xi_meas,
        "xi_corrected": xi_corrected,
        "c_ref": c_ref,
        "c_corrected": c_corrected,
        "foot_placement": foot_placement,
        "dcm_error": dcm_error,
        "delta_xi": delta_xi,
    }
    return a_ref, teacher_info