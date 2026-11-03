from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _velocity_std_scale(
    env: ManagerBasedRlEnv,
    height_command_name: str | None,
    crouch_std_scale: float,
) -> torch.Tensor | None:
    """Return a per-env scale factor for the velocity-tracking std based on the height command.

    高度指令越低（越接近蹲姿），返回的缩放因子越大：指令 0 → ``crouch_std_scale``，
    指令 1 → 1.0。因为蹲姿下腿大幅弯曲、走不快，速度跟踪要求应相应放宽，避免
    机器人为了追速度而拒绝蹲下。当 ``height_command_name`` 为 None 或
    ``crouch_std_scale <= 1.0`` 时返回 None（不缩放）。
    """
    if height_command_name is None or crouch_std_scale <= 1.0:
        return None
    height_cmd = env.command_manager.get_command(height_command_name)
    assert height_cmd is not None, f"Command '{height_command_name}' not found."
    alpha = torch.clip(height_cmd[:, 0], 0.0, 1.0)  # [B], 0=蹲姿, 1=站姿
    return 1.0 + (1.0 - alpha) * (crouch_std_scale - 1.0)  # [B]


def _no_walk_gate(
    env: ManagerBasedRlEnv,
    height_command_name: str | None,
    no_walk_height_threshold: float,
) -> torch.Tensor | None:
    """Return a per-env 0/1 mask that zeroes the velocity command below a height.

    高度指令低于 ``no_walk_height_threshold`` 时返回 0（不要求走动），否则返回 1。
    当 ``height_command_name`` 为 None 或阈值 <= 0 时返回 None（不门控）。
    """
    if height_command_name is None or no_walk_height_threshold <= 0.0:
        return None
    height_cmd = env.command_manager.get_command(height_command_name)
    assert height_cmd is not None, f"Command '{height_command_name}' not found."
    alpha = torch.clip(height_cmd[:, 0], 0.0, 1.0)  # [B]
    return (alpha >= no_walk_height_threshold).float()  # [B]


def track_lin_vel_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    height_command_name: str | None = None,
    crouch_std_scale: float = 1.0,
    no_walk_height_threshold: float = 0.0,
) -> torch.Tensor:
    """Reward for tracking the commanded base linear velocity.

    The commanded z velocity is assumed to be zero. When ``height_command_name`` is
    provided, the std is relaxed toward ``crouch_std_scale * std`` as the height command
    approaches the crouch pose (command 0), so low-posture walking is not over-penalized.
    Additionally, when the height command is below ``no_walk_height_threshold``, the
    velocity command is zeroed so the robot is not required to walk while crouching.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    # Zero the velocity command when the height command is low (do not walk).
    gate = _no_walk_gate(env, height_command_name, no_walk_height_threshold)
    if gate is not None:
        command = command * gate.unsqueeze(1)
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    lin_vel_error = xy_error + z_error
    scale = _velocity_std_scale(env, height_command_name, crouch_std_scale)
    eff_std = std * scale if scale is not None else std
    return torch.exp(-lin_vel_error / eff_std**2)


def track_ang_vel_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    height_command_name: str | None = None,
    crouch_std_scale: float = 1.0,
    no_walk_height_threshold: float = 0.0,
) -> torch.Tensor:
    """Reward heading error for heading-controlled envs, angular velocity for others.

    The commanded xy angular velocities are assumed to be zero. When
    ``height_command_name`` is provided, the std is relaxed toward
    ``crouch_std_scale * std`` as the height command approaches the crouch pose.
    Additionally, when the height command is below ``no_walk_height_threshold``, the
    velocity command is zeroed so the robot is not required to walk while crouching.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b
    # Zero the velocity command when the height command is low (do not walk).
    gate = _no_walk_gate(env, height_command_name, no_walk_height_threshold)
    if gate is not None:
        command = command * gate.unsqueeze(1)
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    ang_vel_error = z_error + xy_error
    scale = _velocity_std_scale(env, height_command_name, crouch_std_scale)
    eff_std = std * scale if scale is not None else std
    return torch.exp(-ang_vel_error / eff_std**2)


def feet_gait(
    env: ManagerBasedRlEnv,
    period: float,  # 步态周期（单位：秒），定义一个完整步态循环的时间长度
    offset: list[float],  # 各腿的相位偏移列表，如 [0.0, 0.5] 表示左右足反相
    sensor_name: str,  # 足部接触传感器名称
    threshold: float = 0.5,  # 支撑相占比阈值（相位 < threshold 时应触地）
    command_name=None,  # 关联的运动指令名称，用于判断是否应激活奖励
) -> torch.Tensor:
    # 获取接触传感器
    sensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0  # [num_envs, num_legs]
    # 计算全局相位 [num_envs, 1]
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    offset_tensor = torch.tensor(offset, device=env.device, dtype=torch.float)
    leg_phase = (global_phase + offset_tensor) % 1.0  # [num_envs, num_legs]
    # 期望支撑相：相位 < 阈值 应该触地
    is_stance = leg_phase < threshold
    reward = (~(is_stance ^ is_contact)).sum(dim=-1).float()
    # 指令门控：只有在运动时才给奖励
    if command_name is not None:
        cmd = env.command_manager.get_command(command_name)
        cmd_norm = torch.norm(cmd, dim=1)
        reward *= (cmd_norm > 0.1).float()
    return reward


def joint_torque_soft_limit(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    threshold: float = 1.0,  # 扭矩阈值（Nm）
) -> torch.Tensor:
    """
    按比例惩罚关节扭矩超限：
    惩罚 = max( |扭矩| - 阈值, 0 ) / 阈值
    对所有目标关节求和后返回。
    """
    # 获取目标关节扭矩
    asset = env.scene[asset_cfg.name]
    tau = asset.data.actuator_force[:, asset_cfg.actuator_ids]  # [num_envs, num_joints]
    # 取绝对值（正负扭矩都惩罚）
    tau_abs = torch.abs(tau)
    # 计算超出量：> threshold 才开始算
    exceedance = torch.max(tau_abs - threshold, torch.zeros_like(tau_abs))
    # 按比例惩罚：(超出量) / 阈值
    penalty_per_joint = exceedance / threshold
    # 对所有目标关节的惩罚求和 → [num_envs]
    total_penalty = penalty_per_joint.sum(dim=-1)
    return total_penalty


def shoulder_thigh_coordination(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gain: float = 1.0,
    std: float = 0.5,
    hip_scale: float = 1.5,
    hip_joint_names: tuple[str, ...] = ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    arm_joint_names: tuple[str, ...] = ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"),
) -> torch.Tensor:
    """手臂与大腿摆动反向协调奖励。

    - 双臂机器人（默认 4 个关节）：左臂对右腿、右臂对左腿交叉协调。
    - 单臂机器人（``arm_joint_names`` 只有 1 个关节）：手臂对双腿摆动差
      ``(l_hip - r_hip) / 2`` 反向协调。
    """
    asset = env.scene[asset_cfg.name]
    # 关节索引
    hip_ids = asset.find_joints(hip_joint_names)[0]
    arm_ids = asset.find_joints(arm_joint_names)[0]
    # 偏移量
    hip_offset = asset.data.joint_pos[:, hip_ids] - asset.data.default_joint_pos[:, hip_ids]
    arm_offset = asset.data.joint_pos[:, arm_ids] - asset.data.default_joint_pos[:, arm_ids]
    l_hip, r_hip = hip_offset.unbind(dim=-1)
    if arm_offset.shape[-1] == 1:
        # 单臂：以双腿摆动差（即腿部摆动的振荡分量）作为协调目标
        (arm,) = arm_offset.unbind(dim=-1)
        mean_err = torch.abs(arm + hip_scale * (l_hip - r_hip) / 2.0)
    else:
        # 双臂：左右臂分别与前向摆动的对侧大腿反向
        l_arm, r_arm = arm_offset.unbind(dim=-1)
        # 协调目标：手臂 = -hip_scale × 大腿摆动
        target_l_shoulder = -hip_scale * r_hip
        target_r_shoulder = -hip_scale * l_hip
        # 实际误差
        err_left = torch.abs(l_arm - target_l_shoulder)
        err_right = torch.abs(r_arm - target_r_shoulder)
        mean_err = (err_left + err_right) / 2
    # 高斯奖励
    reward = gain * torch.exp(-0.5 * (mean_err / std) ** 2)
    return reward


def feet_stumble(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene[sensor_name]
    forces_z = torch.abs(contact_sensor.data.force[..., 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.force[..., :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


class variable_posture:
    """Penalize deviation from a height-command-dependent desired pose.

    目标姿态 ``desired_joint_pos`` 由高度指令二值选择：指令为 0 时对应蹲姿
    （``crouch_joint_pos``），指令为 1 时对应站姿（``default_joint_pos``），不再插值。
    不再直接跟踪高度，高度指令完全通过该 pose 奖励来响应。

    关节容差 ``std`` 由速度档位决定（standing/walking/running），但当指令趋向蹲姿时，
    蹲姿相对站姿的关节偏离很大，若仍用 ``std_standing``（如 0.05）会因误差远大于 std
    而导致高斯奖励梯度消失，机器人静态站立时不遵从指令。因此额外引入 ``std_crouch``，
    令 ``std = max(速度档位 std, (1-指令) * std_crouch)``，蹲姿指令下放宽容差。
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        self.stand_joint_pos = default_joint_pos

        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

        _, _, std_standing = resolve_matching_names_values(
            data=cfg.params["std_standing"], list_of_strings=joint_names
        )
        self.std_standing = torch.tensor(std_standing, device=env.device, dtype=torch.float32)

        _, _, std_walking = resolve_matching_names_values(
            data=cfg.params["std_walking"], list_of_strings=joint_names
        )
        self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

        _, _, std_running = resolve_matching_names_values(
            data=cfg.params["std_running"], list_of_strings=joint_names
        )
        self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

        # 蹲姿指令下的关节容差（用于容纳蹲姿相对站姿的关节偏离）。
        _, _, std_crouch = resolve_matching_names_values(
            data=cfg.params["std_crouch"], list_of_strings=joint_names
        )
        self.std_crouch = torch.tensor(std_crouch, device=env.device, dtype=torch.float32)

        # 蹲姿（高度指令为 0 时的目标关节弧度）。
        _, _, crouch_values = resolve_matching_names_values(
            data=cfg.params["crouch_joint_pos"], list_of_strings=joint_names
        )
        self.crouch_joint_pos = torch.tensor(
            crouch_values, device=env.device, dtype=torch.float32
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std_standing,
        std_walking,
        std_running,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        crouch_joint_pos=None,
        std_crouch=None,
        height_command_name: str | None = None,
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running  # Resolved in __init__.
        del crouch_joint_pos, std_crouch  # Resolved in __init__.

        asset: Entity = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        assert command is not None

        linear_speed = torch.norm(command[:, :2], dim=1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < walking_threshold).float()
        walking_mask = (
            (total_speed >= walking_threshold) & (total_speed < running_threshold)
        ).float()
        running_mask = (total_speed >= running_threshold).float()

        std = (
            self.std_standing * standing_mask.unsqueeze(1)
            + self.std_walking * walking_mask.unsqueeze(1)
            + self.std_running * running_mask.unsqueeze(1)
        )

        # 目标姿态二值：0 → 蹲姿，1 → 站姿（不再插值）。
        stand_pos = self.stand_joint_pos[:, asset_cfg.joint_ids]  # [B, J]
        crouch_pos = self.crouch_joint_pos  # [J]
        if height_command_name is not None:
            height_cmd = env.command_manager.get_command(height_command_name)  # [B, 1]
            assert height_cmd is not None
            alpha = torch.clip(height_cmd[:, 0], 0.0, 1.0)  # [B]
        else:
            alpha = torch.ones(stand_pos.shape[0], device=env.device)  # 默认站姿

        # 蹲姿指令下放宽容差：指令越接近蹲姿，std 越大，避免 std_standing 过小导致梯度消失。
        std_height = (1.0 - alpha).unsqueeze(1) * self.std_crouch  # [B, J]
        std = torch.maximum(std, std_height)

        desired_joint_pos = torch.where(
            alpha.unsqueeze(1) > 0.5, stand_pos, crouch_pos.expand_as(stand_pos)
        )  # [B, J]

        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [B, J]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)

        return torch.exp(-torch.mean(error_squared / (std**2), dim=1))
