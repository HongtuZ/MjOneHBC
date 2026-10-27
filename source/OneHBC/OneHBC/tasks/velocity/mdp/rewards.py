from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_lin_vel_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for tracking the commanded base linear velocity.

    The commanded z velocity is assumed to be zero.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    lin_vel_error = xy_error + z_error
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward heading error for heading-controlled envs, angular velocity for others.

    The commanded xy angular velocities are assumed to be zero.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    ang_vel_error = z_error + xy_error
    return torch.exp(-ang_vel_error / std**2)


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


def low_speed_sway_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """当命令速度低于阈值时，惩罚线速度和角速度。

    此函数在命令速度非常小时惩罚机器人的运动（包括线速度和角速度），
    鼓励机器人在低速命令期间保持静止。
    """
    # 提取使用的量（以启用类型提示）                                                              # 获取场景中的刚体资产对象
    asset: Entity = env.scene[asset_cfg.name]

    # 获取命令速度                                                                                   # 从命令管理器获取指定名称的命令张量
    command = env.command_manager.get_command(command_name)
    command_speed = torch.norm(command[:, :2], dim=1)  # 计算命令在水平面 (x, y) 上的速度范数
    # 惩罚 xy 平面内的线速度                                                                         # 计算机器人本体坐标系下水平线速度的平方和
    lin_vel_penalty = torch.sum(torch.square(asset.data.root_link_lin_vel_b[:, :2]), dim=1)
    # 惩罚角速度                                                                                     # 计算机器人本体坐标系下角速度的平方和
    ang_vel_penalty = torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)
    # 总速度惩罚                                                                                     # 合并线速度和角速度的惩罚项
    vel_penalty = lin_vel_penalty + ang_vel_penalty
    # 仅当命令速度低于阈值时应用惩罚                                                                 # 生成掩码并转换为浮点数，低速时保留惩罚值，高速时归零
    return vel_penalty * (command_speed < command_threshold).float()


def feet_stumble(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene[sensor_name]
    forces_z = torch.abs(contact_sensor.data.force[..., 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.force[..., :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward
