from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
    resolve_matching_names_values,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


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


def self_collision_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize self-collisions.

    When the sensor provides force history (from ``history_length > 0``),
    counts substeps where any contact force exceeds *force_threshold*.
    Falls back to the instantaneous ``found`` count otherwise.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        # force_history: [B, N, H, 3]
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
        return hit.sum(dim=-1).float()  # [B]
    assert data.found is not None
    return data.found.sum(dim=-1).float()


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
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    # 关节索引
    joint_ids = asset.find_joints(
        [
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
        ]
    )[0]
    # 偏移量
    joint_pos = asset.data.joint_pos[:, joint_ids]
    default_pos = asset.data.default_joint_pos[:, joint_ids]
    offset = joint_pos - default_pos
    # 解包
    l_hip, r_hip, l_shoulder, r_shoulder = offset.unbind(dim=-1)
    # 协调目标：手臂 = -hip_scale × 大腿摆动
    target_l_shoulder = -hip_scale * r_hip
    target_r_shoulder = -hip_scale * l_hip
    # 实际误差
    err_left = torch.abs(l_shoulder - target_l_shoulder)
    err_right = torch.abs(r_shoulder - target_r_shoulder)
    # 高斯奖励
    reward = gain * torch.exp(-0.5 * ((err_left + err_right) / 2 / std) ** 2)
    return reward


def angular_momentum_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalize whole-body angular momentum to encourage natural arm swing."""
    angmom_sensor: BuiltinSensor = env.scene[sensor_name]
    angmom = angmom_sensor.data
    angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
    angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
    env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
    return angmom_magnitude_sq


def feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_name: str | None = None,
    command_threshold: float = 0.5,
) -> torch.Tensor:
    """Reward feet air time."""
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    current_air_time = sensor_data.current_air_time
    assert current_air_time is not None
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)
    in_air = current_air_time > 0
    num_in_air = torch.sum(in_air.float())
    mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(num_in_air, min=1)
    env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


def feet_clearance(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize deviation from target clearance height, weighted by foot velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    height_sensor = env.scene[height_sensor_name]
    assert isinstance(
        height_sensor, TerrainHeightSensor
    ), f"feet_clearance requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    foot_height = height_sensor.data.heights  # [B, F]
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
    delta = torch.abs(foot_height - target_height)  # [B, F]
    cost = torch.sum(delta * vel_norm, dim=1)  # [B]
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            active = (total_command > command_threshold).float()
            cost = cost * active
    return cost


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


class feet_swing_height:
    """Penalize deviation from target swing height, evaluated at landing."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        height_sensor = env.scene[cfg.params["height_sensor_name"]]
        assert isinstance(
            height_sensor, TerrainHeightSensor
        ), f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
        num_feet = height_sensor.num_frames
        self.peak_heights = torch.zeros((env.num_envs, num_feet), device=env.device, dtype=torch.float32)
        self.step_dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        height_sensor_name: str,
        target_height: float,
        command_name: str,
        command_threshold: float,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene[sensor_name]
        command = env.command_manager.get_command(command_name)
        assert command is not None
        height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
        foot_heights = height_sensor.data.heights
        in_air = contact_sensor.data.found == 0
        self.peak_heights = torch.where(
            in_air,
            torch.maximum(self.peak_heights, foot_heights),
            self.peak_heights,
        )
        first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm
        active = (total_command > command_threshold).float()
        error = self.peak_heights / target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
        num_landings = torch.sum(first_contact.float())
        peak_heights_at_landing = self.peak_heights * first_contact.float()
        mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(num_landings, min=1)
        env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
        self.peak_heights = torch.where(
            first_contact,
            torch.zeros_like(self.peak_heights),
            self.peak_heights,
        )
        return cost


def feet_slide(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize foot sliding (xy velocity while in contact)."""
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    assert contact_sensor.data.found is not None
    in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
    vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
    vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
    cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
    num_in_contact = torch.sum(in_contact)
    mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(num_in_contact, min=1)
    env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
    return cost


def feet_stumble(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene[sensor_name]
    forces_z = torch.abs(contact_sensor.data.force[..., 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.force[..., :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def soft_landing(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize high impact forces at landing to encourage soft footfalls."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = contact_sensor.data
    assert sensor_data.force is not None
    forces = sensor_data.force  # [B, N, 3]
    force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
    landing_impact = force_magnitude * first_contact.float()  # [B, N]
    cost = torch.sum(landing_impact, dim=1)  # [B]
    num_landings = torch.sum(first_contact.float())
    mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            active = (total_command > command_threshold).float()
            cost = cost * active
    return cost


class variable_posture:
    """Penalize deviation from default pose with speed-dependent tolerance.

    Uses per-joint standard deviations to control how much each joint can deviate
    from default pose. Smaller std = stricter (less deviation allowed), larger
    std = more forgiving. The reward is: exp(-mean(error² / std²))

    Three speed regimes (based on linear + angular command velocity):
      - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
      - std_walking (walking_threshold <= speed < running_threshold): Moderate.
      - std_running (speed >= running_threshold): Loose tolerance for large motion.

    Tune std values per joint based on how much motion that joint needs at each
    speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        self.default_joint_pos = default_joint_pos

        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

        _, _, std_standing = resolve_matching_names_values(
            data=cfg.params["std_standing"],
            list_of_strings=joint_names,
        )
        self.std_standing = torch.tensor(std_standing, device=env.device, dtype=torch.float32)

        _, _, std_walking = resolve_matching_names_values(
            data=cfg.params["std_walking"],
            list_of_strings=joint_names,
        )
        self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

        _, _, std_running = resolve_matching_names_values(
            data=cfg.params["std_running"],
            list_of_strings=joint_names,
        )
        self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std_standing,
        std_walking,
        std_running,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running  # Unused.

        asset: Entity = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        assert command is not None

        linear_speed = torch.norm(command[:, :2], dim=1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < walking_threshold).float()
        walking_mask = ((total_speed >= walking_threshold) & (total_speed < running_threshold)).float()
        running_mask = (total_speed >= running_threshold).float()

        std = (
            self.std_standing * standing_mask.unsqueeze(1)
            + self.std_walking * walking_mask.unsqueeze(1)
            + self.std_running * running_mask.unsqueeze(1)
        )

        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)

        return torch.exp(-torch.mean(error_squared / (std**2), dim=1))
