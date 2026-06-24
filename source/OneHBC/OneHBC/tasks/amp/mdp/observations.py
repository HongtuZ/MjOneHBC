"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_inverse,
    subtract_frame_transforms,
)

from OneHBC.utils.motion_loader import MotionLoader

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def robot_body_pos_b(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    base_pos_w = asset.data.root_link_pos_w
    base_quat_w = asset.data.root_link_quat_w
    body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]  # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]  # (num_envs, num_bodies, 4)

    num_bodies = body_pos_w.shape[1]
    pos_b, _ = subtract_frame_transforms(
        base_pos_w[:, None, :].expand(-1, num_bodies, -1),
        base_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    base_pos_w = asset.data.root_link_pos_w
    base_quat_w = asset.data.root_link_quat_w
    body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]  # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]  # (num_envs, num_bodies, 4)

    num_bodies = body_pos_w.shape[1]
    _, ori_b = subtract_frame_transforms(
        base_pos_w[:, None, :].expand(-1, num_bodies, -1),
        base_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_lin_vel_b(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    body_lin_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]  # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]  # (num_envs, num_bodies, 4)

    num_bodies = body_lin_vel_w.shape[1]
    body_lin_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_lin_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)
    return body_lin_vel_b.reshape(env.num_envs, -1)


def robot_body_ang_vel_b(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    body_ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids]  # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]  # (num_envs, num_bodies, 4)

    num_bodies = body_ang_vel_w.shape[1]
    body_ang_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_ang_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)
    return body_ang_vel_b.reshape(env.num_envs, -1)


def gait_phase(env: ManagerBasedRlEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):  # 检查环境是否已存在回合步数缓冲区
        env.episode_length_buf = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )  # 初始化回合步数计数器（每个子环境独立计数）

    global_phase = (
        (env.episode_length_buf * env.step_dt) % period / period
    )  # 计算全局相位：[0, 1) 区间内循环（基于仿真时间对周期取模）

    phase = torch.zeros(env.num_envs, 2, device=env.device)  # 初始化相位张量：形状 [num_envs, 2]
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)  # 第0维：正弦分量（2π周期）
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)  # 第1维：余弦分量（2π周期）
    return phase


class amp_motion_data_obs:
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
        motion_data_dir = cfg.params.get("motion_data_dir")
        motion_data_weights = cfg.params.get("motion_data_weights")
        self.motion_loader = MotionLoader(
            motion_data_dir=motion_data_dir, motion_data_weights=motion_data_weights, device=env.device
        )

    def __call__(self, env: ManagerBasedRlEnv, n_steps: int, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, **kwargs):
        # Sample motions
        body_names = asset_cfg.body_names
        motion_ids = self.motion_loader.sample_motion_ids(env.num_envs)
        motion_seq_times = self.motion_loader.sample_motion_seq_times(motion_ids, n_steps, env.step_dt)
        motion_data = self.motion_loader.get_motion_seq_data(
            motion_ids, motion_seq_times, body_names=body_names
        )  # (num_env, n_step, dim)
        # concat
        mat = matrix_from_quat(motion_data.body_quat_b)
        body_ori_b = mat[..., :2].reshape(mat.shape[0], -1)
        body_pos_b = motion_data.body_pos_b.reshape(env.num_envs, -1)
        body_lin_vel_b = motion_data.body_lin_vel_b.reshape(env.num_envs, -1)
        body_ang_vel_b = motion_data.body_ang_vel_b.reshape(env.num_envs, -1)
        return torch.concat([body_pos_b, body_ori_b, body_lin_vel_b, body_ang_vel_b], dim=-1)
