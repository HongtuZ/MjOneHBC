"""Useful methods for MDP events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from OneHBC.utils.motion_loader import MotionLoader

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class reset_from_motion_data:
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        motion_data_dir = cfg.params.get("motion_data_dir")
        motion_data_weights = cfg.params.get("motion_data_weights")
        self.motion_loader = MotionLoader(
            motion_data_dir=motion_data_dir, motion_data_weights=motion_data_weights, device=env.device
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        **kwargs,
    ):
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
        if len(env_ids) == 0:
            return
        # Sample motions
        asset = env.scene[asset_cfg.name]
        joint_names = asset_cfg.joint_names
        motion_ids = self.motion_loader.sample_motion_ids(env_ids.shape[0])
        motion_times = self.motion_loader.sample_motion_times(motion_ids)
        motion_data = self.motion_loader.get_motion_data(
            motion_ids, motion_times, joint_names=joint_names
        )  # (num_env, n_step, dim)
        ref_root_pos_w = motion_data["root_pos_w"] + env.scene.env_origins[env_ids]
        ref_root_pos_w[..., 2] = 0.78  # avoid penetration
        ref_root_quat_w = motion_data["root_quat_w"]
        ref_root_lin_vel_w = motion_data["root_lin_vel_w"]
        ref_root_ang_vel_w = torch.zeros_like(ref_root_lin_vel_w)

        joint_limits = asset.data.soft_joint_pos_limits[env_ids]
        ref_joint_pos = torch.clamp(motion_data["joint_pos"], joint_limits[..., 0], joint_limits[..., 1])

        asset.write_root_link_pose_to_sim(torch.cat([ref_root_pos_w, ref_root_quat_w], dim=-1), env_ids=env_ids)
        asset.write_root_link_velocity_to_sim(
            torch.cat([ref_root_lin_vel_w, ref_root_ang_vel_w], dim=-1), env_ids=env_ids
        )
        asset.write_joint_position_to_sim(ref_joint_pos, env_ids=env_ids)
