"""Reset functions for the hight_velocity task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_SE3_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def _sample_se3_range(
    range_dict: dict[str, tuple[float, float]] | None,
    shape: tuple[int, ...],
    device: str,
) -> torch.Tensor:
    """Sample uniform ``[x, y, z, roll, pitch, yaw]`` offsets.

    ``range_dict`` maps any subset of those keys to ``(min, max)`` ranges; missing
    keys default to ``(0.0, 0.0)`` (no offset). ``None`` is treated as empty.
    """
    range_dict = range_dict or {}
    range_list = [range_dict.get(key, (0.0, 0.0)) for key in _SE3_KEYS]
    ranges = torch.tensor(range_list, device=device)
    return sample_uniform(ranges[:, 0], ranges[:, 1], shape, device=device)


def reset_posture_uniform(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    crouch_joint_pos: dict[str, float],
    crouch_root_height: float = 0.06,
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Reset root pose and joint positions to a randomly sampled crouch or stand pose.

    Each environment randomly picks either the crouch pose (alpha = 0) or the stand
    pose (alpha = 1). Joint positions are set to the crouch or stand pose directly
    (no interpolation). For standing, the root height is taken from the default pose;
    for crouching, the root height is fixed to ``crouch_root_height``.

    Args:
        env: The environment.
        env_ids: Environment IDs to reset. If None, resets all environments.
        asset_cfg: Asset configuration (``joint_names`` should cover the actuated joints).
        crouch_joint_pos: Mapping from joint name to crouch joint position (rad).
        crouch_root_height: Root height (m) above ground for the crouch pose.
        pose_range: Optional uniform ranges for x/y/roll/pitch/yaw offsets.
        velocity_range: Optional uniform ranges for the root velocity (default: zero).
    """
    env_ids = resolve_env_ids(env, env_ids)
    asset: Entity = env.scene[asset_cfg.name]

    # Randomly sample crouch (0) or stand (1); no interpolation.
    alpha = (torch.rand(len(env_ids), device=env.device) < 0.5).float()  # [N]

    # --- Joint positions: crouch + alpha * (stand - crouch) ---
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    _, joint_names = asset.find_joints(asset_cfg.joint_names)
    _, _, crouch_values = resolve_matching_names_values(
        data=crouch_joint_pos, list_of_strings=joint_names
    )
    crouch_pos = torch.tensor(
        crouch_values, device=env.device, dtype=torch.float32
    )  # [J]
    stand_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids]  # [N, J]
    joint_pos = crouch_pos + alpha.unsqueeze(1) * (stand_pos - crouch_pos)  # [N, J]
    joint_vel = torch.zeros_like(joint_pos)

    # --- Root state (x/y/yaw randomized; z is fixed per posture below) ---
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()

    pose_samples = _sample_se3_range(pose_range, (len(env_ids), 6), env.device)
    root_states[:, 0:2] = (
        root_states[:, 0:2]
        + pose_samples[:, 0:2]
        + env.scene.env_origins[env_ids, :2]
    )
    orientations_delta = quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    root_states[:, 3:7] = quat_mul(root_states[:, 3:7], orientations_delta)
    vel_samples = _sample_se3_range(velocity_range, (len(env_ids), 6), env.device)
    root_states[:, 7:13] = root_states[:, 7:13] + vel_samples

    # Root height: crouch -> fixed height; stand -> default pose height.
    root_height = torch.where(
        alpha > 0.5,
        default_root_state[env_ids][:, 2],  # stand
        torch.full_like(alpha, crouch_root_height),  # crouch
    )  # [N]
    root_states[:, 2] = env.scene.env_origins[env_ids, 2] + root_height

    # --- Write joint + root state ---
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)
    asset.write_joint_state_to_sim(
        joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids
    )
    asset.write_root_state_to_sim(root_states, env_ids=env_ids)
