"""Useful methods for MDP terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def bad_orientation(
    env: ManagerBasedRlEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Terminate when the asset's orientation exceeds the limit angle."""
    asset: Entity = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b
    return torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0)).abs() > limit_angle


def unexpected_collision(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Terminate when an unexpected ground contact is detected.

    Contacts are read from the contact sensor ``sensor_name`` (e.g. the one
    built from ``unexpected_ground_contact_cfg``; its ``primary`` match already
    selects which joints/links to monitor). If the sensor records a force
    history (``history_length > 0``), a contact counts only when its force
    magnitude exceeds ``force_threshold``, which also catches transient
    collisions happening between policy steps. Otherwise any contact triggers
    termination.

    Args:
        env: The environment instance.
        sensor_name: Name of the :class:`~mjlab.sensor.ContactSensor` to read.
        force_threshold: Contact force magnitude (N) above which a contact is
            treated as a collision. Only used when a force history is available.

    Returns:
        Boolean tensor of shape ``[num_envs]``.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data

    if data.force_history is not None:
        # force_history: [B, N, H, 3]
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]

    assert data.found is not None
    return torch.any(data.found > 0, dim=-1)  # [B]
