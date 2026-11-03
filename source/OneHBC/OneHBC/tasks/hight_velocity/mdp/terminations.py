"""Useful methods for MDP terminations."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import mjlab.envs.mdp as mj_mdp
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class DelayTermination:
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
        self.max_delay_steps = cfg.params["max_delay_steps"]
        self.delay_counters = torch.zeros(env.num_envs, dtype=torch.int, device=env.device)

    def compute_delay(self, dones: torch.Tensor):
        if self.max_delay_steps <= 0:
            return dones
        self.delay_counters[dones] += 1
        self.delay_counters[~dones] = 0
        dones[self.delay_counters < self.max_delay_steps] = False
        return dones

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.delay_counters[env_ids] = 0

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError


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


class delay_bad_orientation(DelayTermination):
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        max_delay_steps: int,
        limit_angle: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del max_delay_steps  # Unused.
        dones = mj_mdp.bad_orientation(env, limit_angle, asset_cfg)
        return self.compute_delay(dones)


class delay_root_height_below_minimum(DelayTermination):
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        max_delay_steps: int,
        minimum_height: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del max_delay_steps  # Unused.
        dones = mj_mdp.root_height_below_minimum(env, minimum_height, asset_cfg)
        return self.compute_delay(dones)
