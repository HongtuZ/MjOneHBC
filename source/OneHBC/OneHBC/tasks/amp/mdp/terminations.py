from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import mjlab.envs.mdp as mj_mdp
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
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
