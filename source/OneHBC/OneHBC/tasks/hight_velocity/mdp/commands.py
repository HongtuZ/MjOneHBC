"""Height command for tracking a target base height."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class UniformHeightCommand(CommandTerm):
    """Command that samples a target base height for the robot to track.

    The command tensor has shape ``(num_envs, 1)`` and holds the desired base
    (root link) height above the world ground plane. It is resampled uniformly
    from :attr:`UniformHeightCommandCfg.ranges.height` at the configured interval.
    """

    cfg: UniformHeightCommandCfg

    def __init__(self, cfg: UniformHeightCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.height_command

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        low, high = self.cfg.ranges.height
        # Binary command: crouch (low) with probability rel_crouch_envs, otherwise stand (high).
        r = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 1.0)
        self.height_command[env_ids, 0] = torch.where(
            r < self.cfg.rel_crouch_envs,
            torch.full_like(r, low),
            torch.full_like(r, high),
        )

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        # Static command; pure function of the current state, so env_ids is unused.
        del env_ids


@dataclass(kw_only=True)
class UniformHeightCommandCfg(CommandTermCfg):
    entity_name: str

    @dataclass
    class Ranges:
        height: tuple[float, float]

    ranges: Ranges
    rel_crouch_envs: float = 0.5
    """Fraction of environments that receive a crouch command (height = ranges.height[0]).

    The command is binary: crouch with probability ``rel_crouch_envs``, stand
    (``ranges.height[1]``) otherwise.
    """

    def build(self, env: ManagerBasedRlEnv) -> UniformHeightCommand:
        return UniformHeightCommand(self, env)
