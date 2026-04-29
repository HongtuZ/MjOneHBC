"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, RayCastSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


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
