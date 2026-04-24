from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnv
from .one_hbc_env_cfg import OneHBCEnvCfg
from OneHBC.utils.motion_loader import MotionLoader


class OneHBCEnv(ManagerBasedRlEnv):
    cfg: OneHBCEnvCfg

    def __init__(self, cfg: OneHBCEnvCfg, device: str, render_mode: str | None = None, **kwargs):
        self.motion_loader = MotionLoader(
            cfg.motion_loader.motion_data_dir,
            cfg.motion_loader.motion_data_weights,
            device=device,
        )
        super().__init__(cfg=cfg, device=device, render_mode=render_mode, **kwargs)
