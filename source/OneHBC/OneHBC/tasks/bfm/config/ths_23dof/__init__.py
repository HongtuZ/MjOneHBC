from mjlab.tasks.registry import register_mjlab_task

from OneHBC.tasks.bfm.rl import MotionTrackingOnPolicyRunner

from .env_cfg import (
    FalldownTrackingFlatEnvCfg,
    FalldownTrackingFlatPlayEnvCfg,
    FalldownTrackingRoughEnvCfg,
    FalldownTrackingRoughPlayEnvCfg,
    GetupTrackingFlatEnvCfg,
    GetupTrackingFlatPlayEnvCfg,
    GetupTrackingRoughEnvCfg,
    GetupTrackingRoughPlayEnvCfg,
    TrackingFlatEnvCfg,
    TrackingFlatPlayEnvCfg,
    TrackingRoughEnvCfg,
    TrackingRoughPlayEnvCfg,
)
from .rl_cfg import TrackingPPORunnerCfg

register_mjlab_task(
    task_id="BFM-Flat-THS23DOF",
    env_cfg=TrackingFlatEnvCfg(),
    play_env_cfg=TrackingFlatPlayEnvCfg(),
    rl_cfg=TrackingPPORunnerCfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="BFM-Rough-THS23DOF",
    env_cfg=TrackingRoughEnvCfg(),
    play_env_cfg=TrackingRoughPlayEnvCfg(),
    rl_cfg=TrackingPPORunnerCfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)