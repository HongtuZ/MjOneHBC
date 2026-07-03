from mjlab.tasks.registry import register_mjlab_task

from OneHBC.tasks.bfm.rl import MotionTrackingOnPolicyRunner

from .rl_cfg import TrackingPPORunnerCfg


from .env_cfg import (
    Tracking_T2_FlatEnvCfg,
    Tracking_T2_FlatPlayEnvCfg,
)
from .rl_cfg import TrackingPPORunnerCfg

register_mjlab_task(
    task_id="BFM-T2-Flat-THS29DOF",
    env_cfg=Tracking_T2_FlatEnvCfg(),
    play_env_cfg=Tracking_T2_FlatPlayEnvCfg(),
    rl_cfg=TrackingPPORunnerCfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)