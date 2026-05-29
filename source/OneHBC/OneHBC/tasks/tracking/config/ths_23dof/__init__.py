from mjlab.tasks.registry import register_mjlab_task

from OneHBC.tasks.tracking.rl import MotionTrackingOnPolicyRunner

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
    task_id="Tracking-Flat-THS23DOF",
    env_cfg=TrackingFlatEnvCfg(),
    play_env_cfg=TrackingFlatPlayEnvCfg(),
    rl_cfg=TrackingPPORunnerCfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Tracking-Rough-THS23DOF",
    env_cfg=TrackingRoughEnvCfg(),
    play_env_cfg=TrackingRoughPlayEnvCfg(),
    rl_cfg=TrackingPPORunnerCfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="GetupTracking-Flat-THS23DOF",
    env_cfg=GetupTrackingFlatEnvCfg(),
    play_env_cfg=GetupTrackingFlatPlayEnvCfg(),
    rl_cfg=TrackingPPORunnerCfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="GetupTracking-Rough-THS23DOF",
    env_cfg=GetupTrackingRoughEnvCfg(),
    play_env_cfg=GetupTrackingRoughPlayEnvCfg(),
    rl_cfg=TrackingPPORunnerCfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="FalldownTracking-Flat-THS23DOF",
    env_cfg=FalldownTrackingFlatEnvCfg(),
    play_env_cfg=FalldownTrackingFlatPlayEnvCfg(),
    rl_cfg=TrackingPPORunnerCfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="FalldownTracking-Rough-THS23DOF",
    env_cfg=FalldownTrackingRoughEnvCfg(),
    play_env_cfg=FalldownTrackingRoughPlayEnvCfg(),
    rl_cfg=TrackingPPORunnerCfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
