from mjlab.tasks.registry import register_mjlab_task
from OneHBC.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfg import TrackingFlatEnvCfg, TrackingFlatPlayEnvCfg, TrackingRoughEnvCfg, TrackingRoughPlayEnvCfg
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