from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl import MjlabOnPolicyRunner

from .env_cfg import TrackingFlatEnvCfg, TrackingFlatPlayEnvCfg
from .rl_cfg import TrackingPPORunnerCfg

register_mjlab_task(
    task_id="Tracking-Flat-THS23DOF",
    env_cfg=TrackingFlatEnvCfg(),
    play_env_cfg=TrackingFlatPlayEnvCfg(),
    rl_cfg=TrackingPPORunnerCfg(),
    runner_cls=MjlabOnPolicyRunner,
)
