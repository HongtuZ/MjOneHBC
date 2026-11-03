from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import VelocityRoughEnvCfg, VelocityRoughPlayEnvCfg, VelocityFlatEnvCfg, VelocityFlatPlayEnvCfg
from .rl_cfg import VelocityPPORunnerCfg
from OneHBC.tasks.hight_velocity.rl import VelocityOnPolicyRunner

register_mjlab_task(
    task_id="HightVelocity-Rough-THS_OSTRICH",
    env_cfg=VelocityRoughEnvCfg(),
    play_env_cfg=VelocityRoughPlayEnvCfg(),
    rl_cfg=VelocityPPORunnerCfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="HightVelocity-Flat-THS_OSTRICH",
    env_cfg=VelocityFlatEnvCfg(),
    play_env_cfg=VelocityFlatPlayEnvCfg(),
    rl_cfg=VelocityPPORunnerCfg(),
    runner_cls=VelocityOnPolicyRunner,
)
