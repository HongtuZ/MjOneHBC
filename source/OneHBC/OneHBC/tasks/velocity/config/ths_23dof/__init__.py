from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import VelocityRoughEnvCfg, VelocityRoughPlayEnvCfg, VelocityFlatEnvCfg, VelocityFlatPlayEnvCfg
from .rl_cfg import VelocityPPORunnerCfg
from OneHBC.tasks.velocity.rl import VelocityOnPolicyRunner

register_mjlab_task(
    task_id="Velocity-Rough-THS23DOF",
    env_cfg=VelocityRoughEnvCfg(),
    play_env_cfg=VelocityRoughPlayEnvCfg(),
    rl_cfg=VelocityPPORunnerCfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Velocity-Flat-THS23DOF",
    env_cfg=VelocityFlatEnvCfg(),
    play_env_cfg=VelocityFlatPlayEnvCfg(),
    rl_cfg=VelocityPPORunnerCfg(),
    runner_cls=VelocityOnPolicyRunner,
)
