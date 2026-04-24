from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl import MjlabOnPolicyRunner

from .velocity_env_cfg import VelocityRoughEnvCfg, VelocityRoughPlayEnvCfg, VelocityFlatEnvCfg, VelocityFlatPlayEnvCfg
from .agents.rsl_rl_ppo_cfg import VelocityPPORunnerCfg

register_mjlab_task(
    task_id="Velocity-Rough-THS23DOF",
    env_cfg=VelocityRoughEnvCfg(),
    play_env_cfg=VelocityRoughPlayEnvCfg(),
    rl_cfg=VelocityPPORunnerCfg(),
    runner_cls=MjlabOnPolicyRunner,
)

register_mjlab_task(
    task_id="Velocity-Flat-THS23DOF",
    env_cfg=VelocityFlatEnvCfg(),
    play_env_cfg=VelocityFlatPlayEnvCfg(),
    rl_cfg=VelocityPPORunnerCfg(),
    runner_cls=MjlabOnPolicyRunner,
)
