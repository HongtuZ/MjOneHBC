from mjlab.tasks.registry import register_mjlab_task

from OneHBC.tasks.amp.rl import AmpRunner

from .env_cfg import AmpFlatEnvCfg, AmpFlatPlayEnvCfg, AmpRoughEnvCfg, AmpRoughPlayEnvCfg
from .rl_cfg import AmpRunnerCfg

register_mjlab_task(
    task_id="Amp-Rough-ths_t2_29dof",
    env_cfg=AmpRoughEnvCfg(),
    play_env_cfg=AmpRoughPlayEnvCfg(),
    rl_cfg=AmpRunnerCfg(),
    runner_cls=AmpRunner,
)

register_mjlab_task(
    task_id="Amp-Flat-ths_t2_29dof",
    env_cfg=AmpFlatEnvCfg(),
    play_env_cfg=AmpFlatPlayEnvCfg(),
    rl_cfg=AmpRunnerCfg(),
    runner_cls=AmpRunner,
)
