from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from dataclasses import dataclass, field


@dataclass
class VelocityPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    logger: str = "tensorboard"
    num_steps_per_env: int = 24
    max_iterations: int = 30_000
    save_interval: int = 50
    obs_groups: dict = field(default_factory=lambda: {"actor": ["actor"], "critic": ["critic"]})
    experiment_name: str = "ths_23dof_velocity"
    actor: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        )
    )
    critic: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        )
    )
    algorithm: RslRlPpoAlgorithmCfg = field(
        default_factory=lambda: RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    )
