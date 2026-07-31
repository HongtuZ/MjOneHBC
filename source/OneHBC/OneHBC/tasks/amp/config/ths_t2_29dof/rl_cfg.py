from dataclasses import dataclass, field

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@dataclass
class AmpModelCfg(RslRlModelCfg):
    style_reward_scale: float = 1.0
    task_reward_lerp: float = 0.0


@dataclass
class AmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm config extended with AMP discriminator learning rate."""

    discriminator_lr: float = 5e-4


@dataclass
class AmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    logger: str = "tensorboard"
    upload_model = False
    num_steps_per_env: int = 24
    max_iterations: int = 30_000
    save_interval: int = 50
    obs_groups: dict = field(
        default_factory=lambda: {
            "actor": ["actor"],
            "critic": ["critic"],
            "discriminator": ["discriminator"],
            "discriminator_expert": ["discriminator_expert"],
        }
    )
    experiment_name: str = "ths_t2_29dof_amp"
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
    discriminator: AmpModelCfg = field(
        default_factory=lambda: AmpModelCfg(
            hidden_dims=(1024, 512, 256),
            activation="elu",
            obs_normalization=True,
            style_reward_scale=0.5,
            task_reward_lerp=0.3,
        )
    )
    algorithm: AmpAlgorithmCfg = field(
        default_factory=lambda: AmpAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            discriminator_lr=5.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    )
