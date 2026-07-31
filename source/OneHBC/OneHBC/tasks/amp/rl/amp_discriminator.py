import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from torch import autograd


class AmpDiscriminator(MLPModel):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        style_reward_scale: float = 1.0,
        task_reward_lerp: float = 0.0,
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        self.style_reward_scale = style_reward_scale
        self.task_reward_lerp = task_reward_lerp

    def compute_grad_penalty(self, amp_expert_obs: TensorDict, lambda_: float = 10.0) -> torch.Tensor:
        """Compute gradient penalty to regularize the discriminator.

        Enforces that the gradient norm of the discriminator output w.r.t. its input approaches zero,
        preventing the discriminator from being too sharp.
        """
        latent = self.get_latent(amp_expert_obs)
        latent = latent.detach().clone().requires_grad_(True)

        disc = self.mlp(latent)
        grad = autograd.grad(
            outputs=disc,
            inputs=latent,
            grad_outputs=torch.ones_like(disc),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return lambda_ * grad.norm(2, dim=1).pow(2).mean()

    def predict_style_reward(self, amp_obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the AMP style reward from discriminator output.

        Uses the reward formula: r_style = scale * clamp(1 - 0.25 * (D(s) - 1)^2, min=0)

        Args:
            amp_obs: Observation TensorDict containing the discriminator observation group.

        Returns:
            tuple:
                - style_reward: Predicted style reward with shape (batch_size,).
                - disc_score: Raw discriminator output with shape (batch_size,).
        """
        with torch.no_grad():
            self.eval()
            disc_score = self.forward(amp_obs)
            style_reward = self.style_reward_scale * torch.clamp(1.0 - 0.25 * torch.square(disc_score - 1.0), min=0.0)
            self.train()
        return style_reward.squeeze(-1), disc_score.squeeze(-1)

    def lerp_reward(self, style_reward: torch.Tensor, task_reward: torch.Tensor) -> torch.Tensor:
        """Interpolate between style reward and task reward.

        reward = (1 - lerp) * style_reward + lerp * task_reward
        """
        return (1.0 - self.task_reward_lerp) * style_reward + self.task_reward_lerp * task_reward
