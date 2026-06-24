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

    def compute_grad_penalty(self, amp_expert_obs: TensorDict, lambda_=10):
        latent = self.get_latent(amp_expert_obs)
        latent = latent.detach().clone()
        latent.requires_grad_(True)

        disc = self.mlp(latent)
        ones = torch.ones_like(disc, device=disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=latent,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Enforce that the grad norm approaches 0.
        grad_penalty = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_penalty

    def predict_style_reward(self, amp_obs: TensorDict):
        """
        Predict the AMP style reward given current and next states, optionally interpolated with a task reward.

        Args:
            expert_data (torch.Tensor): Expert data.
            lambda_ (float, optional): Gradient penalty coefficient. Defaults to 10.
            task_reward (torch.Tensor): Task-specific reward tensor.
            normalizer (optional): Normalizer object to normalize input states before prediction.

        Returns:
            tuple:
                - style_reward (torch.Tensor): Predicted AMP reward (optionally interpolated) with shape (batch_size,).
                - disc_score (torch.Tensor): Raw discriminator output logits with shape (batch_size, 1).
        """
        with torch.no_grad():
            self.eval()
            disc_score = self.forward(amp_obs)
            style_reward = self.style_reward_scale * torch.clamp(1 - (1.0 / 4) * torch.square(disc_score - 1), min=0)
            self.train()
        return style_reward.squeeze(-1), disc_score.squeeze(-1)

    def lerp_reward(self, style_reward: torch.Tensor, task_reward: torch.Tensor):
        reward = (1.0 - self.task_reward_lerp) * style_reward + self.task_reward_lerp * task_reward
        return reward
