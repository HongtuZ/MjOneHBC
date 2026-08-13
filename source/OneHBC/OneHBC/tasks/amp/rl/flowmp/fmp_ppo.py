from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
from mjlab.utils.lab_api.math import quat_apply_inverse
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer
from tensordict import TensorDict

from OneHBC.utils.motion_loader import MotionLoader

FLOW_CONDITION_KEY = "flow_condition"
"""Key under which the motion-derived state condition is stored inside the rollout observations."""


class FlowExpertDataset:
    """Expert replay dataset of ``(condition, action)`` pairs extracted from motion data.

    The state condition layout is ``[projected_gravity(3), base_ang_vel_b(3), joint_pos_rel(n),
    joint_vel_rel(n), last_action(n)]`` and can be reconstructed identically from the live robot
    state during rollouts (see :meth:`FmpPPO._build_condition_from_env`).

    Expert actions are obtained by inverting the affine action mapping of the joint-position
    action term: ``a* = (joint_pos - offset) / scale``.
    """

    def __init__(
        self,
        motion_loader: MotionLoader,
        env,
        num_samples: int,
        step_dt: float,
        device: str = "cpu",
    ) -> None:
        """Build the expert dataset by sampling motion frames.

        Args:
            motion_loader: Loaded motion dataset.
            env: The unwrapped manager-based environment (used for action space information).
            num_samples: Number of expert frames to extract.
            step_dt: Control timestep of the environment (used to fetch the previous frame).
            device: Target device.
        """
        self.device = device

        # Resolve the joint-position action term to map joint positions <-> actions.
        action_term = env.action_manager.get_term("joint_pos")
        self.action_joint_names: list[str] = list(action_term._target_names)
        scale = action_term.scale
        offset = action_term.offset
        if isinstance(scale, torch.Tensor):
            scale = scale[0] if scale.ndim > 1 else scale
            self._action_scale: torch.Tensor | float = scale.to(device)
        else:
            self._action_scale = float(scale)
        if isinstance(offset, torch.Tensor):
            offset = offset[0] if offset.ndim > 1 else offset
            self._action_offset: torch.Tensor | float = offset.to(device)
        else:
            self._action_offset = float(offset)

        # Sample motion frames: t for the current state and t - step_dt for the previous action.
        motion_ids = motion_loader.sample_motion_ids(num_samples)
        times = motion_loader.sample_motion_times(motion_ids, truncate_time_start=step_dt)
        prev_times = times - step_dt
        data = motion_loader.get_motion_data(motion_ids, times, joint_names=self.action_joint_names)
        prev_data = motion_loader.get_motion_data(motion_ids, prev_times, joint_names=self.action_joint_names)

        # Projected gravity and base angular velocity in the base frame.
        root_quat_w = data["root_quat_w"]
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=device).expand(num_samples, 3)
        projected_gravity = quat_apply_inverse(root_quat_w, gravity_w)
        base_ang_vel_b = data["root_ang_vel_b"]

        joint_pos_rel = data["joint_pos"] - self._action_offset
        joint_vel_rel = data["joint_vel"]
        # Expert actions: invert the affine action mapping of the joint-position action term.
        expert_actions = (data["joint_pos"] - self._action_offset) / self._action_scale
        prev_actions = (prev_data["joint_pos"] - self._action_offset) / self._action_scale

        self.conditions = torch.cat(
            [projected_gravity, base_ang_vel_b, joint_pos_rel, joint_vel_rel, prev_actions], dim=-1
        ).float()
        self.actions = expert_actions.float()
        self.num_samples = self.conditions.shape[0]
        self.condition_dim = self.conditions.shape[-1]
        self.action_dim = self.actions.shape[-1]

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of ``(condition, action)`` expert pairs."""
        idx = torch.randint(0, self.num_samples, (batch_size,), device=self.device)
        return self.conditions[idx], self.actions[idx]


class BcFlowNet(nn.Module):
    """Conditional flow matching velocity network (the BC flow teacher).

    Parameterizes the velocity field ``v(c, a_t, t)`` of the ODE that transports standard Gaussian
    noise into the expert action distribution, trained with the conditional flow matching objective.
    """

    def __init__(
        self,
        condition_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 256),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.action_dim = action_dim
        self.mlp = MLP(condition_dim + action_dim + 1, action_dim, hidden_dims, activation)

    def forward(self, condition: torch.Tensor, noisy_action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the flow velocity at (condition, noisy_action, t)."""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.mlp(torch.cat([condition, noisy_action, t], dim=-1))

    @torch.no_grad()
    def integrate_euler(self, condition: torch.Tensor, noise: torch.Tensor, num_steps: int) -> torch.Tensor:
        """Integrate the flow ODE from ``noise`` (t=0) to actions (t=1) with Euler steps."""
        action = noise
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((action.shape[0], 1), i * dt, device=action.device, dtype=action.dtype)
            action = action + dt * self.forward(condition, action, t)
        return action

    def compute_cfm_loss(self, condition: torch.Tensor, expert_actions: torch.Tensor) -> torch.Tensor:
        """Conditional flow matching loss on expert data.

        Samples ``t ~ U(0, 1)``, interpolates ``a_t = (1 - t) * noise + t * a*`` and regresses the
        velocity towards the conditional vector field ``a* - noise``.
        """
        batch_size = expert_actions.shape[0]
        noise = torch.randn_like(expert_actions)
        t = torch.rand(batch_size, 1, device=expert_actions.device, dtype=expert_actions.dtype)
        noisy_action = (1.0 - t) * noise + t * expert_actions
        target_velocity = expert_actions - noise
        pred_velocity = self.forward(condition, noisy_action, t)
        return nn.functional.mse_loss(pred_velocity, target_velocity)


class OneStepFlowPolicy(MLPModel):
    """One-step flow policy distilled from the BC flow teacher.

    Produces an action with a single Euler step from Gaussian noise:
    ``a = eps + v_theta(c_actor, eps, t=0)``, where ``c_actor`` is the latent of the selected
    observation groups (normalized). The noise ``eps`` is drawn from the configured output
    distribution so that exploration and the (approximate) log-prob machinery of rsl_rl remain
    available for logging.
    """

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
    ) -> None:
        # Default to a Gaussian noise distribution if none is provided.
        if distribution_cfg is None:
            distribution_cfg = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )
        # One-step policy: v_theta(c, eps, t=0), input = obs_latent + noise + flow time.
        self.mlp = MLP(self.obs_dim + output_dim + 1, output_dim, hidden_dims, activation)

    def velocity(self, obs: TensorDict, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the one-step velocity field v_theta(c, eps, t)."""
        latent = self.get_latent(obs)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.mlp(torch.cat([latent, noise, t], dim=-1))

    def deterministic_action(self, obs: TensorDict, noise: torch.Tensor) -> torch.Tensor:
        """One Euler step from ``noise`` at t=0: a = eps + v_theta(c, eps, 0)."""
        t = torch.zeros(noise.shape[0], 1, device=noise.device, dtype=noise.dtype)
        return noise + self.velocity(obs, noise, t)

    def sample_noise(self, batch_size: int, device: str) -> torch.Tensor:
        """Sample flow noise eps ~ N(0, sigma^2) via the configured Gaussian distribution."""
        zeros = torch.zeros(batch_size, self.distribution.output_dim, device=device)
        self.distribution.update(zeros)
        return self.distribution.sample()

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Sample noise from the distribution and map it to an action with one flow step."""
        if stochastic_output:
            noise = self.sample_noise(obs.batch_size[0], obs[FLOW_CONDITION_KEY].device)
        else:
            noise = torch.zeros(obs.batch_size[0], self.distribution.output_dim, device=obs[FLOW_CONDITION_KEY].device)
        return self.deterministic_action(obs, noise)

    @property
    def output_std(self) -> torch.Tensor:
        """Noise (exploration) standard deviation of the policy."""
        return self.distribution.std

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxOneStepFlowPolicy(self)


class _OnnxOneStepFlowPolicy(nn.Module):
    """ONNX-exportable wrapper of the one-step flow policy (deterministic, zero-noise mode)."""

    is_recurrent: bool = False

    def __init__(self, model: OneStepFlowPolicy) -> None:
        super().__init__()
        import copy

        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.action_dim = model.distribution.output_dim
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.obs_normalizer(x)
        noise = torch.zeros(latent.shape[0], self.action_dim, device=x.device)
        t = torch.zeros(latent.shape[0], 1, device=x.device)
        return noise + self.mlp(torch.cat([latent, noise, t], dim=-1))

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class FlowCritic(nn.Module):
    """Q(s, a) critic over the same observation groups as the one-step flow policy."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        action_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
    ) -> None:
        super().__init__()
        self.obs_groups = obs_groups[obs_set]
        self.obs_dim = sum(obs[g].shape[-1] for g in self.obs_groups)
        self.action_dim = action_dim
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()
        self.mlp = MLP(self.obs_dim + action_dim, 1, hidden_dims, activation)
        self.is_recurrent = False

    def get_latent(self, obs: TensorDict) -> torch.Tensor:
        latent = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
        return self.obs_normalizer(latent)

    def forward(self, obs: TensorDict, actions: torch.Tensor | None = None) -> torch.Tensor:
        """Q-value of ``(obs, actions)``. If ``actions`` is None, evaluate the policy's own action."""
        latent = self.get_latent(obs)
        if actions is None:
            raise ValueError("FlowCritic requires explicit actions.")
        return self.mlp(torch.cat([latent, actions], dim=-1))

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            self.obs_normalizer.update(torch.cat([obs[g] for g in self.obs_groups], dim=-1))

    def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
        pass

    def get_hidden_state(self):
        return None


class FmpPPO:
    """Flow Matching Policy with PPO (FMP-PPO).

    Three-stage design:
        1. **BC flow pre-training**: a conditional flow matching network (the BC flow) is trained on
           expert ``(state, action)`` pairs extracted from the motion dataset.
        2. **One-step flow policy distillation**: the deployed policy produces actions with a single
           Euler step from Gaussian noise and is distilled against multi-step ODE rollouts of the
           frozen BC flow teacher.
        3. **PPO task Q maximization**: the same one-step policy is fine-tuned to maximize the
           critic's task Q-value, where the critic is trained with PPO-style clipped value loss on
           GAE returns from the task reward.

    Reference:
        - Lipman et al. "Flow Matching for Generative Modeling." ICLR 2023.
        - Chen et al. "FlowAC: Flow-based Actor-Critic Model for Humanoid Control." 2025.
    """

    actor: OneStepFlowPolicy
    critic: FlowCritic
    bc_flow: BcFlowNet

    def __init__(
        self,
        actor: OneStepFlowPolicy,
        critic: FlowCritic,
        bc_flow: BcFlowNet,
        expert_dataset: FlowExpertDataset | None,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        bc_flow_lr: float = 1e-4,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float | None = None,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # Flow matching / distillation parameters
        bc_pretrain_iters: int = 2000,
        bc_batch_size: int = 4096,
        bc_updates_per_iter: int = 1,
        distill_num_steps: int = 10,
        distill_coeff: float = 1.0,
        q_coeff: float = 0.1,
        exploration_noise: float = 0.0,
        # RND / symmetry / distributed placeholders (kept for runner compatibility)
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        if rnd_cfg is not None:
            raise NotImplementedError("FmpPPO does not support RND.")
        if symmetry_cfg is not None:
            raise NotImplementedError("FmpPPO does not support symmetry augmentation.")
        self.rnd = None
        self.symmetry = None
        self.intrinsic_rewards = None

        # Components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.bc_flow = bc_flow.to(self.device)
        self.expert_dataset = expert_dataset
        self._raw_actor = self.actor
        self._raw_critic = self.critic
        self._raw_bc_flow = self.bc_flow

        # Optimizers: one for policy + critic, one for the BC flow teacher.
        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic.parameters()), lr=learning_rate
        )  # type: ignore
        self.bc_flow_optimizer = resolve_optimizer(optimizer)(self.bc_flow.parameters(), lr=bc_flow_lr)  # type: ignore

        # Storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # Flow matching / distillation parameters
        self.bc_pretrain_iters = bc_pretrain_iters
        self.bc_batch_size = bc_batch_size
        self.bc_updates_per_iter = bc_updates_per_iter
        self.distill_num_steps = distill_num_steps
        self.distill_coeff = distill_coeff
        self.q_coeff = q_coeff
        self.exploration_noise = exploration_noise

        # Rollout bookkeeping
        self._last_actions: torch.Tensor | None = None
        self._last_dones: torch.Tensor | None = None
        self._env = None  # set by construct_algorithm for live condition building

        # Stage 1: pre-train the BC flow on expert data, then initialize the one-step policy from
        # the pre-trained velocity field so that distillation starts from a good policy.
        if self.bc_pretrain_iters > 0:
            self.pretrain_bc_flow()
        self._init_actor_from_bc_flow()

    # ------------------------------------------------------------------ #
    # Stage 1: BC flow pre-training
    # ------------------------------------------------------------------ #
    def _init_actor_from_bc_flow(self) -> None:
        """Copy matching BC flow velocity-network weights into the one-step policy MLP."""
        actor_sd = self.actor.mlp.state_dict()
        bc_sd = self.bc_flow.mlp.state_dict()
        copied = 0
        for key, value in bc_sd.items():
            if key in actor_sd and actor_sd[key].shape == value.shape:
                actor_sd[key] = value.clone()
                copied += 1
        self.actor.mlp.load_state_dict(actor_sd)
        print(f"[FmpPPO] Initialized one-step policy from BC flow ({copied}/{len(bc_sd)} tensors copied).")

    def pretrain_bc_flow(self, num_iters: int | None = None) -> None:
        """Train the BC flow on expert data with the conditional flow matching loss."""
        if self.expert_dataset is None:
            print("[FmpPPO] No expert dataset available, skipping BC flow pre-training.")
            return
        num_iters = num_iters if num_iters is not None else self.bc_pretrain_iters
        self.bc_flow.train()
        log_interval = max(1, num_iters // 10)
        for i in range(num_iters):
            cond, expert_actions = self.expert_dataset.sample(self.bc_batch_size)
            loss = self.bc_flow.compute_cfm_loss(cond, expert_actions)
            self.bc_flow_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.bc_flow.parameters(), self.max_grad_norm)
            self.bc_flow_optimizer.step()
            if i % log_interval == 0 or i == num_iters - 1:
                print(f"[FmpPPO] BC flow pretrain iter {i}/{num_iters}, cfm_loss={loss.item():.4f}")

    # ------------------------------------------------------------------ #
    # Rollout interface
    # ------------------------------------------------------------------ #
    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample one-step flow actions and store transition data."""
        # Reset the previous-action buffer for environments that just terminated.
        if self._last_dones is not None and self._last_actions is not None:
            reset_mask = self._last_dones.squeeze(-1).bool()
            if reset_mask.any():
                self._last_actions[reset_mask] = 0.0
        if self._last_actions is None:
            self._last_actions = torch.zeros(obs.batch_size[0], self.actor.distribution.output_dim, device=self.device)

        # Build and record the motion-style state condition.
        condition = self._build_condition_from_env(obs)
        obs = obs.clone()
        obs[FLOW_CONDITION_KEY] = condition

        self.transition.hidden_states = (None, None)
        # Sample noise and map it to an action with one flow step.
        noise = self.actor.sample_noise(obs.batch_size[0], self.device)
        actions = self.actor.deterministic_action(obs, noise)
        if self.exploration_noise > 0.0:
            actions = actions + self.exploration_noise * torch.randn_like(actions)
        self.transition.actions = actions.detach()
        self.transition.values = self.critic(obs, self.transition.actions).detach()
        # Approximate log-prob of the sampled noise under the Gaussian noise distribution.
        self.transition.actions_log_prob = self.actor.get_output_log_prob(noise).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs
        self._last_actions = self.transition.actions.clone()
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update the normalizers."""
        self.actor.update_normalization(self.transition.observations)  # type: ignore
        self.critic.update_normalization(self.transition.observations)  # type: ignore

        # Task reward only (the BC flow teacher replaces the adversarial style reward).
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self._last_dones = dones.clone()

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        self.storage.add_transition(self.transition)
        self.transition.clear()

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions (GAE)."""
        st = self.storage
        # Bootstrap value of the final state with the critic evaluated on the policy's own action.
        condition = self._build_condition_from_env(obs)
        obs = obs.clone()
        obs[FLOW_CONDITION_KEY] = condition
        with torch.no_grad():
            noise = torch.zeros(obs.batch_size[0], self.actor.distribution.output_dim, device=self.device)
            last_actions = self.actor.deterministic_action(obs, noise)
            last_values = self.critic(obs, last_actions)

        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    # ------------------------------------------------------------------ #
    # Stage 2 & 3: distillation + PPO task Q maximization
    # ------------------------------------------------------------------ #
    def update(self) -> dict[str, float]:
        """Run optimization epochs: PPO critic update, one-step flow policy update
        (distillation + Q maximization) and continued BC flow training."""
        mean_value_loss = 0.0
        mean_distill_loss = 0.0
        mean_q_loss = 0.0
        mean_q_value = 0.0
        mean_bc_flow_loss = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        valid_update_cnt = 0
        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            batch_conditions = batch.observations[FLOW_CONDITION_KEY]

            # ---------------- Critic: PPO clipped value loss on task returns ---------------- #
            values = self.critic(batch.observations, batch.actions)
            if not torch.isfinite(values).all():
                continue
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            # ---------------- Actor: distillation + Q maximization ---------------- #
            noise = torch.randn(batch.actions.shape, device=self.device)
            student_actions = self.actor.deterministic_action(batch.observations, noise)

            # Task Q maximization.
            q_values = self.critic(batch.observations, student_actions)
            q_loss = -q_values.mean()

            # Distillation against the frozen BC flow teacher (multi-step ODE rollout).
            with torch.no_grad():
                teacher_actions = self.bc_flow.integrate_euler(
                    batch_conditions, noise, num_steps=self.distill_num_steps
                )
            distill_loss = nn.functional.mse_loss(student_actions, teacher_actions)

            actor_loss = self.q_coeff * q_loss + self.distill_coeff * distill_loss

            if not torch.isfinite(actor_loss) or not torch.isfinite(value_loss):
                continue

            loss = self.value_loss_coef * value_loss + actor_loss

            # Update policy + critic.
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # ---------------- BC flow teacher: keep refining on expert data ---------------- #
            bc_flow_loss_val = 0.0
            if self.expert_dataset is not None:
                for _ in range(self.bc_updates_per_iter):
                    cond, expert_actions = self.expert_dataset.sample(
                        min(self.bc_batch_size, batch_conditions.shape[0])
                    )
                    bc_flow_loss = self.bc_flow.compute_cfm_loss(cond, expert_actions)
                    self.bc_flow_optimizer.zero_grad()
                    bc_flow_loss.backward()
                    nn.utils.clip_grad_norm_(self.bc_flow.parameters(), self.max_grad_norm)
                    self.bc_flow_optimizer.step()
                    bc_flow_loss_val += bc_flow_loss.item()
                bc_flow_loss_val /= self.bc_updates_per_iter

            mean_value_loss += value_loss.item()
            mean_distill_loss += distill_loss.item()
            mean_q_loss += q_loss.item()
            mean_q_value += q_values.mean().item()
            mean_bc_flow_loss += bc_flow_loss_val
            valid_update_cnt += 1

        num_updates = max(valid_update_cnt, 1)
        loss_dict = {
            "value": mean_value_loss / num_updates,
            "distill": mean_distill_loss / num_updates,
            "q_loss": mean_q_loss / num_updates,
            "q_value": mean_q_value / num_updates,
            "bc_flow": mean_bc_flow_loss / num_updates,
        }
        self.storage.clear()
        return loss_dict

    # ------------------------------------------------------------------ #
    # Condition building
    # ------------------------------------------------------------------ #
    def _build_condition_from_env(self, obs: TensorDict) -> torch.Tensor:
        """Reconstruct the motion-style state condition from the live robot state.

        Layout: [projected_gravity(3), base_ang_vel_b(3), joint_pos_rel(n), joint_vel_rel(n),
        last_action(n)] — identical to the expert condition layout.
        """
        env = self._env
        asset = env.scene["robot"]
        num_envs = obs.batch_size[0]

        root_quat_w = asset.data.root_link_quat_w
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(num_envs, 3)
        projected_gravity = quat_apply_inverse(root_quat_w, gravity_w)
        base_ang_vel_b = quat_apply_inverse(root_quat_w, asset.data.root_link_ang_vel_w)

        action_term = env.action_manager.get_term("joint_pos")
        offset = action_term.offset
        if isinstance(offset, torch.Tensor) and offset.ndim > 1:
            offset = offset[0]
        joint_pos_rel = asset.data.joint_pos - offset
        joint_vel_rel = asset.data.joint_vel

        last_actions = (
            self._last_actions
            if self._last_actions is not None
            else torch.zeros(num_envs, self.actor.distribution.output_dim, device=self.device)
        )
        return torch.cat(
            [projected_gravity, base_ang_vel_b, joint_pos_rel, joint_vel_rel, last_actions], dim=-1
        ).detach()

    # ------------------------------------------------------------------ #
    # Utilities (runner interface)
    # ------------------------------------------------------------------ #
    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()
        self.bc_flow.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()
        self.bc_flow.eval()

    def save(self) -> dict:
        return {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "bc_flow_state_dict": self._raw_bc_flow.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "bc_flow_optimizer_state_dict": self.bc_flow_optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "bc_flow": True,
                "optimizer": True,
                "iteration": True,
            }
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("bc_flow") and "bc_flow_state_dict" in loaded_dict:
            self._raw_bc_flow.load_state_dict(loaded_dict["bc_flow_state_dict"], strict=strict)
            if "bc_flow_optimizer_state_dict" in loaded_dict:
                self.bc_flow_optimizer.load_state_dict(loaded_dict["bc_flow_optimizer_state_dict"])
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> OneStepFlowPolicy:
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        # torch.compile is not applied here to keep the flow ODE integration graph-friendly.
        pass

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> FmpPPO:
        """Construct the FmpPPO algorithm."""
        # Resolve observation groups (actor for the policy, critic for the Q-network).
        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Register the flow condition key so the rollout storage allocates a buffer for it.
        num_envs = obs.batch_size[0]
        cond_dim_probe = 3 + 3 + 3 * env.num_actions  # gravity(3) + ang_vel(3) + 3 x joint dim
        obs = obs.clone()
        obs[FLOW_CONDITION_KEY] = torch.zeros(num_envs, cond_dim_probe, device=device)

        # Resolve the expert dataset from the motion loader of the discriminator_expert obs term.
        unwrapped = env.unwrapped
        motion_loader = None
        try:
            expert_group_cfg = unwrapped.cfg.observations["discriminator_expert"].terms
            for term_cfg in expert_group_cfg.values():
                loader = getattr(term_cfg.func, "motion_loader", None)
                if loader is not None:
                    motion_loader = loader
                    break
        except Exception as e:  # pragma: no cover
            print(f"[FmpPPO] Cannot resolve the motion loader from the environment: {e}")
        expert_dataset = None
        if motion_loader is not None and isinstance(motion_loader, MotionLoader):
            expert_dataset = FlowExpertDataset(
                motion_loader=motion_loader,
                env=unwrapped,
                num_samples=cfg.get("expert_dataset_size", 200_000),
                step_dt=unwrapped.step_dt,
                device=device,
            )
            # Override the probe dimension with the real condition dimension.
            obs[FLOW_CONDITION_KEY] = torch.zeros(num_envs, expert_dataset.condition_dim, device=device)
        else:
            print("[FmpPPO] Expert dataset unavailable: BC flow will not be pre-trained.")

        # Actor: one-step flow policy.
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        actor: OneStepFlowPolicy = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(
            device
        )  # type: ignore
        print(f"One-Step Flow Policy: {actor}")

        # Critic: Q(s, a).
        cfg["critic"].pop("class_name", None)
        critic: FlowCritic = FlowCritic(obs, cfg["obs_groups"], "critic", env.num_actions, **cfg["critic"]).to(device)
        print(f"Flow Critic: {critic}")

        # BC flow teacher.
        bc_flow_cfg = cfg.get("bc_flow", {})
        bc_flow: BcFlowNet = BcFlowNet(
            condition_dim=obs[FLOW_CONDITION_KEY].shape[-1],
            action_dim=env.num_actions,
            hidden_dims=bc_flow_cfg.get("hidden_dims", (512, 256, 256)),
            activation=bc_flow_cfg.get("activation", "elu"),
        ).to(device)
        print(f"BC Flow: {bc_flow}")

        # Storage (includes the flow condition buffer).
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Algorithm.
        cfg["algorithm"].pop("class_name", None)
        alg: FmpPPO = FmpPPO(
            actor,
            critic,
            bc_flow,
            expert_dataset,
            storage,
            device=device,
            multi_gpu_cfg=cfg.get("multi_gpu"),
            **cfg["algorithm"],
        )
        alg._env = unwrapped
        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    def broadcast_parameters(self) -> None:
        model_params = [
            self._raw_actor.state_dict(),
            self._raw_critic.state_dict(),
            self._raw_bc_flow.state_dict(),
        ]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self._raw_actor.load_state_dict(model_params[0])
        self._raw_critic.load_state_dict(model_params[1])
        self._raw_bc_flow.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        params = list(chain(self.actor.parameters(), self.critic.parameters(), self.bc_flow.parameters()))
        self._reduce_gradients(params)

    def _reduce_gradients(self, parameters) -> None:
        all_params = list(parameters)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
