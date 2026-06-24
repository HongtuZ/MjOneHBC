import os
from pathlib import Path

import torch
from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.logger import Logger

from .amp_ppo import AmpPPO


class AmpRunner(OnPolicyRunner):
    """Base runner that persists environment state across checkpoints."""

    alg: AmpPPO
    env: RslRlVecEnvWrapper

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        """Construct the runner, algorithm, and logging stack."""
        # Strip None-valued optional configs so MLPModel doesn't receive them.
        for key in ("actor", "critic", "discriminator"):
            if key in train_cfg:
                for opt in ("cnn_cfg", "distribution_cfg"):
                    if train_cfg[key].get(opt) is None:
                        train_cfg[key].pop(opt, None)
                if train_cfg[key].get("rnn_type") is None:
                    for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                        train_cfg[key].pop(opt, None)
        self.env = env
        self.cfg = train_cfg
        self.device = device

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from the environment for algorithm construction
        obs = self.env.get_observations()

        # Create the algorithm
        self.alg = AmpPPO.construct_algorithm(obs, self.env, self.cfg, self.device)

        # Create the logger
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export policy to ONNX format using legacy export path.

        Overrides the base implementation to set dynamo=False, avoiding warnings about
        dynamic_axes being deprecated with the new TorchDynamo export path
        (torch>=2.9 default).
        """
        onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()
        os.makedirs(path, exist_ok=True)
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),  # type: ignore[operator]
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=onnx_model.input_names,  # type: ignore[arg-type]
            output_names=onnx_model.output_names,  # type: ignore[arg-type]
            dynamic_axes={},
            dynamo=False,
        )

    @staticmethod
    def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
        """Resolve ONNX export paths from a checkpoint path."""
        export_dir = Path(checkpoint_path).parent
        filename = f"{export_dir.name}.onnx"
        return export_dir, filename, export_dir / filename

    def save(self, path: str, infos=None) -> None:
        """Save checkpoint.

        Extends the base implementation to persist the environment's
        common_step_counter and to respect the ``upload_model`` config flag.
        """
        env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
        infos = {**(infos or {}), "env_state": env_state}
        # Inline base OnPolicyRunner.save() to conditionally gate W&B upload.
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        if self.cfg["upload_model"]:
            self.logger.save_model(path, self.current_learning_iteration)
        # Export ONNX
        policy_dir, filename, onnx_path = self._get_export_paths(path)
        try:
            self.export_policy_to_onnx(str(policy_dir), filename)
            metadata = get_base_metadata(self.env.unwrapped, "local")
            attach_metadata_to_onnx(str(onnx_path), metadata)
        except Exception as e:
            print(f"[WARN] ONNX export failed (training continues): {e}")

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        infos = super().load(path, load_cfg, strict, map_location)
        if infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
        return infos
