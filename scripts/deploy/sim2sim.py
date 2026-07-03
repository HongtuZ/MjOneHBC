import argparse
from pathlib import Path

import numpy as np
from controller.base_controller import BaseRobotController
from mujoco_env.mj_env import MujocoEnv

ROOT_DIR = Path(__file__).resolve().parents[2]

class SimController(BaseRobotController):
    """
    MuJoCo 键盘 + 可选手柄.
    手柄在 _on_step_end 中轮询.
    """

    def __init__(self, xml_path: str, model_path: str, obs_names=None, use_joystick=False):
        self.xml_path = xml_path
        self._init_pos, self._init_quat = self._resolve_init_pose(model_path)

        # 可选：同时接入手柄
        self.joystick = None
        if use_joystick:
            from ths_real_env import JoystickReader

            self.joystick = JoystickReader(cpu_id=1)

        super().__init__(model_path, obs_names)

    def _resolve_init_pose(self, model_path: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if "getup" in model_path:
            return (0.0, 0.0, 0.15), (0.70710678, 0.0, 0.70710678, 0.0)
        return (0.0, 0.0, 0.78), (1.0, 0.0, 0.0, 0.0)

    def _create_env(self):
        return MujocoEnv(
            xml_path=self.xml_path,
            sim_dt=0.005,
            decimation=4,
            action_joint_cfg=self.policy.action_joint_cfg,
            keyboard_callback=self.handle_keyboard,
        )

    def _reset_env(self) -> dict[str, np.ndarray | list | tuple]:
        return self.env.reset(root_pos=self._init_pos, root_quat=self._init_quat)

    def _on_step_start(self) -> None:
        """在 env.step 之前更新可视化."""
        self.env.show_command(
            velocity_command=self._obs_info.get("velocity_command"),
            ref_motion=self._obs_info.get("ref_motion"),
        )

    def _on_step_end(self) -> None:
        """轮询手柄（如果已连接）."""
        if self.joystick is not None:
            self.handle_joystick(self.joystick.state)

    def close(self) -> None:
        if self.joystick is not None:
            self.joystick.close()
        super().close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--xml", type=str, default=str(ROOT_DIR / "robot_assets/ths_23dof/urdf/ths_23dof.xml"))
    args = parser.parse_args()

    controller = SimController(xml_path=args.xml, model_path=args.model, use_joystick=False)
    controller.run()
