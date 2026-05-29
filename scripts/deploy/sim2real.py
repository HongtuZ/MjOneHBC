import argparse

import numpy as np
from controller.base_controller import BaseRobotController
from ths_real_env import JoystickReader, ThsRealEnv


class RealController(BaseRobotController):
    """
    真机手柄 + 保留键盘接口.
    手柄在 _on_step_end 中轮询.
    键盘可通过外部线程调用 handle_keyboard 实现.
    """

    def __init__(
        self,
        model_path: str,
        obs_names=None,
        joystick_cpu_id: int = 1,
    ):
        self.joystick = JoystickReader(cpu_id=joystick_cpu_id)
        super().__init__(model_path, obs_names)

    def _create_env(self):
        return ThsRealEnv(
            physic_dt=0.005,
            decimation=4,
            action_joint_cfg=self.policy.action_joint_cfg,
        )

    def _reset_env(self) -> dict[str, np.ndarray | list | tuple]:
        return self.env.reset()

    def _on_step_end(self) -> None:
        """每轮轮询手柄."""
        self.handle_joystick(self.joystick.state)

    def close(self) -> None:
        self.joystick.close()
        super().close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    controller = RealController(model_path=args.model)
    controller.run()
