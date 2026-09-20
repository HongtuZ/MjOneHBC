import argparse
import time

import numpy as np
from controller.base_controller import BaseRobotController
from ths_ostrich_env import JoystickReader, ThsOstrichEnv


def _format(values) -> str:
    """把一组数值格式化成一行，便于在终端观察."""
    return "[" + ", ".join(f"{float(v):+.3f}" for v in values) + "]"


class ThsOstrichController(BaseRobotController):
    """
    真机手柄 + 保留键盘接口.
    手柄在 _on_step_end 中轮询.
    键盘可通过外部线程调用 handle_keyboard 实现.

    test_mode: 只读测试模式。不向电机发送任何指令，只读取电机 / 手柄 / IMU 数据，
               并打印策略输出目标位置与当前电机位置之差。
    """

    def __init__(
        self,
        model_path: str,
        obs_names=None,
        joystick_cpu_id: int = 1,
        imu_type: str = "yis320",
        test_mode: bool = False,
        print_interval: int = 10,
    ):
        self.test_mode = test_mode
        self.imu_type = imu_type
        self.print_interval = max(1, print_interval)
        self._step_count = 0
        self.joystick = JoystickReader(cpu_id=joystick_cpu_id)
        super().__init__(model_path, obs_names)

    def _create_env(self):
        return ThsOstrichEnv(
            physic_dt=0.005,
            decimation=4,
            action_joint_cfg=self.policy.action_joint_cfg,
            imu_type=self.imu_type,
            read_only=self.test_mode,
        )

    def _reset_env(self) -> dict[str, np.ndarray | list | tuple]:
        return self.env.reset()

    def _on_step_end(self) -> None:
        """每轮轮询手柄，并按 print_interval 打印当前策略的速度指令."""
        self.handle_joystick(self.joystick.state)

        self._step_count += 1
        if self._step_count % self.print_interval == 0:
            self._print_velocity_command()

    def _print_velocity_command(self) -> None:
        """打印当前 policy 的速度指令."""
        if self.is_motion_policy:
            print(f"[VEL-CMD] step={self._step_count} (motion policy) "
                  f"motion_step_t={self.policy.motion_step_t} running_mode={self.running_mode}")
        else:
            print(f"[VEL-CMD] step={self._step_count} vel_x={self.policy.vel_x:+.3f} "
                  f"vel_y={self.policy.vel_y:+.3f} ang_vel_z={self.policy.ang_vel_z:+.3f} "
                  f"running_mode={self.running_mode}")

    def close(self) -> None:
        self.joystick.close()
        super().close()

    # ---------------- 只读测试模式 ----------------
    def run(self) -> None:
        if self.test_mode:
            self._run_test()
            return
        super().run()

    def _run_test(self) -> None:
        """只读测试循环：不发送任何电机指令，只采集并打印数据."""
        print("[TEST] 只读测试模式：不会向电机发送任何指令")
        print("[TEST] Mode 键或 Ctrl+C 退出；Select 键重置机器人、Start 键重置策略（均不发送电机指令）")

        self._obs_info = self._reset_env()
        self.policy.reset(
            robot_pos=self._obs_info.get("robot_pos"),
            robot_quat=self._obs_info.get("robot_quat"),
        )

        interval = self.env.dt * self.env.decimation  # 与正常控制模式一致的策略周期
        try:
            while not self._should_exit():
                # 1. 更新高层命令（手柄输入，仅影响策略输入）
                self._obs_info.update(self.policy.get_command(self._obs_info.get("robot_quat")))

                # 2. 组装观测 & 策略推理
                self.obs_buffer.push(self._obs_info)
                action = self.policy.get_action(self.obs_buffer.get_obs())

                # 3. 只读一步：不发指令，只读电机 / IMU，并给出目标与实际位置的差值
                obs_info, extra = self.env.read_only_step(action)
                self._obs_info = obs_info

                # 4. 轮询手柄
                self._on_step_end()

                # 5. 周期性打印（_step_count 已在 _on_step_end 中累加）
                if self._step_count % self.print_interval == 0:
                    show_table = self._step_count % (self.print_interval * 2) == 0
                    self._print_test_status(extra, show_table)

                time.sleep(interval)
        except KeyboardInterrupt:
            print("用户中断 (Ctrl+C)")
        finally:
            self.close()

    def _print_test_status(self, extra: dict, show_joint_table: bool) -> None:
        """打印电机目标 / 实际 / 差值、IMU 与手柄数据."""
        names = self.env.action_joint_names
        target = extra["target_position"]
        position = extra["joint_position"]
        error = extra["position_error"]
        velocity = extra["joint_velocity"]
        abs_error = np.abs(error)
        interval = self.env.dt * self.env.decimation

        print(f"\n[TEST] step={self._step_count} t={self._step_count * interval:.2f}s "
              f"手柄={'已连接' if self.joystick.connected else '未连接'}")
        if show_joint_table:
            print(f"{'关节':<26}{'目标(rad)':>12}{'实际(rad)':>12}{'差值(rad)':>12}{'速度(rad/s)':>13}")
            for i, name in enumerate(names):
                print(f"{name:<26}{target[i]:>12.4f}{position[i]:>12.4f}{error[i]:>12.4f}{velocity[i]:>13.4f}")
        print(f"[TARGET-DIFF] max={abs_error.max():.4f} mean={abs_error.mean():.4f} "
              f"最大关节={names[int(abs_error.argmax())]}")

        print(f"[IMU] quat(wxyz)={_format(self.env.imu.quaternion)} "
              f"gyro(rad/s)={_format(self.env.imu.gyro)} "
              f"euler(rad)={_format(self.env.imu.euler)}")

        joy = self.joystick.state
        print(f"[JOY] left={_format(joy.left_stick)} right={_format(joy.right_stick)} hat={joy.hats} "
              f"cross={int(joy.button_cross)} triangle={int(joy.button_triangle)} "
              f"select={int(joy.button_select)} start={int(joy.button_start)} mode={int(joy.button_mode)}")

        if self.is_motion_policy:
            print(f"[CMD] motion_step_t={self.policy.motion_step_t} running_mode={self.running_mode}")
        else:
            print(f"[CMD] vel_x={self.policy.vel_x:+.2f} vel_y={self.policy.vel_y:+.2f} "
                  f"ang_vel_z={self.policy.ang_vel_z:+.2f} running_mode={self.running_mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--imu", type=str, default="yis320", help="IMU 型号: wit / yuansheng / yis320")
    parser.add_argument(
        "--test",
        action="store_true",
        help="只读测试模式：不向电机发送任何指令，只读取电机/手柄/IMU 并打印策略目标与电机位置差值",
    )
    parser.add_argument("--print-interval", type=int, default=10, help="测试模式打印间隔（策略步数，10 步 ≈ 5Hz）")
    args = parser.parse_args()

    controller = ThsOstrichController(
        model_path=args.model,
        imu_type=args.imu,
        test_mode=args.test,
        print_interval=args.print_interval,
    )
    controller.run()