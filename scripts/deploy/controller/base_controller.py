from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .helper import ObsBuffer, OnnxPolicy


# ==================== 常量 ====================
class KeyCode:
    ESC = 256
    SPACE = 32
    UP = 265
    DOWN = 264
    LEFT = 263
    RIGHT = 262


VEL_OBS_NAMES = [
    "base_ang_vel",
    "projected_gravity",
    "velocity_command",
    "joint_pos",
    "joint_vel",
    "last_action",
]

MOTION_OBS_NAMES = [
    "motion_command",
    "motion_anchor_ori_b",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "last_action",
]


# ==================== 工具 ====================
@dataclass
class ButtonState:
    select: bool = False
    start: bool = False
    cross: bool = False
    triangle: bool = False
    mode: bool = False


def stick2cmd(x: float, cmd_min: float, cmd_max: float) -> float:
    """将 [-1, 1] 摇杆输入映射到 [cmd_min, cmd_max]，cmd_min 应为负值."""
    return x * (cmd_max if x > 0 else -cmd_min)


# ==================== 抽象基类 ====================
class BaseRobotController(ABC):
    """
    模板方法模式.
    子类只需实现: _create_env, _reset_env.
    输入处理已全部在基类完成.
    """

    def __init__(
        self,
        model_path: str,
        obs_names: list[str] | None = None,
        obs_history_length=1,
        is_real_env=False,
    ):
        # 1. 策略
        self.policy = OnnxPolicy(
            onnx_policy_path=model_path,
            is_real_env=is_real_env,
        )
        self.is_motion_policy: bool = self.policy.command_name == "motion"

        # 2. 观测名（可自定义，默认自动推断）
        if obs_names is None:
            obs_names = MOTION_OBS_NAMES if self.is_motion_policy else VEL_OBS_NAMES
        self.obs_buffer = ObsBuffer(
            obs_names=obs_names,
            history_length=obs_history_length,
            concatenate=True,
        )

        # 3. 环境（子类创建）
        self.env = self._create_env()

        # 4. 内部状态
        self._obs_info: dict[str, np.ndarray] = {}
        self._done = False
        self._exit_flag = False
        self.running_mode = False
        self._prev_joy_btn = ButtonState()

    # ---------------- 子类必须实现 ----------------
    @abstractmethod
    def _create_env(self):
        """创建并返回环境实例."""
        ...

    @abstractmethod
    def _reset_env(self) -> dict[str, np.ndarray | list | tuple]:
        """重置环境，返回初始 obs_info."""
        ...

    # ---------------- 可选钩子 ----------------
    def _on_step_start(self) -> None:
        """env.step() 之前调用."""

    def _on_step_end(self) -> None:
        """env.step() 之后调用（子类应在此轮询手柄）."""

    def close(self) -> None:
        self.env.close()

    # ---------------- 统一输入处理 ----------------
    def handle_keyboard(self, keycode: int) -> None:
        """键盘事件入口. Sim 由 GLFW 回调触发，Real 也可手动调用."""
        if keycode == KeyCode.ESC:
            self._exit_flag = True
            return

        if self.is_motion_policy:
            if keycode == KeyCode.SPACE:
                self.policy.motion_step_t = 0
        else:
            if keycode == KeyCode.SPACE:
                self.policy.vel_x = 0.0
                self.policy.vel_y = 0.0
                self.policy.ang_vel_z = 0.0
            elif keycode == KeyCode.UP:
                self.policy.vel_x += 0.1
            elif keycode == KeyCode.DOWN:
                self.policy.vel_x -= 0.1
            elif keycode == KeyCode.LEFT:
                self.policy.ang_vel_z += 0.1
            elif keycode == KeyCode.RIGHT:
                self.policy.ang_vel_z -= 0.1

    def handle_joystick(self, joy) -> None:
        """手柄轮询入口. 包含上升沿检测与连续命令映射."""
        # --- 上升沿事件 ---
        if joy.button_select and not self._prev_joy_btn.select:
            self._on_joystick_select()
        if joy.button_start and not self._prev_joy_btn.start:
            self._on_joystick_start()
        if joy.button_cross and not self._prev_joy_btn.cross:
            self._on_joystick_cross()
        if joy.button_triangle and not self._prev_joy_btn.triangle:
            self._on_joystick_triangle()
        if joy.button_mode and not self._prev_joy_btn.mode:
            self._on_joystick_mode()

        # 同步上一帧状态
        self._prev_joy_btn = ButtonState(
            select=joy.button_select,
            start=joy.button_start,
            cross=joy.button_cross,
            triangle=joy.button_triangle,
            mode=joy.button_mode,
        )

        # --- 连续命令更新 ---
        if not self.running_mode:
            if self.is_motion_policy:
                self.policy.motion_step_t = 0
            else:
                self.policy.vel_x = 0.0
                self.policy.vel_y = 0.0
                self.policy.ang_vel_z = 0.0
            return

        if not self.is_motion_policy:
            self.policy.vel_x = stick2cmd(-joy.left_stick[1], -0.5, 2.5)
            self.policy.vel_y = stick2cmd(-joy.left_stick[0], -1.0, 1.0)
            self.policy.ang_vel_z = stick2cmd(-joy.right_stick[0], -1.0, 1.0)

    # --- 手柄事件钩子（子类可覆盖） ---
    def _on_joystick_select(self):
        print("[BTN] Select -> 重置机器人")
        self._obs_info = self._reset_env()
        self.policy.reset(
            robot_pos=self._obs_info.get("robot_pos"),
            robot_quat=self._obs_info.get("robot_quat"),
        )
        self.obs_buffer.reset()
        self.running_mode = False

    def _on_joystick_start(self):
        print("[BTN] Start -> 重置策略")
        if self.env is not None and hasattr(self.env, "attribute_exists"):
            self.env.enable_control = True
        self.policy.reset(
            robot_pos=self._obs_info.get("robot_pos"),
            robot_quat=self._obs_info.get("robot_quat"),
        )
        self.obs_buffer.reset()
        self.running_mode = False

    def _on_joystick_cross(self):
        print("[BTN] Cross -> 停止指令响应")
        self.running_mode = False

    def _on_joystick_triangle(self):
        print("[BTN] Triangle -> 启动指令响应")
        self.running_mode = True

    def _on_joystick_mode(self):
        print("[BTN] Mode -> 退出程序")
        self._exit_flag = True

    # ---------------- 核心模板方法 ----------------
    def run(self) -> None:
        self._obs_info = self._reset_env()
        self.policy.reset(
            robot_pos=self._obs_info.get("robot_pos"),
            robot_quat=self._obs_info.get("robot_quat"),
        )

        try:
            while not self._should_exit():
                # 1. 更新高层命令
                self._obs_info.update(self.policy.get_command(self._obs_info.get("robot_quat")))

                # 2. 组装观测 & 策略推理
                self.obs_buffer.push(self._obs_info)
                obs = self.obs_buffer.get_obs()
                action = self.policy.get_action(obs)

                # 3. 步进前钩子
                self._on_step_start()

                # 4. 环境步进（内部已控频）
                self._obs_info, self._done = self.env.step(action)

                # 5. 步进后钩子（轮询输入设备）
                self._on_step_end()

                if self._done:
                    break

        except KeyboardInterrupt:
            print("用户中断 (Ctrl+C)")
        finally:
            self.close()

    def _should_exit(self) -> bool:
        return self._exit_flag

    def __del__(self):
        self.close()
