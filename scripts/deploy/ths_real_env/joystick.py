import contextlib
import multiprocessing as mp
import os
import time
from dataclasses import dataclass

import pygame


@dataclass
class JoystickState:
    """摇杆当前完整状态"""

    left_stick: tuple[float, float]
    right_stick: tuple[float, float]
    hats: tuple[int, int]
    button_cross: bool
    button_circle: bool
    button_square: bool
    button_triangle: bool
    button_lb: bool
    button_rb: bool
    button_select: bool
    button_start: bool
    button_mode: bool
    timestamp: float  # 最后更新时间


class JoystickReader:
    """
    纯状态采集器：只读取并保存摇杆/按键/轴的原始状态。
    不做任何速度计算、模式切换等业务逻辑。
    """

    # 支持的最大按键/轴数量
    MAX_BUTTONS = 32
    MAX_AXES = 8
    MAX_HATS = 4

    def __init__(
        self,
        cpu_id: int | None = None,
        deadzone: float = 0.05,  # 摇杆死区
        poll_interval: float = 0.001,  # 轮询间隔（秒）
    ):
        self.cpu_id = cpu_id
        self.deadzone = deadzone
        self.poll_interval = poll_interval

        # 共享内存：原始状态
        self._axes = mp.Array("d", self.MAX_AXES)  # 8 个轴
        self._buttons = mp.Array("b", self.MAX_BUTTONS)  # 32 个按键
        self._hats_x = mp.Array("i", self.MAX_HATS)  # 4 个方向键 X
        self._hats_y = mp.Array("i", self.MAX_HATS)  # 4 个方向键 Y
        self._timestamp = mp.Value("d", 0.0)
        self._connected = mp.Value("b", False)  # 是否连接
        self._num_axes = mp.Value("i", 0)  # 实际轴数
        self._num_buttons = mp.Value("i", 0)  # 实际按键数
        self._num_hats = mp.Value("i", 0)  # 实际方向键数

        self._running = mp.Value("b", False)
        self._process: mp.Process | None = None

        self.start()
        time.sleep(0.1)

    # ------------------------------------------------------------------
    # 进程控制
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._running.value = True
        self._process = mp.Process(
            target=self._read_loop,
            args=(
                self._axes,
                self._buttons,
                self._hats_x,
                self._hats_y,
                self._timestamp,
                self._connected,
                self._num_axes,
                self._num_buttons,
                self._num_hats,
                self._running,
                self.cpu_id,
                self.deadzone,
                self.poll_interval,
            ),
            daemon=True,
        )
        self._process.start()

    def close(self) -> None:
        self._running.value = False
        if self._process is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
            self._process = None

    # ------------------------------------------------------------------
    # 外部访问接口：原始状态
    # ------------------------------------------------------------------
    @property
    def state(self) -> JoystickState:
        """获取当前完整状态（拷贝）"""
        with self._axes.get_lock(), self._buttons.get_lock(), self._hats_x.get_lock(), self._hats_y.get_lock():
            return JoystickState(
                left_stick=self.left_stick,
                right_stick=self.right_stick,
                hats=self.hats[0],
                button_cross=self.button_cross(),
                button_circle=self.button_circle(),
                button_square=self.button_square(),
                button_triangle=self.button_triangle(),
                button_lb=self.button_lb(),
                button_rb=self.button_rb(),
                button_select=self.button_select(),
                button_start=self.button_start(),
                button_mode=self.button_mode(),
                timestamp=self._timestamp.value,
            )

    @property
    def axes(self) -> list[float]:
        """所有轴的当前值（已应用死区）"""
        with self._axes.get_lock():
            return list(self._axes[: self._num_axes.value])

    @property
    def buttons(self) -> list[bool]:
        """所有按键的当前状态"""
        with self._buttons.get_lock():
            return list(self._buttons[: self._num_buttons.value])

    @property
    def hats(self) -> list[tuple]:
        """所有方向键的当前状态"""
        with self._hats_x.get_lock(), self._hats_y.get_lock():
            return [(self._hats_x[i], self._hats_y[i]) for i in range(self._num_hats.value)]

    @property
    def connected(self) -> bool:
        """摇杆是否已连接"""
        return bool(self._connected.value)

    @property
    def last_update(self) -> float:
        """最后更新时间戳"""
        return self._timestamp.value

    def is_fresh(self, max_age: float = 0.5) -> bool:
        """数据是否在指定时间内更新"""
        return (time.time() - self._timestamp.value) < max_age

    # ------------------------------------------------------------------
    # 便捷访问：常用按键/轴（按 Xbox 手柄布局）
    # ------------------------------------------------------------------
    @property
    def left_stick(self) -> tuple:
        """左摇杆 (x, y)"""
        a = self.axes
        return (a[0], a[1]) if len(a) >= 2 else (0.0, 0.0)

    @property
    def right_stick(self) -> tuple:
        """右摇杆 (x, y)"""
        a = self.axes
        return (a[3], a[4]) if len(a) >= 5 else (0.0, 0.0)

    @property
    def left_trigger(self) -> float:
        """左扳机 (LT)"""
        a = self.axes
        return a[2] if len(a) >= 3 else 0.0

    @property
    def right_trigger(self) -> float:
        """右扳机 (RT)"""
        a = self.axes
        return a[5] if len(a) >= 6 else 0.0

    # 索尼按键映射（常见布局）
    def button_cross(self) -> bool:
        return self._button_safe(0)

    def button_circle(self) -> bool:
        return self._button_safe(1)

    def button_square(self) -> bool:
        return self._button_safe(2)

    def button_triangle(self) -> bool:
        return self._button_safe(3)

    def button_lb(self) -> bool:
        return self._button_safe(4)

    def button_rb(self) -> bool:
        return self._button_safe(5)

    def button_select(self) -> bool:
        return self._button_safe(6)

    def button_start(self) -> bool:
        return self._button_safe(7)

    def button_mode(self) -> bool:
        return self._button_safe(8)  # 左摇杆按下

    def button_lstick(self) -> bool:
        return self._button_safe(9)  # 左摇杆按下

    def button_rstick(self) -> bool:
        return self._button_safe(10)  # 右摇杆按下

    def _button_safe(self, idx: int) -> bool:
        b = self.buttons
        return b[idx] if idx < len(b) else False

    # ------------------------------------------------------------------
    # 子进程循环：只读不写逻辑
    # ------------------------------------------------------------------
    @staticmethod
    def _read_loop(
        axes,
        buttons,
        hats_x,
        hats_y,
        timestamp,
        connected,
        num_axes,
        num_buttons,
        num_hats,
        running,
        cpu_id,
        deadzone,
        poll_interval,
    ):
        # CPU 绑定
        if cpu_id is not None:
            with contextlib.suppress(AttributeError):
                os.sched_setaffinity(0, {cpu_id})

        # 无显示器兼容
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.joystick.init()

        # 等待手柄连接
        joystick = None
        while running.value:
            count = pygame.joystick.get_count()
            if count > 0:
                joystick = pygame.joystick.Joystick(0)
                joystick.init()
                connected.value = True
                print(f"[Joystick] 已连接: {joystick.get_name()}")
                print(
                    f"[Joystick] 轴: {joystick.get_numaxes()}, 按键: {joystick.get_numbuttons()}, 方向键: {joystick.get_numhats()}"
                )
                break
            time.sleep(0.5)

        if joystick is None:
            print("[Joystick] 未检测到手柄，退出")
            connected.value = False
            pygame.quit()
            return

        # 记录实际数量
        num_axes.value = min(joystick.get_numaxes(), 8)
        num_buttons.value = min(joystick.get_numbuttons(), 32)
        num_hats.value = min(joystick.get_numhats(), 4)

        # 主循环：只读取状态，不处理事件
        while running.value:
            # 刷新 pygame 事件队列（必须调用，否则手柄数据不更新）
            pygame.event.pump()

            # 读取所有轴
            with axes.get_lock():
                for i in range(num_axes.value):
                    val = joystick.get_axis(i)
                    # 应用死区
                    if abs(val) < deadzone:
                        val = 0.0
                    axes[i] = val

            # 读取所有按键
            with buttons.get_lock():
                for i in range(num_buttons.value):
                    buttons[i] = joystick.get_button(i)

            # 读取所有方向键
            with hats_x.get_lock(), hats_y.get_lock():
                for i in range(num_hats.value):
                    hx, hy = joystick.get_hat(i)
                    hats_x[i] = hx
                    hats_y[i] = hy

            # 更新时间戳
            timestamp.value = time.time()

            time.sleep(poll_interval)

        connected.value = False
        pygame.quit()
        print("[Joystick] 读取进程已退出")


if __name__ == "__main__":
    joystick = JoystickReader(cpu_id=1)
    for i in range(10000):
        print(joystick.state)
        time.sleep(0.01)
    joystick.close()
