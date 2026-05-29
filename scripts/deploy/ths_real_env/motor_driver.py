import atexit
import multiprocessing as mp
import struct
import time
from multiprocessing import shared_memory

import can
import numpy as np

from .motor_config import THS_MOTORS, MotorConfig

# ═══════════════════════════════════════════════════
# 协议转换函数
# ═══════════════════════════════════════════════════


def float_to_uint16(x, x_min, x_max):
    """float 转 0~65535"""
    if x > x_max:
        x = x_max
    elif x < x_min:
        x = x_min
    return int((x - x_min) / (x_max - x_min) * 65535)


def uint16_to_float(u, u_min, u_max):
    """0~65535 转 float（注意：按原始公式，u_min 仅用于算跨度）"""
    return float((u - 32767) / 65535) * (u_max - u_min)


def float_to_P4hex(x):
    """float 转 4 字节（小端）"""
    return struct.pack("<f", x)


def P4hex_to_float(x):
    """4 字节 hex 转 float（大端）"""
    return struct.unpack("f", x.to_bytes(4, "big"))[0]


def slow_can_io(bus, motor_id, arbitration_id, data):
    """
    发送后阻塞等待回复，最多尝试 2 次
    """
    for _ in range(2):
        bus.send(can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=True))
        rx = bus.recv(timeout=0.01)
        if rx is not None:
            rx_id = (rx.arbitration_id >> 8) & 0xFF
            rx_func = rx.arbitration_id >> 24
            if rx_id != 0 and rx_func != 0 and rx_id == motor_id:
                return 0, list(rx.data)
    return 1, [0] * 8


def set_motion_mode(bus, motor_id):
    """功能码 0x12，主 ID 0xfd"""
    arb = 0x1200FD00 | motor_id
    state, _ = slow_can_io(bus, motor_id, arb, [0] * 8)
    time.sleep(0.0005)
    return state


def set_motion_enable(bus, motor_id):
    """功能码 0x03，主 ID 0xfd"""
    arb = 0x0300FD00 | motor_id
    state, _ = slow_can_io(bus, motor_id, arb, [0] * 8)
    time.sleep(0.0005)
    return state


def read_motor_single_data(bus, motor_id, address):
    """功能码 0x11，主 ID 0xfd"""
    arb = 0x1100FD00 | motor_id
    data = [0] * 8
    data[0] = address & 0x00FF
    data[1] = address >> 8
    state, rx = slow_can_io(bus, motor_id, arb, data)
    if state == 0:
        read_data = rx[4] << 24 | rx[5] << 16 | rx[6] << 8 | rx[7]
        return 0, P4hex_to_float(read_data)
    return 1, 0.0


def send_one(bus, cfg, pos, vel, kp, kd, tau_ff):
    """发送单个电机指令"""
    tau_u = float_to_uint16(tau_ff, cfg.t_min, cfg.t_max)
    arb = 0x01000000 | (tau_u << 8) | cfg.motor_id

    buf = bytearray(8)
    pos_u = float_to_uint16(pos, cfg.p_min, cfg.p_max)
    buf[0] = pos_u >> 8
    buf[1] = pos_u & 0xFF

    vel_u = float_to_uint16(vel, cfg.v_min, cfg.v_max)
    buf[2] = vel_u >> 8
    buf[3] = vel_u & 0xFF

    kp_u = float_to_uint16(kp, cfg.kp_min, cfg.kp_max)
    buf[4] = kp_u >> 8
    buf[5] = kp_u & 0xFF

    kd_u = float_to_uint16(kd, cfg.kd_min, cfg.kd_max)
    buf[6] = kd_u >> 8
    buf[7] = kd_u & 0xFF

    bus.send(can.Message(arbitration_id=arb, data=bytes(buf), is_extended_id=True))
    time.sleep(0.0005)  # 2000 HZ


# ═══════════════════════════════════════════════════
# 驱动类
# ═══════════════════════════════════════════════════


class MotorDriver:
    """
    多路 CAN 电机驱动， 每路 CAN 一个进程，子进程内完成初始化，共享内存交换数据
    """

    # 共享内存行
    ROW_QPOS = 0
    ROW_QVEL = 1
    ROW_TAU = 2
    ROW_TEMP = 3
    ROW_CTRL_POS = 4
    ROW_CTRL_VEL = 5
    ROW_CTRL_KP = 6
    ROW_CTRL_KD = 7
    ROW_CTRL_TAU = 8
    NUM_ROWS = 9

    DTYPE = np.float32

    def __init__(self, motor_configs: list[MotorConfig], joint_order: list[str] | None = None):
        self.num_motors = len(motor_configs)
        motor_names = [cfg.joint_name for cfg in motor_configs]

        # 外部控制顺序
        self.joint_order = joint_order or motor_names.copy()
        self.num_joints = len(self.joint_order)

        # 映射：外部索引 <-> 电机索引
        self.joint_to_motor = np.array([motor_names.index(n) for n in self.joint_order], dtype=int)
        self.motor_to_joint = np.full(self.num_motors, -1, dtype=int)
        for j, m in enumerate(self.joint_to_motor):
            self.motor_to_joint[m] = j

        # 按 bus_name 分组
        bus_groups = {}
        for i, cfg in enumerate(motor_configs):
            bus_groups.setdefault(cfg.bus_name, []).append(i)

        # 共享内存
        total_bytes = self.NUM_ROWS * self.num_motors * np.dtype(self.DTYPE).itemsize
        self.shared_mem = shared_memory.SharedMemory(create=True, size=total_bytes)
        self.state = np.ndarray((self.NUM_ROWS, self.num_motors), dtype=self.DTYPE, buffer=self.shared_mem.buf)
        self.state[:] = 0.0

        # 启动各 CAN 进程
        self.running = mp.Value("b", True)
        self.processes = []

        for bus_name, motor_indices in bus_groups.items():
            configs = [motor_configs[i] for i in motor_indices]
            proc = mp.Process(
                target=self._can_worker,
                args=(bus_name, motor_indices, configs, self.shared_mem.name, self.state.shape, self.running),
            )
            proc.start()
            self.processes.append(proc)

        atexit.register(self.close)
        time.sleep(0.3)  # 等待子进程完成初始化

    # ── 读取接口（按外部 joint_order 排序）──

    @property
    def qpos(self) -> np.ndarray:
        return self.state[self.ROW_QPOS].copy()[self.motor_to_joint]

    @property
    def qvel(self) -> np.ndarray:
        return self.state[self.ROW_QVEL].copy()[self.motor_to_joint]

    @property
    def tau(self) -> np.ndarray:
        return self.state[self.ROW_TAU].copy()[self.motor_to_joint]

    @property
    def temp(self) -> np.ndarray:
        return self.state[self.ROW_TEMP].copy()[self.motor_to_joint]

    # ── 控制接口（外部顺序写入，内部自动转换）──

    def set_ctrl(self, pos=None, vel=None, kp=None, kd=None, tau=None):
        """
        设置控制量，支持按名字或按数组赋值
        未传入的参数保持前值
        """
        if pos is not None:
            self._write_row(self.ROW_CTRL_POS, pos)
        if vel is not None:
            self._write_row(self.ROW_CTRL_VEL, vel)
        if kp is not None:
            self._write_row(self.ROW_CTRL_KP, kp)
        if kd is not None:
            self._write_row(self.ROW_CTRL_KD, kd)
        if tau is not None:
            self._write_row(self.ROW_CTRL_TAU, tau)

    def _write_row(self, row: int, value):
        """把外部顺序的值写入共享内存的电机顺序"""
        value = np.asarray(value, dtype=self.DTYPE)
        if value.shape == () or value.shape == (self.num_joints,):
            self.state[row, self.joint_to_motor] = value
        else:
            raise ValueError(f"Shape must be scalar or ({self.num_joints},), got {value.shape}")

    def close(self):
        if not self.running.value:
            return
        self.running.value = False
        for p in self.processes:
            p.join(timeout=3.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
        self.shared_mem.close()
        self.shared_mem.unlink()
        atexit.unregister(self.close)
        print(f"MotorDriver closed ({self.num_joints} joints)")

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False

    # ═══════════════════════════════════════════════════
    # CAN 工作进程（每路 bus 一个）
    # ═══════════════════════════════════════════════════

    @staticmethod
    def _can_worker(bus_name, motor_indices, motor_configs, shm_name, state_shape, running):
        # 连接共享内存
        shm = shared_memory.SharedMemory(name=shm_name)
        state = np.ndarray(state_shape, dtype=MotorDriver.DTYPE, buffer=shm.buf)

        n_local = len(motor_indices)
        motor_ids = [cfg.motor_id for cfg in motor_configs]

        # 创建 CAN 总线
        bus = can.interface.Bus(channel=bus_name, interface="socketcan", bitrate=1000000)

        # ═══════ 初始化：设模式 + 使能 + 读当前角度 ═══════
        init_pos = np.zeros(n_local, dtype=np.float64)
        default_pos = np.array([cfg.default_pos for cfg in motor_configs], dtype=np.float64)

        for local_idx, cfg in enumerate(motor_configs):
            mid = cfg.motor_id
            if set_motion_mode(bus, mid) != 0 or set_motion_enable(bus, mid) != 0:
                print(f"[{bus_name}] 电机 {cfg.joint_name} 初始化失败")
                continue
            err, pos = read_motor_single_data(bus, mid, 0x7019)
            if err == 0 and abs(pos) < 1.8:
                init_pos[local_idx] = pos
                print(f"[{bus_name}] {cfg.joint_name} 初始角度: {pos:.4f} rad")
            else:
                init_pos[local_idx] = cfg.default_pos  # 读不到就用默认值
                print(
                    f"[{bus_name}] {cfg.joint_name} 读角度异常 err: {err} 角度: {pos:.4f}，使用默认值: {cfg.default_pos:.4f}"
                )

        # ═══════ 过渡阶段：按最大速度平滑渐变到 default_pos ═══════
        # 预定义最大角速度 (rad/s)，根据你的电机性能调整
        MAX_TRANSITION_VELOCITY = 1.0  # 例如 1 rad/s ≈ 57 deg/s

        # 计算每个电机的角度差异
        angle_diffs = np.abs(default_pos - init_pos)
        max_diff = np.max(angle_diffs)

        # 计算所需过渡时间（最大差异 / 最大速度）
        if max_diff < 0.001:  # 已经到位，无需过渡
            transition_time = 0.0
            transition_steps = 1
        else:
            transition_time = max_diff / MAX_TRANSITION_VELOCITY
            # 至少 50ms，最多 3s
            transition_time = np.clip(transition_time, 0.05, 3.0)
            # 按 2kHz 控制频率计算步数
            transition_steps = max(int(transition_time * 2000.0 / n_local), 1)

        print(
            f"[{bus_name}] 最大角度差异: {max_diff:.4f} rad, 过渡时间: {transition_time:.3f}s, 步数: {transition_steps}"
        )
        for step in range(transition_steps):
            alpha = step / transition_steps
            target = init_pos * (1.0 - alpha) + default_pos * alpha

            for local_idx, cfg in enumerate(motor_configs):
                try:
                    send_one(bus, cfg, target[local_idx], 0.0, cfg.default_kp, cfg.default_kd, 0.0)
                except Exception as e:
                    print(f"Error send_one() to {cfg.joint_name}: {e}")
        print(f"[{bus_name}] 过渡完成，已进入 default_pos")

        # 把最终状态写入共享内存
        state[MotorDriver.ROW_QPOS, motor_indices] = default_pos.astype(MotorDriver.DTYPE)
        state[MotorDriver.ROW_CTRL_POS, motor_indices] = default_pos.astype(MotorDriver.DTYPE)

        # 设置默认 KP/KD
        for local_idx, cfg in enumerate(motor_configs):
            state[MotorDriver.ROW_CTRL_KP, motor_indices[local_idx]] = cfg.default_kp
            state[MotorDriver.ROW_CTRL_KD, motor_indices[local_idx]] = cfg.default_kd

        # ═══════ 高速运行阶段 ═══════
        id_to_local = {mid: i for i, mid in enumerate(motor_ids)}

        local_qpos = default_pos.copy()
        local_qvel = np.zeros(n_local, dtype=np.float64)
        local_tau = np.zeros(n_local, dtype=np.float64)

        # 心跳计时
        alive_check_time_start = time.perf_counter()
        no_response_max_s = 1.0
        while running.value:
            # ── 接收阶段 ──
            received = False
            while True:
                msg = bus.recv(timeout=0)
                if msg is None:
                    break

                # 过滤：功能码必须为 0x02（运控反馈帧）
                if msg.arbitration_id >> 24 != 0x02:
                    continue

                rx_motor_id = (msg.arbitration_id >> 8) & 0xFF
                if rx_motor_id not in id_to_local:
                    continue

                local_idx = id_to_local[rx_motor_id]
                cfg = motor_configs[local_idx]
                data = msg.data

                # 解析反馈（忠实复刻原始代码）
                pos_raw = (data[0] << 8) | data[1]
                vel_raw = (data[2] << 8) | data[3]
                tau_raw = (data[4] << 8) | data[5]

                local_qpos[local_idx] = uint16_to_float(pos_raw, cfg.p_min, cfg.p_max)
                local_qvel[local_idx] = uint16_to_float(vel_raw, cfg.v_min, cfg.v_max)
                local_tau[local_idx] = uint16_to_float(tau_raw, cfg.t_min, cfg.t_max)

                # 温度监控
                temperature = ((data[6] << 8) | data[7]) / 10.0
                if temperature > 80.0:
                    print(f"[{bus_name}] 电机 {cfg.joint_name} 过热: {temperature:.1f}°C")

                # 角度超限检查（对应原始 cof.joint_pmin/pmax 检查）
                if not (cfg.joint_pmin <= local_qpos[local_idx] <= cfg.joint_pmax):
                    print(f"[{bus_name}] 电机 {cfg.joint_name} 角度超限: {local_qpos[local_idx]:.4f}")

                received = True
                alive_check_time_start = time.perf_counter()
                no_response_max_s = 1.0

            if not received:
                no_response_duration = time.perf_counter() - alive_check_time_start
                if no_response_duration > no_response_max_s:
                    print(f"[{bus_name}] 通信异常：超过1s未收到反馈")
                    no_response_max_s += 1.0

            # ── 发送阶段 ──
            for local_idx, global_idx in enumerate(motor_indices):
                cfg = motor_configs[local_idx]
                # 从共享内存读取目标值
                target_pos = state[MotorDriver.ROW_CTRL_POS, global_idx]
                target_vel = state[MotorDriver.ROW_CTRL_VEL, global_idx]
                target_kp = state[MotorDriver.ROW_CTRL_KP, global_idx]
                target_kd = state[MotorDriver.ROW_CTRL_KD, global_idx]
                target_tau = state[MotorDriver.ROW_CTRL_TAU, global_idx]
                # 发送指令
                send_one(bus, cfg, target_pos, target_vel, target_kp, target_kd, target_tau)

            # ── 同步到共享内存 ──
            state[MotorDriver.ROW_QPOS, motor_indices] = local_qpos.astype(MotorDriver.DTYPE)
            state[MotorDriver.ROW_QVEL, motor_indices] = local_qvel.astype(MotorDriver.DTYPE)
            state[MotorDriver.ROW_TAU, motor_indices] = local_tau.astype(MotorDriver.DTYPE)

        bus.shutdown()
        shm.close()


# ═══════════════════════════════════════════════════
# 使用示例
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    # 外部控制顺序（与仿真器 / RL 策略一致）
    control_order = [
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle_pitch",
        "left_ankle_roll",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle_pitch",
        "right_ankle_roll",
        "waist_yaw",
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
        "left_wrist_yaw",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
        "right_wrist_yaw",
    ]

    with MotorDriver(THS_MOTORS, control_order) as driver:
        print(f"电机数量: {driver.num_joints}")
        target = np.zeros(driver.num_joints)
        for cfg in THS_MOTORS:
            target[control_order.index(cfg.joint_name)] = cfg.default_pos
        # 控制循环
        for _ in range(100):
            t1 = time.perf_counter()
            qpos = driver.qpos
            qvel = driver.qvel

            # PD 控制
            driver.set_ctrl(
                pos=target,
                vel=np.zeros(driver.num_joints),
                tau=np.zeros(driver.num_joints),
            )
            t2 = time.perf_counter()
            print(f"{t2 - t1}s")
            time.sleep(0.001)
