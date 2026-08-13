import atexit
import multiprocessing as mp
import os
import struct
import threading
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


# 说明书故障码 0x3022（faultSta）/ 故障反馈帧（通信类型21）各比特含义
FAULT_BIT_NAMES = {
    0: "电机过温",
    1: "驱动芯片故障",
    2: "欠压",
    3: "过压",
    7: "编码器未标定",
    14: "堵转i²t过载",
}

# 反馈帧（通信类型2）/ 使能应答帧 ID 中 bit21~16 故障位含义（与 0x3022 布局不同！）
FB_FAULT_BIT_NAMES = {
    0: "欠压故障",  # bit16
    1: "过流/驱动故障",  # bit17
    2: "过温",  # bit18
    3: "磁编码故障",  # bit19
    4: "堵转过载故障",  # bit20
    5: "编码器未标定",  # bit21
}


def decode_fault_bits(fault: int) -> list[str]:
    """把 0x3022 / 故障反馈帧的故障位图解析成故障描述列表"""
    return [name for bit, name in FAULT_BIT_NAMES.items() if fault & (1 << bit)]


def decode_fb_fault_bits(fault: int) -> list[str]:
    """把反馈帧 ID bit21~16 的故障位图解析成故障描述列表"""
    return [name for bit, name in FB_FAULT_BIT_NAMES.items() if fault & (1 << bit)]


def slow_can_io(bus, motor_id, arbitration_id, data, expect_func=None):
    """
    发送后阻塞等待回复，最多尝试 2 次
    expect_func: 期望的应答帧功能码，收到不匹配的帧（如残留的运控反馈帧）直接丢弃继续等待
    返回 (state, rx_data, rx_arbitration_id)
    """
    for _ in range(2):
        bus.send(can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=True))
        deadline = time.perf_counter() + 0.01
        while time.perf_counter() < deadline:
            rx = bus.recv(timeout=max(0.0, deadline - time.perf_counter()))
            if rx is None:
                break
            rx_id = (rx.arbitration_id >> 8) & 0xFF
            rx_func = rx.arbitration_id >> 24
            if rx_id != motor_id or rx_func == 0:
                continue
            if expect_func is not None and rx_func != expect_func:
                continue  # 丢弃功能码不匹配的帧，避免把残留反馈帧当成应答
            return 0, list(rx.data), rx.arbitration_id
    return 1, [0] * 8, 0


def set_motion_mode(bus, motor_id):
    """功能码 0x12，主 ID 0xfd，应答为反馈帧（0x02）"""
    arb = 0x1200FD00 | motor_id
    state, _, _ = slow_can_io(bus, motor_id, arb, [0] * 8, expect_func=0x02)
    time.sleep(0.0005)
    return state


def set_motion_stop(bus, motor_id, clear_fault=False):
    """通信类型4（0x04）电机停止运行，Byte0=1 时同时清除故障，应答为反馈帧（0x02）"""
    arb = 0x0400FD00 | motor_id
    data = [0x01, 0, 0, 0, 0, 0, 0, 0] if clear_fault else [0] * 8
    state, _, _ = slow_can_io(bus, motor_id, arb, data, expect_func=0x02)
    time.sleep(0.0005)
    return state


def set_motion_enable(bus, motor_id):
    """
    通信类型3（0x03）电机使能运行
    应答为反馈帧（通信类型2），校验 bit22~23 模式状态 == 2（Motor 模式[运行]），
    同时检查 bit21~16 故障位。返回 (state, mode, fault_bits)
    """
    arb = 0x0300FD00 | motor_id
    state, _, rx_arb = slow_can_io(bus, motor_id, arb, [0] * 8, expect_func=0x02)
    time.sleep(0.0005)
    if state != 0:
        return 1, -1, 0
    mode = (rx_arb >> 22) & 0x3
    fault_bits = (rx_arb >> 16) & 0x3F
    return 0, mode, fault_bits


def ensure_motor_enabled(bus, cfg, max_retry=5):
    """
    确保电机使能（全程不发送任何运控指令帧 0x01，只发状态帧）：
    1. 直接发送使能帧（通信类型3）判断状态：该帧不含位置/速度/力矩指令，
       对已使能的电机是幂等操作，不会引起任何动作；
       应答反馈帧模式状态 == 2 即表示已处于使能运行状态
    2. 若未使能（故障/复位状态），再执行 停止(清故障) -> 清运控模式 -> 使能
       序列重试，最多 max_retry 次
    返回 (success, detail)
    """

    def _describe(state, mode, fault_bits):
        if state != 0:
            return "使能帧无应答（通信异常）"
        if fault_bits:
            return "反馈故障位: " + "、".join(decode_fb_fault_bits(fault_bits))
        return f"使能后模式状态为 {mode}（0=复位/1=标定/2=运行），未进入运行状态"

    # 发送使能帧前清空队列残留帧（已使能电机的判断不会发任何运控指令）
    while bus.recv(timeout=0.005) is not None:
        pass

    state, mode, fault_bits = set_motion_enable(bus, cfg.motor_id)
    if state == 0 and mode == 2 and not fault_bits:
        return True, "已使能（含本次使能成功）"

    # 未使能：停止(清故障) -> 清运控模式 -> 使能，重试 max_retry 次
    last_detail = _describe(state, mode, fault_bits)
    for _ in range(max_retry):
        while bus.recv(timeout=0.005) is not None:
            pass
        set_motion_stop(bus, cfg.motor_id, clear_fault=True)  # 切模式须在失能状态，同时清除残留故障
        set_motion_mode(bus, cfg.motor_id)
        state, mode, fault_bits = set_motion_enable(bus, cfg.motor_id)
        if state == 0 and mode == 2 and not fault_bits:
            return True, "使能成功"
        last_detail = _describe(state, mode, fault_bits)
        time.sleep(0.02)  # 重试前稍作等待
    return False, last_detail


def read_motor_raw(bus, motor_id, address):
    """功能码 0x11，主 ID 0xfd，返回原始 Byte4~7（低字节在前）"""
    arb = 0x1100FD00 | motor_id
    data = [address & 0xFF, (address >> 8) & 0xFF, 0, 0, 0, 0, 0, 0]
    state, rx, rx_arb = slow_can_io(bus, motor_id, arb, data, expect_func=0x11)
    if state == 0:
        # 应答帧 Bit23~16: 00 读取成功，01 读取失败
        if (rx_arb >> 16) & 0xFF:
            return 1, 0
        return 0, rx[4] | (rx[5] << 8) | (rx[6] << 16) | (rx[7] << 24)
    return 1, 0


def read_motor_single_data(bus, motor_id, address):
    """单个参数读取，按 float 解析（如 0x7019 = mechPos 负载端机械角度）"""
    state, raw = read_motor_raw(bus, motor_id, address)
    if state == 0:
        return 0, struct.unpack("<f", raw.to_bytes(4, "little"))[0]
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

        # ═══════ 启动子进程前预检：先使能所有电机，再检查通信读数 ═══════
        # 注意：失败路径一律 os._exit 硬退出，避免残留的守护子进程（手柄/IMU）导致解释器无法退出
        for bus_name, motor_indices in bus_groups.items():
            configs = [motor_configs[i] for i in motor_indices]
            try:
                if not self._precheck_bus(bus_name, configs):
                    os._exit(1)
            except RuntimeError as e:
                print(e)
                os._exit(1)
            except OSError as e:
                print(f"错误: CAN 接口 {bus_name} 打开失败: {e}")
                os._exit(1)

        # 共享内存
        total_bytes = self.NUM_ROWS * self.num_motors * np.dtype(self.DTYPE).itemsize
        self.shared_mem = shared_memory.SharedMemory(create=True, size=total_bytes)
        self.state = np.ndarray((self.NUM_ROWS, self.num_motors), dtype=self.DTYPE, buffer=self.shared_mem.buf)
        self.state[:] = 0.0

        # 启动各 CAN 进程
        self.running = mp.Value("b", True)
        self.fatal_comm = mp.Value("b", False)
        self.fatal_msg = mp.Array("c", 512)
        self.ready_count = mp.Value("i", 0)  # 已完成默认姿态过渡的 bus 数
        self.processes = []

        for bus_name, motor_indices in bus_groups.items():
            configs = [motor_configs[i] for i in motor_indices]
            proc = mp.Process(
                target=self._can_worker,
                args=(
                    bus_name,
                    motor_indices,
                    configs,
                    self.shared_mem.name,
                    self.state.shape,
                    self.running,
                    self.fatal_comm,
                    self.fatal_msg,
                    self.ready_count,
                ),
            )
            proc.start()
            self.processes.append(proc)

        # 通信异常监控：子进程上报后立即停止程序
        threading.Thread(target=self._fatal_monitor, daemon=True).start()

        atexit.register(self.close)

        # 等待所有 bus 完成默认姿态过渡（致命异常时由监控线程终止程序）
        deadline = time.perf_counter() + 10.0
        while self.ready_count.value < len(self.processes):
            if self.fatal_comm.value:
                self._abort()  # 子进程已上报致命异常，立即退出
            if time.perf_counter() > deadline:
                print("错误: 等待电机进入默认姿态超时，程序退出")
                self._abort()
            time.sleep(0.05)
        print(f"所有 {self.num_motors} 个电机就绪，已进入默认姿态")

    # ── 启动预检 ──

    @staticmethod
    def _precheck_bus(bus_name, configs):
        """
        主进程内对一路 CAN 做启动预检（全程只发使能/停止/模式状态帧与读取帧，
        不发送任何运控指令帧 0x01，第一条运控指令由子进程过渡到默认姿态时下发）：
        1. 逐个检查并使能电机（已使能的直接跳过），失败则报出具体电机并终止
        2. 逐个读取角度与故障状态，检查通信是否正常
        3. 读取所有电机当前位置，打印与默认姿态的角度差异
        返回 True 表示预检通过
        """
        bus = can.interface.Bus(channel=bus_name, interface="socketcan", bitrate=1000000)
        # 清空总线上残留帧
        while bus.recv(timeout=0.01) is not None:
            pass

        # ── 1. 检查并使能所有电机（已使能的跳过，最多尝试 5 次）──
        enable_failures = []
        for cfg in configs:
            ok, detail = ensure_motor_enabled(bus, cfg, max_retry=5)
            if ok:
                print(f"[{bus_name}] 电机 {cfg.joint_name} (id=0x{cfg.motor_id:02X}) {detail}")
            else:
                enable_failures.append(f"{cfg.joint_name} (id=0x{cfg.motor_id:02X}): {detail}")
        if enable_failures:
            bus.shutdown()
            raise RuntimeError("电机使能失败，程序退出:\n  " + "\n  ".join(enable_failures))

        # ── 2. 检查所有电机通信读数（仅读取，不下发任何指令）──
        comm_failures = []
        positions = {}
        for cfg in configs:
            err, pos = read_motor_single_data(bus, cfg.motor_id, 0x7019)  # mechPos
            if err != 0:
                comm_failures.append(f"{cfg.joint_name} (id=0x{cfg.motor_id:02X}): 角度读取无应答")
                continue
            if not np.isfinite(pos) or abs(pos) > 1.8:
                comm_failures.append(f"{cfg.joint_name} (id=0x{cfg.motor_id:02X}): 角度读数异常 {pos:.4f} rad")
            positions[cfg.joint_name] = pos
            err, raw = read_motor_raw(bus, cfg.motor_id, 0x3022)  # faultSta
            if err == 0 and raw != 0:
                faults = decode_fault_bits(raw)
                comm_failures.append(
                    f"{cfg.joint_name} (id=0x{cfg.motor_id:02X}): 故障状态 0x{raw:08X}"
                    + (f"（{'、'.join(faults)}）" if faults else "")
                )
        if comm_failures:
            bus.shutdown()
            print("电机通信检查异常，以下关节读数有问题，程序退出:")
            for item in comm_failures:
                print(f"  {item}")
            return False
        print(f"[{bus_name}] 通信检查通过，{len(configs)} 个电机读数正常")

        # ── 3. 打印当前位置与默认姿态的角度差异（仅读取，不下发任何指令）──
        print(f"[{bus_name}] 当前角度与默认姿态差异:")
        for cfg in configs:
            pos = positions[cfg.joint_name]
            diff = pos - cfg.default_pos
            print(
                f"  {cfg.joint_name:<24} 当前 {pos:+.4f} rad | 默认 {cfg.default_pos:+.4f} rad | 差异 {diff:+.4f} rad"
            )

        bus.shutdown()
        return True

    def _fatal_monitor(self):
        """监控致命通信异常与子进程存活，一旦触发立即停止程序"""
        while True:
            if self.fatal_comm.value:
                msg = self.fatal_msg.value.decode(errors="ignore")
                print(f"\n通信异常，程序立即停止: {msg}")
                atexit.unregister(self.close)
                os._exit(1)
            if not self.running.value:
                break
            # 子进程意外退出（未上报致命异常）也要立即停止程序
            for p in self.processes:
                if p.exitcode is not None and p.exitcode != 0:
                    print(f"\n通信子进程异常退出 (exitcode={p.exitcode})，程序立即停止")
                    atexit.unregister(self.close)
                    os._exit(1)
            time.sleep(0.05)

    def _abort(self):
        """异常终止：停掉所有子进程并释放资源后退出"""
        self.running.value = False
        for p in self.processes:
            p.terminate()
            p.join(timeout=1.0)
        self.shared_mem.close()
        self.shared_mem.unlink()
        atexit.unregister(self.close)
        os._exit(1)

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
    def _can_worker(
        bus_name, motor_indices, motor_configs, shm_name, state_shape, running, fatal_comm, fatal_msg, ready_count
    ):
        def fatal_stop(reason):
            """通信致命异常：上报主进程并立即退出本子进程"""
            print(f"[{bus_name}] 通信异常: {reason}")
            fatal_msg.value = reason.encode()[:511]
            fatal_comm.value = True
            running.value = False

        # 连接共享内存
        shm = shared_memory.SharedMemory(name=shm_name)
        state = np.ndarray(state_shape, dtype=MotorDriver.DTYPE, buffer=shm.buf)

        n_local = len(motor_indices)
        motor_ids = [cfg.motor_id for cfg in motor_configs]
        id_to_local = {mid: i for i, mid in enumerate(motor_ids)}
        id_to_name = {cfg.motor_id: cfg.joint_name for cfg in motor_configs}

        # 创建 CAN 总线（使能/通信预检已在主进程完成）
        try:
            bus = can.interface.Bus(channel=bus_name, interface="socketcan", bitrate=1000000)
        except OSError as e:
            fatal_stop(f"CAN 接口 {bus_name} 打开失败: {e}")
            shm.close()
            return

        # 清空残留帧
        while bus.recv(timeout=0.01) is not None:
            pass

        # ═══════ 初始化：读当前角度 ═══════
        init_pos = np.zeros(n_local, dtype=np.float64)
        default_pos = np.array([cfg.default_pos for cfg in motor_configs], dtype=np.float64)

        for local_idx, cfg in enumerate(motor_configs):
            mid = cfg.motor_id
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

        # 按电机粒度的心跳计时（0.0 表示从未收到过反馈）
        FEEDBACK_TIMEOUT_S = 1.0
        last_feedback = np.zeros(n_local, dtype=np.float64)

        # 过渡阶段结束后确认所有电机都有反馈，否则视为通信异常
        deadline = time.perf_counter() + FEEDBACK_TIMEOUT_S
        while time.perf_counter() < deadline:
            msg = bus.recv(timeout=0.05)
            if msg is None:
                if np.all(last_feedback > 0.0):
                    break
                continue
            if msg.arbitration_id >> 24 != 0x02:
                continue
            rx_mid = (msg.arbitration_id >> 8) & 0xFF
            if rx_mid in id_to_local:
                last_feedback[id_to_local[rx_mid]] = time.perf_counter()
            if np.all(last_feedback > 0.0):
                break
        silent = [id_to_name[motor_ids[i]] for i in range(n_local) if last_feedback[i] <= 0.0]
        if silent:
            fatal_stop(f"以下关节无反馈: {', '.join(silent)}")
            bus.shutdown()
            shm.close()
            return

        # 把最终状态写入共享内存
        state[MotorDriver.ROW_QPOS, motor_indices] = default_pos.astype(MotorDriver.DTYPE)
        state[MotorDriver.ROW_CTRL_POS, motor_indices] = default_pos.astype(MotorDriver.DTYPE)

        # 设置默认 KP/KD
        for local_idx, cfg in enumerate(motor_configs):
            state[MotorDriver.ROW_CTRL_KP, motor_indices[local_idx]] = cfg.default_kp
            state[MotorDriver.ROW_CTRL_KD, motor_indices[local_idx]] = cfg.default_kd

        # 本 bus 已就绪（默认姿态过渡完成）
        with ready_count.get_lock():
            ready_count.value += 1

        # ═══════ 高速运行阶段 ═══════
        local_qpos = default_pos.copy()
        local_qvel = np.zeros(n_local, dtype=np.float64)
        local_tau = np.zeros(n_local, dtype=np.float64)

        while running.value:
            # ── 接收阶段 ──
            while True:
                msg = bus.recv(timeout=0)
                if msg is None:
                    break

                rx_func = msg.arbitration_id >> 24
                rx_motor_id = (msg.arbitration_id >> 8) & 0xFF
                if rx_motor_id not in id_to_local:
                    continue

                # 故障反馈帧（通信类型21）
                if rx_func == 0x15:
                    fault = msg.data[0] | (msg.data[1] << 8) | (msg.data[2] << 16) | (msg.data[3] << 24)
                    if fault:
                        name = id_to_name.get(rx_motor_id, f"id=0x{rx_motor_id:02X}")
                        faults = decode_fault_bits(fault)
                        detail = "、".join(faults) if faults else f"0x{fault:08X}"
                        fatal_stop(f"关节 {name} 上报故障: {detail}")
                        bus.shutdown()
                        shm.close()
                        return
                    continue

                # 过滤：功能码必须为 0x02（运控反馈帧）
                if rx_func != 0x02:
                    continue

                local_idx = id_to_local[rx_motor_id]
                cfg = motor_configs[local_idx]
                data = msg.data

                # 反馈帧 ID 中的故障位（bit21~16，0 无 1 有）
                fb_fault = (msg.arbitration_id >> 16) & 0x3F
                if fb_fault:
                    faults = decode_fb_fault_bits(fb_fault)
                    detail = "、".join(faults) if faults else f"0x{fb_fault:02X}"
                    fatal_stop(f"关节 {cfg.joint_name} 反馈帧带故障位: {detail}")
                    bus.shutdown()
                    shm.close()
                    return

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

                last_feedback[local_idx] = time.perf_counter()

            # ── 按电机粒度检查反馈超时 ──
            now = time.perf_counter()
            silent = [id_to_name[motor_ids[i]] for i in range(n_local) if now - last_feedback[i] > FEEDBACK_TIMEOUT_S]
            if silent:
                fatal_stop(f"以下关节通信超时（超过 {FEEDBACK_TIMEOUT_S:.0f}s 未收到反馈）: {', '.join(silent)}")
                bus.shutdown()
                shm.close()
                return

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

    # 预检失败会在 MotorDriver.__init__ 内打印具体电机并硬退出
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
