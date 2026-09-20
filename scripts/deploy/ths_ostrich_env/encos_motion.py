"""
ENCOS Driver 的 Python 对外接口。

本文件只负责三件事：构造 C ABI 配置、把 Python 命令转换为 ctypes 结构、
把底层缓存状态转换回 Python 对象。生产调用链为：
EncosMotion / EncosDriver -> 客户端 so -> Unix IPC -> 通信进程 -> 核心 Worker。
CAN/EtherCAT 收发、协议编解码和超时保护在通信进程执行。
Worker 周期取启动配置，不代表已实测达到确定实时频率。
"""

from __future__ import annotations

import ctypes
import configparser
import logging
import math
import os
import time
from dataclasses import dataclass, replace
from enum import IntEnum, IntFlag
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Mapping, Sequence

# SDK 发布版本与 C ABI 版本分开管理。
SDK_VERSION = "0.5.0"
# ABI 版本只用于阻止旧 Python 结构体误加载新 DLL/SO，不代表项目发布版本。
ABI_VERSION = 2
CAPABILITY_MIT = 1 << 0
CAPABILITY_TORQUE = 1 << 1
CAPABILITY_SERVO_POSITION = 1 << 2
CAPABILITY_SERVO_SPEED = 1 << 3
CAPABILITY_ALL = (
    CAPABILITY_MIT
    | CAPABILITY_TORQUE
    | CAPABILITY_SERVO_POSITION
    | CAPABILITY_SERVO_SPEED
)
# group 名和 joint 名是运控侧稳定语义，CAN ID 映射保留在 C/C++ Driver。
GROUPS = {
    "left_leg": 1,
    "right_leg": 2,
    "arm": 3,
}
GROUP_JOINTS = {
    "left_leg": (
        "left_hip_pitch",
        "left_hip_roll",
        "left_hip_yaw",
        "left_knee_pitch",
        "left_ankle_pitch",
    ),
    "right_leg": (
        "right_hip_pitch",
        "right_hip_roll",
        "right_hip_yaw",
        "right_knee_pitch",
        "right_ankle_pitch",
    ),
    "arm": (
        "arm_j1_shoulder_yaw",
        "arm_j2_shoulder_pitch",
        "arm_j3_shoulder_roll",
        "arm_j4_elbow_pitch",
        "arm_j5_elbow_pitch",
        "arm_j6_wrist_pitch",
        "arm_j7_wrist_yaw",
        "arm_j8_gripper",
    ),
}
JOINT_NAMES = sum(GROUP_JOINTS.values(), ())
JOINT_ID = {name: index for index, name in enumerate(JOINT_NAMES)}
JOINT_CAN_ID = dict(zip(JOINT_NAMES, (
    0x11, 0x12, 0x13, 0x14, 0x15, 0x21, 0x22, 0x23, 0x24,
    0x25, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38)))
JOINT_MODEL = {
    **{name: "EC-A10020-P2-24(20250922)" for name in (
        "left_hip_pitch", "left_hip_roll", "left_knee_pitch",
        "right_hip_pitch", "right_hip_roll", "right_knee_pitch")},
    **{name: "EC-A6408-P2-30.25H" for name in ("left_hip_yaw", "right_hip_yaw")},
    **{name: "EC-A4310-P2-36H" for name in ("left_ankle_pitch", "right_ankle_pitch")},
    **{name: "EC-A4315-P2-36" for name in ("arm_j1_shoulder_yaw", "arm_j2_shoulder_pitch")},
    **{name: "EC-A4310-P2-36" for name in (
        "arm_j3_shoulder_roll", "arm_j4_elbow_pitch", "arm_j5_elbow_pitch")},
    **{name: "EC-A2806-P2-36" for name in (
        "arm_j6_wrist_pitch", "arm_j7_wrist_yaw", "arm_j8_gripper")},
}
RECOMMENDED_MIT_GAINS = {
    **dict(zip(GROUP_JOINTS["left_leg"], (
        (400.0, 5.0), (400.0, 5.0), (150.0, 3.0),
        (400.0, 5.0), (90.0, 2.0)))),
    **dict(zip(GROUP_JOINTS["right_leg"], (
        (400.0, 5.0), (400.0, 5.0), (150.0, 3.0),
        (400.0, 5.0), (90.0, 2.0)))),
    **{name: (15.0, 2.5) for name in GROUP_JOINTS["arm"]},
    "arm_j4_elbow_pitch": (60.0, 3.0), "arm_j8_gripper": (8.0, 1.5),
}
STATUS_NAME = {
    0: "OK", -1: "INVALID_ARGUMENT", -2: "OUT_OF_RANGE", -3: "INVALID_FRAME",
    -4: "UNSUPPORTED", -5: "NOT_RUNNING", -6: "ALREADY_RUNNING", -7: "TRANSPORT",
    -8: "TIMEOUT", -9: "PROFILE_UNAVAILABLE", -10: "ESTOP_LATCHED", -11: "CONFIG",
    -12: "NOT_SUPPORTED", -13: "INTERNAL", -14: "MOTOR_FAULT",
    -15: "FEEDBACK_UNAVAILABLE", -16: "QUERY_FAILED", -17: "NOT_CONFIGURED",
}
_FAULT_LOGGER = None


def fault_log_directory() -> Path:
    return Path(os.environ.get("ENCOS_FAULT_RECORD_DIR", "/home/robot/fault_records"))


def _get_fault_logger() -> logging.Logger:
    global _FAULT_LOGGER
    if _FAULT_LOGGER is not None:
        return _FAULT_LOGGER
    logger = logging.getLogger("encos.motion")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        try:
            directory = fault_log_directory()
            directory.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(directory / "encos_python.log",
                maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
        except OSError:
            handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)sZ level=%(levelname)s sdk=0.5.0 %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S")
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    _FAULT_LOGGER = logger
    return logger


def load_joint_direction_config(
    config_path: str | os.PathLike[str],
) -> dict[str, int]:
    """
    读取完整的关节方向配置。

    配置文件必须只有一个 [direction] 段，并且完整列出 18 个英文关节名。
    值只允许 +1、1 或 -1：+1 表示电机原始正方向与 URDF 关节正方向一致，
    -1 表示相反。函数先完成全部校验，再一次性返回结果，不会部分应用配置。
    """
    path = Path(config_path).expanduser().resolve(strict=True)
    parser = configparser.ConfigParser(
        interpolation=None,
        strict=True,
        delimiters=("=",),
        comment_prefixes=("#", ";"),
        inline_comment_prefixes=("#", ";"),
        empty_lines_in_values=False,
    )
    parser.optionxform = str

    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            parser.read_file(stream)
    except configparser.Error as error:
        raise ValueError(f"invalid direction config {path}: {error}") from error

    if parser.defaults() or parser.sections() != ["direction"]:
        raise ValueError(
            "direction config must contain exactly one [direction] section "
            "and no DEFAULT values"
        )

    values = dict(parser.items("direction", raw=True))
    expected = set(JOINT_NAMES)
    actual = set(values)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"direction config joint set mismatch: missing={missing}, unknown={unknown}"
        )

    directions: dict[str, int] = {}
    for joint_name in JOINT_NAMES:
        text = values[joint_name].strip()
        if text not in {"+1", "1", "-1"}:
            raise ValueError(
                f"direction for {joint_name!r} must be +1 or -1, got {text!r}"
            )
        directions[joint_name] = int(text)
    return directions


class RuntimeProfileState(IntEnum):
    """电机实际参数查询的当前状态。"""
    UNQUERIED = 0
    QUERYING = 1
    READY = 2
    MISMATCH = 3
    FAILED = 4


class RuntimeField(IntFlag):
    """实际参数查询中已收到、失败或不一致的字段位。"""
    NONE = 0
    KT = 1 << 0
    KP = 1 << 1
    KD = 1 << 2
    POSITION = 1 << 3
    VELOCITY = 1 << 4
    TORQUE = 1 << 5
    CURRENT = 1 << 6
    CAN_TIMEOUT = 1 << 7
    ALL = 0xFF


@dataclass(frozen=True, slots=True)
class ProtocolRange:
    """MIT 协议编码范围；它不是机器人机械安全限位。"""

    kp_min: float
    kp_max: float
    kd_min: float
    kd_max: float
    position_min_rad: float
    position_max_rad: float
    velocity_min_rad_s: float
    velocity_max_rad_s: float
    torque_min_nm: float
    torque_max_nm: float
    current_min_a: float
    current_max_a: float


def _protocol_range(kd_max: float, torque_max: float, current_max: float) -> ProtocolRange:
    return ProtocolRange(0.0, 500.0, 0.0, kd_max, -12.5, 12.5, -18.0, 18.0,
                         -torque_max, torque_max, -current_max, current_max)


# 现场固件配置：全部电机 KP=0..500、KD=0..50。必须与电机端量程一致。
# 力矩/电流量程沿用原协议；这些量程用于报文编码，不是软件输出限值。
MOTOR_PROTOCOL = {
    "EC-A10020-P2-24(20250922)": (_protocol_range(50.0, 300.0, 140.0), 2.6),
    "EC-A6408-P2-30.25H": (_protocol_range(50.0, 60.0, 60.0), 2.45),
    "EC-A4310-P2-36H": (_protocol_range(50.0, 30.0, 30.0), 1.4),
    "EC-A4315-P2-36": (_protocol_range(50.0, 70.0, 30.0), 2.8),
    "EC-A4310-P2-36": (_protocol_range(50.0, 30.0, 30.0), 1.4),
    "EC-A2806-P2-36": (_protocol_range(50.0, 12.0, 10.0), 1.35),
}


@dataclass(frozen=True, slots=True)
class JointSafetyProfile:
    """
    单关节静态配置。

    direction=-1 表示电机原始正方向与 URDF 关节正方向相反，1 表示一致；
    方向可由 joint_direction.ini 在启动时覆盖。kinematic_mapping_valid 只决定
    motor_zero_at_robot_zero_rad 是否参与位置换算，未确认机械参数仍可保持无效。
    """

    protocol: ProtocolRange
    physical_position_min_rad: float
    physical_position_max_rad: float
    physical_velocity_max_rad_s: float
    physical_torque_max_nm: float
    physical_current_max_a: float
    application_position_min_rad: float
    application_position_max_rad: float
    application_velocity_max_rad_s: float
    application_torque_max_nm: float
    application_current_max_a: float
    kt_nm_per_a: float
    model_spec_confirmed: bool
    physical_limits_valid: bool
    application_limits_valid: bool
    expected_can_timeout_ms: int
    direction: int
    motor_zero_at_robot_zero_rad: float
    control_capabilities: int
    kinematic_mapping_valid: bool


@dataclass(frozen=True, slots=True)
class MitCommand:
    """单关节 MIT 命令，单位依次为 rad、rad/s、Nm。"""

    position_rad: float = 0.0
    velocity_rad_s: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    torque_ff_nm: float = 0.0


@dataclass(frozen=True, slots=True)
class ServoPositionCommand:
    """伺服位置命令；速度和电流字段是运动上限。"""

    position_rad: float
    speed_limit_rad_s: float
    current_limit_a: float


@dataclass(frozen=True, slots=True)
class ServoSpeedCommand:
    """伺服速度命令，外部统一使用 rad/s 和 A。"""

    velocity_rad_s: float
    current_limit_a: float


@dataclass(frozen=True, slots=True)
class JointState:
    """Driver 最新缓存中的单关节状态，不会触发同步 CAN 查询。"""

    position_rad: float
    velocity_rad_s: float
    phase_current_a: float
    torque_est_nm: float
    motor_temperature_c: float
    mos_temperature_c: float
    motor_error_code: int
    control_enabled: bool
    online: bool
    feedback_valid: bool
    feedback_age_us: int
    state_sequence: int
    command_timeout: bool
    feedback_timeout: bool

    # Public SDK aliases: 运控代码无需记忆底层单位后缀字段名。
    @property
    def position(self) -> float:
        return self.position_rad

    @property
    def velocity(self) -> float:
        return self.velocity_rad_s

    @property
    def current(self) -> float:
        return self.phase_current_a

    @property
    def torque(self) -> float:
        return self.torque_est_nm

    @property
    def error(self) -> int:
        return self.motor_error_code


@dataclass(frozen=True, slots=True)
class GroupState:
    """在同一个缓存锁内取得的一组关节状态快照。"""

    group: str
    snapshot_sequence: int
    joints: dict[str, JointState]

    @property
    def names(self) -> tuple[str, ...]:
        return GROUP_JOINTS[self.group]

    @property
    def position(self) -> tuple[float, ...]:
        return tuple(self.joints[name].position_rad for name in self.names)

    @property
    def velocity(self) -> tuple[float, ...]:
        return tuple(self.joints[name].velocity_rad_s for name in self.names)

    @property
    def current(self) -> tuple[float, ...]:
        return tuple(self.joints[name].phase_current_a for name in self.names)

    @property
    def torque(self) -> tuple[float, ...]:
        return tuple(self.joints[name].torque_est_nm for name in self.names)

    @property
    def error(self) -> tuple[int, ...]:
        return tuple(self.joints[name].motor_error_code for name in self.names)

    @property
    def online(self) -> tuple[bool, ...]:
        return tuple(self.joints[name].online for name in self.names)


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """从真实电机查询到的协议范围、Kt 和 CAN Timeout。"""

    state: RuntimeProfileState
    received_mask: RuntimeField
    mismatch_mask: RuntimeField
    query_failed_mask: RuntimeField
    motor_error_code: int
    updated_age_us: int
    actual_protocol: ProtocolRange
    actual_kt_nm_per_a: float
    actual_can_timeout_ms: int


class EncosError(RuntimeError):
    """底层 C API 返回非零状态码时抛出的统一异常。"""



class _CProtocolRange(ctypes.Structure):
    _fields_ = [(name, ctypes.c_float) for name in (
        "kp_min", "kp_max", "kd_min", "kd_max",
        "position_min_rad", "position_max_rad",
        "velocity_min_rad_s", "velocity_max_rad_s",
        "torque_min_nm", "torque_max_nm", "current_min_a", "current_max_a",
    )]


class _CJointSafetyConfig(ctypes.Structure):
    _fields_ = [
        ("protocol", _CProtocolRange),
        ("physical_position_min_rad", ctypes.c_float),
        ("physical_position_max_rad", ctypes.c_float),
        ("physical_velocity_max_rad_s", ctypes.c_float),
        ("physical_torque_max_nm", ctypes.c_float),
        ("physical_current_max_a", ctypes.c_float),
        ("application_position_min_rad", ctypes.c_float),
        ("application_position_max_rad", ctypes.c_float),
        ("application_velocity_max_rad_s", ctypes.c_float),
        ("application_torque_max_nm", ctypes.c_float),
        ("application_current_max_a", ctypes.c_float),
        ("kt_nm_per_a", ctypes.c_float),
        ("static_profile_valid", ctypes.c_uint32),
        ("model_spec_confirmed", ctypes.c_uint32),
        ("physical_limits_valid", ctypes.c_uint32),
        ("application_limits_valid", ctypes.c_uint32),
        ("expected_can_timeout_ms", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("direction", ctypes.c_int32),
        ("motor_zero_at_robot_zero_rad", ctypes.c_float),
        ("control_capabilities", ctypes.c_uint32),
        ("kinematic_mapping_valid", ctypes.c_uint32),
    ]


class _CDriverConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("control_period_us", ctypes.c_uint32),
        ("command_timeout_us", ctypes.c_uint32),
        ("feedback_timeout_us", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("endpoint", (ctypes.c_char * 64) * 3),
        ("joint", _CJointSafetyConfig * 18),
    ]


class _CMitCommand(ctypes.Structure):
    _fields_ = [(name, ctypes.c_float) for name in (
        "position_rad", "velocity_rad_s", "kp", "kd", "torque_ff_nm",
    )]


class _CTorqueCommand(ctypes.Structure):
    _fields_ = [("torque_nm", ctypes.c_float)]


class _CServoPositionCommand(ctypes.Structure):
    _fields_ = [
        ("position_rad", ctypes.c_float),
        ("speed_limit_rad_s", ctypes.c_float),
        ("current_limit_a", ctypes.c_float),
    ]


class _CServoSpeedCommand(ctypes.Structure):
    _fields_ = [
        ("velocity_rad_s", ctypes.c_float),
        ("current_limit_a", ctypes.c_float),
    ]


class _CRuntimeProfile(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("state", ctypes.c_uint32),
        ("received_mask", ctypes.c_uint32),
        ("mismatch_mask", ctypes.c_uint32),
        ("query_failed_mask", ctypes.c_uint32),
        ("motor_error_code", ctypes.c_uint32),
        ("updated_age_us", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("actual_protocol", _CProtocolRange),
        ("actual_kt_nm_per_a", ctypes.c_float),
        ("actual_can_timeout_ms", ctypes.c_uint32),
    ]


class _CJointState(ctypes.Structure):
    _fields_ = [
        ("joint_id", ctypes.c_uint32),
        ("position_rad", ctypes.c_float),
        ("velocity_rad_s", ctypes.c_float),
        ("phase_current_a", ctypes.c_float),
        ("torque_est_nm", ctypes.c_float),
        ("motor_temperature_c", ctypes.c_float),
        ("mos_temperature_c", ctypes.c_float),
        ("motor_error_code", ctypes.c_uint32),
        ("control_enabled", ctypes.c_uint32),
        ("online", ctypes.c_uint32),
        ("feedback_valid", ctypes.c_uint32),
        ("feedback_age_us", ctypes.c_uint32),
        ("state_sequence", ctypes.c_uint32),
        ("command_timeout", ctypes.c_uint32),
        ("feedback_timeout", ctypes.c_uint32),
    ]


class _CGroupState(ctypes.Structure):
    _fields_ = [
        ("group_id", ctypes.c_uint32),
        ("joint_count", ctypes.c_uint32),
        ("snapshot_sequence", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("joints", _CJointState * 8),
    ]


class EncosDriver:
    """
    运控侧唯一需要持有的对象。

    start 至少提供一路有效 endpoint。空端口表示本次不接入该组；生产守护保留
    已配置组，首次使用未配置组时按服务默认端口建立。底层构造器的默认参数保留
    兼容性；整机接线和 ST 默认值见 examples/encos_st_config.py。
    """

    def __init__(
        self,
        library_path: str | os.PathLike[str],
        profiles: Mapping[str, JointSafetyProfile],
        endpoints: Sequence[str] = ("can0", "can1", "can2"),
        control_period_us: int = 5000,
        command_timeout_us: int = 100_000,
        feedback_timeout_us: int = 100_000,
        direction_config: str | os.PathLike[str] | None = None,
    ) -> None:
        """
        加载动态库并保存配置；start() 经客户端 so 请求守护建立或接入对应组。

        direction_config 可指向完整的 joint_direction.ini。文件在每次 start()
        前重新读取，因此 stop 后修改配置再 start 即可生效；运行过程中不会热更新。
        """
        if set(profiles) != set(JOINT_NAMES):
            raise ValueError("profiles must contain exactly all 18 joint names")
        if len(endpoints) != 3:
            raise ValueError("exactly three CAN endpoints are required")
        self._profiles = dict(profiles)
        self._endpoints = tuple(endpoints)
        self._period_us = int(control_period_us)
        self._command_timeout_us = int(command_timeout_us)
        self._feedback_timeout_us = int(feedback_timeout_us)
        self._direction_config = (
            None
            if direction_config is None
            else Path(direction_config).expanduser().resolve(strict=True)
        )
        self._dll_directory = None

        path = Path(library_path).expanduser().resolve(strict=True)
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            # 句柄必须随 EncosDriver 对象存活，否则后续依赖 DLL 解析可能失效。
            self._dll_directory = os.add_dll_directory(str(path.parent))
        self._lib = ctypes.CDLL(str(path))
        self._bind()
        self._check_layout()
        if self._lib.encos_get_abi_version() != ABI_VERSION:
            raise EncosError("ENCOS ABI version mismatch")

    def _bind(self) -> None:
        """显式声明每个 C 函数的参数和返回类型，避免 ctypes 默认按 int 猜测。"""
        lib = self._lib
        lib.encos_get_abi_version.argtypes = []
        lib.encos_get_abi_version.restype = ctypes.c_uint32
        lib.encos_start.argtypes = [ctypes.POINTER(_CDriverConfig)]
        lib.encos_start.restype = ctypes.c_int32
        lib.encos_stop.argtypes = []
        lib.encos_stop.restype = ctypes.c_int32
        lib.encos_set_group_mit.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(_CMitCommand), ctypes.c_uint32,
        ]
        lib.encos_set_group_mit.restype = ctypes.c_int32
        lib.encos_set_group_torque.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(_CTorqueCommand), ctypes.c_uint32,
        ]
        lib.encos_set_group_torque.restype = ctypes.c_int32
        lib.encos_set_group_servo_position.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(_CServoPositionCommand), ctypes.c_uint32,
        ]
        lib.encos_set_group_servo_position.restype = ctypes.c_int32
        lib.encos_set_group_servo_speed.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(_CServoSpeedCommand), ctypes.c_uint32,
        ]
        lib.encos_set_group_servo_speed.restype = ctypes.c_int32
        lib.encos_get_group_state.argtypes = [ctypes.c_uint32, ctypes.POINTER(_CGroupState)]
        lib.encos_get_group_state.restype = ctypes.c_int32
        lib.encos_set_joint_mit.argtypes = [ctypes.c_uint32, ctypes.POINTER(_CMitCommand)]
        lib.encos_set_joint_mit.restype = ctypes.c_int32
        lib.encos_set_joint_torque.argtypes = [ctypes.c_uint32, ctypes.POINTER(_CTorqueCommand)]
        lib.encos_set_joint_torque.restype = ctypes.c_int32
        lib.encos_set_joint_servo_position.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(_CServoPositionCommand),
        ]
        lib.encos_set_joint_servo_position.restype = ctypes.c_int32
        lib.encos_set_joint_servo_speed.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(_CServoSpeedCommand),
        ]
        lib.encos_set_joint_servo_speed.restype = ctypes.c_int32
        lib.encos_get_joint_state.argtypes = [ctypes.c_uint32, ctypes.POINTER(_CJointState)]
        lib.encos_get_joint_state.restype = ctypes.c_int32
        lib.encos_request_joint_state_field.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        lib.encos_request_joint_state_field.restype = ctypes.c_int32
        lib.encos_set_joint_zero.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        lib.encos_set_joint_zero.restype = ctypes.c_int32
        lib.encos_request_joint_runtime_profile.argtypes = [ctypes.c_uint32]
        lib.encos_request_joint_runtime_profile.restype = ctypes.c_int32
        lib.encos_get_joint_runtime_profile.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(_CRuntimeProfile),
        ]
        lib.encos_get_joint_runtime_profile.restype = ctypes.c_int32
        lib.encos_query_bus_motor_id.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.encos_query_bus_motor_id.restype = ctypes.c_int32
        lib.encos_enable_group.argtypes = [ctypes.c_uint32]
        lib.encos_enable_group.restype = ctypes.c_int32
        lib.encos_disable_group.argtypes = [ctypes.c_uint32]
        lib.encos_disable_group.restype = ctypes.c_int32
        lib.encos_emergency_stop.argtypes = []
        lib.encos_emergency_stop.restype = ctypes.c_int32
        lib.encos_is_healthy.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        lib.encos_is_healthy.restype = ctypes.c_int32

    @staticmethod
    def _check_layout() -> None:
        """在第一次调用 Driver 前检查 Python/C 结构体尺寸是否一致。"""
        expected = {
            _CMitCommand: 20,
            _CTorqueCommand: 4,
            _CServoPositionCommand: 12,
            _CServoSpeedCommand: 8,
            _CJointState: 60,
            _CGroupState: 496,
            _CJointSafetyConfig: 132,
            _CDriverConfig: 2592,
            _CRuntimeProfile: 88,
        }
        for structure, size in expected.items():
            actual = ctypes.sizeof(structure)
            if actual != size:
                raise EncosError(
                    f"ABI layout mismatch for {structure.__name__}: "
                    f"{actual} != {size}"
                )

        # 仅检查总尺寸还不足以发现中间字段错位，关键跨语言字段再核对偏移。
        expected_offsets = {
            (_CDriverConfig, "endpoint"): 24,
            (_CDriverConfig, "joint"): 216,
            (_CJointState, "torque_est_nm"): 16,
            (_CJointState, "control_enabled"): 32,
            (_CJointState, "feedback_age_us"): 44,
            (_CGroupState, "joints"): 16,
            (_CRuntimeProfile, "actual_protocol"): 32,
        }
        for (structure, field_name), expected_offset in expected_offsets.items():
            actual_offset = getattr(structure, field_name).offset
            if actual_offset != expected_offset:
                raise EncosError(
                    f"ABI offset mismatch for {structure.__name__}.{field_name}: "
                    f"{actual_offset} != {expected_offset}"
                )

    @staticmethod
    def _check(rc: int, operation: str) -> None:
        if rc != 0:
            status = STATUS_NAME.get(rc, "UNKNOWN")
            _get_fault_logger().error(
                "event=sdk_call_failed operation=%s status=%d status_name=%s "
                "detail=see_encos_driver.log_for_axis_command_and_limits",
                operation, rc, status)
            raise EncosError(f"{operation} failed with ENCOS status {rc} ({status}); "
                             f"details: {fault_log_directory() / 'encos_driver.log'}")

    def _build_config(self) -> _CDriverConfig:
        """把不可变 Python 配置复制到一次性的 C ABI 启动结构体。"""
        profiles = self._profiles
        if self._direction_config is not None:
            directions = load_joint_direction_config(self._direction_config)
            profiles = {
                name: replace(profile, direction=directions[name])
                for name, profile in self._profiles.items()
            }

        config = _CDriverConfig()
        config.struct_size = ctypes.sizeof(config)
        config.abi_version = ABI_VERSION
        config.control_period_us = self._period_us
        config.command_timeout_us = self._command_timeout_us
        config.feedback_timeout_us = self._feedback_timeout_us
        for index, endpoint in enumerate(self._endpoints):
            encoded = endpoint.encode("utf-8")
            if len(encoded) >= 64:
                raise ValueError("endpoint must be UTF-8 shorter than 64 bytes")
            config.endpoint[index].value = encoded
        for index, name in enumerate(JOINT_NAMES):
            profile = profiles[name]
            target = config.joint[index]
            protocol = profile.protocol
            target.protocol = _CProtocolRange(
                protocol.kp_min, protocol.kp_max, protocol.kd_min, protocol.kd_max,
                protocol.position_min_rad, protocol.position_max_rad,
                protocol.velocity_min_rad_s, protocol.velocity_max_rad_s,
                protocol.torque_min_nm, protocol.torque_max_nm,
                protocol.current_min_a, protocol.current_max_a,
            )
            target.physical_position_min_rad = profile.physical_position_min_rad
            target.physical_position_max_rad = profile.physical_position_max_rad
            target.physical_velocity_max_rad_s = profile.physical_velocity_max_rad_s
            target.physical_torque_max_nm = profile.physical_torque_max_nm
            target.physical_current_max_a = profile.physical_current_max_a
            target.application_position_min_rad = profile.application_position_min_rad
            target.application_position_max_rad = profile.application_position_max_rad
            target.application_velocity_max_rad_s = profile.application_velocity_max_rad_s
            target.application_torque_max_nm = profile.application_torque_max_nm
            target.application_current_max_a = profile.application_current_max_a
            target.kt_nm_per_a = profile.kt_nm_per_a
            target.static_profile_valid = 1
            target.model_spec_confirmed = int(profile.model_spec_confirmed)
            target.physical_limits_valid = int(profile.physical_limits_valid)
            target.application_limits_valid = int(profile.application_limits_valid)
            target.expected_can_timeout_ms = profile.expected_can_timeout_ms
            target.direction = profile.direction
            target.motor_zero_at_robot_zero_rad = profile.motor_zero_at_robot_zero_rad
            target.control_capabilities = profile.control_capabilities
            target.kinematic_mapping_valid = int(profile.kinematic_mapping_valid)
        return config

    def start(self) -> None:
        """提交配置并接入守护；已有组按需更新，未指定组保留原运行状态。"""
        config = self._build_config()
        self._check(self._lib.encos_start(ctypes.byref(config)), "encos_start")

    def stop(self) -> None:
        """生产客户端 so 只断开会话；仍使能的组可由守护超时接管。

        直接链接核心的 mock 库则会停止 Worker。需要停止控制发送时显式 disable；
        需要退出后保持时，不要先 disable 再 stop。
        """
        self._check(self._lib.encos_stop(), "encos_stop")

    def is_healthy(self) -> bool:
        """返回 Driver 和所有已配置 Worker 的健康状态。"""
        value = ctypes.c_uint32()
        self._check(
            self._lib.encos_is_healthy(ctypes.byref(value)),
            "encos_is_healthy",
        )
        return bool(value.value)

    def enable(self, group: str) -> None:
        """打开指定组的软件发送门，不会向电机写永久配置。"""
        group_id, _ = self._group_info(group)
        self._check(self._lib.encos_enable_group(group_id), "encos_enable_group")

    def disable(self, group: str) -> None:
        """关闭指定组的软件发送门，并清除该组未发送的缓存命令。"""
        group_id, _ = self._group_info(group)
        self._check(self._lib.encos_disable_group(group_id), "encos_disable_group")

    def emergency_stop(self) -> None:
        """
        同步关闭所有组的软件发送门。

        当前实现不主动发送制动或零力矩报文，不能替代硬件急停。
        """
        self._check(self._lib.encos_emergency_stop(), "encos_emergency_stop")

    def set_group_mit(self, group: str, commands: Mapping[str, MitCommand]) -> None:
        """校验完整组命令后，一次性提交该组所有关节的 MIT 目标。"""
        group_id, names = self._validate_group_command(group, commands)
        array = (_CMitCommand * len(names))(
            *(self._c_mit(commands[name]) for name in names)
        )
        self._check(
            self._lib.encos_set_group_mit(group_id, array, len(names)),
            f"encos_set_group_mit(group={group})",
        )

    def set_group_torque(self, group: str, torque_nm: Mapping[str, float]) -> None:
        """一次性提交整组独立力矩模式目标，单位 Nm。"""
        group_id, names = self._validate_group_command(group, torque_nm)
        array = (_CTorqueCommand * len(names))(
            *(_CTorqueCommand(torque_nm[name]) for name in names)
        )
        self._check(
            self._lib.encos_set_group_torque(group_id, array, len(names)),
            f"encos_set_group_torque(group={group})",
        )

    def set_group_servo_position(
        self,
        group: str,
        commands: Mapping[str, ServoPositionCommand],
    ) -> None:
        """一次性提交整组伺服位置目标，外部单位为 rad、rad/s、A。"""
        group_id, names = self._validate_group_command(group, commands)
        array = (_CServoPositionCommand * len(names))(
            *(self._c_servo_position(commands[name]) for name in names)
        )
        self._check(
            self._lib.encos_set_group_servo_position(group_id, array, len(names)),
            f"encos_set_group_servo_position(group={group})",
        )

    def set_group_servo_speed(
        self,
        group: str,
        commands: Mapping[str, ServoSpeedCommand],
    ) -> None:
        """一次性提交整组伺服速度目标，外部单位为 rad/s、A。"""
        group_id, names = self._validate_group_command(group, commands)
        array = (_CServoSpeedCommand * len(names))(
            *(self._c_servo_speed(commands[name]) for name in names)
        )
        self._check(
            self._lib.encos_set_group_servo_speed(group_id, array, len(names)),
            f"encos_set_group_servo_speed(group={group})",
        )

    def get_group_state(self, group: str) -> GroupState:
        """读取指定组的最新缓存快照，不发送新的 CAN 查询。"""
        group_id, names = self._group_info(group)
        raw = _CGroupState()
        self._check(
            self._lib.encos_get_group_state(group_id, ctypes.byref(raw)),
            "encos_get_group_state",
        )

        # C 端返回数量异常通常意味着 ABI 不匹配或内存被破坏，不能继续索引数组。
        if raw.joint_count != len(names) or raw.joint_count > len(raw.joints):
            raise EncosError(
                f"invalid joint_count from C driver: {raw.joint_count}, "
                f"expected {len(names)}"
            )

        joints = {
            name: self._joint_state(raw.joints[index])
            for index, name in enumerate(names)
        }
        return GroupState(
            group=group,
            snapshot_sequence=int(raw.snapshot_sequence),
            joints=joints,
        )

    @staticmethod
    def _state_query_codes(fields) -> tuple[int, ...]:
        mapping = {"position": 1, "velocity": 2, "current": 3}
        if isinstance(fields, str):
            fields = (fields,)
        try:
            values = tuple(fields)
        except TypeError as exc:
            raise ValueError("fields must contain position/velocity/current") from exc
        if not values:
            raise ValueError("fields cannot be empty")
        codes = []
        for field in values:
            if field not in mapping:
                raise ValueError(f"unsupported state field: {field!r}")
            if mapping[field] not in codes:
                codes.append(mapping[field])
        return tuple(codes)

    def read_joint_state(
        self,
        joint_name: str,
        *,
        fields=("position",),
        timeout_s: float = 0.1,
    ) -> JointState:
        """主动 Query 指定字段；允许阻塞，适合独立状态线程。"""
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and > 0")
        cached = self.get_joint_state(joint_name)
        if (cached.control_enabled and cached.online and
                not cached.feedback_timeout):
            return cached

        joint_id = self._joint_id(joint_name)
        for code in self._state_query_codes(fields):
            before = self.get_joint_state(joint_name).state_sequence
            self._check(
                self._lib.encos_request_joint_state_field(joint_id, code),
                "encos_request_joint_state_field",
            )
            deadline = time.monotonic() + timeout_s
            while True:
                state = self.get_joint_state(joint_name)
                if state.state_sequence != before:
                    break
                if time.monotonic() >= deadline:
                    raise EncosError(
                        f"state query timeout for {joint_name}, code={code}"
                    )
                time.sleep(0.0005)
        return self.get_joint_state(joint_name)

    def read_group_state(
        self,
        group: str,
        *,
        fields=("position",),
        timeout_s: float = 0.1,
    ) -> GroupState:
        """
        控制反馈新鲜时直接读缓存；空闲/手动拖动时逐轴主动 Query。
        默认只刷新 position，避免状态线程制造过多 CAN 流量。
        """
        cached = self.get_group_state(group)
        if all(
            item.control_enabled and item.online and
            not item.feedback_timeout and not item.command_timeout
            for item in cached.joints.values()
        ):
            return cached
        for name in self._group_info(group)[1]:
            self.read_joint_state(name, fields=fields, timeout_s=timeout_s)
        return self.get_group_state(group)

    def _set_joint_zero_raw(self, joint_name: str, timeout_s: float) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        timeout_us = int(timeout_s * 1_000_000)
        if timeout_us == 0 or timeout_us > 0xFFFFFFFF:
            raise ValueError("timeout_s is outside the supported microsecond range")
        self._check(
            self._lib.encos_set_joint_zero(self._joint_id(joint_name), timeout_us),
            "encos_set_joint_zero",
        )

    @staticmethod
    def _validate_zero_state(joint_name: str, state: JointState) -> None:
        if state.control_enabled:
            raise EncosError(
                f"{joint_name}: zeroing requires group disabled before calibration"
            )
        if not state.online or not state.feedback_valid or state.feedback_timeout:
            raise EncosError(f"{joint_name}: fresh feedback required before zeroing")
        if state.error != 0:
            raise EncosError(f"{joint_name}: motor error={state.error}; zeroing refused")
        if abs(state.velocity) > 0.05:
            raise EncosError(
                f"{joint_name}: velocity={state.velocity:+.4f} rad/s; "
                "motor must be stationary before zeroing"
            )

    def set_joint_zero(self, joint_name: str, timeout_s: float = 2.0) -> JointState:
        """
        把当前机械位置写为该电机内部零点（持久配置）。

        必须先 disable 对应 group。函数会主动读取位置/速度，确认电机静止、
        无故障后执行 V1.19 7.2 零点指令，并遵守至少 550 ms 的后置静默期。
        """
        state = self.read_joint_state(
            joint_name, fields=("position", "velocity"), timeout_s=min(timeout_s, 0.5)
        )
        self._validate_zero_state(joint_name, state)
        self._set_joint_zero_raw(joint_name, timeout_s)
        return self.read_joint_state(
            joint_name, fields=("position", "velocity"), timeout_s=min(timeout_s, 0.5)
        )

    def set_group_zero(self, group: str, timeout_s: float = 2.0) -> GroupState:
        """
        顺序将整组所有电机当前位置写为内部零点。

        每颗电机之间由 Driver 强制满足厂家要求的 >=550 ms 总线静默期。
        该操作会永久改变电机零点，必须在机械臂可靠支撑并且 group 已 disable 时执行。
        """
        names = self._group_info(group)[1]
        state = self.read_group_state(
            group, fields=("position", "velocity"), timeout_s=min(timeout_s, 0.5)
        )
        for name in names:
            self._validate_zero_state(name, state.joints[name])
        for name in names:
            self._set_joint_zero_raw(name, timeout_s)
        return self.read_group_state(
            group, fields=("position", "velocity"), timeout_s=min(timeout_s, 0.5)
        )

    def set_joint_mit(self, joint_name: str, command: MitCommand) -> None:
        """按英文 joint_name 提交单关节 MIT 命令。"""
        raw = self._c_mit(command)
        self._check(
            self._lib.encos_set_joint_mit(
                self._joint_id(joint_name), ctypes.byref(raw)
            ),
            f"encos_set_joint_mit(joint={joint_name})",
        )

    def set_joint_torque(self, joint_name: str, torque_nm: float) -> None:
        """按英文 joint_name 提交单关节力矩命令，单位 Nm。"""
        raw = _CTorqueCommand(torque_nm)
        self._check(
            self._lib.encos_set_joint_torque(
                self._joint_id(joint_name), ctypes.byref(raw)
            ),
            f"encos_set_joint_torque(joint={joint_name})",
        )

    def set_joint_servo_position(
        self,
        joint_name: str,
        command: ServoPositionCommand,
    ) -> None:
        """按英文 joint_name 提交单关节伺服位置命令。"""
        raw = self._c_servo_position(command)
        self._check(
            self._lib.encos_set_joint_servo_position(
                self._joint_id(joint_name), ctypes.byref(raw)
            ),
            f"encos_set_joint_servo_position(joint={joint_name})",
        )

    def set_joint_servo_speed(
        self,
        joint_name: str,
        command: ServoSpeedCommand,
    ) -> None:
        """按英文 joint_name 提交单关节伺服速度命令。"""
        raw = self._c_servo_speed(command)
        self._check(
            self._lib.encos_set_joint_servo_speed(
                self._joint_id(joint_name), ctypes.byref(raw)
            ),
            f"encos_set_joint_servo_speed(joint={joint_name})",
        )

    def get_joint_state(self, joint_name: str) -> JointState:
        """读取指定关节的最新缓存状态。"""
        raw = _CJointState()
        self._check(
            self._lib.encos_get_joint_state(
                self._joint_id(joint_name), ctypes.byref(raw)
            ),
            "encos_get_joint_state",
        )
        return self._joint_state(raw)

    def request_runtime_profile(self, joint_name: str) -> None:
        """异步发起电机实际参数查询；结果由 Worker 写入缓存。"""
        self._check(
            self._lib.encos_request_joint_runtime_profile(
                self._joint_id(joint_name)
            ),
            "encos_request_joint_runtime_profile",
        )

    def get_runtime_profile(self, joint_name: str) -> RuntimeProfile:
        """只读取已缓存的实际参数查询结果，不发送 CAN。"""
        raw = _CRuntimeProfile()
        raw.struct_size = ctypes.sizeof(raw)
        self._check(
            self._lib.encos_get_joint_runtime_profile(
                self._joint_id(joint_name), ctypes.byref(raw)
            ),
            "encos_get_joint_runtime_profile",
        )
        return self._runtime_profile(raw)

    def query_runtime_profile(
        self,
        joint_name: str,
        timeout_s: float = 1.0,
        poll_interval_s: float = 0.005,
    ) -> RuntimeProfile:
        """发起查询并轮询缓存，适合调试流程，不应放进高频控制回调。"""
        if (
            not math.isfinite(timeout_s)
            or not math.isfinite(poll_interval_s)
            or timeout_s <= 0.0
            or poll_interval_s <= 0.0
        ):
            raise ValueError("timeout_s and poll_interval_s must be finite and positive")

        self.request_runtime_profile(joint_name)
        deadline = time.monotonic() + timeout_s
        while True:
            profile = self.get_runtime_profile(joint_name)
            if profile.state != RuntimeProfileState.QUERYING:
                return profile
            if time.monotonic() >= deadline:
                raise EncosError(
                    f"runtime profile query timed out for {joint_name!r}"
                )
            time.sleep(poll_interval_s)

    def query_motor_id(self, group: str, timeout_s: float = 1.0) -> int:
        """
        在指定隔离总线上查询唯一上电电机的 CAN ID。

        多颗电机同时上电时可能同时应答，因此本接口不用于多电机扫描。
        """
        group_id, _ = self._group_info(group)
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        timeout_us = int(timeout_s * 1_000_000)
        if timeout_us == 0 or timeout_us > 0xFFFFFFFF:
            raise ValueError("timeout_s is outside the supported microsecond range")

        motor_id = ctypes.c_uint32()
        self._check(
            self._lib.encos_query_bus_motor_id(
                group_id, timeout_us, ctypes.byref(motor_id)
            ),
            "encos_query_bus_motor_id",
        )
        return int(motor_id.value)

    @staticmethod
    def _group_info(group: str) -> tuple[int, tuple[str, ...]]:
        """一次完成 group 合法性检查，并返回 C group_id 与固定关节顺序。"""
        try:
            return GROUPS[group], GROUP_JOINTS[group]
        except KeyError as error:
            raise ValueError(
                f"unknown group {group!r}; valid={tuple(GROUPS)}"
            ) from error

    @classmethod
    def _validate_group_command(
        cls,
        group: str,
        command: Mapping[str, object],
    ) -> tuple[int, tuple[str, ...]]:
        group_id, names = cls._group_info(group)
        if set(command) != set(names):
            raise ValueError(f"{group} command must contain exactly {names}")
        return group_id, names

    @staticmethod
    def _joint_id(joint_name: str) -> int:
        try:
            return JOINT_ID[joint_name]
        except KeyError as error:
            raise ValueError(f"unknown joint_name {joint_name!r}") from error

    @staticmethod
    def _c_mit(command: MitCommand) -> _CMitCommand:
        return _CMitCommand(
            command.position_rad,
            command.velocity_rad_s,
            command.kp,
            command.kd,
            command.torque_ff_nm,
        )

    @staticmethod
    def _c_servo_position(
        command: ServoPositionCommand,
    ) -> _CServoPositionCommand:
        return _CServoPositionCommand(
            command.position_rad,
            command.speed_limit_rad_s,
            command.current_limit_a,
        )

    @staticmethod
    def _c_servo_speed(command: ServoSpeedCommand) -> _CServoSpeedCommand:
        return _CServoSpeedCommand(
            command.velocity_rad_s,
            command.current_limit_a,
        )

    @staticmethod
    def _joint_state(raw: _CJointState) -> JointState:
        # 使用关键字构造，避免 C 字段新增或调序后出现难以察觉的位置参数错位。
        return JointState(
            position_rad=float(raw.position_rad),
            velocity_rad_s=float(raw.velocity_rad_s),
            phase_current_a=float(raw.phase_current_a),
            torque_est_nm=float(raw.torque_est_nm),
            motor_temperature_c=float(raw.motor_temperature_c),
            mos_temperature_c=float(raw.mos_temperature_c),
            motor_error_code=int(raw.motor_error_code),
            control_enabled=bool(raw.control_enabled),
            online=bool(raw.online),
            feedback_valid=bool(raw.feedback_valid),
            feedback_age_us=int(raw.feedback_age_us),
            state_sequence=int(raw.state_sequence),
            command_timeout=bool(raw.command_timeout),
            feedback_timeout=bool(raw.feedback_timeout),
        )

    @staticmethod
    def _runtime_profile(raw: _CRuntimeProfile) -> RuntimeProfile:
        protocol = raw.actual_protocol
        return RuntimeProfile(
            state=RuntimeProfileState(raw.state),
            received_mask=RuntimeField(raw.received_mask),
            mismatch_mask=RuntimeField(raw.mismatch_mask),
            query_failed_mask=RuntimeField(raw.query_failed_mask),
            motor_error_code=int(raw.motor_error_code),
            updated_age_us=int(raw.updated_age_us),
            actual_protocol=ProtocolRange(
                kp_min=float(protocol.kp_min),
                kp_max=float(protocol.kp_max),
                kd_min=float(protocol.kd_min),
                kd_max=float(protocol.kd_max),
                position_min_rad=float(protocol.position_min_rad),
                position_max_rad=float(protocol.position_max_rad),
                velocity_min_rad_s=float(protocol.velocity_min_rad_s),
                velocity_max_rad_s=float(protocol.velocity_max_rad_s),
                torque_min_nm=float(protocol.torque_min_nm),
                torque_max_nm=float(protocol.torque_max_nm),
                current_min_a=float(protocol.current_min_a),
                current_max_a=float(protocol.current_max_a),
            ),
            actual_kt_nm_per_a=float(raw.actual_kt_nm_per_a),
            actual_can_timeout_ms=int(raw.actual_can_timeout_ms),
        )

# ---------------------------------------------------------------------------
# Public Python SDK
# ---------------------------------------------------------------------------


def _find_bundled_library() -> Path:
    """优先加载与 encos_motion.py 同目录的 Driver 动态库。"""
    override = os.environ.get("ENCOS_DRIVER_LIBRARY")
    if override:
        return Path(override).expanduser().resolve(strict=True)

    root = Path(__file__).resolve().parent
    candidates = (
        root / "libencos_driver.so",
        root / "encos_driver.dll",
        root / "libencos_driver.dylib",
    )
    for path in candidates:
        if path.is_file():
            return path

    names = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(
        f"ENCOS driver library not found beside {Path(__file__).name}; expected one of: {names}"
    )


def _default_profiles(
    directions: Mapping[str, int] | None = None,
) -> dict[str, JointSafetyProfile]:
    """
    按装机轴表选择每个电机型号对应的厂家力控协议档案。

    机械/应用限位尚未冻结，因此只启用协议层范围，不把占位机械参数伪装成正式限位。
    directions 可只覆盖需要反向的关节；未提供的关节默认 +1。
    """
    direction_map = {name: 1 for name in JOINT_NAMES}
    if directions is not None:
        unknown = set(directions) - set(JOINT_NAMES)
        if unknown:
            raise ValueError(f"unknown direction joints: {sorted(unknown)}")
        for name, value in directions.items():
            if value not in (-1, 1):
                raise ValueError(f"direction for {name!r} must be +1 or -1")
            direction_map[name] = int(value)

    profiles = {}
    for name in JOINT_NAMES:
        protocol, kt_nm_per_a = MOTOR_PROTOCOL[JOINT_MODEL[name]]
        profiles[name] = JointSafetyProfile(
            protocol=protocol,
            physical_position_min_rad=0.0,
            physical_position_max_rad=0.0,
            physical_velocity_max_rad_s=0.0,
            physical_torque_max_nm=0.0,
            physical_current_max_a=0.0,
            application_position_min_rad=0.0,
            application_position_max_rad=0.0,
            application_velocity_max_rad_s=0.0,
            application_torque_max_nm=0.0,
            application_current_max_a=0.0,
            kt_nm_per_a=kt_nm_per_a,
            model_spec_confirmed=True,
            physical_limits_valid=False,
            application_limits_valid=False,
            expected_can_timeout_ms=500,
            direction=direction_map[name],
            motor_zero_at_robot_zero_rad=0.0,
            control_capabilities=CAPABILITY_ALL,
            kinematic_mapping_valid=False,
        )
    return profiles


def recommended_mit_gains(group: str):
    """返回厂家 Bring-up/现有 ST 基线；最终参数仍需实机整定。"""
    try:
        names = GROUP_JOINTS[group]
    except KeyError as error:
        raise ValueError(f"unknown group {group!r}; valid={tuple(GROUP_JOINTS)}") from error
    return ({name: RECOMMENDED_MIT_GAINS[name][0] for name in names},
            {name: RECOMMENDED_MIT_GAINS[name][1] for name in names})


def _group_values(
    group: str,
    value,
    field_name: str,
) -> tuple[float, ...]:
    """把 scalar / sequence / joint-name mapping 统一成固定关节顺序的 tuple。"""
    try:
        names = GROUP_JOINTS[group]
    except KeyError as error:
        raise ValueError(f"unknown group {group!r}; valid={tuple(GROUP_JOINTS)}") from error

    if isinstance(value, Mapping):
        if set(value) != set(names):
            raise ValueError(f"{field_name} mapping must contain exactly {names}")
        values = tuple(float(value[name]) for name in names)
    else:
        try:
            iterator = iter(value)
        except TypeError:
            values = (float(value),) * len(names)
        else:
            values = tuple(float(item) for item in iterator)
            if len(values) != len(names):
                raise ValueError(
                    f"{field_name} for {group} requires {len(names)} values, got {len(values)}"
                )

    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{field_name} must contain only finite values")
    return values


class EncosMotion:
    """
    运控正式 Public API。

    最小部署只需要：
        encos_motion.py
        libencos_driver.so

    两个文件放在同一个目录即可；library_path 通常无需传入。
    Python 只更新目标和读取缓存，真实周期收发由 C++ Worker 执行。
    """

    def __init__(
        self,
        *,
        left_leg: str | None = None,
        right_leg: str | None = None,
        arm: str | None = None,
        frequency_hz: float = 200.0,
        command_timeout_ms: float = 100.0,
        feedback_timeout_ms: float = 100.0,
        directions: Mapping[str, int] | None = None,
        library_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be finite and positive")
        if not math.isfinite(command_timeout_ms) or command_timeout_ms <= 0.0:
            raise ValueError("command_timeout_ms must be finite and positive")
        if not math.isfinite(feedback_timeout_ms) or feedback_timeout_ms <= 0.0:
            raise ValueError("feedback_timeout_ms must be finite and positive")

        period_us = int(round(1_000_000.0 / frequency_hz))
        if period_us <= 0:
            raise ValueError("frequency_hz is too high")

        endpoints = (
            left_leg or "",
            right_leg or "",
            arm or "",
        )
        if not any(endpoints):
            raise ValueError("configure at least one of left_leg/right_leg/arm")

        path = (
            _find_bundled_library()
            if library_path is None
            else Path(library_path).expanduser().resolve(strict=True)
        )

        self._frequency_hz = 1_000_000.0 / period_us
        self._driver = EncosDriver(
            library_path=path,
            profiles=_default_profiles(directions),
            endpoints=endpoints,
            control_period_us=period_us,
            command_timeout_us=int(round(command_timeout_ms * 1000.0)),
            feedback_timeout_us=int(round(feedback_timeout_ms * 1000.0)),
            direction_config=None,
        )

    @property
    def frequency_hz(self) -> float:
        return self._frequency_hz

    @property
    def raw_driver(self) -> EncosDriver:
        """高级诊断入口；普通运控不需要使用。"""
        return self._driver

    def start(self) -> "EncosMotion":
        self._driver.start()
        return self

    def stop(self) -> None:
        self._driver.stop()

    def __enter__(self) -> "EncosMotion":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def is_healthy(self) -> bool:
        return self._driver.is_healthy()

    def enable(self, group: str) -> None:
        self._driver.enable(group)

    def disable(self, group: str) -> None:
        self._driver.disable(group)

    def emergency_stop(self) -> None:
        self._driver.emergency_stop()

    @staticmethod
    def joint_names(group: str) -> tuple[str, ...]:
        try:
            return GROUP_JOINTS[group]
        except KeyError as error:
            raise ValueError(f"unknown group {group!r}; valid={tuple(GROUP_JOINTS)}") from error

    def set_joint_zero(self, joint: str, timeout_s: float = 2.0) -> JointState:
        """把当前关节位置写为电机内部零点；这是持久标定操作。"""
        return self._driver.set_joint_zero(joint, timeout_s=timeout_s)

    def set_group_zero(self, group: str, timeout_s: float = 2.0) -> GroupState:
        """顺序将整组当前位置写为电机内部零点；这是持久标定操作。"""
        return self._driver.set_group_zero(group, timeout_s=timeout_s)

    def set_joint_mit(
        self,
        joint: str,
        *,
        position: float,
        velocity: float = 0.0,
        kp: float = 0.0,
        kd: float = 0.0,
        torque: float = 0.0,
    ) -> None:
        self._driver.set_joint_mit(
            joint,
            MitCommand(
                position_rad=float(position),
                velocity_rad_s=float(velocity),
                kp=float(kp),
                kd=float(kd),
                torque_ff_nm=float(torque),
            ),
        )

    def set_group_mit(
        self,
        group: str,
        *,
        position,
        velocity=0.0,
        kp=0.0,
        kd=0.0,
        torque=0.0,
    ) -> None:
        names = self.joint_names(group)
        p = _group_values(group, position, "position")
        v = _group_values(group, velocity, "velocity")
        k_p = _group_values(group, kp, "kp")
        k_d = _group_values(group, kd, "kd")
        tau = _group_values(group, torque, "torque")
        self._driver.set_group_mit(
            group,
            {
                name: MitCommand(p[i], v[i], k_p[i], k_d[i], tau[i])
                for i, name in enumerate(names)
            },
        )

    def set_joint_torque(self, joint: str, torque: float) -> None:
        self._driver.set_joint_torque(joint, float(torque))

    def set_group_torque(self, group: str, torque) -> None:
        names = self.joint_names(group)
        tau = _group_values(group, torque, "torque")
        self._driver.set_group_torque(
            group,
            {name: tau[i] for i, name in enumerate(names)},
        )

    def set_joint_position(
        self,
        joint: str,
        *,
        position: float,
        speed_limit: float,
        current_limit: float,
    ) -> None:
        self._driver.set_joint_servo_position(
            joint,
            ServoPositionCommand(
                position_rad=float(position),
                speed_limit_rad_s=float(speed_limit),
                current_limit_a=float(current_limit),
            ),
        )

    def set_group_position(
        self,
        group: str,
        *,
        position,
        speed_limit,
        current_limit,
    ) -> None:
        names = self.joint_names(group)
        p = _group_values(group, position, "position")
        speed = _group_values(group, speed_limit, "speed_limit")
        current = _group_values(group, current_limit, "current_limit")
        self._driver.set_group_servo_position(
            group,
            {
                name: ServoPositionCommand(p[i], speed[i], current[i])
                for i, name in enumerate(names)
            },
        )

    def set_joint_velocity(
        self,
        joint: str,
        *,
        velocity: float,
        current_limit: float,
    ) -> None:
        self._driver.set_joint_servo_speed(
            joint,
            ServoSpeedCommand(
                velocity_rad_s=float(velocity),
                current_limit_a=float(current_limit),
            ),
        )

    def set_group_velocity(
        self,
        group: str,
        *,
        velocity,
        current_limit,
    ) -> None:
        names = self.joint_names(group)
        speed = _group_values(group, velocity, "velocity")
        current = _group_values(group, current_limit, "current_limit")
        self._driver.set_group_servo_speed(
            group,
            {
                name: ServoSpeedCommand(speed[i], current[i])
                for i, name in enumerate(names)
            },
        )

    def get_joint_state(self, joint: str) -> JointState:
        return self._driver.get_joint_state(joint)

    def get_group_state(self, group: str) -> GroupState:
        return self._driver.get_group_state(group)

    def read_joint_state(
        self,
        joint: str,
        *,
        fields=("position",),
        timeout_s: float = 0.1,
    ) -> JointState:
        return self._driver.read_joint_state(
            joint, fields=fields, timeout_s=timeout_s
        )

    def read_group_state(
        self,
        group: str,
        *,
        fields=("position",),
        timeout_s: float = 0.1,
    ) -> GroupState:
        return self._driver.read_group_state(
            group, fields=fields, timeout_s=timeout_s
        )

    def request_runtime_profile(self, joint: str) -> None:
        self._driver.request_runtime_profile(joint)

    def get_runtime_profile(self, joint: str) -> RuntimeProfile:
        return self._driver.get_runtime_profile(joint)

    def query_runtime_profile(
        self,
        joint: str,
        timeout_s: float = 1.0,
    ) -> RuntimeProfile:
        return self._driver.query_runtime_profile(joint, timeout_s=timeout_s)

    def query_motor_id(self, group: str, timeout_s: float = 1.0) -> int:
        return self._driver.query_motor_id(group, timeout_s=timeout_s)


__all__ = [
    "SDK_VERSION",
    "JOINT_CAN_ID",
    "JOINT_MODEL",
    "MOTOR_PROTOCOL",
    "RECOMMENDED_MIT_GAINS",
    "recommended_mit_gains",
    "fault_log_directory",
    "EncosMotion",
    "EncosError",
    "JointState",
    "GroupState",
    "RuntimeProfile",
    "RuntimeProfileState",
    "RuntimeField",
    "GROUP_JOINTS",
    "JOINT_NAMES",
]
