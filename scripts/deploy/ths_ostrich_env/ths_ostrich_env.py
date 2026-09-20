import time
from typing import Mapping

import numpy as np
from controller.helper import ActionJointCfg, normalize
from scipy.spatial.transform import Rotation as R

from .encos_motion import EncosMotion
from .imu import IMUReader

# ═══════════════════════════════════════════════════════════════════
# EncosMotion 固定分组：left_leg(5) + right_leg(5)。
#
# 策略(ONNX metadata)使用 URDF 关节名（带 _joint 后缀），EncosMotion
# 使用运控侧关节名（knee/ankle 带 _pitch）。两边只是命名习惯不同，
# 这里显式按**关节名字**建立映射，不依赖数组顺序或 CAN id，
# 因此策略关节顺序变化、SDK 组内顺序调整都不会造成错位。
# ═══════════════════════════════════════════════════════════════════
GROUPS = ("left_leg", "right_leg")

# 策略(URDF / ONNX metadata)关节名 -> EncosMotion SDK 关节名
POLICY_TO_SDK_JOINT: dict[str, str] = {
    "left_hip_pitch_joint": "left_hip_pitch",
    "left_hip_roll_joint": "left_hip_roll",
    "left_hip_yaw_joint": "left_hip_yaw",
    "left_knee_joint": "left_knee_pitch",
    "left_ankle_joint": "left_ankle_pitch",
    "right_hip_pitch_joint": "right_hip_pitch",
    "right_hip_roll_joint": "right_hip_roll",
    "right_hip_yaw_joint": "right_hip_yaw",
    "right_knee_joint": "right_knee_pitch",
    "right_ankle_joint": "right_ankle_pitch",
}

JOINT_DIRECTIONS: dict[str, int] = {
    "left_hip_yaw": -1,
    "right_hip_pitch": -1,
    "right_hip_yaw": -1,
    "right_knee_pitch": -1,
    "right_ankle_pitch": -1,
}


def _fmt_values(values) -> str:
    """把标量或一组数值格式化成一行，最多保留 3 位小数。"""
    arr = np.atleast_1d(np.asarray(values, dtype=float)).reshape(-1)
    return "[" + ", ".join(f"{v:+.3f}" for v in arr) + "]"


class ThsOstrichEnv:
    def __init__(
        self,
        physic_dt: float,
        decimation: int,
        action_joint_cfg: list[ActionJointCfg],
        imu_type: str = "yis320",
        left_leg_endpoint: str = "can0",
        right_leg_endpoint: str = "can1",
        frequency_hz: float = 200.0,
        command_timeout_ms: float = 100.0,
        feedback_timeout_ms: float = 100.0,
        directions: Mapping[str, int] | None = JOINT_DIRECTIONS,
        library_path: str | None = None,
        read_only: bool = False,
    ):
        # read_only: 只读测试模式。不打开电机发送门，且任何发送路径都会直接报错，
        #            用于只采集电机 / IMU 数据、观察策略输出与实际位置的偏差。
        self.read_only = read_only
        self.enable_control = False
        # 摔倒判定阈值：机体倾角超过该角度(度)即视为 done
        self.max_tilt_angle_deg = 70.0
        self.dt = physic_dt
        self.decimation = decimation
        self.action_joint_names = [aj_cfg.joint_name for aj_cfg in action_joint_cfg]

        self.default_joint_pos = np.array([aj_cfg.default_joint_pos for aj_cfg in action_joint_cfg], dtype=float)
        self.kp = np.array([aj_cfg.kp for aj_cfg in action_joint_cfg], dtype=float)
        self.kd = np.array([aj_cfg.kd for aj_cfg in action_joint_cfg], dtype=float)
        self.action_scale = np.array([aj_cfg.scale for aj_cfg in action_joint_cfg], dtype=float)
        self.action_clip = np.array(
            [(-np.inf, np.inf) if not aj_cfg.clip else aj_cfg.clip for aj_cfg in action_joint_cfg], dtype=float
        )

        # ── 按关节名字建立 策略索引 <-> EncosMotion 分组 的映射 ──
        policy_index = {name: i for i, name in enumerate(self.action_joint_names)}
        if len(policy_index) != len(self.action_joint_names):
            raise ValueError(f"策略关节名存在重复: {self.action_joint_names}")

        unknown = [name for name in self.action_joint_names if name not in POLICY_TO_SDK_JOINT]
        if unknown:
            raise ValueError(f"策略关节名缺少 EncosMotion 映射: {unknown}")

        sdk_to_policy_index = {
            POLICY_TO_SDK_JOINT[name]: policy_index[name] for name in self.action_joint_names
        }

        # group -> 组内按 EncosMotion 顺序排列的策略索引
        self._group_policy_indices: dict[str, np.ndarray] = {}
        covered: set[str] = set()
        for group in GROUPS:
            sdk_names = EncosMotion.joint_names(group)
            missing = [name for name in sdk_names if name not in sdk_to_policy_index]
            if missing:
                raise ValueError(f"策略缺少 EncosMotion {group} 组关节: {missing}")
            self._group_policy_indices[group] = np.array(
                [sdk_to_policy_index[name] for name in sdk_names], dtype=int
            )
            covered.update(sdk_names)

        uncovered = [
            name for name in self.action_joint_names
            if POLICY_TO_SDK_JOINT[name] not in covered
        ]
        if uncovered:
            raise ValueError(f"策略关节未被任何 EncosMotion 组覆盖: {uncovered}")

        # ── 建立 EncosMotion 连接（左右腿分别映射到 can0 / can1）──
        self.encos = EncosMotion(
            left_leg=left_leg_endpoint,
            right_leg=right_leg_endpoint,
            frequency_hz=frequency_hz,
            command_timeout_ms=command_timeout_ms,
            feedback_timeout_ms=feedback_timeout_ms,
            directions=directions,
            library_path=library_path,
        )
        self.encos.start()
        if self.read_only:
            print("[ENV] 只读模式：不打开电机发送门，全程不会向电机发送任何指令")
        else:
            for group in GROUPS:
                self.encos.enable(group)

        self.imu = IMUReader(cpu_id=0, imu_type=imu_type)

    # ── 内部收发 ─────────────────────────────────────────────────

    def _send_position_command(self, positions: np.ndarray) -> None:
        """positions 为策略顺序的目标关节位置，按关节名映射后分组下发 MIT 命令。"""
        if self.read_only:
            raise RuntimeError("只读测试模式下禁止向电机发送指令")
        for group, policy_indices in self._group_policy_indices.items():
            self.encos.set_group_mit(
                group,
                position=positions[policy_indices],
                kp=self.kp[policy_indices],
                kd=self.kd[policy_indices],
            )

    def _read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """读取关节位置/速度，按关节名映射回策略顺序。

        只读模式下电机未使能，缓存可能不新鲜，改用 read_group_state：
        缓存新鲜时直接读，否则由 Driver 主动发起 State Query（只读查询，不是控制指令）。
        """
        qpos = np.zeros(len(self.action_joint_names), dtype=np.float32)
        qvel = np.zeros(len(self.action_joint_names), dtype=np.float32)
        for group, policy_indices in self._group_policy_indices.items():
            if self.read_only:
                state = self.encos.read_group_state(group, fields=("position", "velocity"))
            else:
                state = self.encos.get_group_state(group)
            qpos[policy_indices] = state.position
            qvel[policy_indices] = state.velocity
        return qpos, qvel

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """展平动作，并按各关节的 clip 上下限裁剪。"""
        action = np.asarray(action, dtype=float).reshape(-1)
        return np.clip(action, self.action_clip[:, 0], self.action_clip[:, 1])

    def _build_obs_info(self, action: np.ndarray, qpos: np.ndarray, qvel: np.ndarray) -> dict:
        """按策略观测名组装观测字典。"""
        base_quat = np.array(self.imu.quaternion, dtype=np.float32)
        base_ang_vel = np.array(self.imu.gyro, dtype=np.float32)
        base_rot_inv = R.from_quat(base_quat, scalar_first=True).inv()
        projected_gravity = normalize(base_rot_inv.apply(np.array([0, 0, -9.81])))
        return {
            "robot_pos": np.zeros(3),
            "robot_quat": base_quat,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "joint_pos": qpos - self.default_joint_pos,
            "joint_vel": qvel,
            "last_action": action,
        }

    def read_only_step(self, action: np.ndarray):
        """只读测试：不向电机发送任何指令，只读取电机 / IMU 状态。

        返回 (obs_info, extra)，extra 含策略目标关节位置、当前电机位置与两者差值，
        用于观察策略输出与真实电机位置的偏差。
        """
        action = self._clip_action(action)
        target = action * self.action_scale + self.default_joint_pos
        qpos, qvel = self._read_joint_state()
        extra = {
            "target_position": target,
            "joint_position": qpos,
            "joint_velocity": qvel,
            "position_error": target - qpos,
        }
        return self._build_obs_info(action, qpos, qvel), extra

    def step(
        self,
        action: np.ndarray,
    ):
        action = self._clip_action(action)
        target = action * self.action_scale + self.default_joint_pos
        for _ in range(self.decimation):
            start_time = time.perf_counter()
            if self.enable_control:
                self._send_position_command(target)
            duration = time.perf_counter() - start_time
            time.sleep(max(0, self.dt - duration))

        qpos, qvel = self._read_joint_state()
        obs_info = self._build_obs_info(action, qpos, qvel)
        # self._print_step_debug(obs_info, qpos, target)
        done = self.should_done(obs_info)
        return obs_info, done

    def should_done(self, obs_info: dict) -> bool:
        """根据 projected_gravity 判断机体是否倾倒。

        projected_gravity 是单位化的重力在机体系下的投影，直立时为 (0, 0, -1)，
        倾角即它与 (0, 0, -1) 的夹角：tilt = arccos(-g_z)。
        倾角超过 max_tilt_angle_deg(70°) 视为摔倒，返回 True。
        """
        gravity = np.asarray(obs_info["projected_gravity"], dtype=float).reshape(-1)
        if gravity.shape[0] < 3:
            raise ValueError(f"projected_gravity 维度不足: {gravity.shape}")
        cos_tilt = np.clip(-gravity[2] / (np.linalg.norm(gravity) + 1e-12), -1.0, 1.0)
        tilt_angle_deg = float(np.degrees(np.arccos(cos_tilt)))
        if tilt_angle_deg > self.max_tilt_angle_deg:
            print(f"[ENV] 机体倾角 {tilt_angle_deg:.1f}° 超过阈值 {self.max_tilt_angle_deg}°，判定摔倒")
            return True
        return False

    def _print_step_debug(self, obs_info: dict, qpos: np.ndarray, target: np.ndarray) -> None:
        """打印本步观测与关节 目标/实际 对比：每个观测 key 一行，每个关节一行。"""
        for key, value in obs_info.items():
            print(f"[STEP-OBS] {key} = {_fmt_values(value)}")
        print(f"[STEP-JOINT] {'joint':<26}{'qpos':>10}{'target':>10}")
        for i, name in enumerate(self.action_joint_names):
            print(f"[STEP-JOINT] {name:<26}{qpos[i]:>10.3f}{target[i]:>10.3f}")

    def reset(self, target_pos=None):
        self.enable_control = False
        if self.read_only:
            # 只读测试模式：不发送归位指令，直接以当前状态作为初始观测
            print("[ENV] 只读模式：跳过归位，不发送任何电机指令")
            qpos, qvel = self._read_joint_state()
            return self._build_obs_info(np.zeros(len(self.default_joint_pos)), qpos, qvel)
        # 归位目标：默认 default_pos，可传入自定义目标（如回放首帧 target）
        if target_pos is None:
            target_pos = self.default_joint_pos
        target_pos = np.asarray(target_pos, dtype=float).reshape(-1)
        if target_pos.shape[0] != len(self.default_joint_pos):
            raise ValueError(
                f"target_pos 维度不匹配: {target_pos.shape[0]} vs {len(self.default_joint_pos)}"
            )
        # ═══════ 按最大速度平滑渐变到 target_pos ═══════
        # 预定义最大角速度 (rad/s)，根据你的电机性能调整
        MAX_TRANSITION_VELOCITY = 0.5  # 例如 1 rad/s ≈ 57 deg/s
        # 计算每个电机的角度差异
        init_pos, _ = self._read_joint_state()
        angle_diffs = np.abs(target_pos - init_pos)
        max_diff = np.max(angle_diffs)

        # 计算所需过渡时间（最大差异 / 最大速度）
        if max_diff < 0.001:  # 已经到位，无需过渡
            transition_time = 0.0
            transition_steps = 1
        else:
            transition_time = max_diff / MAX_TRANSITION_VELOCITY
            # 至少 50ms，最多 3s
            transition_time = np.clip(transition_time, 0.05, 3.0)
            transition_steps = max(int(transition_time / self.dt), 1)

        print(f"最大角度差异: {max_diff:.2f}弧度, 重置时间: {transition_time:.2f}s, 步数: {transition_steps}")
        for step in range(transition_steps):
            alpha = step / transition_steps
            target = init_pos * (1.0 - alpha) + target_pos * alpha
            self._send_position_command(target)
            time.sleep(self.dt)
        print("已进入目标姿态")

        qpos, qvel = self._read_joint_state()
        return self._build_obs_info(np.zeros(len(self.default_joint_pos)), qpos, qvel)
    
    def close(self):
        try:
            self.encos.stop()
        finally:
            self.imu.close()
        time.sleep(0.5)
