import time

import numpy as np
from controller.helper import ActionJointCfg, normalize
from scipy.spatial.transform import Rotation as R

from .imu import IMUReader
from .motor_config import THS_MOTORS
from .motor_driver import MotorDriver


class ThsRealEnv:
    def __init__(
        self,
        physic_dt: float,
        decimation: int,
        action_joint_cfg: list[ActionJointCfg],
        imu_type: str = "wit",
    ):
        self.enable_control = False
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

        # 设置关节电机参数
        for motor_config in THS_MOTORS:
            joint_name = motor_config.joint_name
            idx = self.action_joint_names.index(joint_name)
            motor_config.default_pos = self.default_joint_pos[idx]
            motor_config.default_kp = self.kp[idx]
            motor_config.default_kd = self.kd[idx]

        self.motor_driver = MotorDriver(THS_MOTORS, joint_order=self.action_joint_names)
        self.imu = IMUReader(cpu_id=0, imu_type=imu_type)

    def step(
        self,
        action: np.ndarray,
    ):
        action = action.reshape(-1)
        action = np.clip(action, self.action_clip[:, 0], self.action_clip[:, 1])
        target = action * self.action_scale
        target += self.default_joint_pos
        start_time = time.perf_counter()
        for _ in range(self.decimation):
            if self.enable_control:
                self.motor_driver.set_ctrl(pos=target)
            duration = time.perf_counter() - start_time
            time.sleep(max(0, self.dt - duration))

        # return info
        base_quat = np.array(self.imu.quaternion, dtype=np.float32)
        base_ang_vel = np.array(self.imu.gyro, dtype=np.float32)
        base_rot_inv = R.from_quat(base_quat, scalar_first=True).inv()
        projected_gravity = base_rot_inv.apply(np.array([0, 0, -9.81]))
        projected_gravity = normalize(projected_gravity)
        obs_info = {
            "robot_pos": np.zeros(3),
            "robot_quat": base_quat,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "joint_pos": self.motor_driver.qpos - self.default_joint_pos,
            "joint_vel": self.motor_driver.qvel,
            "last_action": action,
        }
        done = False
        return obs_info, done

    def reset(self):
        self.enable_control = False
        # ═══════ 按最大速度平滑渐变到 default_pos ═══════
        # 预定义最大角速度 (rad/s)，根据你的电机性能调整
        MAX_TRANSITION_VELOCITY = 0.5  # 例如 1 rad/s ≈ 57 deg/s
        # 计算每个电机的角度差异
        init_pos = self.motor_driver.qpos
        angle_diffs = np.abs(self.default_joint_pos - init_pos)
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
            target = init_pos * (1.0 - alpha) + self.default_joint_pos * alpha
            self.motor_driver.set_ctrl(pos=target)
        print("已进入默认姿态")

        base_quat = np.array(self.imu.quaternion, dtype=np.float32)
        base_ang_vel = np.array(self.imu.gyro, dtype=np.float32)
        base_rot = R.from_quat(base_quat, scalar_first=True)
        projected_gravity = base_rot.inv().apply(np.array([0, 0, -9.81]))
        projected_gravity = projected_gravity / np.linalg.norm(projected_gravity)
        return {
            "robot_pos": np.zeros(3),
            "robot_quat": base_quat,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "joint_pos": self.motor_driver.qpos - self.default_joint_pos,
            "joint_vel": self.motor_driver.qvel,
            "last_action": np.zeros(len(self.default_joint_pos)),
        }

    def close(self):
        self.motor_driver.close()
        self.imu.close()
        time.sleep(0.5)
