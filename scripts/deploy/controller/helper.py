from collections import deque
from dataclasses import dataclass

import numpy as np
import onnx
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

# -----------------------------------------------------------------------------
#                              Observation Buffer
# -----------------------------------------------------------------------------


class ObsBuffer:
    """
    观测缓冲区：预定义 obs_names，自动填充历史，支持拼接或字典输出

    Args:
        obs_names: 观测列表，如 ["base_ang_vel", "projected_gravity", ...]
        history_length: 历史帧数, 包含当前帧
        concatenate: get_obs 时是否拼接为单个向量

    使用:
        buffer = ObsBuffer(
            obs_names=["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "last_action"],
            history_length=5,
            concatenate=True
        )
        buffer.push(env.step(action))
        obs = buffer.get_obs()  # 根据 concatenate 返回向量或字典
    """

    def __init__(self, obs_names: list, history_length: int = 0, concatenate: bool = True):
        self.obs_names = list(obs_names)
        self.history_length = history_length
        self.concatenate = concatenate

        # 初始化缓冲区：每个 name 一个固定长度 deque
        self._buffers = {k: deque(maxlen=history_length) for k in self.obs_names}
        self._shapes = {}  # 记录每个 name 的单帧 shape
        self._initialized = {k: False for k in self.obs_names}
        self._ready = False

    def push(self, obs_dict: dict[str, np.ndarray | list | tuple]):
        # 检查缺失的 key，用 assert 或 raise
        missing = set(self.obs_names) - obs_dict.keys()
        assert not missing, f"obs_dict missing keys: {missing}"

        for name in self.obs_names:
            val = np.asarray(obs_dict[name])

            if not self._initialized[name]:
                self._shapes[name] = val.shape
                for _ in range(self.history_length):
                    self._buffers[name].append(val.copy())
                self._initialized[name] = True
            else:
                self._buffers[name].append(val)

        self._ready = all(self._initialized.values())

    def get_obs(self):
        """
        获取观测
        concatenate=True:  返回拼接向量 (total_dim,)
        concatenate=False: 返回字典，每个值 (history_length+1, *shape)
        """
        if not self._ready:
            raise RuntimeError("Buffer 未就绪，所有 obs_names 至少需要 push 一次")

        if self.concatenate:
            parts = []
            for name in self.obs_names:
                stacked = np.stack(self._buffers[name], axis=0)  # (history, *shape)
                parts.append(stacked.reshape(-1))  # (history * dim,)
            return np.concatenate(parts, dtype=np.float32)  # (total_dim,)
        else:
            return {name: np.stack(self._buffers[name], axis=0, dtype=np.float32) for name in self.obs_names}

    @property
    def obs_dim(self):
        """拼接后的总维度（需要至少 push 一次后才能计算）"""
        if not self._ready:
            return None
        total = 0
        for name in self.obs_names:
            total = self.history_length * np.prod(self._shapes[name], dtype=int)
        return total

    def reset(self):
        """清空缓冲区"""
        self._buffers = {k: deque(maxlen=self.history_length) for k in self.obs_names}
        self._shapes = {}
        self._initialized = {k: False for k in self.obs_names}
        self._ready = False

    def is_ready(self):
        return self._ready


# -----------------------------------------------------------------------------
#                               Onnx Policy
# -----------------------------------------------------------------------------


def parse_ndarray(csv_str: str | None, delimiter: str = ",") -> np.ndarray:
    if csv_str:
        return np.fromstring(csv_str, sep=delimiter, dtype=float)
    else:
        return np.array([])


def parse_str_list(csv_str: str | None, delimiter: str = ",") -> list[str]:
    if csv_str:
        return [x.strip() for x in csv_str.split(delimiter) if x.strip() != ""]
    else:
        return []


def normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norm = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    return x / np.clip(norm, a_min=eps, a_max=None)


def yaw_quat(quat: np.ndarray) -> np.ndarray:
    """Extract the yaw component of a quaternion.

    Args:
        quat: The orientation in (w, x, y, z). Shape is (..., 4)

    Returns:
        A quaternion with only yaw component.
    """
    shape = quat.shape
    quat_yaw = quat.reshape(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw = np.zeros_like(quat_yaw)
    quat_yaw[:, 3] = np.sin(yaw / 2)
    quat_yaw[:, 0] = np.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.reshape(shape)


class OnnxPolicy:
    def __init__(self, onnx_policy_path, is_real_env: bool = True, use_gpu: bool = False):
        self.onnx_policy_path = onnx_policy_path
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4  # 2~4
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        options.enable_mem_pattern = is_real_env
        options.enable_cpu_mem_arena = is_real_env

        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_policy_path, sess_options=options, providers=providers)

        # Load metadata
        onnx_model = onnx.load(self.onnx_policy_path)
        metadata = {prop.key: prop.value for prop in onnx_model.metadata_props}
        self.command_name = metadata.get("command_names", "base_velocity")
        joint_names = parse_str_list(metadata.get("joint_names"))
        joint_stiffness = parse_ndarray(metadata.get("joint_stiffness"))
        joint_damping = parse_ndarray(metadata.get("joint_damping"))
        default_joint_pos = parse_ndarray(metadata.get("default_joint_pos"))
        action_scale = parse_ndarray(metadata.get("action_scale"))

        # Used to create env
        try:
            self.action_joint_cfg = [
                ActionJointCfg(
                    joint_name=joint_names[i],
                    default_joint_pos=default_joint_pos[i],
                    kp=joint_stiffness[i],
                    kd=joint_damping[i],
                    scale=action_scale[i],
                )
                for i in range(len(joint_names))
            ]
        except Exception:
            print(
                "[Error]: Please check your onnx export, should include "
                "[command_name, joint_names, joint_stiffness, joint_damping, default_joint_pos, action_scale] "
                "in the onnx model metadata_props!"
            )
            exit()

        # Command
        if self.command_name == "motion":
            self.motion_anchor_body_name = metadata.get("anchor_body_name", "base_link")
            self.motion_body_names = parse_str_list(metadata.get("body_names"))
            self.motion_step_t = 0
            motion_keys = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]
            self.motion = {}
            for init in onnx_model.graph.initializer:
                name = init.name.split(".")[0]
                if name in motion_keys:
                    self.motion[name] = onnx.numpy_helper.to_array(init)
            motion_anchor_body_idx = self.motion_body_names.index(self.motion_anchor_body_name)
            self.motion_joint_pos = self.motion["joint_pos"].copy()
            self.motion_joint_vel = self.motion["joint_vel"].copy()
            self.motion_anchor_pos_w = self.motion["body_pos_w"][:, motion_anchor_body_idx].copy()
            self.motion_anchor_quat_w = self.motion["body_quat_w"][:, motion_anchor_body_idx].copy()
        else:
            self.vel_x, self.vel_y, self.ang_vel_z = 0.0, 0.0, 0.0

    def get_action(self, obs):
        if self.command_name == "motion":
            return self.session.run(
                None,
                {
                    self.session.get_inputs()[0].name: obs.reshape(1, -1),
                    self.session.get_inputs()[1].name: np.array([[self.motion_step_t]], dtype=np.float32),
                },
            )[0].squeeze()
        else:
            return self.session.run(None, {self.session.get_inputs()[0].name: obs.reshape(1, -1)})[0].squeeze()

    def get_command(self, robot_quat=None):
        if self.command_name == "motion":
            if robot_quat is None:
                raise ValueError("robot_quat is needed when command is motion!")
            step_t = min(self.motion["joint_pos"].shape[0] - 1, self.motion_step_t)
            robot_rot = R.from_quat(robot_quat, scalar_first=True)
            motion_rot = R.from_quat(self.motion_anchor_quat_w[step_t], scalar_first=True)
            delta_rot = robot_rot.inv() * motion_rot
            motion_command = {
                "motion_command": np.concatenate(
                    [self.motion_joint_pos[step_t], self.motion_joint_vel[step_t]], axis=-1
                ),
                "motion_anchor_pos_w": self.motion_anchor_pos_w[step_t],
                "motion_anchor_quat_w": self.motion_anchor_quat_w[step_t],
                "motion_anchor_ori_b": delta_rot.as_matrix()[..., :2].reshape(-1),
                "ref_motion": {
                    "base_pos": self.motion_anchor_pos_w[step_t],
                    "base_quat": self.motion_anchor_quat_w[step_t],
                    "joint_pos": self.motion_joint_pos[step_t],
                },
            }
            self.motion_step_t += 1
            return motion_command
        else:
            return {"velocity_command": np.array([self.vel_x, self.vel_y, self.ang_vel_z])}

    def reset(self, robot_pos=None, robot_quat=None):
        # robot_pos (xyz), robot_quat (wxyz): used to compute mition command
        if self.command_name == "motion":
            self.motion_step_t = 0
            robot_rot = R.from_quat(robot_quat, scalar_first=True)
            motion_anchor_rot = R.from_quat(self.motion_anchor_quat_w[self.motion_step_t], scalar_first=True)
            delta_rot = R.from_quat(
                yaw_quat((robot_rot * motion_anchor_rot.inv()).as_quat(scalar_first=True)), scalar_first=True
            )
            self.motion_anchor_pos_w[:, :2] -= self.motion_anchor_pos_w[self.motion_step_t, :2]
            self.motion_anchor_pos_w = delta_rot.apply(self.motion_anchor_pos_w)
            self.motion_anchor_pos_w[:, :2] += robot_pos[:2]
            self.motion_anchor_quat_w = (delta_rot * R.from_quat(self.motion_anchor_quat_w, scalar_first=True)).as_quat(
                scalar_first=True
            )
        else:
            self.vel_x, self.vel_y, self.ang_vel_z = 1.0, 0.0, 0.0


@dataclass
class ActionJointCfg:
    joint_name: str
    default_joint_pos: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    scale: float = 1.0
    clip: tuple[float, float] | None = None
