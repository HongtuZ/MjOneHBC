import base64
import io
from collections import deque
from dataclasses import dataclass

import joblib
import numpy as np
import onnx
import onnxruntime as ort


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


class OnnxPolicy:
    def __init__(
        self, onnx_policy_path, is_velocity_command: bool = True, is_real_env: bool = True, use_gpu: bool = False
    ):
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

        # Command
        self.is_velocity_command = is_velocity_command
        self.vel_x, self.vel_y, self.ang_vel_z = 1.0, 0.0, 0.0
        self.step_t, self.ref_motion_length = 0, 0
        self.ref_motion = self.load_ref_motion()
        if self.ref_motion:
            self.ref_motion_length = len(self.ref_motion[0])

    def get_action(self, obs):
        return self.session.run(None, {self.session.get_inputs()[0].name: obs.reshape(1, -1)})[0].squeeze()

    def get_command(self):
        if self.is_velocity_command:
            return {"velocity_command": np.array([self.vel_x, self.vel_y, self.ang_vel_z])}
        else:
            step_t = min(self.ref_motion_length - 1, self.step_t)
            motion_command = {
                "ref_joint_pos": self.ref_motion["joint_pos"][step_t],
                "ref_joint_vel": self.ref_motion["joint_vel"][step_t],
                "ref_base_pos_b": self.ref_motion["base_pos_b"][step_t],
            }
            self.step_t += 1
            return motion_command

    def reset(self):
        self.vel_x, self.vel_y, self.ang_vel_z = 1.0, 0.0, 0.0
        self.step_t = 0

    def load_ref_motion(self):
        onnx_model = onnx.load(self.onnx_policy_path)
        for prop in onnx_model.metadata_props:
            if prop.key == "ref_motion":
                buffer = io.BytesIO(base64.b64decode(prop.value.encode("ascii")))
                return joblib.load(buffer)
        return None


@dataclass
class ActionJointCfg:
    joint_name: str
    default_joint_pos: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    scale: float = 1.0
    clip: tuple[float, float] | None = None
