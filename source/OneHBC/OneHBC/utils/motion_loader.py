import joblib
import torch
from pathlib import Path
from collections import defaultdict
from mjlab.utils.lab_api import math as math_utils
from dataclasses import dataclass


@dataclass
class Motion:
    time_step_total: int
    joint_pos: torch.Tensor  # (num_frames, num_joints)
    joint_vel: torch.Tensor  # (num_frames, num_joints)
    body_pos_w: torch.Tensor  # (num_frames, 3)
    body_quat_w: torch.Tensor  # (num_frames, 4) wxyz
    body_lin_vel_w: torch.Tensor  # (num_frames, 3)
    body_ang_vel_w: torch.Tensor  # (num_frames, 3)


class MotionLoader:
    def __init__(
        self,
        motion_data_dir: str | None = None,
        motion_data_weights: dict[str, float] | None = None,
        device: str = "cpu",
    ):
        self.motion_data_dir = motion_data_dir
        self.motion_data_weights = motion_data_weights
        self.joint_names = None
        self.joint_name2idx = {}
        self.body_names = None
        self.body_name2idx = {}
        self.device = device
        if self.motion_data_dir is not None:
            self._load_motion_data()

    def _load_motion_data(self):
        motion_data_dir = Path(self.motion_data_dir)
        if not motion_data_dir.exists():
            raise ValueError(f"Motion data directory {str(motion_data_dir)} does not exist.")
        motion_files = list(motion_data_dir.rglob("*.pkl"))
        if len(motion_files) == 0:
            raise ValueError(f"No motion data files with .pkl extension found in {str(motion_data_dir)}")
        motion_name2path = {p.stem: p for p in motion_files}

        if self.motion_data_weights is None:
            print("⚠️ Did not specify the motion data weights, load all with weight 1.0!")
            self.motion_data_weights = {f.stem: 1.0 for f in motion_files}

        # Load motion data
        self.motion_durations = []
        self.motion_fps = []
        self.motion_dt = []
        self.motion_num_frames = []
        self.motion_weights = []

        self.root_pos_w = []
        self.root_quat_w = []  # wxyz
        self.root_lin_vel_w = []
        self.root_ang_vel_w = []
        self.body_pos_w = []
        self.body_quat_w = []  # wxyz
        self.body_lin_vel_w = []
        self.body_ang_vel_w = []
        self.body_pos_b = []
        self.body_quat_b = []  # wxyz
        self.joint_pos = []
        self.joint_vel = []

        # only load the motion data files that are in the motion weights dict
        for motion_name, motion_weight in self.motion_data_weights.items():
            if motion_weight <= 0:
                continue
            # check if the motion file name is valid
            if motion_name not in motion_name2path.keys():
                raise ValueError(
                    f"Motion name {motion_name} defined in motion weights not found in motion data directory {str(motion_data_dir)}. Available names: {motion_name2path.keys()}"
                )

            # load the motion data file
            motion_path = motion_name2path[motion_name]
            print(f"[Motion Data Manager] Loading motion data from {str(motion_path)}...")
            motion_raw_data = joblib.load(str(motion_path))
            if not isinstance(motion_raw_data, dict):
                raise ValueError(f"Motion data file {str(motion_path)} does not contain a valid dictionary.")

            num_frames = len(motion_raw_data["root_pos_w"])
            if num_frames < 2:
                raise ValueError(f"[MotionLoader] Motion has only {num_frames} frames, cannot compute velocity.")

            fps = motion_raw_data["fps"]
            root_pos_w = torch.from_numpy(motion_raw_data["root_pos_w"]).float().to(self.device)  # (num_frames, 3)
            root_quat_w = (
                torch.from_numpy(motion_raw_data["root_quat_w"]).float().to(self.device)
            )  # (num_frames, 4) wxyz
            joint_pos = (
                torch.from_numpy(motion_raw_data["joint_pos"]).float().to(self.device)
            )  # (num_frames, num_joints)
            body_pos_b = (
                torch.from_numpy(motion_raw_data["body_pos_b"]).float().to(self.device)
            )  # (num_frames, num_bodies, 3)
            body_pos_w = math_utils.quat_apply_inverse(
                root_quat_w.unsqueeze(1).expand(-1, body_pos_b.shape[1], -1), body_pos_b
            ) + root_pos_w.unsqueeze(1)
            body_quat_b = (
                torch.from_numpy(motion_raw_data["body_quat_b"]).float().to(self.device)
            )  # (num_frames, num_bodies, 4) wxyz
            body_quat_w = math_utils.quat_mul(
                root_quat_w.unsqueeze(1).expand(-1, body_quat_b.shape[1], -1), body_quat_b
            )
            if not self.body_names:
                self.body_names = motion_raw_data["body_names"]
                self.body_name2idx = {name: i for i, name in enumerate(self.body_names)}
            if self.body_names != motion_raw_data["body_names"]:
                raise ValueError(
                    f"Motion data body names {self.body_names} do not match {motion_raw_data['body_names']}."
                )
            if not self.joint_names:
                self.joint_names = motion_raw_data["joint_names"]
                self.joint_name2idx = {name: i for i, name in enumerate(self.joint_names)}
            if self.joint_names != motion_raw_data["joint_names"]:
                raise ValueError(
                    f"Motion data joint names {self.joint_names} do not match {motion_raw_data['joint_names']}."
                )

            # Calculate vel
            dt = 1.0 / fps

            root_lin_vel_w = torch.zeros_like(root_pos_w)
            root_lin_vel_w[:-1] = (root_pos_w[1:] - root_pos_w[:-1]) / dt
            root_lin_vel_w[-1] = root_lin_vel_w[-2]

            body_lin_vel_w = torch.zeros_like(body_pos_w)
            body_lin_vel_w[:-1] = (body_pos_w[1:] - body_pos_w[:-1]) / dt
            body_lin_vel_w[-1] = body_lin_vel_w[-2]

            root_ang_vel_w = torch.zeros_like(root_pos_w)  # (F,3)
            root_ang_vel_w[:-1] = math_utils.quat_box_minus(root_quat_w[1:], root_quat_w[:-1]) / dt
            root_ang_vel_w[-1] = root_ang_vel_w[-2]

            body_ang_vel_w = torch.zeros_like(body_pos_w)  # (F,B,3)
            body_ang_vel_w[:-1] = math_utils.quat_box_minus(body_quat_w[1:], body_quat_w[:-1]) / dt
            body_ang_vel_w[-1] = body_ang_vel_w[-2]

            joint_vel = torch.zeros_like(joint_pos)
            joint_vel[:-1] = (joint_pos[1:] - joint_pos[:-1]) / dt
            joint_vel[-1] = joint_vel[-2]

            # Add motion data
            self.motion_durations.append(num_frames * dt)
            self.motion_fps.append(fps)
            self.motion_dt.append(dt)
            self.motion_num_frames.append(num_frames)
            self.motion_weights.append(motion_weight)

            self.root_pos_w.append(root_pos_w)
            self.root_quat_w.append(root_quat_w)
            self.root_lin_vel_w.append(root_lin_vel_w)
            self.root_ang_vel_w.append(root_ang_vel_w)
            self.body_pos_w.append(body_pos_w)
            self.body_quat_w.append(body_quat_w)
            self.body_lin_vel_w.append(body_lin_vel_w)
            self.body_ang_vel_w.append(body_ang_vel_w)
            self.joint_pos.append(joint_pos)
            self.joint_vel.append(joint_vel)
            self.body_pos_b.append(body_pos_b)
            self.body_quat_b.append(body_quat_b)

        self.motion_durations = torch.tensor(self.motion_durations, dtype=torch.float, device=self.device)
        self.motion_fps = torch.tensor(self.motion_fps, dtype=torch.float, device=self.device)
        self.motion_dt = torch.tensor(self.motion_dt, dtype=torch.float, device=self.device)
        self.motion_num_frames = torch.tensor(self.motion_num_frames, dtype=torch.long, device=self.device)
        self.motion_weights = torch.tensor(self.motion_weights, dtype=torch.float, device=self.device)

        self.root_pos_w = torch.cat(self.root_pos_w)
        self.root_quat_w = torch.cat(self.root_quat_w)
        self.root_lin_vel_w = torch.cat(self.root_lin_vel_w)
        self.root_ang_vel_w = torch.cat(self.root_ang_vel_w)
        self.body_pos_w = torch.cat(self.body_pos_w)
        self.body_quat_w = torch.cat(self.body_quat_w)
        self.body_lin_vel_w = torch.cat(self.body_lin_vel_w)
        self.body_ang_vel_w = torch.cat(self.body_ang_vel_w)
        self.joint_pos = torch.cat(self.joint_pos)
        self.joint_vel = torch.cat(self.joint_vel)
        self.body_pos_b = torch.cat(self.body_pos_b)
        self.body_quat_b = torch.cat(self.body_quat_b)

        # Some other information
        self.num_joints = self.joint_pos.shape[-1]
        self.num_bodies = self.body_pos_b.shape[-2]

        self.motion_ids = torch.arange(len(self.motion_durations), dtype=torch.long, device=self.device)

        lengths_shifted = self.motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self.motion_start_indices = torch.cumsum(lengths_shifted, dim=0)
        print(f"✅ Load motion data successfully on device {self.device}!")

    def sample_motion_ids(self, n: int) -> torch.Tensor:
        return torch.multinomial(self.motion_weights, num_samples=n, replacement=True)

    def sample_motion_times(
        self,
        motion_ids: torch.Tensor,
        truncate_time_start: float | None = None,
        truncate_time_end: float | None = None,
    ) -> torch.Tensor:
        motion_durations = self.motion_durations[motion_ids]

        # Calculate valid time range
        time_start = torch.zeros_like(motion_durations)
        time_end = motion_durations.clone()

        if truncate_time_start is not None:
            assert (
                truncate_time_start >= 0
            ), f"[MotionLoader] truncate_time_start must be non-negative, but got {truncate_time_start}."
            time_start = torch.clamp(time_start + truncate_time_start, min=0.0, max=motion_durations)

        if truncate_time_end is not None:
            assert (
                truncate_time_end >= 0
            ), f"[MotionLoader] truncate_time_end must be non-negative, but got {truncate_time_end}."
            time_end = torch.clamp(time_end - truncate_time_end, min=0.0)

        # Check if valid range exists
        valid_range = time_end - time_start
        if torch.any(valid_range <= 0.0):
            print("[Warning] Some motions have invalid time range after truncation (start >= end).")
            valid_range = torch.clamp(valid_range, min=1e-6)  # Prevent division by zero

        # Sample time within the valid range
        phase = torch.rand(motion_ids.shape, device=self.device)
        sample_times = time_start + phase * valid_range

        return sample_times

    def sample_motion_seq_times(self, motion_ids: torch.Tensor, n_steps: int, dt: float) -> torch.Tensor:
        motion_seq_duration = n_steps * dt
        start_times = self.sample_motion_times(motion_ids, truncate_time_end=motion_seq_duration)  # (ids,)
        motion_seq_times = (
            start_times.reshape(-1, 1) + torch.arange(n_steps, device=self.device).reshape(1, -1) * dt
        )  # (ids, steps)
        return motion_seq_times

    def get_motion_data(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
        joint_names: list | None = None,
        body_names: list | None = None,
    ) -> dict[str, torch.Tensor]:
        if motion_ids.shape != motion_times.shape:
            raise ValueError(
                f"motion_ids shape {motion_ids.shape} should be equal with motion_times shape {motion_times.shape}"
            )

        phase = motion_times / self.motion_durations[motion_ids]  # (ids,)
        num_frames = self.motion_num_frames[motion_ids]

        frame_idx0 = torch.floor((phase * (num_frames - 1))).long()  # (ids,)
        frame_idx1 = torch.minimum(frame_idx0 + 1, num_frames - 1)  # (ids,)
        blend = (phase * (num_frames - 1) - frame_idx0).reshape(-1, 1)  # (ids, 1)

        frame_idx0 += self.motion_start_indices[motion_ids]
        frame_idx1 += self.motion_start_indices[motion_ids]

        root_pos_w_0 = self.root_pos_w[frame_idx0]  # (ids, 3)
        root_pos_w_1 = self.root_pos_w[frame_idx1]
        root_quat_w_0 = self.root_quat_w[frame_idx0]
        root_quat_w_1 = self.root_quat_w[frame_idx1]
        root_lin_vel_w_0 = self.root_lin_vel_w[frame_idx0]
        root_lin_vel_w_1 = self.root_lin_vel_w[frame_idx1]
        root_ang_vel_w_0 = self.root_ang_vel_w[frame_idx0]
        root_ang_vel_w_1 = self.root_ang_vel_w[frame_idx1]
        body_pos_w_0 = self.body_pos_w[frame_idx0]  # (ids, body, 3)
        body_pos_w_1 = self.body_pos_w[frame_idx1]
        body_quat_w_0 = self.body_quat_w[frame_idx0]
        body_quat_w_1 = self.body_quat_w[frame_idx1]
        body_lin_vel_w_0 = self.body_lin_vel_w[frame_idx0]
        body_lin_vel_w_1 = self.body_lin_vel_w[frame_idx1]
        body_ang_vel_w_0 = self.body_ang_vel_w[frame_idx0]
        body_ang_vel_w_1 = self.body_ang_vel_w[frame_idx1]
        joint_pos_0 = self.joint_pos[frame_idx0]
        joint_pos_1 = self.joint_pos[frame_idx1]
        joint_vel_0 = self.joint_vel[frame_idx0]
        joint_vel_1 = self.joint_vel[frame_idx1]
        body_pos_b_0 = self.body_pos_b[frame_idx0]
        body_pos_b_1 = self.body_pos_b[frame_idx1]
        body_quat_b_0 = self.body_quat_b[frame_idx0]
        body_quat_b_1 = self.body_quat_b[frame_idx1]

        # interpolate the values
        root_pos_w = torch.lerp(root_pos_w_0, root_pos_w_1, blend)
        root_quat_w = self.quat_slerp(root_quat_w_0, root_quat_w_1, blend).float()
        root_lin_vel_w = torch.lerp(root_lin_vel_w_0, root_lin_vel_w_1, blend)
        root_lin_vel_b = math_utils.quat_apply_inverse(root_quat_w, root_lin_vel_w)
        root_ang_vel_w = torch.lerp(root_ang_vel_w_0, root_ang_vel_w_1, blend)
        root_ang_vel_b = math_utils.quat_apply_inverse(root_quat_w, root_ang_vel_w)
        joint_pos = torch.lerp(joint_pos_0, joint_pos_1, blend)
        joint_vel = torch.lerp(joint_vel_0, joint_vel_1, blend)
        body_pos_w = torch.lerp(body_pos_w_0, body_pos_w_1, blend.unsqueeze(-1))
        body_quat_w = self.quat_slerp(body_quat_w_0, body_quat_w_1, blend.unsqueeze(-1)).float()
        body_lin_vel_w = torch.lerp(body_lin_vel_w_0, body_lin_vel_w_1, blend.unsqueeze(-1))
        body_ang_vel_w = torch.lerp(body_ang_vel_w_0, body_ang_vel_w_1, blend.unsqueeze(-1))
        root_quat_w_expanded = root_quat_w.unsqueeze(1).expand(-1, body_lin_vel_w.shape[1], -1)
        body_lin_vel_b = math_utils.quat_apply_inverse(root_quat_w_expanded, body_lin_vel_w)
        body_ang_vel_b = math_utils.quat_apply_inverse(root_quat_w_expanded, body_ang_vel_w)
        body_pos_b = torch.lerp(body_pos_b_0, body_pos_b_1, blend.unsqueeze(-1))
        body_quat_b = self.quat_slerp(body_quat_b_0, body_quat_b_1, blend.unsqueeze(-1)).float()

        if joint_names:
            joint2target_idxs = torch.tensor(
                [self.joint_name2idx[name] for name in joint_names],
                dtype=torch.long,
                device=self.device,
            )
            joint_pos = joint_pos[:, joint2target_idxs]
            joint_vel = joint_vel[:, joint2target_idxs]
        if body_names:
            body2target_idxs = torch.tensor(
                [self.body_name2idx[name] for name in body_names],
                dtype=torch.long,
                device=self.device,
            )
            body_pos_w = body_pos_w[:, body2target_idxs]
            body_quat_w = body_quat_w[:, body2target_idxs]
            body_lin_vel_w = body_lin_vel_w[:, body2target_idxs]
            body_ang_vel_w = body_ang_vel_w[:, body2target_idxs]
            body_lin_vel_b = body_lin_vel_b[:, body2target_idxs]
            body_ang_vel_b = body_ang_vel_b[:, body2target_idxs]
            body_pos_b = body_pos_b[:, body2target_idxs]
            body_quat_b = body_quat_b[:, body2target_idxs]

        return {
            "root_pos_w": root_pos_w,
            "root_quat_w": root_quat_w,
            "root_lin_vel_w": root_lin_vel_w,
            "root_lin_vel_b": root_lin_vel_b,
            "root_ang_vel_w": root_ang_vel_w,
            "root_ang_vel_b": root_ang_vel_b,
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": body_lin_vel_w,
            "body_lin_vel_b": body_lin_vel_b,
            "body_ang_vel_w": body_ang_vel_w,
            "body_ang_vel_b": body_ang_vel_b,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "body_pos_b": body_pos_b,
            "body_quat_b": body_quat_b,
        }

    def get_motion_seq_data(
        self,
        motion_ids: torch.Tensor,  # (ids,)
        motion_seq_times: torch.Tensor,  # (ids, n_step)
        joint_names: list | None = None,
        body_names: list | None = None,
    ) -> dict[str, torch.Tensor]:
        new_motion_ids = motion_ids.unsqueeze(-1).expand(-1, motion_seq_times.shape[-1]).reshape(-1)
        new_motion_seq_times = motion_seq_times.reshape(-1)
        motion_data = self.get_motion_data(new_motion_ids, new_motion_seq_times, joint_names, body_names)
        for k, v in motion_data.items():
            motion_data[k] = v.reshape(*motion_seq_times.shape, *v.shape[1:])
        return motion_data

    def get_one_motion(
        self, motion_id: int, dt: float, joint_names: list | None = None, body_names: list | None = None
    ) -> Motion:
        sampled_times = torch.arange(0, self.motion_durations[motion_id], dt).to(self.device)
        motion_ids = torch.full_like(sampled_times, motion_id, dtype=int).to(self.device)
        motion_data = self.get_motion_data(motion_ids, sampled_times, joint_names, body_names)
        return Motion(
            time_step_total=sampled_times.shape[-1],
            joint_pos=motion_data["joint_pos"],
            joint_vel=motion_data["joint_vel"],
            body_pos_w=motion_data["body_pos_w"],
            body_quat_w=motion_data["body_quat_w"],
            body_lin_vel_w=motion_data["body_lin_vel_w"],
            body_ang_vel_w=motion_data["body_ang_vel_w"],
        )

    # TODO: We implement this function due to isaaclab math_utils does not support parallel quat_slerp,
    # this function should be replaced with math_utils.quat_slerp in the future.
    def quat_slerp(
        self,
        q0: torch.Tensor,
        q1: torch.Tensor,
        blend: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolation between consecutive rotations (Spherical Linear Interpolation).

        Args:
            q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            blend: Interpolation coefficient between 0 (q0) and 1 (q1). Shape is (N, 1) or (N, M, 1).
        Returns:
            Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
        """
        qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()  # type: ignore
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.clamp(torch.sqrt(1.0 - cos_half_theta * cos_half_theta), min=1e-6)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q


if __name__ == "__main__":
    # Test
    device = "cuda:0"
    # device = 'cpu'
    motion_loader = MotionLoader(
        motion_data_dir="/home/robot/hongtu/SimpleAMP/robot_assets/ths_23dof/motion_data",
        device=device,
    )
    for _ in range(3):
        motion_ids = motion_loader.sample_motion_ids(4096)
        motion_seq_times = motion_loader.sample_motion_seq_times(motion_ids=motion_ids, n_steps=3, dt=0.1)
        motion_seq_data = motion_loader.get_motion_seq_data(motion_ids, motion_seq_times)
        amp_input = torch.cat(
            [
                motion_seq_data["root_lin_vel_b"],
                motion_seq_data["root_ang_vel_b"],
                motion_seq_data["joint_pos"],
                motion_seq_data["joint_vel"],
            ],
            dim=-1,
        ).to(
            "cuda:0"
        )  # (num_envs, n_steps, d)
