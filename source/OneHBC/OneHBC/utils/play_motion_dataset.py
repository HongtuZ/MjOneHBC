import argparse
import time

import mujoco
import mujoco.viewer

from OneHBC.utils.motion_loader import MotionLoader

paused = False
motion_num = 0
motion_id = 0
current_motion_id = -1


class MuJoCoMotionPlayer:
    def __init__(self, mjcf_path: str, step_dt: float, keyboard_callback=None):
        # 加载MuJoCo模型和数据
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.step_dt = step_dt
        self.viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=keyboard_callback,
        )
        self.viewer.cam.distance = 5.0  # 相机距离
        self.viewer.cam.azimuth = 90.0  # 相机方位角
        self.viewer.cam.elevation = -20.0  # 相机仰角

    def step(self, root_pos_w, root_quat_w, joint_pos):
        start_time = time.perf_counter()
        self.data.qpos[:3] = root_pos_w
        self.data.qpos[3:7] = root_quat_w
        self.data.qpos[7:] = joint_pos
        mujoco.mj_forward(self.model, self.data)
        self.viewer.cam.lookat = self.data.qpos[:3]
        self.viewer.sync()
        sleep_time = max(0, self.step_dt - (time.perf_counter() - start_time))
        time.sleep(sleep_time)

    def close(self):
        self.viewer.close()
        time.sleep(0.5)


def keyboard_callback(keycode):
    global paused, motion_id, motion_num
    if chr(keycode) == " ":
        paused = not paused
    if chr(keycode) == "[":
        motion_id = (motion_id - 1) % motion_num
    if chr(keycode) == "]":
        motion_id = (motion_id + 1) % motion_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, required=True)
    parser.add_argument("--motion_data_dir", type=str, required=True)
    args = parser.parse_args()

    step_dt = 0.02
    motion_loader = MotionLoader(args.motion_data_dir)
    motion_num = motion_loader.motion_ids.shape[0]
    motion_data = None

    env = MuJoCoMotionPlayer(
        mjcf_path=args.xml,
        step_dt=step_dt,
        keyboard_callback=keyboard_callback,
    )

    frame_idx = 0
    while True:
        # get current motion
        if current_motion_id != motion_id:
            current_motion_id = motion_id
            frame_idx = 0
            motion_data = motion_loader.get_one_motion(motion_id, dt=step_dt, joint_names=())
            print(
                f"Switched to motion {motion_id}: {list(motion_loader.motion_data_weights.keys())[motion_id]}, fps: {motion_loader.motion_fps[motion_id]}, num_frames: {motion_loader.motion_num_frames[motion_id]}"
            )
        if not paused:
            env.step(
                motion_data.root_pos_w[frame_idx], motion_data.root_quat_w[frame_idx], motion_data.joint_pos[frame_idx]
            )
            frame_idx += 1
            if frame_idx >= motion_data.num_frames:
                frame_idx = 0
    env.close()
