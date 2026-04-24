import mujoco
import mujoco.viewer
import numpy as np
import time
import torch

from source.OneHBC.OneHBC.utils.motion_loader import MotionLoader


class MuJoCoMotionPlayer:
    def __init__(self, mjcf_path: str, motion_loader: MotionLoader):
        """
        :param mjcf_path: MuJoCo模型文件路径 (.xml)
        :param motion_loader: 运动数据管理器实例
        """
        # 加载MuJoCo模型和数据
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        # 运动数据管理器
        self.motion_loader = motion_loader

        # 播放参数
        self.fps = 20  # 播放帧率
        self.play_speed = 10  # 播放速度
        self.motion_id = 2  # 要播放的运动ID

    def _set_mujoco_state(self, motion_data: dict):
        """将运动数据设置到MuJoCo的data中"""
        # 1. 转换张量为numpy（CPU）
        root_pos_w = motion_data["root_pos_w"].cpu().numpy()[0]
        root_quat_w = motion_data["root_quat_w"].cpu().numpy()[0]
        joint_pos = motion_data["joint_pos"].cpu().numpy()[0]

        self.data.qpos[:3] = root_pos_w
        self.data.qpos[3:7] = root_quat_w
        self.data.qpos[7:] = joint_pos

        mujoco.mj_forward(self.model, self.data)

    def play(self):
        """开始播放运动"""
        # 计算总时长和时间步
        motion_duration = (
            self.motion_loader.motion_durations[self.motion_id].cpu().item()
        )
        dt = 1.0 / (self.fps * self.play_speed)
        print(dt)

        # 创建Viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 5.0  # 相机距离
            viewer.cam.azimuth = 90.0  # 相机方位角
            viewer.cam.elevation = -20.0  # 相机仰角

            print("开始播放运动，按ESC退出...")

            # 播放循环
            current_time = 0.0
            while viewer.is_running and current_time < motion_duration:
                # 记录开始时间
                start_time = time.time()

                # 1. 获取当前时刻的运动数据
                print(current_time)
                motion_ids = torch.tensor(
                    [self.motion_id], device=self.motion_loader.device
                )
                motion_times = torch.tensor(
                    [current_time], device=self.motion_loader.device
                )
                motion_data = self.motion_loader.get_motion_data(
                    motion_ids, motion_times
                )

                # 2. 设置MuJoCo状态
                self._set_mujoco_state(motion_data)

                # 3. 更新Viewer
                viewer.cam.lookat = self.data.qpos[:3]
                viewer.sync()

                # 4. 控制播放速度
                elapsed = time.time() - start_time
                sleep_time = max(0, dt - elapsed)
                time.sleep(sleep_time)

                # 5. 更新当前时间
                current_time += dt


# --------------------------
# 4. 主函数：运行播放程序
# --------------------------
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    motion_loader = MotionLoader(
        motion_data_dir="/home/robot/hongtu/SimpleAMP/robot_assets/ths_23dof/motion_data",
        device=DEVICE,
    )
    # 配置路径（替换为你的实际路径）
    MJCF_PATH = "/home/robot/hongtu/SimpleAMP/robot_assets/ths_23dof/urdf/ths_23dof.xml"  # MuJoCo模型文件

    # 2. 创建播放器并开始播放
    player = MuJoCoMotionPlayer(MJCF_PATH, motion_loader)
    player.play()
