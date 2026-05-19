import argparse
import math
import sys
from pathlib import Path

from helper import ActionJointCfg, ObsBuffer, OnnxPolicy
from mujoco_env.mj_env import MujocoEnv

deploy_dir = Path(__file__).resolve().parent
if str(deploy_dir) not in sys.path:
    sys.path.insert(0, str(deploy_dir))

# ==================== 常量定义====================
PI = math.pi
ARMATURE = {
    "E00": 0.001 * 2,
    "E02": 0.0042,
    "E03": 0.02,
    "E06": 0.012,
}
NATURAL_FREQ = 20 * PI
DAMPING_RATIO = 2.0

STIFFNESS = {k: v * NATURAL_FREQ**2 for k, v in ARMATURE.items()}
DAMPING = {k: 2 * DAMPING_RATIO * v * NATURAL_FREQ for k, v in ARMATURE.items()}

ACTION_JOINT_CFG = [
    ActionJointCfg(
        joint_name="left_hip_pitch_joint",
        default_joint_pos=0.25,
        kp=STIFFNESS["E03"],
        kd=DAMPING["E03"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="right_hip_pitch_joint",
        default_joint_pos=-0.25,
        kp=STIFFNESS["E03"],
        kd=DAMPING["E03"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="torso_joint",
        default_joint_pos=0,
        kp=STIFFNESS["E06"],
        kd=DAMPING["E06"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="left_hip_roll_joint",
        default_joint_pos=0,
        kp=STIFFNESS["E03"],
        kd=DAMPING["E03"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="right_hip_roll_joint",
        default_joint_pos=0,
        kp=STIFFNESS["E03"],
        kd=DAMPING["E03"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="left_shoulder_pitch_joint",
        default_joint_pos=0,
        kp=STIFFNESS["E02"],
        kd=DAMPING["E02"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="right_shoulder_pitch_joint",
        default_joint_pos=0,
        kp=STIFFNESS["E02"],
        kd=DAMPING["E02"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="left_hip_yaw_joint",
        default_joint_pos=0,
        kp=STIFFNESS["E03"],
        kd=DAMPING["E03"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="right_hip_yaw_joint",
        default_joint_pos=0,
        kp=STIFFNESS["E03"],
        kd=DAMPING["E03"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="left_shoulder_roll_joint",
        default_joint_pos=1.4,
        kp=STIFFNESS["E00"],
        kd=DAMPING["E00"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="right_shoulder_roll_joint",
        default_joint_pos=-1.4,
        kp=STIFFNESS["E00"],
        kd=DAMPING["E00"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="left_knee_joint",
        default_joint_pos=-0.6,
        kp=STIFFNESS["E03"],
        kd=DAMPING["E03"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="right_knee_joint",
        default_joint_pos=0.6,
        kp=STIFFNESS["E03"],
        kd=DAMPING["E03"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="left_shoulder_yaw_joint",
        default_joint_pos=0,
        kp=STIFFNESS["E00"],
        kd=DAMPING["E00"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="right_shoulder_yaw_joint",
        default_joint_pos=0,
        kp=STIFFNESS["E00"],
        kd=DAMPING["E00"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="left_ankle_pitch_joint",
        default_joint_pos=-0.35,
        kp=STIFFNESS["E02"],
        kd=DAMPING["E02"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="right_ankle_pitch_joint",
        default_joint_pos=0.35,
        kp=STIFFNESS["E02"],
        kd=DAMPING["E02"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="left_elbow_joint",
        default_joint_pos=0.3,
        kp=STIFFNESS["E00"],
        kd=DAMPING["E00"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="right_elbow_joint",
        default_joint_pos=-0.3,
        kp=STIFFNESS["E00"],
        kd=DAMPING["E00"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="left_ankle_roll_joint",
        default_joint_pos=0,
        kp=STIFFNESS["E00"],
        kd=DAMPING["E00"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="right_ankle_roll_joint",
        default_joint_pos=0,
        kp=STIFFNESS["E00"],
        kd=DAMPING["E00"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="left_wrist_roll_joint",
        default_joint_pos=0,
        kp=STIFFNESS["E00"],
        kd=DAMPING["E00"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
    ActionJointCfg(
        joint_name="right_wrist_roll_joint",
        default_joint_pos=0,
        kp=STIFFNESS["E00"],
        kd=DAMPING["E00"],
        scale=0.25,
        clip=(-10.0, 10.0),
    ),
]


# ==================== 主仿真 ====================
def main(xml_path: str, model_path: str):
    finish_sim = False

    # Create onnx policy
    onnx_policy = OnnxPolicy(onnx_policy_path=model_path, is_velocity_command=True, is_real_env=False, use_gpu=False)

    def keyboard_callback(keycode):
        nonlocal finish_sim, onnx_policy
        if keycode == 256:  # ESC
            finish_sim = True
        if chr(keycode) == " ":  # 空格
            onnx_policy.vel_x, onnx_policy.vel_y, onnx_policy.ang_vel_z = 0.0, 0.0, 0.0
        if keycode == 265:  # 箭头上
            onnx_policy.vel_x += 0.1
        if keycode == 264:  # 箭头下
            onnx_policy.vel_x -= 0.1
        # if keycode == 263:  # 箭头左
        #     onnx_policy.vel_y -= 0.1
        # if keycode == 262:  # 箭头右
        #     onnx_policy.vel_y += 0.1
        if keycode == 263:
            onnx_policy.ang_vel_z += 0.1
        if keycode == 262:
            onnx_policy.ang_vel_z -= 0.1

    # Create mujoco env
    env = MujocoEnv(
        xml_path=xml_path,
        sim_dt=0.005,
        decimation=4,
        action_joint_cfg=ACTION_JOINT_CFG,
        keyboard_callback=keyboard_callback,
    )
    # Create observation buffer
    obs_buffer = ObsBuffer(
        obs_names=["base_ang_vel", "projected_gravity", "velocity_command", "joint_pos", "joint_vel", "last_action"],
        history_length=1,
        concatenate=True,
    )

    # Run
    obs_info = env.reset()
    while not finish_sim:
        obs_info.update(onnx_policy.get_command())
        obs_buffer.push(obs_info)
        obs = obs_buffer.get_obs()
        action = onnx_policy.get_action(obs)
        env.show_command(
            velocity_command=obs_info.get("velocity_command", None), ref_motion=obs_info.get("ref_motion", None)
        )
        obs_info, done = env.step(action)
        if done:
            break
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="robot_assets/ths_23dof/urdf/ths_23dof.xml")
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args.xml_path, args.model_path)
