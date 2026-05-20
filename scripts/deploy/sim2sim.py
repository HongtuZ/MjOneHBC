import argparse
import sys
from pathlib import Path

from helper import ObsBuffer, OnnxPolicy
from mujoco_env.mj_env import MujocoEnv

# ==================== noqa ====================
deploy_dir = Path(__file__).resolve().parent
if str(deploy_dir) not in sys.path:
    sys.path.insert(0, str(deploy_dir))


# ==================== 主仿真 ====================
def main(xml_path: str, model_path: str):
    finish_sim = False

    # Create onnx policy
    policy = OnnxPolicy(onnx_policy_path=model_path, is_real_env=False)

    def keyboard_callback(keycode):
        nonlocal finish_sim, policy
        if keycode == 256:  # ESC
            finish_sim = True
        if policy.command_name == "motion":
            if chr(keycode) == " ":  # 空格
                policy.motion_step_t = 0
            pass
        else:
            if chr(keycode) == " ":  # 空格
                policy.vel_x, policy.vel_y, policy.ang_vel_z = 0.0, 0.0, 0.0
            if keycode == 265:  # 箭头上
                policy.vel_x += 0.1
            if keycode == 264:  # 箭头下
                policy.vel_x -= 0.1
            # if keycode == 263:  # 箭头左
            #     policy.vel_y -= 0.1
            # if keycode == 262:  # 箭头右
            #     policy.vel_y += 0.1
            if keycode == 263:
                policy.ang_vel_z += 0.1
            if keycode == 262:
                policy.ang_vel_z -= 0.1

    # Create mujoco env
    env = MujocoEnv(
        xml_path=xml_path,
        sim_dt=0.005,
        decimation=4,
        action_joint_cfg=policy.action_joint_cfg,
        keyboard_callback=keyboard_callback,
    )
    # Create observation buffer
    if policy.command_name == "motion":
        obs_buffer = ObsBuffer(
            obs_names=[
                "motion_command",
                "motion_anchor_ori_b",
                "base_ang_vel",
                "joint_pos",
                "joint_vel",
                "last_action",
            ],
            history_length=1,
            concatenate=True,
        )
    else:
        obs_buffer = ObsBuffer(
            obs_names=[
                "base_ang_vel",
                "projected_gravity",
                "velocity_command",
                "joint_pos",
                "joint_vel",
                "last_action",
            ],
            history_length=1,
            concatenate=True,
        )

    # Run
    obs_info = env.reset()
    policy.reset(robot_pos=obs_info.get("robot_pos"), robot_quat=obs_info.get("robot_quat"))
    while not finish_sim:
        obs_info.update(policy.get_command(obs_info.get("robot_quat")))
        obs_buffer.push(obs_info)
        obs = obs_buffer.get_obs()
        action = policy.get_action(obs)
        env.show_command(velocity_command=obs_info.get("velocity_command"), ref_motion=obs_info.get("ref_motion"))
        obs_info, done = env.step(action)
        if done:
            break
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="robot_assets/ths_23dof/urdf/ths_23dof.xml")
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args.xml, args.model)
