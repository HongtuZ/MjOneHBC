import argparse
import time

import numpy as np

from controller.helper import ActionJointCfg
from ths_ostrich_env import ThsOstrichEnv


def build_action_joint_cfg(data) -> list[ActionJointCfg]:
    """从记录 npz 重建 action 关节配置（含 kp/kd/scale/default_pos）。"""
    joint_names = data["joint_names"].tolist()
    kp = data["kp"]
    kd = data["kd"]
    action_scale = data["action_scale"]
    default_joint_pos = data["default_joint_pos"]
    return [
        ActionJointCfg(
            joint_name=joint_names[i],
            default_joint_pos=float(default_joint_pos[i]),
            kp=float(kp[i]),
            kd=float(kd[i]),
            scale=float(action_scale[i]),
        )
        for i in range(len(joint_names))
    ]


def run_replay(env: ThsOstrichEnv, actions: np.ndarray) -> list[np.ndarray]:
    """真实回放：逐帧下发 target 到电机，并记录每步关节角度."""
    recorded_qpos = []
    for action in actions:
        obs_info, _ = env.step(action)
        qpos = np.asarray(obs_info["joint_pos"]) + env.default_joint_pos
        recorded_qpos.append(qpos.astype(np.float32))
    return recorded_qpos


def run_dry_run(env: ThsOstrichEnv, actions: np.ndarray) -> None:
    """只读预览：不使能电机、不发送指令，只打印目标与当前电机位置差异。"""
    interval = env.dt * env.decimation
    for i, action in enumerate(actions):
        _, extra = env.read_only_step(action)
        error = extra["position_error"]
        print(
            f"[DRY] step={i} max_err={np.abs(error).max():.4f} "
            f"mean_err={np.abs(error).mean():.4f}"
        )
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="真机 Replay 记录的 action 轨迹")
    parser.add_argument("--record", type=str, required=True, help="action 轨迹 npz 文件路径")
    parser.add_argument("--imu", type=str, default="yis320", help="IMU 型号: wit / yuansheng / yis320")
    parser.add_argument("--skip-reset", action="store_true", help="跳过上电归位，直接以当前姿态开始回放")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只读预览：不使能电机、不发送指令，只打印目标与当前电机位置差异",
    )
    parser.add_argument("--out", type=str, default="real_replay_qpos.npz", help="回放关节角度保存路径")
    args = parser.parse_args()

    data = np.load(args.record)
    actions = data["actions"]
    action_joint_cfg = build_action_joint_cfg(data)

    env = ThsOstrichEnv(
        physic_dt=0.005,
        decimation=4,
        action_joint_cfg=action_joint_cfg,
        imu_type=args.imu,
        read_only=args.dry_run,
    )

    mode = "只读预览" if args.dry_run else "真实回放"
    print(f"[REAL-REPLAY] 轨迹共 {len(actions)} 步，模式={mode}")
    recorded_qpos = []
    try:
        if args.dry_run:
            run_dry_run(env, actions)
            return

        if not args.skip_reset:
            # 归位到首帧 action 对应的 target，使回放第一帧时电机已基本到位
            first_action = np.clip(actions[0], env.action_clip[:, 0], env.action_clip[:, 1])
            first_target = first_action * env.action_scale + env.default_joint_pos
            print("[REAL-REPLAY] 归位到首帧 target...")
            env.reset(target_pos=first_target)
        env.enable_control = True
        print("[REAL-REPLAY] 已使能电机，开始回放...")
        recorded_qpos = run_replay(env, actions)
    finally:
        if len(recorded_qpos) > 0:
            qpos_arr = np.stack(recorded_qpos, axis=0)
            np.savez(args.out, joint_names=data["joint_names"], qpos=qpos_arr)
            print(f"[REAL-REPLAY] 已保存 {len(recorded_qpos)} 步关节角度到 {args.out}")
        env.enable_control = False
        env.close()
    print("[REAL-REPLAY] 结束")


if __name__ == "__main__":
    main()
