import argparse
from pathlib import Path

import numpy as np

from controller.helper import ActionJointCfg
from mujoco_env.mj_env import MujocoEnv

ROOT_DIR = Path(__file__).resolve().parents[2]


def make_fixed_base_xml(xml_path: str, base_body: str) -> str:
    """生成一份将 base 焊接到 world 的临时 XML，返回其路径.

    通过 weld equality 约束把 ``base_body`` 固定在世界坐标系，
    使机器人悬空、base 不再受重力影响整体移动，仅保留关节运动。
    """
    xml_path = Path(xml_path)
    text = xml_path.read_text()

    weld_line = f'    <weld name="base_fixed" body1="{base_body}"/>'
    if "<equality>" in text:
        # 已有 equality 块，把 weld 插入到第一个 </equality> 之前
        text = text.replace("</equality>", weld_line + "\n  </equality>", 1)
    else:
        block = "\n  <equality>\n" + weld_line + "\n  </equality>\n"
        text = text.replace("</mujoco>", block + "</mujoco>")

    # 临时文件放在原 XML 同目录，保证 meshdir 等相对路径仍可解析
    tmp_path = xml_path.with_name(xml_path.stem + "_fixed_base_replay.xml")
    tmp_path.write_text(text)
    return str(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Replay 记录的 action 轨迹（base 固定在空中）")
    parser.add_argument("--record", type=str, required=True, help="action 轨迹 npz 文件路径")
    parser.add_argument("--xml", type=str, default=str(ROOT_DIR / "robot_assets/ths_23dof/urdf/ths_23dof.xml"))
    parser.add_argument("--base-body", type=str, default="base_link", help="要固定的 base body 名称")
    parser.add_argument("--sim-dt", type=float, default=0.005)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--out", type=str, default="mj_replay_qpos.npz", help="回放关节角度保存路径")
    args = parser.parse_args()

    # 读取记录
    data = np.load(args.record)
    actions = data["actions"]
    joint_names = data["joint_names"].tolist()
    kp = data["kp"]
    kd = data["kd"]
    action_scale = data["action_scale"]
    default_joint_pos = data["default_joint_pos"]

    # 重建 action 关节配置（含 kp/kd）
    action_joint_cfg = [
        ActionJointCfg(
            joint_name=joint_names[i],
            default_joint_pos=float(default_joint_pos[i]),
            kp=float(kp[i]),
            kd=float(kd[i]),
            scale=float(action_scale[i]),
        )
        for i in range(len(joint_names))
    ]

    fixed_xml = make_fixed_base_xml(args.xml, args.base_body)
    env = None
    recorded_qpos = []
    try:
        env = MujocoEnv(
            xml_path=fixed_xml,
            sim_dt=args.sim_dt,
            decimation=args.decimation,
            action_joint_cfg=action_joint_cfg,
            keyboard_callback=None,
        )

        # 初始姿态设为第一帧 action 对应的 target
        first_action = np.clip(actions[0], env.action_clip[:, 0], env.action_clip[:, 1])
        first_target = first_action * env.action_scale + env.default_joint_pos
        env.reset(joint_pos=first_target)

        print(f"[REPLAY] 轨迹共 {len(actions)} 步，base 已固定在空中，开始回放...")
        for i, action in enumerate(actions):
            env.step(action)
            recorded_qpos.append(env.data.qpos[env.jnt_qpos_indices].copy())
    finally:
        if len(recorded_qpos) > 0:
            qpos_arr = np.stack(recorded_qpos, axis=0)
            np.savez(args.out, joint_names=np.array(joint_names), qpos=qpos_arr)
            print(f"[REPLAY] 已保存 {len(recorded_qpos)} 步关节角度到 {args.out}")
        if env is not None:
            env.close()
        Path(fixed_xml).unlink(missing_ok=True)
    print("[REPLAY] 回放结束")


if __name__ == "__main__":
    main()
