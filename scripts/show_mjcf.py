import mujoco
import tyro


def print_qpos_joint_mapping(mjcf: str):
    if mjcf:
        model = mujoco.MjModel.from_xml_path(mjcf)
    else:
        raise ValueError("请提供 xml_path")

    data = mujoco.MjData(model)

    print(f"总关节数 (njnt): {model.njnt}")
    print(f"qpos 维度 (nq): {model.nq}")
    print(f"自由度数量 (nv): {model.nv}")
    print("=" * 50)

    # 关节类型名称映射
    JNT_TYPE_NAMES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}

    joint_names = []
    # 1. 打印每个关节的基本信息
    print("\n【关节列表】")
    for jnt_id in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        jnt_type = model.jnt_type[jnt_id]
        type_name = JNT_TYPE_NAMES.get(jnt_type, f"unknown({jnt_type})")
        if type_name == "hinge":
            joint_names.append(jnt_name)
        qpos_adr = model.jnt_qposadr[jnt_id]  # 该关节在 qpos 中的起始索引
        dof_adr = model.jnt_dofadr[jnt_id]  # 该关节在 qvel 中的起始索引

        # 计算 qpos 占用维度
        if jnt_type == 0:  # free: 位置(3) + 四元数(4) = 7
            qpos_size = 7
        elif jnt_type == 1:  # ball: 四元数(4)
            qpos_size = 4
        else:  # slide/hinge: 1
            qpos_size = 1

        print(
            f"  [{jnt_id}] '{jnt_name}' | type={type_name} | qpos[{qpos_adr}:{qpos_adr + qpos_size}] | dof[{dof_adr}]"
        )

    print(f"\n================={len(joint_names)}关节名list供复制=================")
    print("[")
    for name in joint_names:
        print(f'"{name}",')
    print("]")
    print("==============================================\n")

    print(f"\n================={len(joint_names)}关节名dict供复制=================")
    print("{")
    for name in joint_names:
        print(f'"{name}": 0,')
    print("}")
    print("==============================================\n")

    # 2. 打印每个 qpos 索引对应的关节
    print("\n【qpos 索引明细】")
    for i in range(model.nq):
        for jnt_id in range(model.njnt):
            qpos_adr = model.jnt_qposadr[jnt_id]
            jnt_type = model.jnt_type[jnt_id]

            # 确定该关节占用的 qpos 范围
            if jnt_type == 0:
                size = 7
            elif jnt_type == 1:
                size = 4
            else:
                size = 1

            if qpos_adr <= i < qpos_adr + size:
                jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
                offset = i - qpos_adr

                # 根据关节类型描述具体分量
                if jnt_type == 0:  # free
                    components = ["pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"]
                elif jnt_type == 1:  # ball
                    components = ["quat_w", "quat_x", "quat_y", "quat_z"]
                else:
                    components = ["value"]

                print(f"  qpos[{i:2d}] = {data.qpos[i]:8.4f}  ->  '{jnt_name}.{components[offset]}'")
                break

    # 3. 打印当前 qpos 值（可选）
    print("\n【当前 qpos 值】")
    print(f"  {data.qpos}")

    return model, data


if __name__ == "__main__":
    tyro.cli(print_qpos_joint_mapping)
