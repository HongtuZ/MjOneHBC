import copy
import time

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from controller.helper import ActionJointCfg, normalize
from scipy.spatial.transform import Rotation as R

from mujoco_env.visualizer import MujocoDebugVisualizer


class MujocoEnv:
    def __init__(
        self,
        xml_path: str,
        sim_dt: float,
        decimation: int,
        action_joint_cfg: list[ActionJointCfg],
        keyboard_callback=None,
    ):
        self.dt = sim_dt
        self.decimation = decimation
        self.action_joint_names = [aj_cfg.joint_name for aj_cfg in action_joint_cfg]
        self.default_joint_pos = np.array([aj_cfg.default_joint_pos for aj_cfg in action_joint_cfg], dtype=float)
        self.kp = np.array([aj_cfg.kp for aj_cfg in action_joint_cfg], dtype=float)
        self.kd = np.array([aj_cfg.kd for aj_cfg in action_joint_cfg], dtype=float)
        self.action_scale = np.array([aj_cfg.scale for aj_cfg in action_joint_cfg], dtype=float)
        self.action_clip = np.array(
            [(-np.inf, np.inf) if not aj_cfg.clip else aj_cfg.clip for aj_cfg in action_joint_cfg], dtype=float
        )
        # Load model
        self.xml_path = xml_path
        self.spec = mj.MjSpec.from_file(self.xml_path)
        # Add plane
        if mj.mjtGeom.mjGEOM_PLANE not in [geom.type for geom in self.spec.geoms]:
            self.spec.worldbody.add_light(
                name="sun_light",
                pos=[0, 0, 4],
                dir=[0, 0, -1],
                type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
                ambient=[0.25, 0.25, 0.25],
                diffuse=[0.8, 0.8, 0.8],
                specular=[0.8, 0.8, 0.8],
                castshadow=True,
                intensity=1.0,
            )
            self.spec.add_texture(
                name="ground_texture",
                type=mj.mjtTexture.mjTEXTURE_2D,
                builtin=mj.mjtBuiltin.mjBUILTIN_CHECKER,
                rgb1=[0.1, 0.2, 0.3],
                rgb2=[0.2, 0.3, 0.4],
                markrgb=[0.8, 0.8, 0.8],
                width=300,
                height=300,
            )
            material = self.spec.add_material(
                name="ground_material",
                texrepeat=[4.0, 4.0],
                reflectance=0.2,
            )
            material.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "ground_texture"
            self.spec.worldbody.add_geom(
                name="ground",
                type=mj.mjtGeom.mjGEOM_PLANE,
                size=[0, 0, 0.01],
                conaffinity=1,
                condim=3,
                friction=[1.0, 0.005, 0.0001],
                material="ground_material",
            )
        # Compile model
        self.model = self.spec.compile()
        self.model.opt.timestep = self.dt
        self.data = mj.MjData(self.model)

        # ghost model for debug
        self.ghost_model = copy.deepcopy(self.model)

        # 关节名字 -> qpos索引, qvel索引
        self.jnt_qpos_indices = np.zeros(len(self.action_joint_names), dtype=int)
        self.jnt_qvel_indices = np.zeros(len(self.action_joint_names), dtype=int)

        for i, name in enumerate(self.action_joint_names):
            jnt_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
            self.jnt_qpos_indices[i] = self.model.jnt_qposadr[jnt_id]
            self.jnt_qvel_indices[i] = self.model.jnt_dofadr[jnt_id]

        # Action -> mujoco.data.ctr
        self.action2ctrl_ids = np.zeros(len(self.action_joint_names), dtype=int)
        for i, name in enumerate(self.action_joint_names):
            # 找到驱动该关节的执行器
            for act_id in range(self.model.nu):
                driven_jnt_id = self.model.actuator_trnid[act_id][0]
                driven_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, driven_jnt_id)
                if driven_name == name:
                    self.action2ctrl_ids[i] = act_id
                    break
        self.reset()

        # Set viewer
        self.viewer = mjv.launch_passive(
            model=self.model, data=self.data, show_left_ui=True, show_right_ui=True, key_callback=keyboard_callback
        )
        self.viewer.cam.distance = 3
        self.viewer.cam.elevation = -10  # 正面视角，轻微向下看
        # # self.viewer.cam.azimuth = 180    # 正面朝向机器人
        self.debug_visualizer = MujocoDebugVisualizer(self.viewer.user_scn, self.model)

    def compute_torque(
        self,
        action: np.ndarray,
    ):
        qpos = self.data.qpos[self.jnt_qpos_indices]
        qvel = self.data.qvel[self.jnt_qvel_indices]
        torque = self.kp * (action - qpos) - self.kd * qvel
        return torque

    def step(
        self,
        action: np.ndarray,
    ):
        action = action.reshape(-1)
        action = np.clip(action, self.action_clip[:, 0], self.action_clip[:, 1])
        target = action * self.action_scale
        target += self.default_joint_pos
        start_time = time.perf_counter()
        for _ in range(self.decimation):
            tau = self.compute_torque(target)
            self.data.ctrl[self.action2ctrl_ids] = tau
            self.data.ctrl[:] = np.clip(self.data.ctrl, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
            mj.mj_step(self.model, self.data)
            self.viewer.sync()
        self.viewer.cam.lookat = self.data.qpos[:3]
        duration = time.perf_counter() - start_time
        time.sleep(max(0, self.dt * self.decimation - duration))

        # return info
        base_quat = self.data.qpos[3:7]
        base_ang_vel = self.data.qvel[3:6]
        base_rot_inv = R.from_quat(base_quat, scalar_first=True).inv()
        projected_gravity = base_rot_inv.apply(np.array([0, 0, -9.81]))
        projected_gravity = normalize(projected_gravity)
        obs_info = {
            "robot_pos": self.data.qpos[:3],
            "robot_quat": base_quat,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "joint_pos": self.data.qpos[self.jnt_qpos_indices] - self.default_joint_pos,
            "joint_vel": self.data.qvel[self.jnt_qvel_indices],
            "last_action": action,
        }
        done = False
        return obs_info, done

    def show_command(self, velocity_command=None, ref_motion=None):
        self.debug_visualizer.clear()
        # Helper to transform local to world coordinates.
        base_pos_w = self.data.qpos[:3]
        base_rot = R.from_quat(self.data.qpos[3:7], scalar_first=True)

        def local2world(vec: np.ndarray, pos: np.ndarray = base_pos_w, rot: R = base_rot) -> np.ndarray:
            return pos + rot.apply(vec)

        if velocity_command is not None:
            head = np.array([0, 0, 0.5])
            cmd_from = local2world(head)
            cmd_lin_to = local2world(head + np.array([velocity_command[0], velocity_command[1], 0]))
            cmd_ang_to = local2world(head + np.array([0, 0, velocity_command[2]]))
            act_lin_to = head + base_pos_w + np.array([self.data.qvel[0], self.data.qvel[1], 0])
            act_ang_to = head + base_pos_w + np.array([0, 0, self.data.qvel[5]])
            self.debug_visualizer.add_arrow(cmd_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015)  # blue
            self.debug_visualizer.add_arrow(cmd_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015)  # green
            self.debug_visualizer.add_arrow(cmd_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015)  # cyan
            self.debug_visualizer.add_arrow(cmd_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015)  # lgreen

        if ref_motion is not None:
            for gi in range(self.ghost_model.ngeom):
                if self.ghost_model.geom_contype[gi] != 0 or self.ghost_model.geom_conaffinity[gi] != 0:
                    self.ghost_model.geom_rgba[gi, 3] = 0
                else:
                    self.ghost_model.geom_rgba[gi] = (0.5, 0.7, 0.5, 0.5)
            qpos = np.zeros(self.ghost_model.nq)
            qpos[:3] = ref_motion["base_pos"]
            qpos[3:7] = ref_motion["base_quat"]
            qpos[self.jnt_qpos_indices] = ref_motion["joint_pos"]
            self.debug_visualizer.add_ghost_mesh(
                qpos,
                model=self.ghost_model,
            )

    def reset(self, root_pos=None, root_quat=None):
        mj.mj_resetData(self.model, self.data)
        if root_pos is not None:
            self.data.qpos[:3] = root_pos
        if root_quat is not None:
            self.data.qpos[3:7] = root_quat
        self.data.qpos[self.jnt_qpos_indices] = self.default_joint_pos
        mj.mj_forward(self.model, self.data)
        base_quat = self.data.qpos[3:7]
        base_ang_vel = self.data.qvel[3:6]
        base_rot = R.from_quat(base_quat, scalar_first=True)
        projected_gravity = base_rot.inv().apply(np.array([0, 0, -9.81]))
        projected_gravity = projected_gravity / np.linalg.norm(projected_gravity)
        return {
            "robot_pos": self.data.qpos[:3],
            "robot_quat": base_quat,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "joint_pos": self.data.qpos[self.jnt_qpos_indices] - self.default_joint_pos,
            "joint_vel": self.data.qvel[self.jnt_qvel_indices],
            "last_action": np.zeros(len(self.data.ctrl)),
        }

    def close(self):
        self.viewer.close()
        time.sleep(0.5)
