"""THS23DOF velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

import OneHBC.tasks.one_hbc.mdp as mdp
from OneHBC.assets.robots import THS23DOF_CFG, THS23DOF_ACTION_SCALE
from OneHBC.tasks.one_hbc.one_hbc_env_cfg import OneHBCEnvCfg

from dataclasses import dataclass

# Sensors
foot_height_scan_cfg = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(
        ObjRef(type="site", name="left_foot_site", entity="robot"),
        ObjRef(type="site", name="right_foot_site", entity="robot"),
    ),
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
    viz=TerrainHeightSensorCfg.VizCfg(
        show_rays=True,
        hit_color=(1.0, 0.0, 1.0, 0.8),  # Magenta rays.
        hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
    ),
)
feet_ground_contact_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
        mode="subtree",
        pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
        entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
    history_length=4,
)
self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
)


@dataclass(kw_only=True)
class VelocityRoughEnvCfg(OneHBCEnvCfg):
    def __post_init__(self):
        # Simulation
        self.sim.mujoco.ccd_iterations = 500
        self.sim.contact_sensor_maxmatch = 500
        self.sim.nconmax = 70

        # Scene
        self.scene.entities = {"robot": THS23DOF_CFG}
        self.scene.sensors = (foot_height_scan_cfg, feet_ground_contact_cfg, self_collision_cfg)
        if self.scene.terrain is not None and self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True

        # Action
        joint_pos_action = self.actions["joint_pos"]
        assert isinstance(joint_pos_action, JointPositionActionCfg)
        joint_pos_action.scale = THS23DOF_ACTION_SCALE

        # Viewer
        self.viewer.body_name = "torso_link"
        velocity_cmd = self.commands["base_velocity"]
        assert isinstance(velocity_cmd, UniformVelocityCommandCfg)
        velocity_cmd.viz.z_offset = 1.15

        # Event
        self.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

        # Basic Reward
        self.rewards["is_alive"].weight = 0.15
        self.rewards["is_terminated"].weight = -1.0
        self.rewards["joint_torques_l2"].weight = -1.0e-5
        self.rewards["joint_vel_l2"].weight = -1.0e-3
        self.rewards["joint_acc_l2"].weight = -2.5e-7
        self.rewards["action_rate_l2"].weight = -0.05
        self.rewards["action_acc_l2"].weight = -0.01
        self.rewards["joint_pos_limits"].weight = -5.0
        self.rewards["flat_orientation_l2"].weight = -10.0
        self.rewards["joint_deviation_exp"].weight = -1e-3
        self.rewards["joint_energy"].weight = -2e-5
        self.rewards["track_lin_vel_exp"].weight = 1.5
        self.rewards["track_ang_vel_exp"].weight = 1.5

        # New Reward
        self.rewards["feet_gait"] = RewardTermCfg(
            func=mdp.feet_gait,
            weight=1.0,
            params={
                "period": 0.6,  # 步态周期0.8秒
                "offset": [0.0, 0.5],  # 左右足相位偏移（0.0表示同相，0.5表示反相）
                "threshold": 0.45,  # 接触判断阈值
                "command_name": "base_velocity",  # 关联的速度指令名称
                "sensor_name": "feet_ground_contact",  # 传感器配置：监测脚踝滚转关节的接触力
            },
        )
        self.rewards["feet_clearance"] = RewardTermCfg(
            func=mdp.feet_clearance,
            weight=0.5,
            params={
                "target_height": 0.05,
                "height_sensor_name": "foot_height_scan",
                "command_name": "base_velocity",  # 关联的速度指令名称
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", site_names=".*_foot_site"),
            },
        )
        self.rewards["low_speed_sway_penalty"] = RewardTermCfg(
            func=mdp.low_speed_sway_penalty,
            weight=-1e-2,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.1,
            },
        )
        self.rewards["feet_slide"] = RewardTermCfg(
            func=mdp.feet_slide,
            weight=-0.5,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", site_names=".*_foot_site"),
            },
        )
        self.rewards["feet_stumble"] = RewardTermCfg(
            func=mdp.feet_stumble,
            weight=-0.1,
            params={
                "sensor_name": "feet_ground_contact",
            },
        )
        self.rewards["soft_landing"] = RewardTermCfg(
            func=mdp.soft_landing,
            weight=-1e-5,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "base_velocity",
                "command_threshold": 0.1,
            },
        )
        self.rewards["self_collision"] = RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-1,
            params={
                "force_threshold": 10,
                "sensor_name": "self_collision",
            },
        )

        # 惩罚脚踝关节横滚扭矩过大
        self.rewards["ankle_roll_torque_limit"] = RewardTermCfg(
            func=mdp.joint_torque_soft_limit,
            weight=-1.0,
            params={"threshold": 2.0, "asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle_roll_.*")},
        )

        # 手臂跟踪腿部协调奖励项
        self.rewards["shoulder_thigh_coordination"] = RewardTermCfg(func=mdp.shoulder_thigh_coordination, weight=0.5)

        # -- 惩罚不希望运动的关节
        # std越小惩罚越重
        self.rewards["pose"] = RewardTermCfg(
            func=mdp.variable_posture,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "command_name": "base_velocity",
                "std_standing": {".*": 0.05},  # Set per-robot.
                "std_walking": {
                    # Legs
                    r".*hip_pitch.*": 1.0,
                    r".*hip_roll.*": 0.2,
                    r".*hip_yaw.*": 0.6,
                    r".*knee.*": 1.0,
                    r".*ankle_pitch.*": 0.5,
                    r".*ankle_roll.*": 0.15,
                    # Waist
                    r".*torso.*": 0.3,
                    # Arms
                    r".*shoulder_pitch.*": 1.0,
                    r".*shoulder_roll.*": 1.0,
                    r".*shoulder_yaw.*": 1.0,
                    r".*elbow.*": 1.0,
                    r".*wrist.*": 0.3,
                },  # Set per-robot.
                "std_running": {
                    # Legs
                    r".*hip_pitch.*": 1.0,
                    r".*hip_roll.*": 0.2,
                    r".*hip_yaw.*": 0.6,
                    r".*knee.*": 1.0,
                    r".*ankle_pitch.*": 0.5,
                    r".*ankle_roll.*": 0.15,
                    # Waist
                    r".*torso.*": 0.3,
                    # Arms
                    r".*shoulder_pitch.*": 1.0,
                    r".*shoulder_roll.*": 1.0,
                    r".*shoulder_yaw.*": 1.0,
                    r".*elbow.*": 1.0,
                    r".*wrist.*": 0.3,
                },  # Set per-robot.
                "walking_threshold": 0.1,
                "running_threshold": 1.5,
            },
        )


@dataclass(kw_only=True)
class VelocityRoughPlayEnvCfg(VelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Effectively infinite episode length.
        self.episode_length_s = int(1e9)

        self.observations["actor"].enable_corruption = False
        self.events.pop("push_robot", None)
        self.curriculum = {}
        self.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        if self.scene.terrain is not None and self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.border_width = 10.0


@dataclass(kw_only=True)
class VelocityFlatEnvCfg(VelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.sim.njmax = 300
        self.sim.mujoco.ccd_iterations = 50
        self.sim.contact_sensor_maxmatch = 64
        self.sim.nconmax = None

        # Switch to flat terrain.
        assert self.scene.terrain is not None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None


@dataclass(kw_only=True)
class VelocityFlatPlayEnvCfg(VelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        velocity_cmd = self.commands["base_velocity"]
        assert isinstance(velocity_cmd, UniformVelocityCommandCfg)
        velocity_cmd.ranges.lin_vel_x = (-1.5, 2.0)
        velocity_cmd.ranges.ang_vel_z = (-0.7, 0.7)
