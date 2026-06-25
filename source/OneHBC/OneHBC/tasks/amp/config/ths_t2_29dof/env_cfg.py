"""THS23DOF velocity environment configurations."""

import math
from dataclasses import dataclass

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

import OneHBC.tasks.amp.mdp as mdp
from OneHBC import ONEHBC_ROOT
from OneHBC.assets.robots import THS_T2_29DOF_ACTION_SCALE, THS_T2_29DOF_CFG
from OneHBC.tasks.amp.amp_env_cfg import AmpEnvCfg

# AMP body names
AMP_MOTION_DATA_DIR = str(ONEHBC_ROOT / "robot_assets/ths_t2_29dof/motion_data/walk_run")
AMP_MOTION_DATA_WEIGHTS = None
AMP_BODY_NAMES = (
    "base_link",
    ".*_hip_roll_link",
    ".*_knee_link",
    ".*_ankle_roll_link",
    ".*_shoulder_roll_link",
    ".*_elbow_link",
    ".*_wrist_pitch_link",
)

# Sensors
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
class AmpRoughEnvCfg(AmpEnvCfg):
    def __post_init__(self):
        # Simulation
        self.sim.mujoco.ccd_iterations = 128
        self.sim.contact_sensor_maxmatch = 128
        self.sim.nconmax = 48

        # Scene
        self.scene.entities = {"robot": THS_T2_29DOF_CFG}
        self.scene.sensors = (feet_ground_contact_cfg, self_collision_cfg)
        if self.scene.terrain is not None and self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True

        # Viewer
        self.viewer.body_name = "waist_yaw_link"

        # Event
        self.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
        self.events["base_mass"].params["asset_cfg"].body_names = ("base_link",)
        self.events["body_mass"].params["asset_cfg"].body_names = ("left_.*_link", "right_.*_link")
        self.events["foot_friction"].params["asset_cfg"].geom_names = r"^(left|right)_foot[1-9]_collision$"
        self.events["reset_from_motion_data"].params["motion_data_dir"] = AMP_MOTION_DATA_DIR
        self.events["reset_from_motion_data"].params["motion_data_weights"] = AMP_MOTION_DATA_WEIGHTS

        # Terminations
        self.terminations["delay_bad_orientation"] = TerminationTermCfg(
            func=mdp.delay_bad_orientation, params={"max_delay_steps": 250, "limit_angle": math.radians(70.0)}
        )
        self.terminations["delay_root_height_below_minimum"] = TerminationTermCfg(
            func=mdp.delay_root_height_below_minimum, params={"max_delay_steps": 250, "minimum_height": 0.5}
        )

        # Observation
        self.observations["critic"].terms["body_pos_b"].params["asset_cfg"].body_names = AMP_BODY_NAMES
        self.observations["critic"].terms["body_ori_b"].params["asset_cfg"].body_names = AMP_BODY_NAMES
        self.observations["discriminator"].terms["body_pos_b"].params["asset_cfg"].body_names = AMP_BODY_NAMES
        self.observations["discriminator"].terms["body_ori_b"].params["asset_cfg"].body_names = AMP_BODY_NAMES
        self.observations["discriminator"].terms["body_lin_vel_b"].params["asset_cfg"].body_names = AMP_BODY_NAMES
        self.observations["discriminator"].terms["body_ang_vel_b"].params["asset_cfg"].body_names = AMP_BODY_NAMES
        self.observations["discriminator_expert"].terms["amp_data_discriminator"].params[
            "asset_cfg"
        ].body_names = AMP_BODY_NAMES
        self.observations["discriminator_expert"].terms["amp_data_discriminator"].params["motion_data_dir"] = (
            AMP_MOTION_DATA_DIR
        )
        self.observations["discriminator_expert"].terms["amp_data_discriminator"].params["motion_data_weights"] = (
            AMP_MOTION_DATA_WEIGHTS
        )

        # Action
        self.actions["joint_pos"].scale = THS_T2_29DOF_ACTION_SCALE

        # Basic Reward
        self.rewards["is_alive"].weight = 0
        self.rewards["is_terminated"].weight = -100.0
        self.rewards["joint_torques_l2"].weight = 0
        self.rewards["joint_vel_l2"].weight = -1.0e-5
        self.rewards["joint_acc_l2"].weight = -2.5e-7
        self.rewards["action_rate_l2"].weight = -0.01
        self.rewards["action_acc_l2"].weight = -0.01
        self.rewards["joint_pos_limits"].weight = -10.0
        self.rewards["flat_orientation_l2"].weight = 0
        self.rewards["joint_deviation_exp"].weight = -0.01
        self.rewards["joint_energy"].weight = -2e-5
        self.rewards["track_lin_vel_exp"].weight = 1.0
        self.rewards["track_ang_vel_exp"].weight = 0.5
        # New Reward
        self.rewards["lin_vel_z_l2"] = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-0.1)  # 基座Z 轴 上下线速度
        self.rewards["ang_vel_xy_l2"] = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.1)  # 基座XY轴运动惩罚 -0.1
        # self.rewards["base_height_l2"] = RewardTermCfg(  # 基座高度惩罚
        #     func=mdp.base_height_l2, weight=-0.2, params={"target_height": 0.73}
        # )
        self.rewards["feet_slip"] = RewardTermCfg(  # 脚部滑动惩罚
            func=mdp.feet_slip,
            weight=-0.25,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", site_names=".*_foot_site"),
            },
        )
        self.rewards["self_collision"] = RewardTermCfg(  # 惩罚机器人身体非脚部区域与环境的接触
            func=mdp.self_collision_cost, weight=-0.1, params={"sensor_name": "self_collision", "force_threshold": 10.0}
        )


@dataclass(kw_only=True)
class AmpRoughPlayEnvCfg(AmpRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Effectively infinite episode length.
        self.episode_length_s = int(1e9)

        self.observations["actor"].enable_corruption = False
        self.events.pop("push_robot", None)
        self.curriculum = {}
        self.events["randomize_terrain"] = EventTermCfg(
            func=mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        if self.scene.terrain is not None and self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.border_width = 10.0


@dataclass(kw_only=True)
class AmpFlatEnvCfg(AmpRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.sim.njmax = 1024
        self.sim.mujoco.ccd_iterations = 50
        self.sim.contact_sensor_maxmatch = 256
        self.sim.nconmax = None

        # Switch to flat terrain.
        assert self.scene.terrain is not None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None


@dataclass(kw_only=True)
class AmpFlatPlayEnvCfg(AmpFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        velocity_cmd = self.commands["base_velocity"]
        assert isinstance(velocity_cmd, UniformVelocityCommandCfg)
        velocity_cmd.ranges.lin_vel_x = (-1.5, 3.0)
        velocity_cmd.ranges.lin_vel_y = (-1.0, 1.0)
        velocity_cmd.ranges.ang_vel_z = (-3.14 / 2, 3.14 / 2)
