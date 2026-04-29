import math
from dataclasses import replace, dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from . import mdp

##
# MDP settings
##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

commands: dict[str, CommandTermCfg] = {
    "motion": mdp.MotionCommandCfg(
        entity_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        # Override in robot cfg.
        motion_file="",
        anchor_body_name="",
        body_names=(),
    )
}

actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=0.5,  # Override per-robot.
        use_default_offset=True,
    )
}

observations = {
    "actor": ObservationGroupCfg(
        terms={
            "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "motion"}),
            "motion_anchor_pos_b": ObservationTermCfg(
                func=mdp.motion_anchor_pos_b,
                params={"command_name": "motion"},
                noise=Unoise(n_min=-0.25, n_max=0.25),
            ),
            "motion_anchor_ori_b": ObservationTermCfg(
                func=mdp.motion_anchor_ori_b,
                params={"command_name": "motion"},
                noise=Unoise(n_min=-0.05, n_max=0.05),
            ),
            "base_lin_vel": ObservationTermCfg(
                func=mdp.builtin_sensor,
                params={"sensor_name": "robot/imu_lin_vel"},
                noise=Unoise(n_min=-0.5, n_max=0.5),
            ),
            "base_ang_vel": ObservationTermCfg(
                func=mdp.builtin_sensor,
                params={"sensor_name": "robot/imu_ang_vel"},
                noise=Unoise(n_min=-0.2, n_max=0.2),
            ),
            "joint_pos": ObservationTermCfg(
                func=mdp.joint_pos_rel,
                noise=Unoise(n_min=-0.01, n_max=0.01),
                params={"biased": True},
            ),
            "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)),
            "actions": ObservationTermCfg(func=mdp.last_action),
        },
        history_length=1,
        concatenate_terms=True,
        enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
        terms={
            "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "motion"}),
            "motion_anchor_pos_b": ObservationTermCfg(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}),
            "motion_anchor_ori_b": ObservationTermCfg(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}),
            "body_pos": ObservationTermCfg(func=mdp.robot_body_pos_b, params={"command_name": "motion"}),
            "body_ori": ObservationTermCfg(func=mdp.robot_body_ori_b, params={"command_name": "motion"}),
            "base_lin_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}),
            "base_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}),
            "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
            "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
            "actions": ObservationTermCfg(func=mdp.last_action),
        },
        history_length=3,
        concatenate_terms=True,
        enable_corruption=False,
    ),
}

events = {
    "push_robot": EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    ),
    "base_com": EventTermCfg(
        mode="startup",
        func=dr.body_com_offset,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set in robot cfg.
            "operation": "add",
            "ranges": {
                0: (-0.025, 0.025),
                1: (-0.05, 0.05),
                2: (-0.05, 0.05),
            },
        },
    ),
    "encoder_bias": EventTermCfg(
        mode="startup",
        func=dr.encoder_bias,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "bias_range": (-0.01, 0.01),
        },
    ),
    "foot_friction": EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
            "operation": "abs",
            "ranges": (0.3, 1.2),
            "shared_random": True,  # All foot geoms share the same friction.
        },
    ),
    "pd_gains": EventTermCfg(
        mode="startup",
        func=dr.pd_gains,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_joint")),
            "operation": "scale",
            "kp_range": (0.8, 1.2),
            "kd_range": (0.8, 1.2),
        },
    ),
}

rewards = {
    # -- builtin rewards
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
    "joint_limit": RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # -- task rewards
    "motion_global_root_pos": RewardTermCfg(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=2.5,
        params={"command_name": "motion", "std": 0.3},
    ),
    "motion_global_root_ori": RewardTermCfg(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_pos": RewardTermCfg(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    ),
    "motion_body_ori": RewardTermCfg(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    ),
    "self_collisions": RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
}


terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "anchor_pos": TerminationTermCfg(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
        func=mdp.bad_anchor_ori,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "threshold": 0.8,
        },
    ),
    "ee_body_pos": TerminationTermCfg(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": (),  # Set per-robot.
        },
    ),
}

##
# Environment configuration
##


@dataclass(kw_only=True)
class TrackingEnvCfg(ManagerBasedRlEnvCfg):
    # Scene settings
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            env_spacing=2.5,
        )
    )
    # Basic settings
    observations: dict = field(default_factory=lambda: observations)
    actions: dict = field(default_factory=lambda: actions)
    commands: dict = field(default_factory=lambda: commands)
    # MDP settings
    rewards: dict = field(default_factory=lambda: rewards)
    terminations: dict = field(default_factory=lambda: terminations)
    events: dict = field(default_factory=lambda: events)
    viewer: ViewerConfig = field(
        default_factory=lambda: ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # Set per-robot.
            distance=3.0,
            elevation=-5.0,
            fovy=55.0,
            azimuth=120.0,
        )
    )
    sim: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            nconmax=35,
            njmax=250,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
    )
    decimation: int = 4
    episode_length_s: float = 10.0
