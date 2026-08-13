# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from copy import deepcopy
from dataclasses import dataclass, field, replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from . import mdp

##
# MDP settings
##

AMP_INPUT_STEPS = 3  # Number of past steps to include in the input (for amp discriminator).


commands: dict[str, CommandTermCfg] = {
    "base_velocity": UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.25,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 3.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-math.pi / 2, math.pi / 2),
            heading=(-math.pi / 2, math.pi / 2),
        ),
    ),
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
            "base_ang_vel": ObservationTermCfg(
                func=mdp.base_ang_vel,
                noise=Unoise(n_min=-0.2, n_max=0.2),
            ),
            "projected_gravity": ObservationTermCfg(
                func=mdp.projected_gravity,
                noise=Unoise(n_min=-0.05, n_max=0.05),
            ),
            "velocity_commands": ObservationTermCfg(
                func=mdp.generated_commands,
                params={"command_name": "base_velocity"},
            ),
            "joint_pos": ObservationTermCfg(
                func=mdp.joint_pos_rel,
                noise=Unoise(n_min=-0.01, n_max=0.01),
            ),
            "joint_vel": ObservationTermCfg(
                func=mdp.joint_vel_rel,
                noise=Unoise(n_min=-1.5, n_max=1.5),
            ),
            "actions": ObservationTermCfg(func=mdp.last_action),
        },
        history_length=1,
        concatenate_terms=True,
        enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
        terms={
            "base_lin_vel": ObservationTermCfg(
                func=mdp.base_lin_vel,
            ),
            "base_ang_vel": ObservationTermCfg(
                func=mdp.base_ang_vel,
            ),
            "body_pos_b": ObservationTermCfg(
                func=mdp.robot_body_pos_b,
                params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # set per robot
            ),
            "body_ori_b": ObservationTermCfg(
                func=mdp.robot_body_ori_b,
                params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # set per robot
            ),
            "projected_gravity": ObservationTermCfg(
                func=mdp.projected_gravity,
            ),
            "velocity_commands": ObservationTermCfg(
                func=mdp.generated_commands,
                params={"command_name": "base_velocity"},
            ),
            "joint_pos": ObservationTermCfg(
                func=mdp.joint_pos_rel,
            ),
            "joint_vel": ObservationTermCfg(
                func=mdp.joint_vel_rel,
            ),
            "actions": ObservationTermCfg(func=mdp.last_action),
        },
        history_length=3,
        concatenate_terms=True,
        enable_corruption=False,
    ),
    "discriminator": ObservationGroupCfg(
        terms={
            "body_pos_b": ObservationTermCfg(
                func=mdp.robot_body_pos_b,
                params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # set per robot
            ),
            "body_ori_b": ObservationTermCfg(
                func=mdp.robot_body_ori_b,
                params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # set per robot
            ),
            "body_lin_vel_b": ObservationTermCfg(
                func=mdp.robot_body_lin_vel_b,
                params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # set per robot
            ),
            "body_ang_vel_b": ObservationTermCfg(
                func=mdp.robot_body_ang_vel_b,
                params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # set per robot
            ),
        },
        history_length=AMP_INPUT_STEPS,
        concatenate_terms=True,
        enable_corruption=False,
    ),
    "discriminator_expert": ObservationGroupCfg(
        terms={
            "amp_data_discriminator": ObservationTermCfg(
                func=mdp.amp_motion_data_obs,
                params={
                    "motion_data_dir": None,  # set per robot
                    "motion_data_weights": None,
                    "n_steps": AMP_INPUT_STEPS,
                    "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per robot
                },
            ),
        },
        history_length=1,
        concatenate_terms=True,
        enable_corruption=False,
    ),
}

events = {
    # Start up randomization
    "base_com": EventTermCfg(
        mode="startup",
        func=dr.body_com_offset,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
            "operation": "add",
            "ranges": {
                0: (-0.03, 0.03),
                1: (-0.05, 0.05),
                2: (-0.05, 0.05),
            },
        },
    ),
    "base_mass": EventTermCfg(
        mode="startup",
        func=dr.body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
            "operation": "add",
            "ranges": (-3.0, 3.0),
        },
    ),
    "body_mass": EventTermCfg(
        mode="startup",
        func=dr.body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
            "operation": "scale",
            "ranges": (0.8, 1.2),
        },
    ),
    "encoder_bias": EventTermCfg(
        mode="startup",
        func=dr.encoder_bias,
        params={"bias_range": (-0.01, 0.01)},
    ),
    "foot_friction": EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
            "operation": "abs",
            "ranges": (0.2, 2.0),
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
    # Reset from motion data
    # "reset_from_motion_data": EventTermCfg(
    #     mode="reset",
    #     func=mdp.reset_from_motion_data,
    #     params={
    #         "motion_data_dir": None,  # set per robot
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_joint")),
    #     },
    # ),
    "reset_base": EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.01, 0.05),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    ),
    "reset_robot_joints": EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.2, 0.2),
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    ),
    # Interval randomization
    "push_robot": EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-0.5, -0.5),
                "z": (-0.4, 0.4),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            }
        },
    ),
}

rewards = {
    # -- builtin rewards
    "is_alive": RewardTermCfg(func=mdp.is_alive, weight=1.0),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1.0e-5),
    "joint_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=-1.0e-5),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.01),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "flat_orientation_l2": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-0.01),
    "joint_deviation_exp": RewardTermCfg(
        func=mdp.posture,
        weight=-0.01,
        params={"std": {".*": 0.5}, "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_energy": RewardTermCfg(
        func=mdp.electrical_power_cost, weight=-0.01, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}
    ),
    # -- command tracking
    "track_lin_vel_exp": RewardTermCfg(
        func=mdp.track_lin_vel_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    ),
    "track_ang_vel_exp": RewardTermCfg(
        func=mdp.track_ang_vel_exp, weight=0.5, params={"command_name": "base_velocity", "std": 0.5}
    ),
}


terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
}

##
# Environment configuration
##

metrics = {
    "mean_action_acc": MetricsTermCfg(
        func=mdp.mean_action_acc,
    ),
}

curriculum = {}


@dataclass(kw_only=True)
class AmpEnvCfg(ManagerBasedRlEnvCfg):
    # Scene settings
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=replace(ROUGH_TERRAINS_CFG),
                max_init_terrain_level=5,
            ),
            num_envs=1,
            env_spacing=2.5,
        )
    )
    # Basic settings
    observations: dict = field(default_factory=lambda: deepcopy(observations))
    actions: dict = field(default_factory=lambda: deepcopy(actions))
    commands: dict = field(default_factory=lambda: deepcopy(commands))
    curriculum: dict = field(default_factory=lambda: deepcopy(curriculum))
    # MDP settings
    rewards: dict = field(default_factory=lambda: deepcopy(rewards))
    terminations: dict = field(default_factory=lambda: deepcopy(terminations))
    events: dict = field(default_factory=lambda: deepcopy(events))
    metrics: dict = field(default_factory=lambda: deepcopy(metrics))
    viewer: ViewerConfig = field(
        default_factory=lambda: ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # Set per-robot.
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
            width=720,
            height=480,
        )
    )
    sim: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            nconmax=35,
            njmax=1500,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
    )
    decimation: int = 4
    episode_length_s: float = 20.0
