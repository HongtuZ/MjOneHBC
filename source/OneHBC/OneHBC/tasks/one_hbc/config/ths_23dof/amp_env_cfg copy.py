"""THS23DOF velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
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
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from OneHBC.assets.robots import THS23DOF_CFG, THS23DOF_ACTION_SCALE
from OneHBC.tasks.one_hbc.one_hbc_env_cfg import OneHBCEnvCfg

from dataclasses import dataclass

# Sensors
feet_ground_cfg = ContactSensorCfg(
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


@dataclass(kw_only=True)
class VelocityRoughEnvCfg(OneHBCEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Simulation
        self.sim.mujoco.ccd_iterations = 500
        self.sim.contact_sensor_maxmatch = 500
        self.sim.nconmax = 70

        # Scene
        self.scene.entities = {"robot": THS23DOF_CFG}
        self.scene.sensors = (self.scene.sensors or ()) + (feet_ground_cfg,)
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
        self.rewards["is_alive"].weight = 1.0
        self.rewards["is_terminated"].weight = -1.0
        self.rewards["joint_torques_l2"].weight = -1.0e-5
        self.rewards["joint_vel_l2"].weight = -1.0e-5
        self.rewards["joint_acc_l2"].weight = -2.5e-7
        self.rewards["action_rate_l2"].weight = -0.01
        self.rewards["action_acc_l2"].weight = -0.01
        self.rewards["joint_pos_limits"].weight = -0.01
        self.rewards["flat_orientation_l2"].weight = -0.01
        self.rewards["track_linear_velocity"].weight = 1.0
        self.rewards["track_angular_velocity"].weight = 0.5

        # New Reward
        self.rewards["feet_air"] = RewardTermCfg(
            func=mdp.air_time_cost,
            weight=-1.0,
            params={"sensor_name": feet_ground_cfg.name, "command_name": "base_velocity", "command_threshold": 0.2},
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
