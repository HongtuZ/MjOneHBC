"""THS Ostrich velocity environment configurations."""

from dataclasses import dataclass

from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from OneHBC.assets.robots import THS_OSTRICH_ACTION_SCALE, THS_OSTRICH_CFG
from OneHBC.tasks.velocity import mdp
from OneHBC.tasks.velocity.velocity_env_cfg import VelocityEnvCfg
from mjlab.managers.termination_manager import TerminationTermCfg

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
        pattern=r"^(left_foot_link|right_foot_link)$",
        entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
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

unexpected_ground_contact_cfg = ContactSensorCfg(
    name="unexpected_ground_contact",
    primary=ContactMatch(
        mode="body",
        pattern=r"^(left_hip_pitch_link|right_hip_pitch_link|left_hip_yaw_link|right_hip_yaw_link|base_link)$",
        entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
)


@dataclass(kw_only=True)
class VelocityRoughEnvCfg(VelocityEnvCfg):
    def __post_init__(self):
        # Simulation
        self.sim.mujoco.ccd_iterations = 500
        self.sim.contact_sensor_maxmatch = 500
        self.sim.nconmax = 70

        # Scene
        self.scene.entities = {"robot": THS_OSTRICH_CFG}
        self.scene.sensors = (foot_height_scan_cfg, feet_ground_contact_cfg, self_collision_cfg, unexpected_ground_contact_cfg)
        if self.scene.terrain is not None and self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True

        # Action
        joint_pos_action = self.actions["joint_pos"]
        assert isinstance(joint_pos_action, JointPositionActionCfg)
        joint_pos_action.scale = THS_OSTRICH_ACTION_SCALE

        # Viewer
        self.viewer.body_name = "base_link"
        velocity_cmd = self.commands["base_velocity"]
        assert isinstance(velocity_cmd, UniformVelocityCommandCfg)
        velocity_cmd.viz.z_offset = 1.0

        # Event
        self.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
        self.events["foot_friction"].params["asset_cfg"].geom_names = r"^(left|right)_foot[1-3][0-9]_collision"

        # Termination
        self.terminations["unexpected_ground_collision"] = TerminationTermCfg(
            func=mdp.unexpected_collision,
            params={"sensor_name": unexpected_ground_contact_cfg.name},
        )


        # Reward
        self.rewards["pose"].params["std_standing"] = {".*": 0.05}
        self.rewards["pose"].params["std_walking"] = {
            r".*hip_pitch.*": 0.3,
            r".*hip_roll.*": 0.15,
            r".*hip_yaw.*": 0.15,
            r".*knee.*": 0.35,
            r".*ankle.*": 0.25,
        }
        self.rewards["pose"].params["std_running"] = {
            # Lower body.
            r".*hip_pitch.*": 0.6,
            r".*hip_roll.*": 0.3,
            r".*hip_yaw.*": 0.3,
            r".*knee.*": 0.7,
            r".*ankle.*": 0.5,
        }

        self.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
        self.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

        for reward_name in ["foot_clearance", "foot_slip"]:
            self.rewards[reward_name].params["asset_cfg"].site_names = ("left_foot_site", "right_foot_site")

        self.rewards["body_ang_vel"].weight = -0.05
        self.rewards["angular_momentum"].weight = -0.02

        self.rewards["self_collisions"] = RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-1.0,
            params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
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
        self.sim.njmax = 500
        self.sim.mujoco.ccd_iterations = 50
        self.sim.contact_sensor_maxmatch = 128
        self.sim.nconmax = 128

        # Switch to flat terrain.
        assert self.scene.terrain is not None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.terminations.pop("out_of_terrain_bounds", None)
        self.curriculum.pop("terrain_levels", None)


@dataclass(kw_only=True)
class VelocityFlatPlayEnvCfg(VelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        velocity_cmd = self.commands["base_velocity"]
        assert isinstance(velocity_cmd, UniformVelocityCommandCfg)
        velocity_cmd.ranges.lin_vel_x = (-1.5, 2.0)
        velocity_cmd.ranges.ang_vel_z = (-0.7, 0.7)
