"""THS23DOF velocity environment configurations."""

from copy import deepcopy
from dataclasses import dataclass

from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
)
from mjlab.terrains import BoxFlatTerrainCfg, HfRandomUniformTerrainCfg, TerrainGeneratorCfg

from OneHBC import ONEHBC_ROOT
from OneHBC.assets.robots import THS_T2_29DOF_CFG, THS_T2_29DOF_ACTION_SCALE
from OneHBC.tasks.tracking.tracking_env_cfg import TrackingEnvCfg

# -----------------------------------------------------------------------------
#                                  Motions
# -----------------------------------------------------------------------------

motion_file = str(ONEHBC_ROOT / "robot_assets/ths_t2_29dof/motion_data/side_flip/90_08_stageii.pkl")

# -----------------------------------------------------------------------------
#                                   Sensors
# -----------------------------------------------------------------------------

self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
)

# -----------------------------------------------------------------------------
#                                  Terrain
# -----------------------------------------------------------------------------

half_rough_terrain_cfg = TerrainGeneratorCfg(
    size=(5.0, 5.0),
    border_width=20.0,
    num_rows=40,
    num_cols=40,
    # curriculum=True,
    color_scheme="none",
    difficulty_range=(0.0, 1.0),
    add_lights=True,
    sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=0.5),
        "rough": HfRandomUniformTerrainCfg(
            proportion=0.5,
            noise_range=(0, 0.02),
            noise_step=0.0025,
            downsampled_scale=0.1,
            border_width=0.0,
            horizontal_scale=0.1,
            vertical_scale=0.0025,
        ),
    },
)


# -----------------------------------------------------------------------------
#                               Environment
# -----------------------------------------------------------------------------


@dataclass(kw_only=True)
class Tracking_T2_FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        # Simulation
        self.sim.mujoco.ccd_iterations = 500
        self.sim.contact_sensor_maxmatch = 500
        self.sim.nconmax = 70

        # Scene
        self.scene.entities = {"robot": THS_T2_29DOF_CFG}
        self.scene.sensors = (deepcopy(self_collision_cfg),)

        # Viewer
        self.viewer.body_name = "waist_roll_link"

        # Event
        self.events["base_com"].params["asset_cfg"].body_names = ("waist_roll_link",)
        self.events["base_mass"].params["asset_cfg"].body_names = ("waist_roll_link",)
        self.events["body_mass"].params["asset_cfg"].body_names = ("left_.*_link", "right_.*_link")
        self.events["foot_friction"].params["asset_cfg"].geom_names = r"^(left|right)_foot[1-9]_collision$"

        # MDP
        self.observations["actor"].terms.pop("base_lin_vel", None)
        self.observations["actor"].terms.pop("motion_anchor_pos_b", None)
        self.actions["joint_pos"].scale = THS_T2_29DOF_ACTION_SCALE

        # Terminations
        self.terminations["ee_body_pos"].params["body_names"] = (".*_ankle_roll_link", ".*_wrist_pitch_link")

        # Command
        self.commands["motion"].motion_file = motion_file
        self.commands["motion"].anchor_body_name = "base_link"
        self.commands["motion"].body_names = (
            "base_link",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "waist_roll_link",
            "head_pitch_link",
            "left_shoulder_yaw_link",
            "left_elbow_link",
            "left_wrist_pitch_link",
            "right_shoulder_yaw_link",
            "right_elbow_link",
            "right_wrist_pitch_link",
        )


@dataclass(kw_only=True)
class Tracking_T2_FlatPlayEnvCfg(Tracking_T2_FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = int(1e9)

        self.observations["actor"].enable_corruption = False
        self.events.pop("push_robot", None)
        self.events["foot_friction"].params["ranges"] = (1.5, 1.5)

        # Disable RSI randomization.
        self.commands["motion"].pose_range = {}
        self.commands["motion"].velocity_range = {}
        self.commands["motion"].joint_position_range = (0.0, 0.0)
        self.commands["motion"].sampling_mode = "start"
