"""THS23DOF velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
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

from OneHBC import ONEHBC_ROOT
import OneHBC.tasks.tracking.mdp as mdp
from OneHBC.assets.robots import THS23DOF_CFG, THS23DOF_ACTION_SCALE
from OneHBC.tasks.tracking.tracking_env_cfg import TrackingEnvCfg

from dataclasses import dataclass

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
#                               Environment
# -----------------------------------------------------------------------------


@dataclass(kw_only=True)
class TrackingFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        # Simulation
        self.sim.mujoco.ccd_iterations = 500
        self.sim.contact_sensor_maxmatch = 500
        self.sim.nconmax = 70

        # Scene
        self.scene.entities = {"robot": THS23DOF_CFG}
        self.scene.sensors = (self_collision_cfg,)

        # Viewer
        self.viewer.body_name = "torso_link"

        # Event
        self.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
        self.events["foot_friction"].params["asset_cfg"].geom_names = r"^(left|right)_foot[1-5]_collision$"

        # MDP
        self.observations["actor"].terms.pop("base_lin_vel", None)
        self.observations["actor"].terms.pop("motion_anchor_pos_b", None)
        self.actions["joint_pos"].scale = THS23DOF_ACTION_SCALE

        # Terminations
        self.terminations["ee_body_pos"].params["body_names"] = (".*_ankle_roll_link", ".*_wrist_roll_link")

        # Command
        self.commands["motion"].motion_file = str(
            ONEHBC_ROOT / "robot_assets/ths_23dof/motion_data/dance/dance1_subject2.pkl"
        )
        self.commands["motion"].anchor_body_name = "torso_link"
        self.commands["motion"].body_names = (
            "base_link",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_yaw_link",
            "left_elbow_link",
            "left_wrist_roll_link",
            "right_shoulder_yaw_link",
            "right_elbow_link",
            "right_wrist_roll_link",
        )


@dataclass(kw_only=True)
class TrackingFlatPlayEnvCfg(TrackingFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = int(1e9)

        self.observations["actor"].enable_corruption = False
        self.events.pop("push_robot", None)

        # Disable RSI randomization.
        self.commands["motion"].pose_range = {}
        self.commands["motion"].velocity_range = {}
        self.commands["motion"].sampling_mode = "start"
