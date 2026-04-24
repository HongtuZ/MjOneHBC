"""THS23DOF constants."""

import mujoco

from OneHBC import ONEHBC_ROOT
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

ARMATURE_E00 = 0.001 * 2
ARMATURE_E02 = 0.0042
ARMATURE_E03 = 0.02
ARMATURE_E06 = 0.012

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz     = 62.83185307    平方 = 3947.8417602
DAMPING_RATIO = 2.0

STIFFNESS_E00 = ARMATURE_E00 * NATURAL_FREQ**2  # 3.94784176       # 4
STIFFNESS_E02 = ARMATURE_E02 * NATURAL_FREQ**2  # 16.58093539      # 16.6
STIFFNESS_E03 = ARMATURE_E03 * NATURAL_FREQ**2  # 78.956835204     # 79
STIFFNESS_E06 = ARMATURE_E06 * NATURAL_FREQ**2  # 47.3741011224    #

DAMPING_E00 = 2.0 * DAMPING_RATIO * ARMATURE_E00 * NATURAL_FREQ  # 0.2513274122
DAMPING_E02 = 2.0 * DAMPING_RATIO * ARMATURE_E02 * NATURAL_FREQ  # 1.0555751315
DAMPING_E03 = 2.0 * DAMPING_RATIO * ARMATURE_E03 * NATURAL_FREQ  # 5.0265482456
DAMPING_E06 = 2.0 * DAMPING_RATIO * ARMATURE_E06 * NATURAL_FREQ  # 3.01592894736

##
# MJCF and assets.
##

THS23DOF_CFG = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_file(str(ONEHBC_ROOT / "robot_assets/ths_23dof/urdf/ths_23dof.xml")),
    init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            "left_hip_pitch_joint": 0.25,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": -0.6,
            "left_ankle_pitch_joint": -0.35,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.25,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.6,
            "right_ankle_pitch_joint": 0.35,
            "right_ankle_roll_joint": 0.0,
            "torso_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 1.4,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.3,
            "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -1.4,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -0.3,
            "right_wrist_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
        soft_joint_pos_limit_factor=0.90,
        actuators=(
            BuiltinPositionActuatorCfg(
                target_names_expr=(
                    ".*_ankle_roll_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                ),
                stiffness=STIFFNESS_E00,
                damping=DAMPING_E00,
                armature=ARMATURE_E00,
                effort_limit=12,
                delay_min_lag=0,
                delay_max_lag=2,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(
                    ".*_ankle_pitch_joint",
                    ".*_shoulder_pitch_joint",
                ),
                stiffness=STIFFNESS_E02,
                damping=DAMPING_E02,
                armature=ARMATURE_E02,
                effort_limit=15,
                delay_min_lag=0,
                delay_max_lag=2,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(
                    ".*_hip_pitch_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_yaw_joint",
                    ".*_knee_joint",
                ),
                stiffness=STIFFNESS_E03,
                damping=DAMPING_E03,
                armature=ARMATURE_E03,
                effort_limit=50,
                delay_min_lag=0,
                delay_max_lag=2,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=("torso_joint",),
                stiffness=STIFFNESS_E06,
                damping=DAMPING_E06,
                armature=ARMATURE_E06,
                effort_limit=20,
                delay_min_lag=0,
                delay_max_lag=2,
            ),
        ),
    ),
)

THS23DOF_ACTION_SCALE: dict[str, float] = {}
for a in THS23DOF_CFG.articulation.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.target_names_expr
    assert e is not None
    for n in names:
        THS23DOF_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(THS23DOF_CFG)

    viewer.launch(robot.spec.compile())
