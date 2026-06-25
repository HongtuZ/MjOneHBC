"""THS_T2_29DOF constants."""

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from OneHBC import ONEHBC_ROOT

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

EC_A6416_EFFORT = 132
EC_A4315_EFFORT = 75
EC_A6408_EFFORT = 45
EC_A4310_EFFORT = 36
EC_A2806_EFFORT = 12

EC_A6416_ARMATURE = 0.10268
EC_A4315_ARMATURE = 0.02555
EC_A6408_ARMATURE = 0.06346
EC_A4310_ARMATURE = 0.01869
EC_A2806_ARMATURE = 0.00430

EC_A6416_STIFFNESS = EC_A6416_ARMATURE * NATURAL_FREQ**2
EC_A4315_STIFFNESS = EC_A4315_ARMATURE * NATURAL_FREQ**2
EC_A6408_STIFFNESS = EC_A6408_ARMATURE * NATURAL_FREQ**2
EC_A4310_STIFFNESS = EC_A4310_ARMATURE * NATURAL_FREQ**2
EC_A2806_STIFFNESS = EC_A2806_ARMATURE * NATURAL_FREQ**2

EC_A6416_DAMPPING = 2.0 * DAMPING_RATIO * EC_A6416_ARMATURE * NATURAL_FREQ
EC_A4315_DAMPPING = 2.0 * DAMPING_RATIO * EC_A4315_ARMATURE * NATURAL_FREQ
EC_A6408_DAMPPING = 2.0 * DAMPING_RATIO * EC_A6408_ARMATURE * NATURAL_FREQ
EC_A4310_DAMPPING = 2.0 * DAMPING_RATIO * EC_A4310_ARMATURE * NATURAL_FREQ
EC_A2806_DAMPPING = 2.0 * DAMPING_RATIO * EC_A2806_ARMATURE * NATURAL_FREQ

##
# MJCF and assets.
##

THS_T2_29DOF_CFG = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_file(str(ONEHBC_ROOT / "robot_assets/ths_t2_29dof/urdf/ths_t2_29dof.xml")),
    init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            "left_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.0,
            "left_ankle_pitch_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.0,
            "right_ankle_pitch_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "waist_roll_joint": 0.0,
            "head_yaw_joint": 0.0,
            "head_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
        soft_joint_pos_limit_factor=0.90,
        actuators=(
            BuiltinPositionActuatorCfg(
                target_names_expr=(
                    ".*_hip_pitch_joint",
                    ".*_hip_roll_joint",
                    ".*_knee_joint",
                ),
                stiffness=EC_A6416_EFFORT,
                damping=EC_A6416_DAMPPING,
                armature=EC_A6416_ARMATURE,
                effort_limit=EC_A6416_EFFORT,
                delay_min_lag=0,
                delay_max_lag=2,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(
                    "waist_pitch_joint",
                    "waist_roll_joint",
                ),
                stiffness=EC_A4315_STIFFNESS,
                damping=EC_A4315_DAMPPING,
                armature=EC_A4315_ARMATURE,
                effort_limit=EC_A4315_EFFORT,
                delay_min_lag=0,
                delay_max_lag=2,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(".*_hip_yaw_joint",),
                stiffness=EC_A6408_STIFFNESS,
                damping=EC_A6408_DAMPPING,
                armature=EC_A6408_ARMATURE,
                effort_limit=EC_A6408_EFFORT,
                delay_min_lag=0,
                delay_max_lag=2,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(
                    ".*_ankle_pitch_joint",
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                ),
                stiffness=EC_A4310_STIFFNESS,
                damping=EC_A4310_DAMPPING,
                armature=EC_A4310_ARMATURE,
                effort_limit=EC_A4310_EFFORT,
                delay_min_lag=0,
                delay_max_lag=2,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(
                    ".*_ankle_roll_joint",
                    ".*_wrist_pitch_joint",
                    "head_yaw_joint",
                    "head_pitch_joint",
                ),
                stiffness=EC_A2806_STIFFNESS,
                damping=EC_A2806_DAMPPING,
                armature=EC_A2806_ARMATURE,
                effort_limit=EC_A2806_EFFORT,
                delay_min_lag=0,
                delay_max_lag=2,
            ),
        ),
    ),
)

THS_T2_29DOF_ACTION_SCALE: dict[str, float] = {}
for a in THS_T2_29DOF_CFG.articulation.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.target_names_expr
    assert e is not None
    for n in names:
        THS_T2_29DOF_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(THS_T2_29DOF_CFG)

    viewer.launch(robot.spec.compile())
