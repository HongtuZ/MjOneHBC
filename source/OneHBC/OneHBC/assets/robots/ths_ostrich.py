"""THS_OSTRICH constants（手臂已在 ostrich.xml 中焊死，模型只有腿部 10 个自由度）。"""

import mujoco

from OneHBC import ONEHBC_ROOT
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

ARMATURE_EC_A10020 = 0.277 * 0.1
ARMATURE_EC_A6408 = 0.058 * 0.2
ARMATURE_EC_A4310 = 0.024 * 0.2

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz     = 62.83185307    平方 = 3947.8417602
DAMPING_RATIO = 2.0

# STIFFNESS_EC_A10020 = ARMATURE_EC_A10020 * NATURAL_FREQ**2  # 218.71043352
# STIFFNESS_EC_A6408 = ARMATURE_EC_A6408 * NATURAL_FREQ**2  # 45.79496442
# STIFFNESS_EC_A4310 = ARMATURE_EC_A4310 * NATURAL_FREQ**2  # 18.94964045

# DAMPING_EC_A10020 = 2.0 * DAMPING_RATIO * ARMATURE_EC_A10020 * NATURAL_FREQ  # 13.92353864
# DAMPING_EC_A6408 = 2.0 * DAMPING_RATIO * ARMATURE_EC_A6408 * NATURAL_FREQ  # 2.91539798
# DAMPING_EC_A4310 = 2.0 * DAMPING_RATIO * ARMATURE_EC_A4310 * NATURAL_FREQ  # 1.20637158

STIFFNESS_EC_A10020 = 120
STIFFNESS_EC_A6408 = 60
STIFFNESS_EC_A4310 = 30

DAMPING_EC_A10020 = 10
DAMPING_EC_A6408 = 5
DAMPING_EC_A4310 = 3

##
# MJCF and assets.
##

THS_OSTRICH_CFG = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_file(str(ONEHBC_ROOT / "robot_assets/ths_ostrich/urdf/ostrich.xml")),
    init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.62),
        joint_pos={
            "left_hip_pitch_joint": 1.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.35,
            "left_knee_joint": -1.57,
            "left_ankle_joint": 0.6,
            "right_hip_pitch_joint": 1.0,
            "right_hip_roll_joint": -0.0,
            "right_hip_yaw_joint": -0.35,
            "right_knee_joint": -1.57,
            "right_ankle_joint": 0.6,
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
                stiffness=STIFFNESS_EC_A10020,
                damping=DAMPING_EC_A10020,
                armature=ARMATURE_EC_A10020,
                effort_limit=330,
                delay_min_lag=0,
                delay_max_lag=2,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(".*_hip_yaw_joint",),
                stiffness=STIFFNESS_EC_A6408,
                damping=DAMPING_EC_A6408,
                armature=ARMATURE_EC_A6408,
                effort_limit=70,
                delay_min_lag=0,
                delay_max_lag=2,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(".*_ankle_joint",),
                stiffness=STIFFNESS_EC_A4310,
                damping=DAMPING_EC_A4310,
                armature=ARMATURE_EC_A4310,
                effort_limit=36,
                delay_min_lag=0,
                delay_max_lag=2,
            ),
        ),
    ),
)

THS_OSTRICH_ACTION_SCALE: dict[str, float] = {}
for a in THS_OSTRICH_CFG.articulation.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.target_names_expr
    assert e is not None
    for n in names:
        # THS_OSTRICH_ACTION_SCALE[n] = 0.25 * e / s
        THS_OSTRICH_ACTION_SCALE[n] = 0.25


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(THS_OSTRICH_CFG)

    viewer.launch(robot.spec.compile())
