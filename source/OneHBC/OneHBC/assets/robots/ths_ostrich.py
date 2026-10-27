"""THS_OSTRICH constants（手臂已在 ostrich.xml 中焊死，模型只有腿部 10 个自由度）。"""

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from OneHBC import ONEHBC_ROOT

ARMATURE_EC_A10020 = 0.277 * 0.1
ARMATURE_EC_A6416 = 0.0658 * 0.2
ARMATURE_EC_A6408 = 0.058 * 0.2
ARMATURE_EC_A4315 = 0.033 * 0.2

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz     = 62.83185307    平方 = 3947.8417602
DAMPING_RATIO = 2.0

# STIFFNESS_EC_A10020 = ARMATURE_EC_A10020 * NATURAL_FREQ**2  # 218.71043352
# STIFFNESS_EC_A6416 = ARMATURE_EC_A6416 * NATURAL_FREQ**2  # 45.79496442
# STIFFNESS_EC_A6408 = ARMATURE_EC_A6408 * NATURAL_FREQ**2  # 45.79496442
# STIFFNESS_EC_A4315 = ARMATURE_EC_A4315 * NATURAL_FREQ**2  # 18.94964045

# DAMPING_EC_A10020 = 2.0 * DAMPING_RATIO * ARMATURE_EC_A10020 * NATURAL_FREQ  # 13.92353864
# DAMPING_EC_A6416 = 2.0 * DAMPING_RATIO * ARMATURE_EC_A6416 * NATURAL_FREQ  # 2.91539798
# DAMPING_EC_A6408 = 2.0 * DAMPING_RATIO * ARMATURE_EC_A6408 * NATURAL_FREQ  # 2.91539798
# DAMPING_EC_A4315 = 2.0 * DAMPING_RATIO * ARMATURE_EC_A4315 * NATURAL_FREQ  # 1.20637158

STIFFNESS_EC_A10020 = 120
STIFFNESS_EC_A6416 = 100
STIFFNESS_EC_A6408 = 80
STIFFNESS_EC_A4315 = 80

DAMPING_EC_A10020 = 12
DAMPING_EC_A6416 = 10
DAMPING_EC_A6408 = 8
DAMPING_EC_A4315 = 8

##
# MJCF and assets.
##

THS_OSTRICH_CFG = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_file(str(ONEHBC_ROOT / "robot_assets/ths_ostrich/urdf/ostrich.xml")),
    init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        # pos=(0.0, 0.0, 0.725), # for ostrich_final.xml
        joint_pos={
            "left_hip_pitch_joint": 0.52,
            "left_hip_roll_joint": -0.10,
            "left_hip_yaw_joint": 0.31,
            "left_knee_joint": -0.52,
            "left_ankle_pitch_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.52,
            "right_hip_roll_joint": 0.10,
            "right_hip_yaw_joint": -0.31,
            "right_knee_joint": -0.52,
            "right_ankle_pitch_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
        soft_joint_pos_limit_factor=0.95,
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
                effort_limit=300,
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
                target_names_expr=(".*_ankle_pitch_joint",),
                stiffness=STIFFNESS_EC_A6416,
                damping=DAMPING_EC_A6416,
                armature=ARMATURE_EC_A6416,
                effort_limit=120,
                delay_min_lag=0,
                delay_max_lag=2,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(".*_ankle_roll_joint",),
                stiffness=STIFFNESS_EC_A4315,
                damping=DAMPING_EC_A4315,
                armature=ARMATURE_EC_A4315,
                effort_limit=75,
                delay_min_lag=0,
                delay_max_lag=2,
            ),
        ),
    ),
    collisions=(
        CollisionCfg(
            geom_names_expr=(".*_collision",),
            contype=1,
            conaffinity=1,
            condim={r"^(left|right)_ankle_.*_collision$": 3, ".*_collision": 1},
            priority={r"^(left|right)_ankle_.*_collision$": 1, ".*": 0},
            friction={r"^(left|right)_ankle_.*_collision$": (0.8,)},
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
    from mjlab.entity.entity import Entity
    from mujoco import viewer

    robot = Entity(THS_OSTRICH_CFG)

    viewer.launch(robot.spec.compile())
