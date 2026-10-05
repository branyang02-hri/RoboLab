# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two facing DROID robots and their joint-position action interface."""

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from robolab.robots.droid import (
    BinaryJointPositionZeroToOneActionCfg,
    DroidCfg,
    arm_joint_pos,
    gripper_pos,
)


_partner_robot = DroidCfg().robot.copy()
_partner_robot.prim_path = "{ENV_REGEX_NS}/partner_robot"
_partner_robot.init_state = _partner_robot.init_state.copy()
# The task objects are near x=0.5. Put the second base on the opposite side and
# rotate it to face the first robot.
_partner_robot.init_state.pos = (1.0, 0.0, 0.0)
_partner_robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)

_partner_wrist_cam = DroidCfg().wrist_cam.copy()
_partner_wrist_cam.prim_path = (
    "{ENV_REGEX_NS}/partner_robot/Gripper/Robotiq_2F_85/base_link/partner_wrist_cam"
)


@configclass
class DualDroidCfg(DroidCfg):
    """The stock DROID plus an identical robot across the task table."""

    partner_robot = _partner_robot
    partner_wrist_cam = _partner_wrist_cam


# Labels consumed by the dynamic environment factory. There is one shared table
# fixture; both fixed-base articulations live in the same scene.
DualDroidCfg.table_fixture = DroidCfg.table_fixture
DualDroidCfg.ee_recorder_bodies = {}


@configclass
class DualDroidJointPositionActionCfg:
    """A flat 16D action: primary 7+1 followed by partner 7+1."""

    primary_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
    )
    primary_gripper = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )
    partner_arm = mdp.JointPositionActionCfg(
        asset_name="partner_robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
    )
    partner_gripper = BinaryJointPositionZeroToOneActionCfg(
        asset_name="partner_robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )


@configclass
class DualDroidProprioceptionObservationCfg(ObsGroup):
    """Separate proprioception terms suitable for a future two-robot policy."""

    primary_arm_joint_pos = ObsTerm(
        func=arm_joint_pos, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    primary_gripper_pos = ObsTerm(
        func=gripper_pos, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    partner_arm_joint_pos = ObsTerm(
        func=arm_joint_pos, params={"asset_cfg": SceneEntityCfg("partner_robot")}
    )
    partner_gripper_pos = ObsTerm(
        func=gripper_pos, params={"asset_cfg": SceneEntityCfg("partner_robot")}
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False


dual_contact_gripper = {
    "primary_gripper": "{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/left_inner_finger",
    "partner_gripper": "{ENV_REGEX_NS}/partner_robot/Gripper/Robotiq_2F_85/left_inner_finger",
    "gripper": ["primary_gripper", "partner_gripper"],
}


@configclass
class DualDroidViewportCameraCfg:
    """Wide side view centered between the two facing robots."""

    dual_droid_viewport_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/dual_droid_viewport_camera",
        height=480,
        width=864,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            vertical_aperture=15.29,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.5, 1.6, 1.0),
            rot=(0.0, 0.0, 0.5314, 0.8471),
            convention="opengl",
        ),
    )


@configclass
class PartnerOverShoulderCameraCfg:
    """Partner-relative mirror of the DROID left exterior camera."""

    partner_over_shoulder_left_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/partner_over_shoulder_left_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            # 180-degree transform of the primary camera about the workspace
            # center at x=0.5, so the partner receives a robot-relative view.
            pos=(0.95, -0.57, 0.66),
            rot=(0.805, 0.399, 0.195, 0.393),
            convention="opengl",
        ),
    )
