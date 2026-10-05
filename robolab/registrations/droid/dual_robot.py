"""Task-local registration for the facing dual-DROID environment."""

from robolab.constants import TASK_DIR


def register_dual_droid_banana_then_cube() -> str:
    from robolab.core.environments.factory import create_env_cfg
    from robolab.core.observations.observation_utils import (
        generate_image_obs_from_cameras,
        generate_obs_cfg,
    )
    from robolab.robots.dual_droid import (
        DualDroidCfg,
        DualDroidJointPositionActionCfg,
        DualDroidProprioceptionObservationCfg,
        DualDroidViewportCameraCfg,
        PartnerOverShoulderCameraCfg,
        dual_contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import OverShoulderLeftCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    ImageObsCfg = generate_image_obs_from_cameras(
        [DualDroidCfg, OverShoulderLeftCameraCfg, PartnerOverShoulderCameraCfg]
    )
    ViewportObsCfg = generate_image_obs_from_cameras(DualDroidViewportCameraCfg)
    ObservationCfg = generate_obs_cfg(
        {
            "image_obs": ImageObsCfg(),
            "proprio_obs": DualDroidProprioceptionObservationCfg(),
            "viewport_cam": ViewportObsCfg(),
        }
    )

    env_name = "DualDroidBananaThenRubiksCubeTask"
    create_env_cfg(
        "BananaThenRubiksCubeTask",
        task_dir=TASK_DIR,
        env_name=env_name,
        tags=["dual_robot"],
        observations_cfg=ObservationCfg(),
        actions_cfg=DualDroidJointPositionActionCfg(),
        robot_cfg=DualDroidCfg,
        camera_cfg=[
            OverShoulderLeftCameraCfg,
            PartnerOverShoulderCameraCfg,
            DualDroidViewportCameraCfg,
        ],
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=dual_contact_gripper,
        dt=1 / 120,
        render_interval=8,
        decimation=8,
        seed=1,
    )
    return env_name
