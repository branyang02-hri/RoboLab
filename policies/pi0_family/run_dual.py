"""Evaluate two simultaneous DROID policies using one Pi0.5 server."""

import argparse
import sys
import traceback

import cv2  # noqa: F401 -- must precede Isaac Lab
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--remote-host", default="localhost")
parser.add_argument("--remote-port", type=int, default=8000)
parser.add_argument("--remote-uri", default=None)
parser.add_argument("--open-loop-horizon", type=int, default=None)
from robolab.eval.runner import add_common_eval_args

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.task = ["DualDroidBananaThenRubiksCubeTask"]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from policies.pi0_family.dual_client import DualPi05Client  # noqa: E402
from robolab.registrations.droid.dual_robot import register_dual_droid_banana_then_cube  # noqa: E402

register_dual_droid_banana_then_cube()

def main() -> None:
    import os

    import robolab.constants
    from robolab.constants import PACKAGE_DIR, get_timestamp, set_output_dir
    from robolab.core.environments.runtime import create_env
    from robolab.eval.episode import run_episode

    output_name = args_cli.output_folder_name or f"{get_timestamp()}_pi05_dual"
    output_dir = os.path.join(PACKAGE_DIR, "output", output_name)
    os.makedirs(output_dir, exist_ok=True)
    set_output_dir(output_dir)
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True

    env, env_cfg = create_env(
        "DualDroidBananaThenRubiksCubeTask",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        policy="pi05_dual",
        renderer=args_cli.renderer,
        rendering_mode=args_cli.rendering_type,
    )
    kwargs = {
        "remote_host": args_cli.remote_host,
        "remote_port": args_cli.remote_port,
        "remote_uri": args_cli.remote_uri,
        "open_loop_horizon": args_cli.open_loop_horizon,
        "policy_variant": "pi05",
    }
    client = DualPi05Client(**{k: v for k, v in kwargs.items() if v is not None})
    try:
        results, subtask_status, timing = run_episode(
            env,
            env_cfg,
            0,
            client,
            headless=args_cli.headless,
            save_videos=args_cli.video_mode != "none",
            video_mode=args_cli.video_mode,
        )
        result = {
            "results": results,
            "mode": "simultaneous",
            "final_subtask_status": subtask_status[-1] if subtask_status else None,
            "timing": timing,
        }
        result_path = os.path.join(output_dir, "dual_eval_result.json")
        import json
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"DUAL_PI05_RESULT={result_path}", flush=True)
        print(json.dumps(result, indent=2), flush=True)
    finally:
        client.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Dual Pi0.5 evaluation failed: {exc}", flush=True)
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
