"""Two DROID robots querying one Pi0.5 server concurrently."""

import numpy as np

from policies.pi0_family.client import Pi0DroidJointposClient


class DualPi05Client:
    """Combine two independent 8D Pi0.5 outputs into one 16D environment action."""

    PRIMARY_PROMPT = "Pick up the Rubik's cube and place it in the red bowl. Do not pick up the banana."
    PARTNER_PROMPT = "Pick up the yellow banana and place it in the red bowl. Do not pick up the Rubik's cube."

    def __init__(self, **client_kwargs):
        # Separate clients retain separate action chunks while sharing one endpoint.
        self.primary = Pi0DroidJointposClient(**client_kwargs)
        self.partner = Pi0DroidJointposClient(**client_kwargs)

    @staticmethod
    def _robot_obs(obs: dict, *, partner: bool) -> dict:
        prefix = "partner" if partner else "primary"
        exterior = "partner_over_shoulder_left_camera" if partner else "over_shoulder_left_camera"
        wrist = "partner_wrist_cam" if partner else "wrist_cam"
        return {
            "image_obs": {
                "over_shoulder_left_camera": obs["image_obs"][exterior],
                "wrist_cam": obs["image_obs"][wrist],
            },
            "proprio_obs": {
                "arm_joint_pos": obs["proprio_obs"][f"{prefix}_arm_joint_pos"],
                "gripper_pos": obs["proprio_obs"][f"{prefix}_gripper_pos"],
            },
        }

    def begin_episode(self, episode_idx: int) -> None:
        self.primary.begin_episode(episode_idx)
        self.partner.begin_episode(episode_idx)
        print(f"[DualPi05] primary_prompt={self.PRIMARY_PROMPT!r}", flush=True)
        print(f"[DualPi05] partner_prompt={self.PARTNER_PROMPT!r}", flush=True)

    def infer(self, obs, _instruction: str, *, env_id: int = 0) -> dict:
        primary = self.primary.infer(
            self._robot_obs(obs, partner=False), self.PRIMARY_PROMPT, env_id=env_id
        )
        partner = self.partner.infer(
            self._robot_obs(obs, partner=True), self.PARTNER_PROMPT, env_id=env_id
        )
        action = np.concatenate((primary["action"], partner["action"]))
        if action.shape != (16,):
            raise ValueError(f"Expected two 8D Pi0.5 actions, got combined shape {action.shape}")

        viz = None
        if primary.get("viz") is not None and partner.get("viz") is not None:
            viz = np.concatenate((primary["viz"], partner["viz"]), axis=0)
        return {"action": action, "viz": viz}

    def infer_batch(self, obs, instruction: str, *, env_ids: list[int]) -> dict[int, dict]:
        return {env_id: self.infer(obs, instruction, env_id=env_id) for env_id in env_ids}

    def reset(self, *, env_id: int | None = None) -> None:
        self.primary.reset(env_id=env_id)
        self.partner.reset(env_id=env_id)

    def close(self) -> None:
        self.primary.close()
        self.partner.close()
