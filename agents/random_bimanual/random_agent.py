from __future__ import annotations

from typing import List

import numpy as np
import torch
from yarr.agents.agent import ActResult, Agent, Summary


class RandomBimanualAgent(Agent):
    """Small-noise random policy for RLBench2 pipeline smoke tests."""

    def __init__(
        self,
        position_noise: float = 0.01,
        gripper_flip_prob: float = 0.05,
        include_ignore_collisions: bool = True,
        seed: int = 0,
    ):
        self.position_noise = float(position_noise)
        self.gripper_flip_prob = float(gripper_flip_prob)
        self.include_ignore_collisions = bool(include_ignore_collisions)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def build(self, training: bool, device: torch.device = None):
        self._device = device or torch.device("cpu")

    def reset(self):
        pass

    def act(self, step: int, observation: dict, deterministic=False) -> ActResult:
        if "left_joint_positions" in observation and "right_joint_positions" in observation:
            return ActResult(self._act_joint_position(observation))

        left_pose = observation["left_gripper_pose"].detach().cpu().numpy().reshape(-1).astype(np.float32)
        right_pose = observation["right_gripper_pose"].detach().cpu().numpy().reshape(-1).astype(np.float32)
        left_open = float(observation["left_gripper_open"].detach().cpu().numpy().reshape(-1)[0])
        right_open = float(observation["right_gripper_open"].detach().cpu().numpy().reshape(-1)[0])

        left_pose[:3] += self._rng.normal(0.0, self.position_noise, size=3).astype(np.float32)
        right_pose[:3] += self._rng.normal(0.0, self.position_noise, size=3).astype(np.float32)
        left_pose[3:7] = self._normalize_quat(left_pose[3:7])
        right_pose[3:7] = self._normalize_quat(right_pose[3:7])

        if self._rng.random() < self.gripper_flip_prob:
            left_open = 1.0 - left_open
        if self._rng.random() < self.gripper_flip_prob:
            right_open = 1.0 - right_open

        action = np.concatenate(
            [
                right_pose[:7],
                np.array([np.clip(right_open, 0.0, 1.0), 1.0], dtype=np.float32),
                left_pose[:7],
                np.array([np.clip(left_open, 0.0, 1.0), 1.0], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        return ActResult(action)

    def _act_joint_position(self, observation: dict) -> np.ndarray:
        left_joints = observation["left_joint_positions"].detach().cpu().numpy().reshape(-1).astype(np.float32)
        right_joints = observation["right_joint_positions"].detach().cpu().numpy().reshape(-1).astype(np.float32)
        left_open = float(observation["left_gripper_open"].detach().cpu().numpy().reshape(-1)[0])
        right_open = float(observation["right_gripper_open"].detach().cpu().numpy().reshape(-1)[0])

        left_joints += self._rng.normal(0.0, self.position_noise, size=left_joints.shape).astype(np.float32)
        right_joints += self._rng.normal(0.0, self.position_noise, size=right_joints.shape).astype(np.float32)

        if self._rng.random() < self.gripper_flip_prob:
            left_open = 1.0 - left_open
        if self._rng.random() < self.gripper_flip_prob:
            right_open = 1.0 - right_open

        if self.include_ignore_collisions:
            return np.concatenate(
                [
                    right_joints[:7],
                    np.array([np.clip(right_open, 0.0, 1.0), 1.0], dtype=np.float32),
                    left_joints[:7],
                    np.array([np.clip(left_open, 0.0, 1.0), 1.0], dtype=np.float32),
                ],
                axis=0,
            ).astype(np.float32)

        return np.concatenate(
            [
                right_joints[:7],
                np.array([np.clip(right_open, 0.0, 1.0)], dtype=np.float32),
                left_joints[:7],
                np.array([np.clip(left_open, 0.0, 1.0)], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)

    def update(self, step: int, replay_sample: dict) -> dict:
        return {}

    def update_summaries(self) -> List[Summary]:
        return []

    def act_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str):
        pass

    def save_weights(self, savedir: str):
        pass

    @staticmethod
    def _normalize_quat(quat):
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        quat = quat / norm
        if quat[0] < 0:
            quat = -quat
        return quat.astype(np.float32)
