from typing import List, Optional

import torch
import numpy as np

from core.types import KBBatchProtocol
from .utils import is_similar


class TrajectoryBandit:
    """A bandit which selects which trajectory from the knowledge base the agent should use."""

    def __init__(self, knowledge_base: "KnowledgeBase"):  # type: ignore
        self.knowledge_base = knowledge_base
        self.arms = {}  # {traj_id: Arm}

    def select_trajectory(
        self, init_latent_obs: torch.Tensor
    ) -> Optional[KBBatchProtocol]:
        """Extracts a subset of candidate trajectories from the knowledge base."""
        trajectories = self.knowledge_base.get_all_trajectories()

        matching_arms = []
        for buffer_traj in trajectories:
            for traj in buffer_traj:
                if is_similar(init_latent_obs, traj[0].latent_obs):
                    traj_id = traj[0].traj_id
                    if traj_id not in self.arms:
                        self.arms[traj_id] = Arm(traj)
                    matching_arms.append(self.arms[traj_id])
        if not matching_arms:
            return None

        selected_arm = self._UCB1(matching_arms)
        return selected_arm.trajectory

    def _UCB1(self, matching_arms: List[KBBatchProtocol]):
        """Implements the UCB1 algorithm (https://link.springer.com/article/10.1023/A:1013689704352)."""
        total_pulls = sum(arm.pulls for arm in matching_arms)
        ucb_values = [
            (
                arm.estimated_value + np.sqrt((2 * np.log10(total_pulls)) / arm.pulls)
                if arm.pulls > 0
                else float("inf")
            )
            for arm in matching_arms
        ]
        selected_arm = matching_arms[np.argmax(ucb_values)]
        return selected_arm

    def update(self, traj_id: int, reward: int):
        """Updates the statistics of each arm."""
        arm = self.arms.get(traj_id)
        if arm:
            arm.update(reward)


class Arm:
    def __init__(
        self,
        trajectory: KBBatchProtocol,
        cumulative_reward: float = 0.0,
        pulls: int = 0,
    ):
        self.trajectory = trajectory
        self.traj_id = trajectory[0].traj_id  # all traj_ids are the same
        self.cumulative_reward = cumulative_reward
        self.pulls = pulls

    def update(self, reward: int):
        """Updates the statistics associated with each arm."""
        self.cumulative_reward += reward
        self.pulls += 1

    @property
    def estimated_value(self):
        return self.cumulative_reward / self.pulls if self.pulls > 0 else 0.0
