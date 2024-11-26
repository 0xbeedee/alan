from typing import List
from core.types import KBBatchProtocol

import torch

from .knowledge_base import KnowledgeBase
from .utils import is_similar


class TrajectoryBandit:
    """A bandit which selects which trajectory from the knowledge base the agent should use."""

    # TODO

    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self.arms = {}  # map trajectory indices to bandit values

    def select_trajectory(self, init_latent_obs: torch.Tensor) -> List[KBBatchProtocol]:
        """Extracts a subset of candidate trajectories from the knowledge base."""
        trajectories = self.knowledge_base.get_all_trajectories()
        # find matching trajectories
        matching_indices = []
        # TODO this n^2 nesting is not too pleasant...
        for idx, traj_buffers in enumerate(trajectories):
            for traj in traj_buffers:
                if is_similar(init_latent_obs, traj[0].latent_obs):
                    matching_indices.append(idx)
        if not matching_indices:
            return None

        # select trajectory using bandit algorithm
        selected_idx = bandit_algorithm(matching_indices, self.arms)
        return trajectories[selected_idx]

    def update_bandit(self, trajectory_idx, reward):
        # TODO update the bandit values based on the received reward
        pass


def bandit_algorithm(matching_indices, arms):
    # TODO bandit algorithm
    pass
