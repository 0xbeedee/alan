from typing import List, Optional, Dict, Tuple
from core.types import KBBatchProtocol

import torch
import numpy as np

from .utils import is_similar


class Arm:
    def __init__(self, trajectory: KBBatchProtocol) -> None:
        # one arm per trajectory
        self.trajectory = trajectory
        # per-trajectory statistics
        self.cumulative_reward = sum(trajectory.rew)
        self.pulls = 0

    def update_stats(self, reward: float) -> None:
        """Updates the statistics of the arm."""
        self.cumulative_reward += reward
        self.pulls += 1  # one pull per trajectory update

    @property
    def estimated_value(self) -> torch.Tensor:
        """Computes the estimated value of the arm."""
        return self.cumulative_reward / self.pulls if self.pulls > 0 else 0.0


class TrajectoryBandit:
    def __init__(self, knowledge_base: "KnowledgeBase"):  # type:ignore
        self.knowledge_base = knowledge_base
        self.arms = [{} for _ in range(self.knowledge_base.buffer_num)]

    def select_trajectories(
        self, init_latent_obs: torch.Tensor, ready_env_ids: np.ndarray
    ) -> Tuple[List[Optional[KBBatchProtocol]], List[int]]:
        """Extracts a subset of candidate trajectories from the knowledge base.

        It returns a list of said trajectories, and a list of containing the buffer_id of the buffer from which these trajectories have been extracted.
        """
        selected_trajectories = []
        # keep track of the buffers from which we extracted the trajectories
        selected_buffer_ids = []
        # TODO this assumes that the traj_id always correspond to the same trajectories over time, but that is only possible with an infinitely sized buffer!
        # (could i maybe cache the currently saved arms somehow?)
        # only update the data for the ready environments (note that env_id and buffer_id are in a bijective correspondance)
        for buffer_id in ready_env_ids:
            matching_arms = []
            buffer_arms = self.arms[buffer_id]
            for traj_id in range(self.knowledge_base.n_trajectories):
                traj = self.knowledge_base.get_single_trajectory(traj_id, buffer_id)
                if traj is None or len(traj) == 0:
                    # skip over the trajectory if it is None OR if it has no transitions
                    continue
                if is_similar(
                    init_latent_obs[buffer_id].unsqueeze(0),
                    traj.latent_obs[0].unsqueeze(0),
                ):
                    if traj_id not in buffer_arms:
                        buffer_arms[traj_id] = Arm(traj)
                    matching_arms.append(buffer_arms[traj_id])
            if matching_arms:
                # select one arm from each buffer
                selected_arm = self._UCB1(matching_arms)
                # TODO technically, we could only save the actions in this array, they are the only ones we need in policy.forward()
                selected_trajectories.append(selected_arm.trajectory)
                selected_buffer_ids.append(buffer_id)

        return selected_trajectories, selected_buffer_ids

    def _UCB1(self, matching_arms: List[Arm]) -> Arm:
        """Implements the UCB1 algorithm (https://link.springer.com/article/10.1023/A:1013689704352)."""
        total_pulls = sum(arm.pulls for arm in matching_arms)
        ucb_values = []
        for arm in matching_arms:
            arm_pulls = arm.pulls
            if arm_pulls == 0:
                bonus = float("inf")  # encourage exploration of untried arms
            else:
                bonus = np.sqrt((2 * np.log(max(total_pulls, 1))) / arm_pulls)
            ucb_value = arm.estimated_value + bonus
            ucb_values.append(ucb_value)
        selected_arm = matching_arms[np.argmax(ucb_values)]
        return selected_arm

    def update(self, traj_rewards: Dict[Tuple[int, int], float]) -> None:
        """Updates the data associated with each arm based on the chosen trajectories."""
        for (buffer_id, traj_id), reward in traj_rewards.items():
            buffer_arms = self.arms[buffer_id]
            arm = buffer_arms.get(traj_id)
            if arm:
                arm.update_stats(reward)
