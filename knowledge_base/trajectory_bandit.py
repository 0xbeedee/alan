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
        self, init_latent_obs: torch.Tensor
    ) -> Tuple[List[Optional[KBBatchProtocol]], List[int]]:
        selected_trajectories = []
        # we need to keep track of the buffers from which we extracted the trajectories
        selected_buffer_ids = []
        # TODO this assumes that the traj_id always correspond to the same trajectories over time, but that is only possible with an infinitely sized buffer!
        # (could i maybe cache the currently saved arms somehow?)
        for buffer_id in range(self.knowledge_base.buffer_num):
            matching_arms = []
            buffer_arms = self.arms[buffer_id]
            for traj_id in range(self.knowledge_base.n_trajectories):
                traj = self.knowledge_base.get_single_trajectory(traj_id, buffer_id)
                if traj is None:
                    continue
                # TODO sometimes init_latent_obs does not have correct batch dims (num_ready_envs in the Collector might be the issue)
                if is_similar(
                    init_latent_obs[buffer_id].unsqueeze(0),
                    traj.latent_obs[0].unsqueeze(0),
                ):
                    if traj_id not in buffer_arms:
                        # if traj_id is new, create the entry; it traj_id is already present, overwrite the entry
                        # TODO doing it this way is somewhat inefficient
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
        total_pulls = sum(arm.pulls for arm in matching_arms)
        ucb_values = []
        for arm in matching_arms:
            arm_pulls = arm.pulls
            arm_estimated_value = arm.estimated_value
            # add an epsilon to avoid division by 0
            bonus = np.sqrt((2 * np.log(total_pulls + 1e-5)) / (arm_pulls + 1e-7))
            ucb_value = arm_estimated_value + bonus
            ucb_values.append(ucb_value)
        selected_arm = matching_arms[np.argmax(ucb_values)]
        return selected_arm

    def update(self, traj_rewards: List[Dict[int, float]]) -> None:
        for buffer_id, buffer_rewards in enumerate(traj_rewards):
            buffer_arms = self.arms[buffer_id]
            for traj_id, reward in buffer_rewards.items():
                arm = buffer_arms.get(traj_id)
                if arm:
                    arm.update_stats(reward)
