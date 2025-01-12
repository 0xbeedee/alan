from typing import List, Optional, Dict, Tuple
from core.types import KBBatchProtocol

import torch
import numpy as np
from torch import nn
from tianshou.data import Batch

from .utils import is_similar


class Arm:
    def __init__(self, trajectory: Optional[KBBatchProtocol]) -> None:
        # one arm per trajectory
        self.trajectory = trajectory
        # per-trajectory statistics
        if self.trajectory is not None:
            self.cumulative_reward = sum(trajectory.rew)
            self.traj_length = len(trajectory)
        else:
            self.cumulative_reward = 0.0
            self.traj_length = 0
        self.pulls = 0

    def update_stats(self, reward: float) -> None:
        """Updates the statistics of the arm."""
        self.cumulative_reward += reward
        self.pulls += 1  # one pull per trajectory update

    @property
    def estimated_value(self) -> torch.Tensor:
        """Computes the estimated value of the arm."""
        if self.pulls > 0 and self.traj_length > 0:
            # include traj_length to penalise longer trajectories
            return self.cumulative_reward / (self.pulls * self.traj_length)
        else:
            return 0.0


class TrajectoryBandit:
    def __init__(self, knowledge_base: "KnowledgeBase"):  # type:ignore
        self.knowledge_base = knowledge_base
        self.global_arm_id = 1
        # global dict tracking all the available arms with a unique ID
        self.all_arms = {}  # {arm_id: Arm}
        # per-buffer mapping from traj_id to global arm_id
        self.arm_ids_by_buffer = {}  # {buffer_id: {traj_id: arm_id}}

        # add a "null" arm, allowing the bandit to ignore the trajectories available
        self.null_arm_id = 0
        self.all_arms[self.null_arm_id] = Arm(None)

    def select_trajectories(
        self, init_env_obs: Batch, ready_env_ids: np.ndarray, obs_net: nn.Module
    ) -> Tuple[List[Optional[KBBatchProtocol]], List[int]]:
        """Extracts a subset of candidate trajectories from the knowledge base.

        It returns a list of said trajectories, and a list containing the buffer_ids of the buffers from which these trajectories have been extracted.
        """
        selected_trajectories = []
        # keep track of the buffers from which we extracted the trajectories
        selected_buffer_ids = []
        for buffer_id in ready_env_ids:
            # only update the data for the ready environments (env_id and buffer_id are one-to-one)
            if buffer_id not in self.arm_ids_by_buffer:
                self.arm_ids_by_buffer[buffer_id] = {}
            arm_ids_by_traj = self.arm_ids_by_buffer[buffer_id]

            # always include the null arm
            matching_arms = [self.all_arms[self.null_arm_id]]
            for traj_id in range(self.knowledge_base.n_trajectories - 1):
                # only consider complete trajectories
                traj = self.knowledge_base.get_single_trajectory(traj_id, buffer_id)
                if traj is None or len(traj) == 0 or sum(traj.rew) <= 0:
                    continue
                if is_similar(obs_net, init_env_obs[buffer_id], traj.obs[0]):
                    # check if it's been assigned a global arm ID
                    if traj_id in arm_ids_by_traj:
                        arm_id = arm_ids_by_traj[traj_id]
                        arm = self.all_arms[arm_id]
                        if arm.trajectory != traj:
                            # overwrite in global dict and buffer reference
                            updated_arm = Arm(traj)
                            self.all_arms[arm_id] = updated_arm
                            matching_arms.append(updated_arm)
                        else:
                            # same arm as before
                            matching_arms.append(arm)
                    else:
                        # new trajectory, so assign a new global ID and store it
                        self.global_arm_id += 1
                        new_arm = Arm(traj)
                        self.all_arms[self.global_arm_id] = new_arm
                        arm_ids_by_traj[traj_id] = self.global_arm_id
                        matching_arms.append(new_arm)
            if matching_arms:
                # select one arm from each buffer
                selected_arm = self._UCB1(matching_arms)
                selected_trajectories.append(selected_arm.trajectory)
                selected_buffer_ids.append(buffer_id)

        return selected_trajectories, selected_buffer_ids

    def get_arm_id(self, buffer_id: int, traj_id: int) -> Optional[int]:
        """Returns the unique arm ID, given the buffer ID and the trajectory ID, or None if the arm ID does not exist."""
        if buffer_id in self.arm_ids_by_buffer:
            if traj_id in self.arm_ids_by_buffer[buffer_id]:
                return self.arm_ids_by_buffer[buffer_id][traj_id]
        return None

    def update(self, traj_rewards: Dict[int, float]) -> None:
        """Updates the data associated with each arm based on the chosen trajectories."""
        for arm_id, reward in traj_rewards.items():
            arm = self.all_arms[arm_id]
            arm.update_stats(reward)

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
