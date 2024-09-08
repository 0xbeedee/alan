from tianshou.data.types import ObsBatchProtocol
from core.types import GoalBatchProtocol, ObservationNet
from core import GoalReplayBuffer
import gymnasium as gym

import torch
from torch import nn

import numpy as np

from .her import HER


class SelfModel:
    """The SelfModel represents an agent's model of itself.

    It is, fundamentally, a container for all things that should happen exclusively within an agent, independently of the outside world.
    """

    def __init__(
        self,
        obs_net: ObservationNet,
        action_space: gym.Space,
        intrinsic_module: nn.Module,
        buffer: GoalReplayBuffer,
    ) -> None:
        self.intrinsic_module = intrinsic_module(obs_net, action_space)
        self.obs_net = obs_net
        self.buffer = buffer

    # TODO where do I get the horizon from?
    def her(self, batch_size: int, goal_selection_strategy: str = "future"):
        indices = self.buffer.sample_indices(batch_size)
        her = HER(self.buffer, indices, horizon=3)

        achieved_goal = self.sample_goals(self.desired_goal)

        her.rewrite_transitions(achieved_goal, self.desired_goal)

    @torch.no_grad()
    def select_goal(self, batch_obs: ObsBatchProtocol) -> torch.Tensor:
        """Selects a goal for the agent to pursue based on the batch of observations it receives in input."""
        batch_latent_obs = self.obs_net.forward(batch_obs)

        # TODO placeholder (although randomness can work better than expected at times...)
        random_idx = torch.randint(0, batch_latent_obs.shape[0], (1,)).item()
        goal = batch_latent_obs[random_idx]
        # need to return a batch of goals, not a single one
        return goal.repeat(batch_latent_obs.shape[0], 1)

    @torch.no_grad()
    def compute_intrinsic_reward(self, batch: GoalBatchProtocol) -> np.ndarray:
        return self.intrinsic_module.forward(batch)

    def __call__(self, batch: GoalBatchProtocol, sleep: bool = False):
        # modify the buffer by adding the goals
        # self.her(len(batch))
        # TODO do something with sleep
        pass

    # def sample_goals(
    #     self, indices: np.ndarray, goal_selection_strategy: str
    # ) -> np.ndarray:
    #     """Sample goals based on goal_selection_strategy."""

    #     batch_ep_start = self.ep_start[indices]
    #     batch_ep_length = self.ep_length[indices]

    #     if goal_selection_strategy == "final":
    #         # Replay with final state of current episode
    #         transition_indices_in_episode = batch_ep_length - 1

    #     elif goal_selection_strategy == "future":
    #         # Replay with random state which comes from the same episode and was observed after current transition
    #         # Note: our implementation is inclusive: current transition can be sampled
    #         current_indices_in_episode = (indices - batch_ep_start) % self.buffer_size
    #         transition_indices_in_episode = np.random.randint(
    #             current_indices_in_episode, batch_ep_length
    #         )

    #     elif goal_selection_strategy == "episode":
    #         # Replay with random state which comes from the same episode as current transition
    #         transition_indices_in_episode = np.random.randint(0, batch_ep_length)

    #     else:
    #         raise ValueError(
    #             f"Strategy {goal_selection_strategy} for sampling goals not supported!"
    #         )

    #     transition_indices = (
    #         transition_indices_in_episode + batch_ep_start
    #     ) % self.buffer_size
    #     return self.next_observations["achieved_goal"][transition_indices]
