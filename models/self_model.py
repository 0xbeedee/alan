from typing import Self, Any
from tianshou.data.types import ObsBatchProtocol
from core.types import (
    GoalBatchProtocol,
    ObservationNetProtocol,
    GoalReplayBufferProtocol,
    ObsActNextBatchProtocol,
)

from tianshou.policy.base import TrainingStats
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
        obs_net: ObservationNetProtocol,
        action_space: gym.Space,
        buffer: GoalReplayBufferProtocol,
        intrinsic_module: nn.Module,
        her_horizon: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.obs_net = obs_net.to(device)
        self.buffer = buffer

        # fast intrinsic
        self.intrinsic_module = intrinsic_module(obs_net, action_space, device=device)
        # slow intrinsic
        self.her = HER(self.buffer, horizon=her_horizon)

    @torch.no_grad()
    def select_goal(self, batch_obs: ObsBatchProtocol) -> np.ndarray:
        """Selects a goal for the agent to pursue based on the batch of observations it receives in input."""
        # TODO what if I want to select goals based on something more than observations? => should this be moved to process_fn??
        # TODO should pull form the KB (obs) (?)
        batch_latent_obs = self.obs_net.forward(batch_obs)

        # TODO placeholder (although randomness can work better than expected at times...)
        random_idx = torch.randint(0, batch_latent_obs.shape[0], (1,)).item()
        goal = batch_latent_obs[random_idx]
        # TODO these could actually be different => multi-goal approach?
        # need to return a batch of goals (in numpy format for consistency with the other Batch entries)
        return (
            goal.repeat(batch_latent_obs.shape[0], 1).cpu().numpy().astype(np.float32)
        )

    @torch.no_grad
    def fast_intrinsic_reward(self, batch: ObsActNextBatchProtocol) -> np.ndarray:
        """A fast system for computing intrinsic motivation, inspired by the dual process theory (https://en.wikipedia.org/wiki/Dual_process_theory).

        This intrinsic computation happens at collect time, and is somewhat conceptually analogous to Kahneman's System 1.
        """
        return self.intrinsic_module.get_reward(batch)

    @torch.no_grad()
    def slow_intrinsic_reward_(self, indices: np.ndarray) -> np.ndarray:
        """A slow system for computing intrinsic motivation, inspired by the dual process theory (https://en.wikipedia.org/wiki/Dual_process_theory).

        This intrinsic computation happens at update time, and is somewhat conceptually analogous to Kahneman's System 2.
        """
        # get_future_observation_ alters the indices
        future_obs = self.her.get_future_observation_(indices)
        latent_future_goal = self.obs_net.forward(future_obs)
        # we cannot return the reward here because modifying the buffer requires access to its internals
        self.her.rewrite_transitions_(latent_future_goal.cpu().numpy())

    def learn(self, batch: GoalBatchProtocol, **kwargs: Any) -> TrainingStats:
        return self.intrinsic_module.learn(batch, **kwargs)

    def __call__(self, batch: GoalBatchProtocol, sleep: bool = False):
        # TODO
        pass

    def to(self, device: torch.device) -> Self:
        self.device = device
        self.obs_net = self.obs_net.to(self.device)
        self.intrinsic_module = self.intrinsic_module.to(self.device)
        # note: We don't move self.buffer or self.her as they typically don't contain torch tensors
        return self
