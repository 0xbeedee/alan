from typing import Any
from tianshou.data.types import ObsBatchProtocol
from core.types import (
    GoalBatchProtocol,
    ObsActNextBatchProtocol,
    FastIntrinsicModuleProtocol,
    SlowIntrinsicModuleProtocol,
)

from tianshou.policy.base import TrainingStats
import numpy as np
import torch
from torch import nn


class SelfModel:
    """The SelfModel represents an agent's model of itself.

    It is, fundamentally, a container for all things that should happen exclusively within an agent, independently of the outside world.
    """

    def __init__(
        self,
        obs_net: nn.Module,
        fast_intrinsic_module: FastIntrinsicModuleProtocol,
        slow_intrinsic_module: SlowIntrinsicModuleProtocol,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.obs_net = obs_net.to(device)
        self.fast_intrinsic_module = fast_intrinsic_module
        self.slow_intrinsic_module = slow_intrinsic_module

    @torch.no_grad()
    def select_goal(self, batch_obs: ObsBatchProtocol) -> np.ndarray:
        """Selects a goal for the agent to pursue based on the batch of observations it receives in input."""
        # the batch_obs contains one (obs, info) pair per environment
        # TODO should pull form the KB (obs)?
        batch_latent_obs = self.obs_net(batch_obs)
        # a basic goal selection mechanism: simply add some gaussian noise
        goal = batch_latent_obs + torch.randn_like(batch_latent_obs)
        # return in numpy format for consistency with the other Batch entries
        return goal.cpu().numpy().astype(np.float32)

    @torch.no_grad
    def fast_intrinsic_reward(self, batch: ObsActNextBatchProtocol) -> np.ndarray:
        """A fast system for computing intrinsic motivation, inspired by the dual process theory (https://en.wikipedia.org/wiki/Dual_process_theory).

        This intrinsic computation happens at collect time, and is somewhat conceptually analogous to Kahneman's System 1.
        """
        return self.fast_intrinsic_module.get_reward(batch)

    @torch.no_grad()
    def slow_intrinsic_reward_(self, indices: np.ndarray) -> np.ndarray:
        """A slow system for computing intrinsic motivation, inspired by the dual process theory (https://en.wikipedia.org/wiki/Dual_process_theory).

        This intrinsic computation happens at update time, and is somewhat conceptually analogous to Kahneman's System 2.
        """
        # we cannot return the reward here because modifying the buffer requires access to its internals
        self.slow_intrinsic_module.rewrite_rewards_(indices)

    def learn(self, batch: GoalBatchProtocol, **kwargs: Any) -> TrainingStats:
        return self.fast_intrinsic_module.learn(batch, **kwargs)
