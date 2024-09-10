from typing import Any, Literal, cast
from core.types import (
    GoalBatchProtocol,
    SelfModelProtocol,
    EnvModelProtocol,
)

from tianshou.data import ReplayBuffer
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ObsBatchProtocol
from tianshou.policy.modelfree.ppo import PPOPolicy, TPPOTrainingStats
from tianshou.policy.base import TLearningRateScheduler

import numpy as np
import gymnasium as gym

import torch

from networks import GoalNetHackActor, GoalNetHackCritic
from core import CorePolicy


class PPOBasedPolicy(CorePolicy):
    """A policy combining a Tianshou PPOPolicy and the CorePolicy.

    It is mostly meant as a blueprint/example for future integration of my code with Tianshou.
    """

    # TODO no lifelong learning yet!

    def __init__(
        self,
        *,
        self_model: SelfModelProtocol,
        env_model: EnvModelProtocol,
        act_net: GoalNetHackActor,
        critic_net: GoalNetHackCritic,
        optim: torch.optim.Optimizer,
        action_space: gym.Space,
        observation_space: gym.Space | None,
        action_scaling: bool = False,
        action_bound_method: None | Literal["clip"] | Literal["tanh"] = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        super().__init__(
            self_model=self_model,
            env_model=env_model,
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )

        self.ppo_policy = PPOPolicy(
            actor=act_net,
            critic=critic_net,
            optim=optim,
            action_space=action_space,
            # we can confidently hardcode these two because we intend to use NLE
            dist_fn=self._dist_fn,
            action_scaling=False,
        )

    def learn(
        self,
        batch: GoalBatchProtocol,
        batch_size: int | None,
        repeat: int,
        *args: Any,
        **kwargs: Any,
    ) -> TPPOTrainingStats:
        return self.ppo_policy.learn(batch, batch_size, repeat, *args, **kwargs)

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> GoalBatchProtocol:
        latent_goal = super().forward(batch, state)

        result = self.ppo_policy.forward(batch, state, **kwargs)
        result.latent_goal = latent_goal
        return cast(GoalBatchProtocol, result)

    def process_fn(
        self,
        batch: GoalBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> GoalBatchProtocol:
        batch = super().process_fn(batch, buffer, indices)
        return self.ppo_policy.process_fn(batch, buffer, indices)

    def _dist_fn(self, logits: torch.Tensor):
        return torch.distributions.Categorical(logits=logits)
