from typing import Any, Literal

from core.types import (
    GoalBatchProtocol,
    SelfModelProtocol,
    EnvModelProtocol,
)

from tianshou.data import ReplayBuffer
from tianshou.data.types import ObsBatchProtocol
from tianshou.policy.modelfree.dqn import TDQNTrainingStats
from tianshou.policy.base import TLearningRateScheduler

from torch import nn
import numpy as np
import gymnasium as gym

import torch

from networks import GoalActor
from core import CorePolicy
from .dqn_prime import DQNPrimePolicy


class GoalDQN(CorePolicy):
    """A policy based based Tianshou's (double) DQN policy.

    (To use vanilla DQN one must simply disable all the extra modules at experimentation time.)
    """

    def __init__(
        self,
        *,
        self_model: SelfModelProtocol,
        env_model: EnvModelProtocol,
        obs_net: nn.Module,
        model: GoalActor,
        optim: torch.optim.Optimizer,
        action_space: gym.Space,
        observation_space: gym.Space | None,
        action_scaling: bool = False,
        action_bound_method: None | Literal["clip"] | Literal["tanh"] = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
        target_update_freq: int = 0,
        is_double: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            self_model=self_model,
            env_model=env_model,
            obs_net=obs_net,
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )

        self.dqn_policy = DQNPrimePolicy(
            model=model,
            optim=optim,
            action_space=action_space,
            observation_space=observation_space,
            target_update_freq=target_update_freq,
            is_double=is_double,
            lr_scheduler=lr_scheduler,
            **kwargs,
        )

    def _forward(
        self,
        batch: ObsBatchProtocol,
        state: torch.Tensor = None,
        **kwargs: Any,
    ) -> GoalBatchProtocol:
        # somewhat hacky, but it provides a cleaner interface with Tianshou
        batch.obs["latent_goal"] = self.latent_goal

        result = self.dqn_policy.forward(batch, state, **kwargs)
        result.act = torch.as_tensor(result.act)
        result.latent_goal = self.latent_goal
        return result

    def learn(
        self,
        batch: GoalBatchProtocol,
        *args: Any,
        **kwargs: Any,
    ) -> TDQNTrainingStats:
        return self.dqn_policy.learn(batch, *args, **kwargs)

    def process_fn(
        self,
        batch: GoalBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> GoalBatchProtocol:
        batch = super().process_fn(batch, buffer, indices)
        batch.obs["latent_goal"] = batch.latent_goal
        # one goal per observation
        batch.obs_next["latent_goal"] = batch.latent_goal_next
        return self.dqn_policy.process_fn(batch, buffer, indices)
