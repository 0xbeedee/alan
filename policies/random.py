from typing import Any, Literal
from core.types import (
    RandomActBatchProtocol,
    GoalBatchProtocol,
    GoalReplayBufferProtocol,
    SelfModelProtocol,
    EnvModelProtocol,
)

from tianshou.data.types import ObsBatchProtocol
from tianshou.data.batch import Batch, BatchProtocol
from tianshou.policy.base import TLearningRateScheduler

from torch import nn
import torch
import gymnasium as gym
import numpy as np

from core import CorePolicy


class RandomPolicy(CorePolicy):
    """A policy which selects actions at random.

    This policy is mostly meant for offline training, i.e., we use it to collect experience with which we then train the world model, as done in Ha and Schmidhuber (https://arxiv.org/abs/1803.10122).
    """

    def __init__(
        self,
        *,
        self_model: SelfModelProtocol,
        env_model: EnvModelProtocol,
        obs_net: nn.Module,
        action_space: gym.Space,
        observation_space: gym.Space | None,
        action_scaling: bool = False,
        action_bound_method: None | Literal["clip"] | Literal["tanh"] = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
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

    def _forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        pass

    def learn(
        self,
        batch: GoalBatchProtocol,
        batch_size: int | None,
        repeat: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> RandomActBatchProtocol:
        result = Batch()
        result.act = torch.tensor([self.action_space.sample()])

        return result
