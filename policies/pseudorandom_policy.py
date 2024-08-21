from typing import Any, Literal, TypeVar

from tianshou.data import ReplayBuffer
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    ActBatchProtocol,
    ActStateBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.data.buffer.base import TBuffer
from policies.core import CoreTrainingStats
from tianshou.policy.base import TLearningRateScheduler, TrainingStats

import gymnasium as gym
import numpy as np
import torch

from models import SelfModel, EnvModel
from .core import CorePolicy


class PseudorandomPolicy(CorePolicy):
    """An extremely simple and highly sub-optimal policy, mostly meant to illustrate how the various components are meant to fit together."""

    def __init__(
        self,
        *,
        action_space: gym.Space,
        observation_space: gym.Space | None = None,
        action_scaling: bool = False,
        action_bound_method: None | Literal["clip"] | Literal["tanh"] = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
        self_model: SelfModel,
        env_model: EnvModel
    ) -> None:
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
            self_model=self_model,
            env_model=env_model,
        )
        # TODO

    def learn(
        self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any
    ) -> CoreTrainingStats:
        raise NotImplementedError

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any
    ) -> ActBatchProtocol | ActStateBatchProtocol:
        raise NotImplementedError

    _TArrOrActBatch = TypeVar("_TArrOrActBatch", bound="np.ndarray | ActBatchProtocol")

    def exploration_noise(
        self, act: _TArrOrActBatch, batch: ObsBatchProtocol
    ) -> _TArrOrActBatch:
        raise NotImplementedError

    def process_fn(
        self, batch: RolloutBatchProtocol, buffer: ReplayBuffer, indices: np.ndarray
    ) -> RolloutBatchProtocol:
        raise NotImplementedError

    def process_buffer(self, buffer: TBuffer) -> TBuffer:
        raise NotImplementedError
