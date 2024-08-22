from collections.abc import Callable
from typing import Literal
from abc import abstractmethod

import gymnasium as gym

import numpy as np
from tianshou.data import ReplayBuffer
from tianshou.data.types import BatchWithReturnsProtocol, RolloutBatchProtocol
from tianshou.policy import BasePolicy
from tianshou.policy.base import (
    TLearningRateScheduler,
    TrainingStatsWrapper,
)
from torch import Tensor

from models import SelfModel, EnvModel


class CoreTrainingStats(TrainingStatsWrapper):
    def __init__(self, wrapped_stats):
        # TODO should I add more to this?
        super().__init__(wrapped_stats)


class CorePolicy(BasePolicy[CoreTrainingStats]):
    def __init__(
        self,
        *,
        self_model: SelfModel,
        env_model: EnvModel,
        action_space: gym.Space,
        observation_space: gym.Space | None,
        action_scaling: bool = False,
        action_bound_method: None | Literal["clip"] | Literal["tanh"] = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )

        self.self_model = self_model
        self.env_model = env_model

    @abstractmethod
    def combine_reward(self, batch: RolloutBatchProtocol) -> np.ndarray:
        """Combines the intrinsic and extrinsic rewards into a single scalar value."""

    @classmethod
    def compute_episodic_return(
        cls,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
        v_s_: np.ndarray | Tensor | None = None,
        v_s: np.ndarray | Tensor | None = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        # by default, batch.rew contains the reward provided by the env
        batch.rew = cls.combine_reward(batch)
        return BasePolicy.compute_episodic_return(
            batch, buffer, indices, v_s_, v_s, gamma, gae_lambda
        )

    @classmethod
    def compute_nstep_return(
        cls,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
        target_q_fn: Callable[[ReplayBuffer, np.ndarray], Tensor],
        gamma: float = 0.99,
        n_step: int = 1,
        rew_norm: bool = False,
    ) -> BatchWithReturnsProtocol:
        # by default, batch.rew contains the reward provided by the env
        batch.rew = cls.combine_reward(batch)
        return BasePolicy.compute_nstep_return(
            batch, buffer, indices, target_q_fn, gamma, n_step, rew_norm
        )
