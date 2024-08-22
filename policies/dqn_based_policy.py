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
from tianshou.policy.modelfree.dqn import DQNPolicy
from policies.core import CoreTrainingStats
from tianshou.policy.base import TLearningRateScheduler
from tianshou.utils.net.common import Net

import gymnasium as gym
import numpy as np
import torch

from models import SelfModel, EnvModel
from .core import CorePolicy


class DQNBasedPolicy(CorePolicy):
    """A simple example illustrating how to combine a policy (DQN, in this case) and all the additional machinery I've added (i.e, intrinsic motivation and lifelong learning)"""

    # TODO no lifelong learning yet!

    def __init__(
        self,
        *,
        self_model: SelfModel,
        env_model: EnvModel,
        net: torch.nn.Module | Net,
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

        self.dqn_policy = DQNPolicy(
            model=net,
            optim=optim,
            action_space=action_space,
            observation_space=observation_space,
            # the rest of the values we'll leave to their defaults
        )

    # TODO might need to put this in other methods as well
    def combine_reward(self, batch: RolloutBatchProtocol) -> np.ndarray:
        beta = 0.01  # TODO don't hardcode this!
        i_rew = self.self_model(batch)
        return batch.rew + beta * i_rew

    def learn(
        self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any
    ) -> CoreTrainingStats:
        batch.rew = self.combine_reward(batch)
        return self.dqn_policy.learn(batch, *args, **kwargs)

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        model: Literal["model", "model_old"] = "model",
        **kwargs: Any,
    ) -> ActBatchProtocol | ActStateBatchProtocol:
        return self.dqn_policy.forward(batch, state, model, **kwargs)

    _TArrOrActBatch = TypeVar("_TArrOrActBatch", bound="np.ndarray | ActBatchProtocol")

    def exploration_noise(
        self, act: _TArrOrActBatch, batch: ObsBatchProtocol
    ) -> _TArrOrActBatch:
        return self.dqn_policy.exploration_noise(act, batch)

    def process_fn(
        self, batch: RolloutBatchProtocol, buffer: ReplayBuffer, indices: np.ndarray
    ) -> RolloutBatchProtocol:
        return super().compute_nstep_return(
            batch=batch,
            buffer=buffer,
            indices=indices,
            target_q_fn=self.dqn_policy._target_q,
            gamma=self.dqn_policy.gamma,
            n_step=self.dqn_policy.n_step,
            rew_norm=self.dqn_policy.rew_norm,
        )

    def process_buffer(self, buffer: TBuffer) -> TBuffer:
        return self.dqn_policy.process_buffer(buffer)
