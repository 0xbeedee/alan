from typing import Any, Literal, cast

from tianshou.data import ReplayBuffer, Batch, to_torch_as
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    DistBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
    BatchWithAdvantagesProtocol,
    LogpOldProtocol,
)
from tianshou.policy.modelfree.ppo import PPOPolicy, TPPOTrainingStats
from tianshou.policy.base import TLearningRateScheduler
from tianshou.utils.net.common import Net

import numpy as np
from gymnasium import Space

import torch
from torch import nn

from models import SelfModel, EnvModel
from core import CorePolicy


class PPOBasedPolicy(CorePolicy):
    """A simple example illustrating how to combine a Tianshou policy (PPO, in this case) and all the additional machinery I've added (i.e, intrinsic motivation and lifelong learning)"""

    # TODO no lifelong learning yet!

    def __init__(
        self,
        *,
        self_model: SelfModel,
        env_model: EnvModel,
        act_net: nn.Module | Net,
        critic_net: nn.Module | Net,
        optim: torch.optim.Optimizer,
        action_space: Space,
        observation_space: Space | None,
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
        batch: RolloutBatchProtocol,
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
    ) -> DistBatchProtocol:
        """Compute action over the given batch data by applying the actor."""

        # we compute the latent observations here for two reasons
        # 1. this way we only compute them once
        # 2. it makes sense for this computation to happen within an agent
        latent_goal = self.self_model.select_goal(batch.obs)
        # TODO this is somewhat hacky, but it provides a cleaner interface with Tianshou
        batch.obs["latent_goal"] = latent_goal

        result = self.ppo_policy.forward(batch, state, **kwargs)
        # result is a Batch
        result.latent_goal = latent_goal
        # TODO this should be a custom type with latent goal (see todo in core/buffer.py)
        return cast(DistBatchProtocol, result)

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:
        batch = super().process_fn(batch, buffer, indices)
        return self.ppo_policy.process_fn(batch, buffer, indices)

    def _dist_fn(self, logits: torch.Tensor):
        return torch.distributions.Categorical(logits=logits)
