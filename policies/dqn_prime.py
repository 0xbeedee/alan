from typing import Any, Literal, cast

from tianshou.data import Batch, ReplayBuffer, to_numpy
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    ModelOutputBatchProtocol,
    ObsBatchProtocol,
)
from tianshou.policy.modelfree.dqn import DQNPolicy

import numpy as np
import torch


class DQNPrimePolicy(DQNPolicy):
    """A modified DQN policy that better fits ALAN's API."""

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        model: Literal["model", "model_old"] = "model",
        **kwargs: Any,
    ) -> ModelOutputBatchProtocol:
        action_values_BA, hidden_BH = self.model(
            batch.obs, state=state, info=batch.info
        )
        q = self.compute_q_value(action_values_BA, getattr(batch.obs, "mask", None))
        if self.max_action_num is None:
            self.max_action_num = q.shape[1]
        act_B = to_numpy(q.argmax(dim=1))
        result = Batch(logits=action_values_BA, act=act_B, state=hidden_BH)
        return cast(ModelOutputBatchProtocol, result)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=[None] * len(indices),
        )
        obs_next_batch.obs["latent_goal"] = buffer[indices].latent_goal
        result = self(obs_next_batch)
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self(obs_next_batch, model="model_old").logits
        else:
            target_q = result.logits
        if self.is_double:
            return target_q[np.arange(len(result.act)), result.act]
        # Nature DQN, over estimate
        return target_q.max(dim=1)[0]
