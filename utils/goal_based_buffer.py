from typing import Any

import numpy as np

from tianshou.data import (
    ReplayBuffer,
    ReplayBufferManager,
)
from tianshou.data.types import RolloutBatchProtocol


class GoalReplayBuffer(ReplayBuffer):
    _reserved_keys = (
        "obs",
        "act",
        "latent_obs",
        "latent_goal",
        "rew",
        "terminated",
        "truncated",
        "done",
        "obs_next",
        "info",
        "policy",
    )

    _input_keys = (
        "obs",
        "act",
        "latent_obs",
        "latent_goal",
        "rew",
        "terminated",
        "truncated",
        "obs_next",
        "info",
        "policy",
    )


class GoalReplayBufferManager(ReplayBufferManager, GoalReplayBuffer):
    def __init__(self, buffer_list: list[GoalReplayBuffer]) -> None:
        super().__init__(buffer_list)

    def add(
        self,
        batch: RolloutBatchProtocol,
        buffer_ids: np.ndarray | list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return ReplayBufferManager.add(self, batch, buffer_ids)


class GoalVectorReplayBuffer(GoalReplayBufferManager):
    """Goal based variant of the Tianshou buffer."""

    def __init__(self, total_size: int, buffer_num: int, **kwargs: Any) -> None:
        assert buffer_num > 0
        size = int(np.ceil(total_size / buffer_num))
        buffer_list = [GoalReplayBuffer(size, **kwargs) for _ in range(buffer_num)]
        super().__init__(buffer_list)
