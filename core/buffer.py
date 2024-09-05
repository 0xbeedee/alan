from typing import Any, cast

import numpy as np

from tianshou.data import ReplayBuffer, ReplayBufferManager, Batch
from tianshou.data.types import RolloutBatchProtocol


class GoalReplayBuffer(ReplayBuffer):
    _reserved_keys = (
        "obs",
        "act",
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
        "latent_goal",
        "rew",
        "terminated",
        "truncated",
        "obs_next",
        "info",
        "policy",
    )

    def __getitem__(
        self, index: slice | int | list[int] | np.ndarray
    ) -> RolloutBatchProtocol:
        if isinstance(index, slice):  # change slice to np array
            # buffer[:] will get all available data
            indices = (
                self.sample_indices(0)
                if index == slice(None)
                else self._indices[: len(self)][index]
            )
        else:
            indices = index  # type: ignore
        # raise KeyError first instead of AttributeError,
        # to support np.array([ReplayBuffer()])
        obs = self.get(indices, "obs")
        if self._save_obs_next:
            obs_next = self.get(indices, "obs_next", Batch())
        else:
            obs_next = self.get(self.next(indices), "obs", Batch())
        batch_dict = {
            "obs": obs,
            "act": self.act[indices],
            "latent_goal": self.latent_goal[indices],
            "rew": self.rew[indices],
            "terminated": self.terminated[indices],
            "truncated": self.truncated[indices],
            "done": self.done[indices],
            "obs_next": obs_next,
            "info": self.get(indices, "info", Batch()),
            "policy": self.get(indices, "policy", Batch()),
        }
        for key in self._meta.__dict__:
            if key not in self._input_keys:
                batch_dict[key] = self._meta[key][indices]
        # TODO should make new goal-aware types! (and apply them everywhere!!)
        return cast(RolloutBatchProtocol, Batch(batch_dict))


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
