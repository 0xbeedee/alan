from typing import Any, cast
from .types import GoalBatchProtocol

import numpy as np

from tianshou.data import ReplayBuffer, ReplayBufferManager, Batch
from tianshou.data.batch import alloc_by_keys_diff, create_value


class GoalReplayBuffer(ReplayBuffer):
    """The Buffer stores the data generated from the interaction between policy and environment.

    For details, see https://tianshou.org/en/stable/03_api/data/buffer/base.html. On top of Tianshou's implementation, we add the possibility of storing latent goals and intrinsic rewards.
    """

    _reserved_keys = (
        "obs",
        "latent_goal",
        "act",
        "obs_next",
        "latent_goal_next",
        "rew",
        "int_rew",
        "terminated",
        "truncated",
        "done",
        "info",
        "policy",
    )

    _input_keys = (
        "obs",
        "latent_goal",
        "act",
        "obs_next",
        "latent_goal_next",
        "rew",
        "int_rew",
        "terminated",
        "truncated",
        "info",
        "policy",
    )

    def __init__(
        self,
        size: int,
        stack_num: int = 1,
        ignore_obs_next: bool = False,
        save_only_last_obs: bool = False,
        sample_avail: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(
            size,
            stack_num=stack_num,
            ignore_obs_next=ignore_obs_next,
            save_only_last_obs=save_only_last_obs,
            sample_avail=sample_avail,
            **kwargs
        )

    def __getitem__(
        self, index: slice | int | list[int] | np.ndarray
    ) -> GoalBatchProtocol:
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
            latent_goal_next = self.get(indices, "latent_goal_next", Batch())
        else:
            obs_next = self.get(self.next(indices), "obs", Batch())
            latent_goal_next = self.get(indices, "latent_goal", Batch())

        batch_dict = {
            "obs": obs,
            "latent_goal": self.latent_goal[indices],
            "act": self.act[indices],
            "obs_next": obs_next,
            "latent_goal_next": latent_goal_next,
            "rew": self.rew[indices],
            "int_rew": self.int_rew[indices],
            "terminated": self.terminated[indices],
            "truncated": self.truncated[indices],
            "done": self.done[indices],
            "info": self.get(indices, "info", Batch()),
            "policy": self.get(indices, "policy", Batch()),
        }

        for key in self._meta.__dict__:
            if key not in self._input_keys:
                batch_dict[key] = self._meta[key][indices]
        return cast(GoalBatchProtocol, Batch(batch_dict))


class GoalReplayBufferManager(GoalReplayBuffer, ReplayBufferManager):
    """GoalReplayBufferManager contains a list of GoalReplayBuffers, each with the exact same configuration.

    For details, see https://tianshou.org/en/stable/03_api/data/buffer/manager.html."""

    def __init__(self, buffer_list: list[GoalReplayBuffer]) -> None:
        ReplayBufferManager.__init__(self, buffer_list)  # type: ignore

    def add(
        self,
        batch: GoalBatchProtocol,
        buffer_ids: np.ndarray | list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Add a batch of data into the GoalReplayBufferManager.

        If the episode isn't finished, the return value of episode_length and
        episode_reward is 0.
        """
        # preprocess batch
        new_batch = Batch()
        for key in set(self._reserved_keys).intersection(batch.get_keys()):
            new_batch.__dict__[key] = batch[key]
        batch = new_batch
        batch.__dict__["done"] = np.logical_or(batch.terminated, batch.truncated)
        assert {
            "obs",
            "latent_goal",
            "act",
            "obs_next",
            "latent_goal_next",
            "rew",
            "int_rew",
            "terminated",
            "truncated",
            "done",
        }.issubset(batch.get_keys())

        # get index
        if buffer_ids is None:
            buffer_ids = np.arange(self.buffer_num)

        ptrs, ep_lens, ep_rews, ep_idxs = [], [], [], []
        for batch_idx, buffer_id in enumerate(buffer_ids):
            ptr, ep_rew, ep_len, ep_idx = self.buffers[buffer_id]._add_index(
                batch.rew[batch_idx],
                batch.done[batch_idx],
            )
            ptrs.append(ptr + self._offset[buffer_id])
            ep_lens.append(ep_len)
            ep_rews.append(ep_rew.astype(np.float32))
            ep_idxs.append(ep_idx + self._offset[buffer_id])
            self.last_index[buffer_id] = ptr + self._offset[buffer_id]
            self._lengths[buffer_id] = len(self.buffers[buffer_id])
        ptrs = np.array(ptrs)
        try:
            self._meta[ptrs] = batch
        except ValueError:
            batch.rew = batch.rew.astype(np.float32)
            batch.done = batch.done.astype(bool)
            batch.terminated = batch.terminated.astype(bool)
            batch.truncated = batch.truncated.astype(bool)
            if len(self._meta.get_keys()) == 0:
                self._meta = create_value(batch, self.maxsize, stack=False)  # type: ignore
            else:  # dynamic key pops up in batch
                alloc_by_keys_diff(self._meta, batch, self.maxsize, False)
            self._set_batch_for_children()
            self._meta[ptrs] = batch
        return ptrs, np.array(ep_rews), np.array(ep_lens), np.array(ep_idxs)


class GoalVectorReplayBuffer(GoalReplayBufferManager):
    """The GoalVectorReplayBuffer contains `n` GoalReplayBuffers of the same size.

    For details, see https://tianshou.org/en/stable/03_api/data/buffer/vecbuf.html."""

    def __init__(self, total_size: int, buffer_num: int, **kwargs: Any) -> None:
        assert buffer_num > 0
        size = int(np.ceil(total_size / buffer_num))
        buffer_list = [GoalReplayBuffer(size, **kwargs) for _ in range(buffer_num)]
        super().__init__(buffer_list)
