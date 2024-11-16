from typing import Any, cast
from core.types import KBBatchProtocol

from tianshou.data import ReplayBuffer, ReplayBufferManager, Batch
from tianshou.data.batch import alloc_by_keys_diff, create_value
import numpy as np


class KnowledgeBase(ReplayBuffer):
    """A replay buffer that represents the agent's knowledge base.

    It stores transitions and aggregates them into trajectories.
    """

    # store the an (o_t, a_t, r_t) tuple with an additional initial_obs entry to keep track of the beginning of a trajectory, and a final_obs entry to keep track of the end of said trajectory
    _reserved_keys = ("obs", "act", "rew", "init_obs", "traj_id")
    _input_keys = ("obs", "act", "rew", "init_obs", "traj_id")

    def __init__(
        self,
        size: int,
        stack_num: int = 1,
        ignore_obs_next: bool = True,  # we do not need the obs_next
        save_only_last_obs: bool = True,  # we only need the last observation
        sample_avail: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(
            size, stack_num, ignore_obs_next, save_only_last_obs, sample_avail, **kwargs
        )

    def add(
        self,
        batch: KBBatchProtocol,
        buffer_ids: np.ndarray | list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Adds a batch of data into replay buffer."""
        # preprocess batch
        new_batch = Batch()
        for key in batch.get_keys():
            new_batch.__dict__[key] = batch[key]
        batch = new_batch

        assert {"obs", "act", "rew", "init_obs", "traj_id"}.issubset(
            batch.get_keys(),
        )

        stacked_batch = buffer_ids is not None
        if stacked_batch:
            assert len(batch) == 1
        if self._save_only_last_obs:
            batch.obs = batch.obs[:, -1] if stacked_batch else batch.obs[-1]

        # get ptr
        rew = batch.rew[0] if stacked_batch else batch.rew
        ptr, ep_rew, ep_len, ep_idx = (
            np.array([x]) for x in self._add_index(rew, done=False)
        )
        try:
            self._meta[ptr] = batch
        except ValueError:
            stack = not stacked_batch
            batch.rew = batch.rew.astype(np.float32)
            batch.init_obs = batch.init_obs
            batch.traj_id = batch.traj_id.astype(int)
            if len(self._meta.get_keys()) == 0:
                self._meta = create_value(batch, self.maxsize, stack)  # type: ignore
            else:  # dynamic key pops up in batch
                alloc_by_keys_diff(self._meta, batch, self.maxsize, stack)
            self._meta[ptr] = batch
        return ptr, ep_rew, ep_len, ep_idx

    def __getitem__(
        self, index: slice | int | list[int] | np.ndarray
    ) -> KBBatchProtocol:
        if isinstance(index, slice):  # change slice to np array
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

        batch_dict = {
            "obs": obs,
            "act": self.act[indices],
            "rew": self.rew[indices],
            "init_obs": self.init_obs[indices],
            "traj_id": self.traj_id[indices],
        }

        for key in self._meta.__dict__:
            if key not in self._input_keys:
                batch_dict[key] = self._meta[key][indices]
        return cast(KBBatchProtocol, Batch(batch_dict))


class KnowledgeBaseManager(KnowledgeBase, ReplayBufferManager):
    """A class for managing vectorised knowledge bases."""

    def __init__(self, buffer_list: list[KnowledgeBase]) -> None:
        ReplayBufferManager.__init__(self, buffer_list)  # type: ignore


class VectorKnowledgeBase(KnowledgeBaseManager):
    """A class containing `buffer_num` knowledge bases.

    Note that, conceptually speaking, the knowledge base is only one. This class is merely an implementation-level convenience, its point being to provide a frictionless interaction with Tianshou.
    """

    def __init__(self, total_size: int, buffer_num: int, **kwargs: Any) -> None:
        assert buffer_num > 0
        size = int(np.ceil(total_size / buffer_num))
        buffer_list = [KnowledgeBase(size, **kwargs) for _ in range(buffer_num)]
        super().__init__(buffer_list)
