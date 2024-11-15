from typing import Any
from core.types import GoalBatchProtocol

from tianshou.data import ReplayBuffer, ReplayBufferManager, Batch
from tianshou.data.batch import alloc_by_keys_diff
import numpy as np


class KnowledgeBase(ReplayBuffer):
    """A replay buffer that represents the agent's knowledge base.

    It stores transitions and aggregates them into trajectories.
    """

    # store the an (o_t, a_t, r_t) tuple with an additional initial_obs entry to keep track of the beginning of a trajectory, and a final_obs entry to keep track of the end of said trajectory
    _reserved_keys = ("obs", "act", "rew", "init_obs")
    _input_keys = ("obs", "act", "rew", "init_obs")

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
        batch: GoalBatchProtocol,
        buffer_ids: np.ndarray | list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Adds a batch of data into replay buffer."""
        # TODO
        # extract obs, act, rew from batch
        # assign to each triplet the correct initial_obs
        #   if the buffer is empty => the initial obs matches obs
        #   else ...

        # preprocess batch
        new_batch = Batch()
        for key in batch.get_keys():
            new_batch.__dict__[key] = batch[key]
        batch = new_batch
        batch.__dict__["done"] = np.logical_or(batch.terminated, batch.truncated)
        assert {"obs", "act", "rew", "terminated", "truncated", "done"}.issubset(
            batch.get_keys(),
        )  # important to do after preprocess batch
        stacked_batch = buffer_ids is not None
        if stacked_batch:
            assert len(batch) == 1
        if self._save_only_last_obs:
            batch.obs = batch.obs[:, -1] if stacked_batch else batch.obs[-1]
        if not self._save_obs_next:
            batch.pop("obs_next", None)
        elif self._save_only_last_obs:
            batch.obs_next = (
                batch.obs_next[:, -1] if stacked_batch else batch.obs_next[-1]
            )
        # get ptr
        if stacked_batch:
            rew, done = batch.rew[0], batch.done[0]
        else:
            rew, done = batch.rew, batch.done
        ptr, ep_rew, ep_len, ep_idx = (
            np.array([x]) for x in self._add_index(rew, done)
        )
        try:
            self._meta[ptr] = batch
        except ValueError:
            stack = not stacked_batch
            batch.rew = batch.rew.astype(float)
            batch.done = batch.done.astype(bool)
            batch.terminated = batch.terminated.astype(bool)
            batch.truncated = batch.truncated.astype(bool)
            if len(self._meta.get_keys()) == 0:
                self._meta = create_value(batch, self.maxsize, stack)  # type: ignore
            else:  # dynamic key pops up in batch
                alloc_by_keys_diff(self._meta, batch, self.maxsize, stack)
            self._meta[ptr] = batch
        return ptr, ep_rew, ep_len, ep_idx


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
