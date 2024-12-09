from typing import Any, List, Tuple, Optional, cast
from core.types import KBBatchProtocol

from tianshou.data import ReplayBuffer, ReplayBufferManager, Batch
from tianshou.data.utils.converter import to_hdf5, from_hdf5
from tianshou.data.batch import alloc_by_keys_diff, create_value
import h5py
import numpy as np


class KnowledgeBase(ReplayBuffer):
    """A replay buffer that represents the agent's knowledge base.

    It stores transitions and aggregates them into trajectories.
    """

    # (observation, action, reward, trajectory identifier)
    _reserved_keys = ("obs", "act", "rew", "traj_id")
    _input_keys = ("obs", "act", "rew", "traj_id")

    def __init__(
        self,
        size: int,
        stack_num: int = 1,
        ignore_obs_next: bool = True,  # we do not need the obs_next
        save_only_last_obs: bool = True,  # we only need the last observation
        sample_avail: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            size, stack_num, ignore_obs_next, save_only_last_obs, sample_avail, **kwargs
        )

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
            "traj_id": self.traj_id[indices],
        }

        for key in self._meta.__dict__:
            if key not in self._input_keys:
                batch_dict[key] = self._meta[key][indices]
        return cast(KBBatchProtocol, Batch(batch_dict))


class KnowledgeBaseManager(KnowledgeBase, ReplayBufferManager):
    """A class for managing vectorised knowledge bases."""

    def __init__(self, buffer_list: list[KnowledgeBase]) -> None:
        ReplayBufferManager.__init__(self, buffer_list)
        self._traj_meta = {}  # {traj_id: [[i_0, i_1, ..., i_n], [..., ...]]}

    @property
    def n_trajectories(self):
        return len(self._traj_meta)

    def add(
        self,
        batch: KBBatchProtocol,
        buffer_ids: np.ndarray | list[int] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Adds a batch of data into replay buffer."""
        # preprocess batch
        new_batch = Batch()
        for key in self._reserved_keys:
            new_batch.__dict__[key] = batch[key]
        batch = new_batch

        assert {"obs", "act", "rew", "traj_id"}.issubset(
            batch.get_keys(),
        )

        # get index
        if buffer_ids is None:
            buffer_ids = np.arange(self.buffer_num)

        ptrs, ep_lens, ep_rews, ep_idxs = [], [], [], []
        for batch_idx, buffer_id in enumerate(buffer_ids):
            ptr, ep_rew, ep_len, ep_idx = self.buffers[buffer_id]._add_index(
                batch.rew[batch_idx],
                done=False,
            )
            ptrs.append(ptr + self._offset[buffer_id])
            ep_lens.append(ep_len)
            ep_rews.append(ep_rew.astype(np.float32))
            ep_idxs.append(ep_idx + self._offset[buffer_id])

            self.last_index[buffer_id] = ptr + self._offset[buffer_id]
            self._lengths[buffer_id] = len(self.buffers[buffer_id])

            traj_id = batch.traj_id[batch_idx]
            if traj_id not in self._traj_meta:
                self._traj_meta[traj_id] = [[] for _ in range(self.buffer_num)]

            idx_list = self._traj_meta[traj_id][buffer_id]
            # the indices in _traj_meta are monotonically increasing, so if last_index[buffer_id] < idx_list[-1] we're overwriting data
            if idx_list and self.last_index[buffer_id] < idx_list[-1]:
                # clear the old index list to eliminate stale data
                idx_list.clear()
            idx_list.append(self.last_index[buffer_id])
        ptrs = np.array(ptrs)

        try:
            self._meta[ptrs] = batch
        except ValueError:
            batch.rew = batch.rew.astype(np.float32)
            if len(self._meta.get_keys()) == 0:
                self._meta = create_value(batch, self.maxsize, stack=False)  # type: ignore
            else:  # dynamic key pops up in batch
                alloc_by_keys_diff(self._meta, batch, self.maxsize, False)
            self._set_batch_for_children()
            self._meta[ptrs] = batch

        return (
            ptrs,
            np.array(ep_rews),
            np.array(ep_lens),
            np.array(ep_idxs),
        )

    def get_trajectories_by_id(
        self, traj_id: int, ensure_uniform: bool = False
    ) -> Optional[Batch] | List[Optional[KBBatchProtocol]]:
        """
        Retrieves the trajectory data for the given traj_id from each buffer.

        If ensure_uniform is True, it returns a Batch object containing the trajectory data, eliminating all the buffers with any None trajectories and truncating the trajectories to the same length.
        If ensure_uniform is False, it returns a list of trajectory data, where each element in the list represents the trajectory data with the given traj_id from a given buffer (None if the buffer doesn't have any trajectory with the specified ID):
        """
        trajectory_data_per_buffer = []
        for buffer_id, buffer in enumerate(self.buffers):
            indices = self._traj_meta[traj_id][buffer_id]
            if indices:
                # adjust indices relative to the buffer
                buffer_indices = [idx - self._offset[buffer_id] for idx in indices]
                # retrieve data and ensure it belongs to the correct traj_id
                data = buffer[buffer_indices]
                data = data[data.traj_id == traj_id]
                trajectory_data_per_buffer.append(data)
            else:
                # no matching trajectory in the buffer
                trajectory_data_per_buffer.append(None)

        if ensure_uniform:
            if None in trajectory_data_per_buffer:
                return None
            min_length = min(len(traj) for traj in trajectory_data_per_buffer)
            return Batch([traj[:min_length] for traj in trajectory_data_per_buffer])

        return trajectory_data_per_buffer

    def get_all_trajectories(self) -> List[List[Optional[KBBatchProtocol]]]:
        """Returns all the trajectories stored in the knowledge base."""
        trajectories = []
        for traj_id in range(len(self._traj_meta)):
            trajectories.append(self.get_trajectories_by_id(traj_id))
        return trajectories

    def get_single_trajectory(
        self, traj_id: int, buffer_id: int
    ) -> Optional[KBBatchProtocol]:
        """Extracts a single trajectory from the knowledge base, if it exists."""
        traj_per_buffer = self.get_trajectories_by_id(traj_id)
        if traj_per_buffer[buffer_id] is not None:
            return traj_per_buffer[buffer_id]
        return None

    def save_hdf5(self, path: str, compression: str | None = None) -> None:
        """Saves all the data within the knowledge base to an HDF5 file."""
        with h5py.File(path, "w") as f:
            f.attrs["buffer_num"] = self.buffer_num
            f.attrs["total_size"] = self.maxsize

            buf_grp = f.create_group("buffers")
            for i, buf in enumerate(self.buffers):
                grp = buf_grp.create_group(str(i))
                to_hdf5(buf.__dict__, grp, compression=compression)

            traj_grp = f.create_group("traj_meta")
            for tid, traj_lists in self._traj_meta.items():
                tgrp = traj_grp.create_group(str(tid))
                for j, indices in enumerate(traj_lists):
                    tgrp.create_dataset(str(j), data=np.array(indices, dtype=np.int64))

            f.create_dataset("offset", data=self._offset)
            f.create_dataset("last_index", data=self.last_index)
            f.create_dataset("lengths", data=self._lengths)

    @classmethod
    def load_hdf5(cls, path: str, device: str | None = None):
        """Loads all the data within the knowledge base from an HDF5 file."""
        with h5py.File(path, "r") as f:
            buffer_num = f.attrs["buffer_num"]
            total_size = f.attrs["total_size"]

            buffers = []
            buf_grp = f["buffers"]
            for i in range(buffer_num):
                state = from_hdf5(buf_grp[str(i)], device=device)
                kb = KnowledgeBase(state["maxsize"], **state["options"])
                kb.__setstate__(state)
                buffers.append(kb)

            # cls should be a VectorKnowledgeBase
            kbm = cls(total_size, buffer_num)
            kbm.buffers = buffers

            kbm._traj_meta = {}
            traj_grp = f["traj_meta"]
            for tid_str in traj_grp.keys():
                tid = int(tid_str)
                traj_lists = []
                for j_str in traj_grp[tid_str].keys():
                    traj_lists.append(list(traj_grp[tid_str][j_str][...]))
                kbm._traj_meta[tid] = traj_lists

            kbm._offset = f["offset"][...]
            kbm.last_index = f["last_index"][...]
            kbm._lengths = f["lengths"][...]

            return kbm


class VectorKnowledgeBase(KnowledgeBaseManager):
    """A class containing `buffer_num` knowledge bases.

    Note that, conceptually speaking, the knowledge base is only one. This class is merely an implementation-level convenience, its point being to provide a frictionless interaction with Tianshou.
    """

    def __init__(self, total_size: int, buffer_num: int, **kwargs: Any) -> None:
        assert buffer_num > 0
        size = int(np.ceil(total_size / buffer_num))
        buffer_list = [KnowledgeBase(size, **kwargs) for _ in range(buffer_num)]
        super().__init__(buffer_list)
