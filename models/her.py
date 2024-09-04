import numpy as np

from tianshou.data.batch import BatchProtocol, Batch
from tianshou.data import ReplayBuffer


class HER:
    """An implementation of Hindsight Experience Replay (https://arxiv.org/abs/1707.01495).

    This version doesn't necessitate goal-based environments, as current implementations in Tianshou (https://tianshou.org/en/v1.1.0/03_api/data/buffer/her.html#tianshou.data.buffer.her.HERReplayBuffer) and Baselines (https://stable-baselines3.readthedocs.io/en/master/modules/her.html) do.
    This is advantageous from a conceptual point of view: HER is something that should happen within an agent's "head".
    """

    def __init__(
        self,
        buffer: ReplayBuffer,
        indices: np.ndarray,
        horizon: int,
        future_k: float = 8.0,
    ) -> None:
        self.buffer = buffer
        self.indices = indices
        # TODO keep these?
        self.horizon = horizon
        self.future_p = 1 - 1 / future_k

        self._original_meta = Batch()
        self._altered_indices = np.array([])

    # TODO I might be able to avoid using ReplayBuffer private attributes in this! => would be nice...
    def rewrite_transitions(self) -> None:
        """Re-write the goal of some sampled transitions' episodes according to HER."""
        if self.indices.size == 0:
            return

        # Sort self.indices keeping chronological order
        self.indices[self.indices < self.buffer._index] += self.buffer.maxsize
        self.indices = np.sort(self.indices)
        self.indices[self.indices >= self.buffer.maxsize] -= self.buffer.maxsize

        # Construct episode trajectories
        self.indices = [self.indices]
        for _ in range(self.horizon - 1):
            self.indices.append(self.next(self.indices[-1]))
        self.indices = np.stack(self.indices)

        # Calculate future timestep to use
        current = self.indices[0]
        terminal = self.indices[-1]
        episodes_len = (terminal - current + self.buffer.maxsize) % self.buffer.maxsize
        future_offset = np.random.uniform(size=len(self.indices[0])) * episodes_len
        future_offset = np.round(future_offset).astype(int)
        future_t = (current + future_offset) % self.buffer.maxsize

        # Compute indices
        #   open indices are used to find longest, unique trajectories among
        #   presented episodes
        unique_ep_open_indices = np.sort(np.unique(terminal, return_index=True)[1])
        unique_ep_indices = self.indices[:, unique_ep_open_indices]
        #   close indices are used to find max future_t among presented episodes
        unique_ep_close_indices = np.hstack(
            [(unique_ep_open_indices - 1)[1:], len(terminal) - 1]
        )
        #   episode indices that will be altered
        her_ep_indices = np.random.choice(
            len(unique_ep_open_indices),
            size=int(len(unique_ep_open_indices) * self.future_p),
            replace=False,
        )

        # Cache original meta
        self._altered_indices = unique_ep_indices.copy()
        self._original_meta = self.buffer._meta[self._altered_indices].copy()

        # Copy original obs, ep_rew (and obs_next), and obs of future time step
        ep_obs = self.buffer[unique_ep_indices].obs
        # to satisfy mypy
        ep_rew = self.buffer[unique_ep_indices].rew
        if self.buffer._save_obs_next:
            ep_obs_next = self.buffer[unique_ep_indices].obs_next
            future_obs = self.buffer[future_t[unique_ep_close_indices]].obs_next
        else:
            future_obs = self.buffer[self.next(future_t[unique_ep_close_indices])].obs

        # Re-assign goals and rewards via broadcast assignment
        # TODO this should be for desired_goal
        ep_obs[:, her_ep_indices] = future_obs[None, her_ep_indices]
        if self.buffer._save_obs_next:
            # TODO desired_goal on left, achieved_goal on right
            ep_obs_next[:, her_ep_indices] = future_obs[
                None,
                her_ep_indices,
            ]
            ep_rew[:, her_ep_indices] = self._compute_reward(ep_obs_next)[
                :, her_ep_indices
            ]
        else:
            tmp_ep_obs_next = self.buffer[self.next(unique_ep_indices)].obs
            ep_rew[:, her_ep_indices] = self._compute_reward(tmp_ep_obs_next)[
                :, her_ep_indices
            ]

        # Sanity check
        # TODO this should be checked for desired and achieved goals
        assert ep_obs.shape[:2] == unique_ep_indices.shape
        assert ep_obs.shape[:2] == unique_ep_indices.shape
        assert ep_rew.shape == unique_ep_indices.shape

        # Re-write meta
        self.buffer._meta.obs[unique_ep_indices] = ep_obs
        if self.buffer._save_obs_next:
            self.buffer._meta.obs_next[unique_ep_indices] = ep_obs_next  # type: ignore
        self.buffer._meta.rew[unique_ep_indices] = ep_rew.astype(np.float32)

    # TODO this should get desired and achieved goals as input
    def _compute_reward(self, obs: BatchProtocol, lead_dims: int = 2) -> np.ndarray:
        lead_shape = obs.shape[:lead_dims]
        # TODO the two statements below should operate on the goals, not on the observations
        # desired_goal = obs.reshape(-1, *obs[lead_dims:])
        # achieved_goal = obs.reshape(-1, *obs[lead_dims:])
        desired_goal = obs.reshape(-1, obs.shape[-1])
        achieved_goal = obs.reshape(-1, obs.shape[-1])
        rewards = self.compute_reward_fn(achieved_goal, desired_goal)
        return rewards.reshape(*lead_shape)
