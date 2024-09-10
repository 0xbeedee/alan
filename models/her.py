from tianshou.data.batch import BatchProtocol
from core.types import GoalReplayBufferProtocol
import numpy as np

from tianshou.data.buffer.her import HERReplayBuffer


class HER(HERReplayBuffer):
    """An implementation of Hindsight Experience Replay (https://arxiv.org/abs/1707.01495).

    This version doesn't necessitate goal-based environments, as do current implementations in Tianshou (https://tianshou.org/en/v1.1.0/03_api/data/buffer/her.html#tianshou.data.buffer.her.HERReplayBuffer) and Stable Baselines (https://stable-baselines3.readthedocs.io/en/master/modules/her.html).
    This is advantageous from a conceptual point of view: HER is something that should happen within an agent's "head", not depend on an environment.
    """

    def __init__(
        self,
        buffer: GoalReplayBufferProtocol,
        horizon: int,
        future_k: float = 8.0,
    ) -> None:
        super().__init__(
            size=buffer.maxsize,
            # we perform the reward computation directly in this class
            compute_reward_fn=None,
            horizon=horizon,
            future_k=future_k,
        )

    def rewrite_transitions(self, indices: np.ndarray) -> None:
        """Re-write the goal of some sampled transitions' episodes according to HER."""
        if indices.size == 0:
            return

        # Sort self.indices keeping chronological order
        indices[indices < self._index] += self.maxsize
        indices = np.sort(indices)
        indices[indices >= self.maxsize] -= self.maxsize

        # Construct episode trajectories
        indices = [indices]
        for _ in range(self.horizon - 1):
            indices.append(self.next(indices[-1]))
        indices = np.stack(indices)

        # Calculate future timestep to use
        current = indices[0]
        terminal = indices[-1]
        episodes_len = (terminal - current + self.maxsize) % self.maxsize
        future_offset = np.random.uniform(size=len(indices[0])) * episodes_len
        future_offset = np.round(future_offset).astype(int)
        future_t = (current + future_offset) % self.maxsize

        # Compute indices
        #   open indices are used to find longest, unique trajectories among
        #   presented episodes
        unique_ep_open_indices = np.sort(np.unique(terminal, return_index=True)[1])
        unique_ep_indices = indices[:, unique_ep_open_indices]
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
        self._original_meta = self._meta[self._altered_indices].copy()

        # Copy original obs, ep_rew (and obs_next), and obs of future time step
        ep_obs = self[unique_ep_indices].obs
        # to satisfy mypy
        ep_rew = self[unique_ep_indices].rew
        if self._save_obs_next:
            ep_obs_next = self[unique_ep_indices].obs_next
            future_obs = self[future_t[unique_ep_close_indices]].obs_next
        else:
            future_obs = self[self.next(future_t[unique_ep_close_indices])].obs

        # Re-assign goals and rewards via broadcast assignment
        # TODO this should be for desired_goal
        ep_obs[:, her_ep_indices] = future_obs[None, her_ep_indices]
        if self._save_obs_next:
            # TODO desired_goal on left, achieved_goal on right
            ep_obs_next[:, her_ep_indices] = future_obs[
                None,
                her_ep_indices,
            ]
            ep_rew[:, her_ep_indices] = self._compute_reward(ep_obs_next)[
                :, her_ep_indices
            ]
        else:
            tmp_ep_obs_next = self[self.next(unique_ep_indices)].obs
            ep_rew[:, her_ep_indices] = self._compute_reward(tmp_ep_obs_next)[
                :, her_ep_indices
            ]

        # Sanity check
        # TODO this should be checked for desired and achieved goals
        assert ep_obs.shape[:2] == unique_ep_indices.shape
        assert ep_obs.shape[:2] == unique_ep_indices.shape
        assert ep_rew.shape == unique_ep_indices.shape

        # Re-write meta
        self._meta.obs[unique_ep_indices] = ep_obs
        if self._save_obs_next:
            self._meta.obs_next[unique_ep_indices] = ep_obs_next  # type: ignore
        self._meta.rew[unique_ep_indices] = ep_rew.astype(np.float32)

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
