from core.types import GoalReplayBufferProtocol

from tianshou.data import Batch
import numpy as np


class HER:
    """An implementation of Hindsight Experience Replay (https://arxiv.org/abs/1707.01495) with the "future" strategy (it has been shown to perform best and it subsumes the "episode" one).

    This version doesn't necessitate goal-based environments, as do current implementations in Tianshou (https://tianshou.org/en/v1.1.0/03_api/data/buffer/her.html#tianshou.data.buffer.her.HERReplayBuffer) and Stable Baselines (https://stable-baselines3.readthedocs.io/en/master/modules/her.html). This is advantageous from a conceptual point of view: HER is something that should happen within an agent's "head", not depend on an external environment.
    """

    def __init__(
        self,
        buffer: GoalReplayBufferProtocol,
        horizon: int,
        future_k: float = 8.0,
        epsilon: float = 0.001,
    ) -> None:
        assert (
            buffer._save_obs_next == True
        ), "obs_next is needed for HER to work properly"
        self.buf = buffer

        self.horizon = horizon
        self.future_p = 1 - 1 / future_k
        self.epsilon = epsilon

    def get_future_observation_(self, indices: np.ndarray) -> Batch:
        # we need to keep the chronological order
        indices[indices < (self.buf.last_index[0] + 1)] += self.buf.maxsize
        indices = np.sort(indices)
        indices[indices >= self.buf.maxsize] -= self.buf.maxsize

        # trajectories
        indices = [indices]
        for _ in range(self.horizon - 1):
            indices.append(self.buf.next(indices[-1]))
        indices = np.stack(indices)

        current = indices[0]
        terminal = indices[-1]
        episodes_len = (terminal - current + self.buf.maxsize) % self.buf.maxsize
        future_offset = np.random.uniform(size=len(indices[0])) * episodes_len
        future_offset = np.round(future_offset).astype(int)
        # future timestep to use
        future_t = (current + future_offset) % self.buf.maxsize

        # open indices are used to find longest unique trajectories
        unique_open_indices = np.sort(np.unique(terminal, return_index=True)[1])
        self.unique_indices = indices[:, unique_open_indices]
        # close indices are used to find max future_t among available episodes
        unique_close_indices = np.hstack(
            [(unique_open_indices - 1)[1:], len(terminal) - 1]
        )
        # indices used for re-assigning goals
        self.her_indices = np.random.choice(
            len(unique_open_indices),
            size=int(len(unique_open_indices) * self.future_p),
            replace=False,
        )

        future_obs = self.buf[future_t[unique_close_indices]].obs_next
        return future_obs

    def rewrite_transitions_(self, future_achieved_goal: np.ndarray) -> np.ndarray:
        next_desired_goal = self.buf[self.unique_indices].latent_goal_next
        reassigned_desired_goal = next_desired_goal.copy()
        reassigned_desired_goal[:, self.her_indices] = future_achieved_goal[
            None, self.her_indices
        ]

        rew = self.buf[self.unique_indices].rew
        # we add instead of assigning because we want to keep the fast intrinsic bonus
        # (as for the sparsity argument in the HER paper: we decrease the fast intrinsic contribution over time, so we will eventually reach a situation in which we'll only operate with env rewards and the sparse binary ones provided by HER)
        rew[:, self.her_indices] += self._compute_reward(
            next_desired_goal, reassigned_desired_goal
        )[:, self.her_indices]
        # unlike in the Tianshou case, we don't need to restore anything
        self.buf._meta.rew[self.unique_indices] = rew

    def _compute_reward(
        self,
        desired_goal: np.ndarray,
        achieved_goal: np.ndarray,
        lead_dims: int = 2,
    ) -> np.ndarray:
        """Computes the hindsight experience replay reward.

        It is implemented using the L1 norm (as in the original paper) and a reward of 1 if an agent reaches the goal and 0 otherwise.

        (For a discussion of why using a reward of -1 [as is done in the original paper] doesn't speed up training, see section 3.5 of Zhao's "Mathematical Foundations of Reinforcement Learning".)
        """
        lead_shape = desired_goal.shape[:lead_dims]
        desired_goal = desired_goal.reshape(-1, *desired_goal.shape[lead_dims:])
        achieved_goal = achieved_goal.reshape(-1, *achieved_goal.shape[lead_dims:])

        distances = np.sum(np.abs(desired_goal - achieved_goal), axis=1)
        rewards = (distances <= self.epsilon).astype(np.float32)
        return rewards.reshape(*lead_shape)
