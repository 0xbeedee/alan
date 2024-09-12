from core.types import GoalReplayBufferProtocol
import numpy as np
import torch


class HER:
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
        self.buf = buffer
        self.horizon = horizon
        self.future_p = 1 - 1 / future_k
        self.future_k = future_k

    # TODO turn this into a function => I only care about behaviour, not attributes! (i.e., I only need it to calculate the rewards derived from the hindsight replay, that's it)
    # TODO HER is intrinsic, and we only care about the reward coming form the hindsight replay => no need to manage state, we can keep things pure
    def rewrite_transitions(self, indices: np.ndarray) -> None:
        if indices.size == 0:
            return

        # Sort self.indices keeping chronological order
        indices[indices < self.buf._index] += self.buf.maxsize
        indices = np.sort(indices)
        indices[indices >= self.buf.maxsize] -= self.buf.maxsize

        # Construct episode trajectories
        indices = [indices]
        for _ in range(self.horizon - 1):
            indices.append(self.buf.next(indices[-1]))
        indices = np.stack(indices)

        # Calculate future timestep to use
        current = indices[0]
        terminal = indices[-1]
        episodes_len = (terminal - current + self.buf.maxsize) % self.buf.maxsize
        future_offset = np.random.uniform(size=len(indices[0])) * episodes_len
        future_offset = np.round(future_offset).astype(int)
        future_t = (current + future_offset) % self.buf.maxsize

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

        # copy original obs, ep_rew, obs_next (if available), and obs at future time step
        ep_obs = self.buf[unique_ep_indices].obs
        ep_rew = self.buf[unique_ep_indices].rew
        if self.buf._save_obs_next:
            # if we have an obs_next, use it
            ep_obs_next = self[unique_ep_indices].obs_next
            future_obs = self[future_t[unique_ep_close_indices]].obs_next
        else:
            # else, construct the future_obs from the current one
            future_obs = self.buf[self.buf.next(future_t[unique_ep_close_indices])].obs

        # re-assign goals and rewards
        # all the desired goals are latent_goals, the achieved goals are computed with a forward pass of the observation net
        # get_latent_goal will basically pass the current observation through an obs_net
        self.buf.latent_goal[:, her_ep_indices] = get_achieved_goal(
            future_obs[None, her_ep_indices]
        )
        if self.buf._save_obs_next:
            # if _save_obs_next is True, then there is a latent_goal_next (check core/buffer.py)
            self.buf.latent_goal_next[:, her_ep_indices] = get_achieved_goal(
                future_obs[
                    None,
                    her_ep_indices,
                ]
            )
            ep_rew[:, her_ep_indices] = self._compute_reward(
                self.buf.latent_goal_next, get_achieved_goal(future_obs)
            )[:, her_ep_indices]
        else:
            # TODO
            tmp_ep_obs_next = self[self.next(unique_ep_indices)].obs
            ep_rew[:, her_ep_indices] = self._compute_reward(tmp_ep_obs_next)[
                :, her_ep_indices
            ]

        # Sanity check
        assert ep_obs.shape[:2] == unique_ep_indices.shape
        assert ep_obs.shape[:2] == unique_ep_indices.shape
        assert ep_rew.shape == unique_ep_indices.shape

        return ep_rew.astype(np.float32)

    # TODO are the types correct?
    def _compute_reward(
        self,
        desired_goal: torch.Tensor,
        achieved_goal: torch.Tensor,
        epsilon: float = 0.001,
    ) -> np.ndarray:
        """Computes the rewards for achieving goals.

        It is implemented exactly as specified in the paper, so using the L1 norm and a reward of 0 if an agent reaches the goal and -1 otherwise.
        """
        distances = torch.sum(torch.abs(desired_goal - achieved_goal), dim=1)
        return (distances <= epsilon).float() - 1


# def compute_reward_fn(
#     achieved_goal: np.ndarray, desired_goal: np.ndarray
# ) -> np.ndarray:
#     num_flattened_lead_dims = achieved_goal.shape[0]
#     rewards = np.ones(num_flattened_lead_dims)
#     return rewards
