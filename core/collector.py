from typing import Any, cast, Union
import time
from copy import copy

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import (
    Batch,
    to_numpy,
    Collector,
    AsyncCollector,
    VectorReplayBuffer,
)
from tianshou.data.types import ObsBatchProtocol, ActBatchProtocol
from tianshou.env import BaseVectorEnv

from .types import (
    GoalBatchProtocol,
    KBBatchProtocol,
    GoalReplayBufferProtocol,
    CorePolicyProtocol,
    TArrLike,
)
from .stats import EpNStepCollectStats


class GoalCollector(Collector):
    """The Collector collects data from a (possibly vectorised) Gymnasium Enviroment using a specified policy, and stores this data into a buffer.

    For details, see https://tianshou.org/en/stable/03_api/data/collector.html."""

    def __init__(
        self,
        policy: CorePolicyProtocol,
        env: gym.Env | BaseVectorEnv,
        buffer: GoalReplayBufferProtocol | None = None,
        knowledge_base: VectorReplayBuffer | None = None,
        bandit: Union["TrajectoryBandit", None] = None,  # type:ignore
        exploration_noise: bool = False,
    ) -> None:
        super().__init__(policy, env, buffer, exploration_noise=exploration_noise)

        self.knowledge_base = knowledge_base
        self.bandit = bandit

        if self.env.is_async:
            raise ValueError(
                f"Please use {AsyncCollector.__name__} for asynchronous environments. "
                f"Env class: {self.env.__class__.__name__}.",
            )

    @torch.no_grad()
    def _collect(
        self,
        n_step: int | None = None,
        n_episode: int | None = None,
        random: bool = False,
        render: float | None = None,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> EpNStepCollectStats:
        """Collects a specified number of steps or episodes.

        Note that the collect() method in BaseCollector calls this method to do the actual collecting.
        """
        if n_step is not None:
            ready_env_ids_R = np.arange(self.env_num)
        elif n_episode is not None:
            ready_env_ids_R = np.arange(min(self.env_num, n_episode))
        else:
            raise ValueError("Either n_step or n_episode should be set.")

        start_time = time.time()
        if self._pre_collect_obs_RO is None or self._pre_collect_info_R is None:
            raise ValueError(
                "Initial obs and info should not be None. "
                "Either reset the collector (using reset or reset_env) or pass reset_before_collect=True to collect.",
            )

        # get the first obs to be the current obs in the n_step case as
        # episodes as a new call to collect does not restart trajectories
        # (which we also really don't want)
        step_count = 0
        num_collected_episodes = 0
        episode_returns: list[float] = []
        episode_lens: list[int] = []
        episode_start_indices: list[int] = []
        # to track the episodic intrinsic reward
        episode_intrinsic_returns: list[float] = []

        # to track the reward if we do not manage to finish one episode
        nstep_returns: list[float] = []
        # to track the n_step (fast) intrinsic reward
        nstep_intrinsic_returns: list[float] = []

        # in case we select fewer episodes than envs, we run only some of them
        last_obs_RO = _nullable_slice(self._pre_collect_obs_RO, ready_env_ids_R)
        last_info_R = _nullable_slice(self._pre_collect_info_R, ready_env_ids_R)
        last_hidden_state_RH = _nullable_slice(
            self._pre_collect_hidden_state_RH,
            ready_env_ids_R,
        )

        cur_traj_id = np.zeros(self.env_num, dtype=np.uint16)
        # set as class attribute because we need it across methods
        self.selected_trajectories = None

        while True:
            (
                latent_last_obs_RO,
                act_RA,
                act_normalized_RA,
                latent_goal_R,
                policy_R,
                hidden_state_RH,
            ) = self._compute_action_policy_hidden(
                random=random,
                ready_env_ids_R=ready_env_ids_R,
                last_obs_RO=last_obs_RO,
                last_info_R=last_info_R,
                last_hidden_state_RH=last_hidden_state_RH,
            )

            obs_next_RO, rew_R, terminated_R, truncated_R, info_R = self.env.step(
                act_normalized_RA,
                ready_env_ids_R,
            )
            # for MPS compatibility
            rew_R = rew_R.astype(np.float32)
            nstep_returns.extend(rew_R)

            if self.selected_trajectories is not None:
                for i, env_idx in enumerate(self.updated_envs):
                    traj = self.selected_trajectories[i]
                    if traj is None:
                        arm_id = self.bandit.null_arm_id
                    else:
                        arm_id = self.bandit.get_arm_id(env_idx, traj.traj_id[0])
                    self.trajectory_rewards[arm_id] += rew_R[env_idx]

            # the obs_next MUST be passed through the obs_net only here
            latent_obs_next_RO = self.policy.obs_net(Batch(obs_next_RO))

            int_rew_R = self.policy.self_model.fast_intrinsic_reward(
                Batch(
                    latent_obs=latent_last_obs_RO,
                    act=act_RA,
                    latent_obs_next=latent_obs_next_RO,
                    # we need obs_next for BeBold
                    obs_next=obs_next_RO,
                )
            )
            nstep_intrinsic_returns.extend(int_rew_R)

            latent_goal_next_R = self.policy.self_model.select_goal(latent_obs_next_RO)

            if isinstance(info_R, dict):  # type: ignore[unreachable]
                # This can happen if the env is an envpool env. Then the info returned by step is a dict
                info_R = _dict_of_arr_to_arr_of_dicts(info_R)  # type: ignore[unreachable]
            done_R = np.logical_or(terminated_R, truncated_R)

            # construct the goal-aware batch
            current_iteration_batch = cast(
                GoalBatchProtocol,
                Batch(
                    obs=last_obs_RO,
                    latent_goal=latent_goal_R,
                    act=act_RA,
                    obs_next=obs_next_RO,
                    latent_goal_next=latent_goal_next_R,
                    # extrinsic reward
                    rew=rew_R,
                    int_rew=int_rew_R,
                    terminated=terminated_R,
                    truncated=truncated_R,
                    done=done_R,
                    info=info_R,
                    policy=policy_R,
                ),
            )

            # add data into the buffer
            _, ep_rew_R, ep_int_rew_R, ep_len_R, ep_idx_R = self.buffer.add(
                current_iteration_batch,
                buffer_ids=ready_env_ids_R,
            )

            # update the knowledge base, if one is specified
            if self.knowledge_base is not None:
                kb_batch = cast(
                    KBBatchProtocol,
                    Batch(
                        obs=last_obs_RO,
                        act=act_RA,
                        rew=rew_R,
                        traj_id=cur_traj_id[ready_env_ids_R],
                    ),
                )

                # increment the trajectory ID if the agent got a positive reward OR the episode is over
                combined_condition = (rew_R > 0) | done_R
                idxs_to_update = ready_env_ids_R & combined_condition
                if np.any(idxs_to_update):
                    cur_traj_id[idxs_to_update] += 1

                self.knowledge_base.add(kb_batch, buffer_ids=ready_env_ids_R)

            # collect statistics
            num_episodes_done_this_iter = np.sum(done_R)
            num_collected_episodes += num_episodes_done_this_iter
            step_count += len(ready_env_ids_R)

            # preparing for the next iteration
            # obs_next, info and hidden_state will be modified inplace in the code below, so we copy to not affect the data in the buffer
            last_obs_RO = copy(obs_next_RO)
            last_info_R = copy(info_R)
            last_hidden_state_RH = copy(hidden_state_RH)

            # Preparing last_obs_RO, last_info_R, last_hidden_state_RH for the next while-loop iteration
            # Resetting envs that reached done, or removing some of them from the collection if needed (see below)
            if num_episodes_done_this_iter > 0:
                # D - number of envs that reached done in the rollout above
                env_ind_local_D = np.where(done_R)[0]
                env_ind_global_D = ready_env_ids_R[env_ind_local_D]

                episode_lens.extend(ep_len_R[env_ind_local_D])
                episode_returns.extend(ep_rew_R[env_ind_local_D])
                episode_intrinsic_returns.extend(ep_int_rew_R[env_ind_local_D])
                episode_start_indices.extend(ep_idx_R[env_ind_local_D])

                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                gym_reset_kwargs = gym_reset_kwargs or {}
                obs_reset_DO, info_reset_D = self.env.reset(
                    env_id=env_ind_global_D,
                    **gym_reset_kwargs,
                )

                # Set the hidden state to zero or None for the envs that reached done
                self._reset_hidden_state_based_on_type(
                    env_ind_local_D, last_hidden_state_RH
                )

                # preparing for the next iteration
                last_obs_RO[env_ind_local_D] = obs_reset_DO
                last_info_R[env_ind_local_D] = info_reset_D

                # Handling the case when we have more ready envs than desired and are not done yet
                #
                # This can only happen if we are collecting a fixed number of episodes
                # If we have more ready envs than there are remaining episodes to collect,
                # we will remove some of them for the next rollout
                # One effect of this is the following: only envs that have completed an episode
                # in the last step can ever be removed from the ready envs.
                # Thus, this guarantees that each env will contribute at least one episode to the
                # collected data (the buffer). This effect was previous called "avoiding bias in selecting environments"
                # However, it is not at all clear whether this is actually useful or necessary.
                # Additional naming convention:
                # S - number of surplus envs
                #   Changing R to R-S highly increases the complexity of the code.
                if n_episode:
                    remaining_episodes_to_collect = n_episode - num_collected_episodes
                    surplus_env_num = (
                        len(ready_env_ids_R) - remaining_episodes_to_collect
                    )
                    if surplus_env_num > 0:
                        # R becomes R-S here, preparing for the next iteration in while loop
                        # Everything that was of length R needs to be filtered and become of length R-S.
                        # Note that this won't be the last iteration, as one iteration equals one
                        # step and we still need to collect the remaining episodes to reach the breaking condition.

                        # creating the mask
                        env_to_be_ignored_ind_local_S = env_ind_local_D[
                            :surplus_env_num
                        ]
                        env_should_remain_R = np.ones_like(ready_env_ids_R, dtype=bool)
                        env_should_remain_R[env_to_be_ignored_ind_local_S] = False
                        # stripping the "idle" indices, shortening the relevant quantities from R to R-S
                        ready_env_ids_R = ready_env_ids_R[env_should_remain_R]
                        last_obs_RO = last_obs_RO[env_should_remain_R]
                        last_info_R = last_info_R[env_should_remain_R]
                        if hidden_state_RH is not None:
                            last_hidden_state_RH = last_hidden_state_RH[env_should_remain_R]  # type: ignore[index]

            if (n_step and step_count >= n_step) or (
                n_episode and num_collected_episodes >= n_episode
            ):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += num_collected_episodes
        collect_time = max(time.time() - start_time, 1e-9)
        self.collect_time += collect_time

        if n_step:
            # persist for future collect iterations
            self._pre_collect_obs_RO = last_obs_RO
            self._pre_collect_info_R = last_info_R
            self._pre_collect_hidden_state_RH = last_hidden_state_RH
        elif n_episode:
            # reset envs and the _pre_collect fields
            self.reset_env(gym_reset_kwargs)

        return EpNStepCollectStats.with_autogenerated_stats(
            # episode stats
            ep_returns=np.array(episode_returns, dtype=np.float32),
            ep_int_returns=np.array(episode_intrinsic_returns, dtype=np.float32),
            lens=np.array(episode_lens),
            n_collected_episodes=num_collected_episodes,
            # n-step stats
            returns=np.array(nstep_returns, dtype=np.float32),
            int_returns=np.array(nstep_intrinsic_returns, dtype=np.float32),
            n_collected_steps=step_count,
            collect_time=collect_time,
            collect_speed=step_count / collect_time,
        )

    @torch.no_grad()
    def _compute_action_policy_hidden(
        self,
        random: bool,
        ready_env_ids_R: np.ndarray,
        last_obs_RO: np.ndarray,
        last_info_R: np.ndarray,
        last_hidden_state_RH: np.ndarray | torch.Tensor | Batch | None = None,
    ) -> tuple[np.ndarray, np.ndarray, Batch, np.ndarray | torch.Tensor | Batch | None]:
        """Returns the action, the normalized action, a "policy" entry, the hidden state and a latent goal.

        The suffixes at the end of the variable names have their own, Tianshou-specific semantics. See the comments in the source code of Tianshou's Collector for details.
        """
        if random:
            try:
                act_normalized_RA = np.array(
                    [self._action_space[i].sample() for i in ready_env_ids_R],
                )
            except TypeError:  # envpool's action space is not for per-env
                act_normalized_RA = np.array(
                    [self._action_space.sample() for _ in ready_env_ids_R]
                )

            act_RA = self.policy.map_action_inverse(np.array(act_normalized_RA))
            policy_R = Batch()
            hidden_state_RH = None

        else:
            info_batch = _create_info_batch(last_info_R)
            obs_batch_R = cast(
                ObsBatchProtocol, Batch(obs=last_obs_RO, info=info_batch)
            )

            # the last_obs MUST be passed through the obs_net only here
            latent_last_obs_RO = self.policy.obs_net(obs_batch_R.obs)

            # call the policy's forward() method
            act_batch_RA = self.policy(
                obs_batch_R, last_hidden_state_RH, latent_obs=latent_last_obs_RO
            )

            # pass the actions through the bandit, if one is specified
            if self.bandit is not None:
                # cannot use obs_batch_R.obs because a latent_goal is added after the pass through the policy
                self._pass_through_bandit_(
                    Batch(last_obs_RO), act_batch_RA, ready_env_ids_R
                )

            act_RA = to_numpy(act_batch_RA.act)
            if self.exploration_noise:
                act_RA = self.policy.exploration_noise(act_RA, obs_batch_R)
            act_normalized_RA = self.policy.map_action(act_RA)

            policy_R = act_batch_RA.get("policy", Batch())
            if not isinstance(policy_R, Batch):
                raise RuntimeError(
                    f"The policy result should be a {Batch}, but got {type(policy_R)}",
                )

            hidden_state_RH = act_batch_RA.get("state", None)
            if hidden_state_RH is not None:
                policy_R.hidden_state = (
                    hidden_state_RH  # save state into buffer through policy attr
                )

            latent_goal_R = act_batch_RA.get("latent_goal", None)
            if latent_goal_R is None:
                raise RuntimeError("The latent goals should not be None!")

        return (
            latent_last_obs_RO,
            act_RA,
            act_normalized_RA,
            latent_goal_R,
            policy_R,
            hidden_state_RH,
        )

    def _pass_through_bandit_(
        self,
        last_obs_RO: Batch,
        act_batch_RA: ActBatchProtocol,
        ready_env_ids_R: np.ndarray,
    ) -> None:
        """Uses the bandit to obtain the action, overwriting the one chosen by the policy if necessary.

        The underscore at the end of the name indicates that this function modifies an object it uses for computation (i.e., it isn't pure). In this case, we modify the act_batch_RA object.
        """
        if self.selected_trajectories is None:
            self.selected_trajectories, self.updated_envs = (
                self.bandit.select_trajectories(
                    last_obs_RO, ready_env_ids_R, self.policy.obs_net
                )
            )
            # init all the necessary variables
            self.act_index = np.zeros(len(self.selected_trajectories), dtype=int)
            self.trajectory_rewards = {}
            for i, trajectory in enumerate(self.selected_trajectories):
                if trajectory is None:
                    arm_id = self.bandit.null_arm_id
                else:
                    arm_id = self.bandit.get_arm_id(
                        self.updated_envs[i], trajectory.traj_id[0]
                    )
                self.trajectory_rewards[arm_id] = 0.0
        else:
            for i, env_idx in enumerate(self.updated_envs):
                trajectory = self.selected_trajectories[i]
                if trajectory is None:
                    continue
                if self.act_index[i] < len(trajectory.act):
                    # there are still transitions in the trajectory
                    action = trajectory.act[self.act_index[i]]
                    act_batch_RA.act[env_idx] = action
                    self.act_index[i] += 1
            if all(
                self.act_index[i] >= len(self.selected_trajectories[i].act)
                for i in range(len(self.selected_trajectories))
                if self.selected_trajectories[i] is not None
            ):
                # all the trajectories have finished
                self.bandit.update(self.trajectory_rewards)
                # resetting selected_trajectories is enough because the rest of the variables depend on it
                self.selected_trajectories = None


def _create_info_batch(info_array: np.ndarray) -> Batch:
    # this exists for backwards compatibility with Tianshou
    if info_array.dtype != np.dtype("O"):
        raise ValueError(
            f"Expected info_array to have dtype=object, but got {info_array.dtype}.",
        )

    truthy_info_indices = info_array.nonzero()[0]
    falsy_info_indices = set(range(len(info_array))) - set(truthy_info_indices)
    falsy_info_indices = np.array(list(falsy_info_indices), dtype=int)

    if len(falsy_info_indices) == len(info_array):
        return Batch()

    some_nonempty_info = None
    for info in info_array:
        if info:
            some_nonempty_info = info
            break

    info_array = copy(info_array)
    info_array[falsy_info_indices] = some_nonempty_info
    result_batch_parent = Batch(info=info_array)
    result_batch_parent.info[falsy_info_indices] = {}
    return result_batch_parent.info


def _nullable_slice(obj: TArrLike, indices: np.ndarray) -> TArrLike:
    """Return None, or the values at the given indices if the object is not None."""
    if obj is not None:
        return obj[indices]  # type: ignore[index, return-value]
    return None  # type: ignore[unreachable]
