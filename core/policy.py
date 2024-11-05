from typing import Literal, Any, Tuple
from .types import (
    GoalReplayBufferProtocol,
    SelfModelProtocol,
    EnvModelProtocol,
    GoalBatchProtocol,
)
from tianshou.data.types import (
    ObsBatchProtocol,
    ActStateBatchProtocol,
    ActBatchProtocol,
)

from abc import abstractmethod

from tianshou.data import Batch
from tianshou.data.batch import BatchProtocol
from tianshou.policy import BasePolicy
from tianshou.policy.base import TLearningRateScheduler
from tianshou.utils.torch_utils import torch_train_mode

from torch import nn
import gymnasium as gym
import numpy as np
import torch
import time

from .stats import CoreTrainingStats
from models.utils import sample_mdn


class CorePolicy(BasePolicy[CoreTrainingStats]):
    """CorePolicy is the base class for all the policies we wish to implement.

    It is analogous to Tianshou's BasePolicy (see https://tianshou.org/en/stable/03_api/policy/base.html), in that each policy we create must inherit from it.
    """

    def __init__(
        self,
        *,
        self_model: SelfModelProtocol,
        env_model: EnvModelProtocol,
        obs_net: nn.Module,
        action_space: gym.Space,
        observation_space: gym.Space | None,
        action_scaling: bool = False,
        action_bound_method: None | Literal["clip"] | Literal["tanh"] = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
        beta: float = 0.314,
    ) -> None:
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )
        self.self_model = self_model
        self.env_model = env_model
        self.obs_net = obs_net
        self.beta = beta

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol | ActStateBatchProtocol:
        """Computes the action given a batch of data.

        Note this is just a template method, the actual computation happens in _forward().
        """
        assert "latent_obs" in kwargs
        # deciding on the goal here:
        # 1) makes the actor goal-aware (which is desirable, seeing as we'd like the agent to learn to use goals)
        # 2) centralises goal selection
        self.latent_goal = self.self_model.select_goal(kwargs["latent_obs"])
        result = self._forward(batch, state, **kwargs)
        self._handle_state_(result, state, **kwargs)
        return result

    @abstractmethod
    def _forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol | ActStateBatchProtocol:
        """Carries out the actual computation of the policy's forward() method That is, it computes an action given the current observation and the current hidden state."""

    def process_fn(
        self,
        batch: GoalBatchProtocol,
        buffer: GoalReplayBufferProtocol,
        indices: np.ndarray,
    ) -> GoalBatchProtocol:
        """Pre-processes the data from the provided replay buffer.

        It is meant to be overwritten by the policy. The current implementation simply adds the fast intrinsic reward.
        """
        # TODO the docstring is wrong
        self.combine_fast_reward_(batch)
        batch.latent_obs = self.obs_net(batch.obs)
        batch.latent_obs_next = self.obs_net(batch.obs_next)
        return super().process_fn(batch, buffer, indices)

    def update(
        self,
        sample_size: int | None,
        buffer: GoalReplayBufferProtocol | None,
        **kwargs: Any,
    ) -> CoreTrainingStats:
        """Updates the policy network and replay buffer."""
        if buffer is None:
            return CoreTrainingStats()  # type: ignore[return-value]

        start_time = time.time()

        indices = buffer.sample_indices(sample_size)
        # we copy the indices because they get modified within combine_slow_reward_()
        self.combine_slow_reward_(indices.copy())
        batch = buffer[indices]

        # perform the update
        self.updating = True
        batch = self.process_fn(batch, buffer, indices)
        with torch_train_mode(self):
            policy_stats = self.learn(batch, **kwargs)
            self_model_stats = self.self_model.learn(batch, **kwargs)
            env_model_stats = self.env_model.learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indices)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.updating = False

        train_time = time.time() - start_time
        return CoreTrainingStats(
            policy_stats=policy_stats,
            self_model_stats=self_model_stats,
            env_model_stats=env_model_stats,
            train_time=train_time,
        )

    def post_process_fn(
        self,
        batch: GoalBatchProtocol,
        buffer: GoalReplayBufferProtocol,
        indices: np.ndarray,
    ) -> None:
        """Post-processes the data from the provided replay buffer."""
        super().post_process_fn(batch, buffer, indices)
        # original_rew is guaranteed to exist because process_fn() always gets called before post_process_fn()
        batch.rew = batch.original_rew
        del batch.original_rew

    def combine_fast_reward_(self, batch: GoalBatchProtocol) -> None:
        """Combines the fast intrinsic reward (int_rew) and the extrinsic reward (rew) into a single scalar value, in place.

        By "fast intrinsic reward" we mean the reward as computed by SelfModel's fast_compute_reward() method.

        The underscore at the end of the name indicates that this function modifies an object it uses for computation (i.e., it isn't pure). In this case, we modify the batch.
        """
        batch.original_rew = batch.rew.copy()
        batch.rew += self.beta * batch.int_rew

    def combine_slow_reward_(self, indices: np.ndarray) -> np.ndarray:
        """Combines the slow intrinsic reward and the extrinsic reward into a single scalar value, in place.

        By "slow intrinsic reward" we mean the reward as computed by SelfModel's slow_compute_reward() method.

        The underscore at the end of the name indicates that this function modifies an object it uses for computation (i.e., it isn't pure). In this case, we modify the buffer.
        """
        self.self_model.slow_intrinsic_reward_(indices)

    def _handle_state_(
        self,
        result: ActBatchProtocol | ActStateBatchProtocol,
        state: torch.Tensor,
        **kwargs: Any,
    ):
        """Handles the hidden state.

        The underscore at the end of the name indicates that this function modifies an object it uses for computation (i.e., it isn't pure). In this case, we modify the result object.
        """
        assert "latent_obs" in kwargs
        latent = kwargs["latent_obs"]
        state = self._split_state(state)
        action = result.act.unsqueeze(1)

        outs = self.env_model.mdnrnn.pass_through_rnn(action, latent, hidden=state)

        result.state = self._cat_state(outs)

    def _split_state(
        self, state: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Splits the tensor representing the hidden state into an (h, c) tuple, as expected by the RNN."""
        if state is not None:
            # the RNN expects an (h, c) tuple as the hidden state
            state = torch.split(state, state.shape[1] // 2, dim=1)
        return state

    def _cat_state(self, rnn_outs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Concats the (h, c) tensors returned by the RNN to obtain a state compatible with the rest of the Tianshou pipeline."""
        return torch.cat(rnn_outs, dim=1)
