from typing import Literal, Any, Self

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

from tianshou.data.batch import BatchProtocol
from tianshou.policy import BasePolicy
from tianshou.policy.base import TLearningRateScheduler
from tianshou.utils.torch_utils import torch_train_mode

import torch
import gymnasium as gym
import numpy as np
import time

from .stats import CoreTrainingStats


class CorePolicy(BasePolicy[CoreTrainingStats]):
    """CorePolicy is the base class for all the policies we wish to implement.

    It is analogous to Tianshou's BasePolicy (see https://tianshou.org/en/stable/03_api/policy/base.html), in that each policy we create must inherit from it.
    """

    def __init__(
        self,
        *,
        self_model: SelfModelProtocol,
        env_model: EnvModelProtocol,
        action_space: gym.Space,
        observation_space: gym.Space | None,
        action_scaling: bool = False,
        action_bound_method: None | Literal["clip"] | Literal["tanh"] = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
        beta0: float = 0.314,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )
        self.self_model = self_model.to(device)
        self.env_model = env_model
        self._beta = beta0
        self.device = device

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        if value <= 0:
            raise ValueError("the beta parameter must be greater than zero.")
        self._beta = value

    def get_beta(self) -> float:
        """A getter method for the beta parameter.

        Override this method in subclasses to implement custom beta calculation logic.
        """
        return self.beta

    def combine_fast_reward_(self, batch: GoalBatchProtocol) -> None:
        """Combines the fast intrinsic reward (int_rew) and the extrinsic reward (rew) into a single scalar value, in place.

        By "fast intrinsic reward" we mean the reward as computed by SelfModel's fast_compute_reward() method.

        The underscore at the end of the name indicates that this function modifies an object it uses for computation (i.e., it isn't pure). In our case, we modify the batch (and, specifically, the "rew" entry), and we add an additional entry to keep track of the original reward.
        """
        batch.original_rew = batch.rew.copy()
        batch.rew += self.get_beta() * batch.int_rew

    def combine_slow_reward_(self, indices: np.ndarray) -> np.ndarray:
        """Combines the slow intrinsic reward and the extrinsic reward into a single scalar value, in place.

        By "slow intrinsic reward" we mean the reward as computed by SelfModel's slow_compute_reward() method.

        The underscore at the end of the name indicates that this function modifies an object it uses for computation (i.e., it isn't pure). In our case, we modify the buffer (and, specifically, the "rew" entry).
        """
        self.self_model.slow_intrinsic_reward_(indices)

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol | ActStateBatchProtocol:
        """Compute action over the given batch of data.

        The default implementation simply selects the latent goal. It must be overridden.

        (Note that this method could easily function with torch.no_grad(), but there is no need for us to specify it here: this method is called by the Collector, and the collection process is already decorated with no_grad().)
        """
        # we must compute the latent_goals here because
        # 1) it makes the actor goal-aware (which is desirable, seeing as we'd like the agent to learn to use goals)
        # 2) it centralises goal selection
        # 3) it makes conceptual sense
        latent_goal = self.self_model.select_goal(batch.obs)
        return latent_goal

    def process_fn(
        self,
        batch: GoalBatchProtocol,
        buffer: GoalReplayBufferProtocol,
        indices: np.ndarray,
    ) -> GoalBatchProtocol:
        """Pre-processes the data from the provided replay buffer.

        It is meant to be overwritten by the policy. The current implementation simply adds the fast intrinsic reward.
        """
        self.combine_fast_reward_(batch)
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
        # we copy the indices because they get modified within combine_slow_reward_
        self.combine_slow_reward_(indices.copy())
        batch = buffer[indices]

        # perform the update
        self.updating = True
        batch = self.process_fn(batch, buffer, indices)
        with torch_train_mode(self):
            policy_stats = self.learn(batch, **kwargs)
            self_model_stats = self.self_model.learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indices)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.updating = False

        train_time = time.time() - start_time
        return CoreTrainingStats(
            policy_stats=policy_stats,
            self_model_stats=self_model_stats,
            env_model_stats=None,  # TODO
            train_time=train_time,
        )

    def post_process_fn(
        self,
        batch: GoalBatchProtocol,
        buffer: GoalReplayBufferProtocol,
        indices: np.ndarray,
    ) -> None:
        """Post-process the data from the provided replay buffer."""
        # we do not check for existence of original_rew because we're guaranteed to have it
        super().post_process_fn(batch, buffer, indices)
        batch.rew = batch.original_rew
        del batch.original_rew

    def to(self, device: torch.device) -> Self:
        self.device = device
        self.self_model = self.self_model.to(device)
        return self
