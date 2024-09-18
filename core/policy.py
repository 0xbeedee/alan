from typing import Literal, Any, Self
from .types import (
    GoalReplayBufferProtocol,
    SelfModelProtocol,
    EnvModelProtocol,
)
from tianshou.data.types import (
    ObsBatchProtocol,
    ActStateBatchProtocol,
    ActBatchProtocol,
)


from tianshou.data.batch import BatchProtocol
from tianshou.policy import BasePolicy
from tianshou.policy.base import (
    TLearningRateScheduler,
    TrainingStatsWrapper,
    TrainingStats,
)
from tianshou.utils.torch_utils import torch_train_mode

import torch
import gymnasium as gym
import numpy as np
import time


class CoreTrainingStats(TrainingStatsWrapper):
    # TODO should I add more to this?
    def __init__(self, wrapped_stats: TrainingStats):
        super().__init__(wrapped_stats)


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
        # TODO should I initialise the models here?
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
            raise ValueError("The beta parameter must be greater than zero.")
        self._beta = value

    def get_beta(self) -> float:
        """A getter method for the beta parameter.

        Override this method in subclasses to implement custom beta calculation logic.
        """
        return self.beta

    def combine_fast_reward(self, rew: np.ndarray, int_rew: np.ndarray) -> np.ndarray:
        """Combines the fast intrinsic reward (int_rew) and the extrinsic reward (rew) into a single scalar value.

        By "fast intrinsic reward" we mean the reward as computed by SelfModel's fast_compute_reward() method.
        """
        return rew + self.get_beta() * int_rew

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

        The default implementation simply selects the latent goal and attaches it to batch.obs. It must be overridden.

        (Note that this method could easily function with torch.no_grad(), but there is no need for us to specify it here: this method is called by the Collector, and the collection process is already decorated with no_grad().)
        """
        # we must compute the latent_goals here because
        # 1) it makes the actor goal-aware (which is desirable, seeing as we'd like the agent to learn to use goals)
        # 2) it centralises goal selection
        # 3) it makes conceptual sense
        latent_goal = self.self_model.select_goal(batch.obs)
        return latent_goal

    def update(
        self,
        sample_size: int | None,
        buffer: GoalReplayBufferProtocol | None,
        **kwargs: Any,
    ) -> CoreTrainingStats:
        """Updates the policy network and replay buffer."""
        if buffer is None:
            return TrainingStats()  # type: ignore[return-value]

        start_time = time.time()

        indices = buffer.sample_indices(sample_size)
        # we copy the indices because they get modified within combine_slow_reward_
        self.combine_slow_reward_(indices.copy())
        batch = buffer[indices]

        # perform the update
        self.updating = True
        batch = self.process_fn(batch, buffer, indices)
        with torch_train_mode(self):
            training_stat = self.learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indices)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.updating = False

        training_stat.train_time = time.time() - start_time

        return training_stat

    def to(self, device: torch.device) -> Self:
        self.device = device
        self.self_model = self.self_model.to(device)
        return self
