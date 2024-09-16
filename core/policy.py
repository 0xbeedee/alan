from typing import Literal, Any
from .types import (
    GoalBatchProtocol,
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

import gymnasium as gym
import numpy as np


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
    ) -> None:
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )
        # TODO should I initialise the models here?
        self.self_model = self_model
        self.env_model = env_model
        self._beta = beta0

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        if value <= 0:
            raise ValueError("The beta parameter must be greater than zero!")
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

    def combine_slow_reward_(self, batch: GoalBatchProtocol) -> np.ndarray:
        """Combines the slow intrinsic reward and the extrinsic reward into a single scalar value, in place.

        By "slow intrinsic reward" we mean the reward as computed by SelfModel's slow_compute_reward() method.

        The underscore at the end of the name indicates that this function modifies an object it uses for computation (i.e., it isn't pure). In our case, we modify the buffer (and, specifically, the "rew" entry).
        """
        # TODO is this the correct way to get batch size?
        self.self_model.slow_intrinsic_reward_(len(batch))

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

    def process_fn(
        self,
        batch: GoalBatchProtocol,
        buffer: GoalReplayBufferProtocol,
        indices: np.ndarray,
    ) -> GoalReplayBufferProtocol:
        """Pre-processes the data from the specified buffer before updating the policy.

        This method gets called as soon as data collection is done and we wish to use this data to improve our agent.
        """
        # TODO I could pass the indices to HER directly?
        self.combine_slow_reward_(batch)

        return batch
