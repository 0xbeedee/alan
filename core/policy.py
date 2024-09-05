from typing import Literal, Any
import time

from gymnasium import Space
import numpy as np

from tianshou.data import ReplayBuffer
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import (
    RolloutBatchProtocol,
    ObsBatchProtocol,
    ActStateBatchProtocol,
    ActBatchProtocol,
)
from tianshou.policy import BasePolicy
from tianshou.policy.base import (
    TLearningRateScheduler,
    TrainingStatsWrapper,
)
from tianshou.utils.torch_utils import torch_train_mode

from core.buffer import GoalReplayBuffer
from models import SelfModel, EnvModel


class CoreTrainingStats(TrainingStatsWrapper):
    def __init__(self, wrapped_stats):
        super().__init__(wrapped_stats)
        # TODO should I add more to this?


class CorePolicy(BasePolicy[CoreTrainingStats]):
    """This is the Core policy object we'll use throughout.

    It is analogous to Tianshou's BasePolicy, in that each policy we create MUST inherit from it.
    """

    def __init__(
        self,
        *,
        self_model: SelfModel,
        env_model: EnvModel,
        action_space: Space,
        observation_space: Space | None,
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

    def combine_reward(self, batch: RolloutBatchProtocol) -> np.ndarray:
        """Combines the intrinsic and extrinsic rewards into a single scalar value in-place.

        A default implementation is provided, but this method is meant to be overridden.
        """
        i_rew = self.self_model(batch)
        batch.rew += self.get_beta() * i_rew

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol | ActStateBatchProtocol:
        """Compute action over the given batch data.

        The default implementation simply selects the latent goal and attaches it to batch.obs. It must be overridden!
        """
        # we must compute the latent_goals here because
        # 1) it makes the actor goal-aware (which is desirable, seeing as we'd like the agent to learn to use goals)
        # 2) it centralises goal selection
        # 3) it makes conceptual sense
        latent_goal = self.self_model.select_goal(batch.obs)
        # TODO this is somewhat hacky, but it provides a cleaner interface with Tianshou
        batch.obs["latent_goal"] = latent_goal

    # TODO better docs throughout! (at least one expalanatory line per method)
    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:
        # it is sufficient to call combine_reward here because process_fn() gets called before all the learning happens
        self.combine_reward(batch)
        # TODO this is somewhat hacky, but it provides a cleaner interface with Tianshou
        batch.obs["latent_goal"] = batch.latent_goal
        batch.obs_next["latent_goal"] = batch.latent_goal
        return batch

    def update(
        self,
        sample_size: int | None,
        buffer: GoalReplayBuffer | None,
        **kwargs: Any,
    ) -> CoreTrainingStats:
        if buffer is None:
            return TrainingStats()  # type: ignore[return-value]
        start_time = time.time()
        batch, indices = buffer.sample(sample_size)
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
