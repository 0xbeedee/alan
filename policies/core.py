from typing import Literal

import gymnasium as gym

from tianshou.policy import BasePolicy
from tianshou.policy.base import (
    TLearningRateScheduler,
    TrainingStatsWrapper,
)

from models import SelfModel, EnvModel


class CoreTrainingStats(TrainingStatsWrapper):
    def __init__(self, wrapped_stats):
        # TODO should I add more to this?
        super().__init__(wrapped_stats)


class CorePolicy(BasePolicy[CoreTrainingStats]):
    def __init__(
        self,
        *,
        action_space: gym.Space,
        observation_space: gym.Space | None = None,
        action_scaling: bool = False,
        action_bound_method: None | Literal["clip"] | Literal["tanh"] = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
        self_model: SelfModel,
        env_model: EnvModel
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
