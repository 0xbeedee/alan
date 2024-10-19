from typing import (
    Protocol,
    TypeVar,
    Any,
    Union,
    List,
    Dict,
    Optional,
    Literal,
    Callable,
)
from tianshou.data.types import (
    RolloutBatchProtocol,
    ObsBatchProtocol,
    ActStateBatchProtocol,
    ActBatchProtocol,
)
from tianshou.data import (
    Batch,
    ReplayBuffer,
    CollectStats,
    EpochStats,
)
from tianshou.data.batch import BatchProtocol
from tianshou.policy import BasePolicy
from tianshou.policy.base import (
    TLearningRateScheduler,
    TrainingStatsWrapper,
    TrainingStats,
)
from tianshou.trainer.base import BaseTrainer
from tianshou.utils import BaseLogger

import torch
from torch import nn
import numpy as np
import gymnasium as gym

TArrLike = Union[np.ndarray, torch.Tensor, Batch, None]


class ObsActNextBatchProtocol(BatchProtocol, Protocol):
    """A BatchProtocol containing an observation, an action, and the observation after it.

    Usually used by the intrinsic module and obtained from the Collector.
    """

    obs: Union[TArrLike, BatchProtocol]
    act: np.ndarray
    obs_next: Union[TArrLike, BatchProtocol]


class IntrinsicBatchProtocol(RolloutBatchProtocol, Protocol):
    """A RolloutBatchProtocol with added intrinsic rewards.

    For details on RolloutBatchProtocol, see https://tianshou.org/en/stable/_modules/tianshou/data/types.html.
    """

    int_rew: np.ndarray


class GoalBatchProtocol(IntrinsicBatchProtocol, Protocol):
    """An IntrinsicBatchProtocol with latent goals for the current and the next observation.

    Usually obtained from sampling a GoalReplayBuffer.
    """

    latent_goal: np.ndarray
    latent_goal_next: np.ndarray


RB = TypeVar("RB", bound=ReplayBuffer)


class GoalReplayBufferProtocol(Protocol[RB]):
    _reserved_keys: tuple
    _input_keys: tuple

    def __getitem__(
        self, index: Union[slice, int, List[int], np.ndarray]
    ) -> GoalBatchProtocol: ...


class FastIntrinsicModuleProtocol(Protocol):
    def get_reward(self, batch: ObsActNextBatchProtocol) -> np.ndarray: ...
    def learn(self, data: GoalBatchProtocol, **kwargs: Any) -> TrainingStats: ...


class SlowIntrinsicModuleProtocol(Protocol):
    def get_future_observation_(self, indices: np.ndarray) -> Batch: ...
    def rewrite_transitions_(self, future_achieved_goal: np.ndarray) -> None: ...


class SelfModelProtocol(Protocol):
    obs_net: nn.Module
    fast_intrinsic_module: FastIntrinsicModuleProtocol
    slow_intrinsic_module: SlowIntrinsicModuleProtocol

    def __init__(
        self,
        obs_net: nn.Module,
        fast_intrinsic_module: FastIntrinsicModuleProtocol,
        slow_intrinsic_module: SlowIntrinsicModuleProtocol,
        device: torch.device,
    ) -> None: ...

    @torch.no_grad()
    def select_goal(self, batch_obs: ObsBatchProtocol) -> torch.Tensor: ...

    @torch.no_grad
    def fast_intrinsic_reward(self, batch: ObsActNextBatchProtocol) -> np.ndarray: ...

    @torch.no_grad()
    def slow_intrinsic_reward_(self, indices: np.ndarray) -> np.ndarray: ...

    def learn(self, batch: GoalBatchProtocol, **kwargs: Any) -> TrainingStats: ...

    def __call__(self, batch: GoalBatchProtocol, sleep: bool = False) -> None: ...


class EnvModelProtocol(Protocol):
    vae: nn.Module
    mdnrnn: nn.Module
    vae_trainer: "VAETrainer"  # type: ignore
    mdnrnn_trainer: "MDNRNNTrainer"  # type: ignore
    device: torch.device

    def __init__(
        self,
        vae: nn.Module,
        mdnrnn: nn.Module,
        vae_trainer: "VAETrainer",  # type:ignore
        mdnrnn_trainer: "MDNRNNTrainer",  # type:ignore
        device: torch.device = torch.device("cpu"),
    ) -> None: ...

    def learn(
        self, data: Union[GoalBatchProtocol, GoalReplayBufferProtocol], **kwargs: Any
    ) -> TrainingStats: ...


TW = TypeVar("TW", bound=TrainingStatsWrapper)


class CoreTrainingStatsProtocol(Protocol[TW]):
    def __init__(self, wrapped_stats: TrainingStats): ...


TS = TypeVar("TS", bound=CoreTrainingStatsProtocol)
BP = TypeVar("BP", bound=BasePolicy[TS])


class CorePolicyProtocol(Protocol[BP]):
    self_model: SelfModelProtocol
    env_model: EnvModelProtocol

    def __init__(
        self,
        *,
        self_model: SelfModelProtocol,
        env_model: EnvModelProtocol,
        action_space: gym.Space,
        observation_space: Optional[gym.Space],
        action_scaling: bool = False,
        action_bound_method: Optional[Literal["clip", "tanh"]] = "clip",
        lr_scheduler: Optional[TLearningRateScheduler] = None,
        beta: float,
    ) -> None: ...

    @property
    def beta(self) -> float: ...

    @beta.setter
    def beta(self, value: float) -> None: ...

    def get_beta(self) -> float: ...

    def combine_fast_reward(
        self, rew: np.ndarray, int_rew: np.ndarray
    ) -> np.ndarray: ...

    def combine_slow_reward_(self, batch: GoalBatchProtocol) -> np.ndarray: ...

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: Optional[Union[Dict, BatchProtocol, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[ActBatchProtocol, ActStateBatchProtocol]: ...

    def process_fn(
        self,
        batch: GoalBatchProtocol,
        buffer: GoalReplayBufferProtocol,
        indices: np.ndarray,
    ) -> GoalBatchProtocol: ...


CS = TypeVar("CS", bound=CollectStats)


class GoalCollectStatsProtocol(Protocol[CS]):
    int_returns: np.ndarray

    @classmethod
    def with_autogenerated_stats(
        cls,
        returns: np.ndarray,
        int_returns: np.ndarray,
        lens: np.ndarray,
        n_collected_episodes: int,
        n_collected_steps: int,
        collect_time: float,
        collect_speed: float,
    ) -> "GoalCollectStatsProtocol": ...


class GoalCollectorProtocol(Protocol):
    def __init__(
        self,
        policy: CorePolicyProtocol,
        env: Union[gym.Env, gym.vector.VectorEnv],
        buffer: Optional[GoalReplayBufferProtocol] = None,
        exploration_noise: bool = False,
    ) -> None: ...

    def _collect(
        self,
        n_step: Optional[int],
        n_episode: Optional[int],
        random: bool,
        render: Optional[float],
        gym_reset_kwargs: Optional[Dict[str, Any]],
    ) -> GoalCollectStatsProtocol: ...

    def _compute_action_policy_hidden(
        self,
        random: bool,
        ready_env_ids_R: np.ndarray,
        last_obs_RO: np.ndarray,
        last_info_R: np.ndarray,
        last_hidden_state_RH: Optional[Union[np.ndarray, torch.Tensor, Batch]] = None,
    ) -> tuple[
        np.ndarray, np.ndarray, Batch, Optional[Union[np.ndarray, torch.Tensor, Batch]]
    ]: ...


BT = TypeVar("BT", bound=BaseTrainer)


class GoalTrainerProtocol(Protocol[BT]):
    def __init__(
        self,
        policy: CorePolicyProtocol,
        max_epoch: int,
        batch_size: Optional[int],
        train_collector: Optional[GoalCollectorProtocol],
        test_collector: Optional[GoalCollectorProtocol],
        buffer: Optional[GoalReplayBufferProtocol],
        step_per_epoch: Optional[int],
        repeat_per_collect: Optional[int],
        episode_per_test: Optional[int],
        update_per_step: float,
        step_per_collect: Optional[int],
        episode_per_collect: Optional[int],
        train_fn: Optional[Callable[[int, int], None]],
        test_fn: Optional[Callable[[int, Optional[int]], None]],
        stop_fn: Optional[Callable[[float], bool]],
        save_best_fn: Optional[Callable[[CorePolicyProtocol], None]],
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]],
        resume_from_log: bool,
        reward_metric: Optional[Callable[[np.ndarray], np.ndarray]],
        logger: BaseLogger,
        verbose: bool,
        show_progress: bool,
        test_in_train: bool,
    ) -> None: ...

    def __next__(self) -> EpochStats: ...

    def _collect_training_data(self) -> CollectStats: ...
