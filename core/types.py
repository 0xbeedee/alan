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
    EpochStats,
)
from tianshou.data.batch import BatchProtocol, TArr
from tianshou.policy import BasePolicy
from tianshou.policy.base import (
    TLearningRateScheduler,
    TrainingStatsWrapper,
    TrainingStats,
)
from tianshou.trainer.base import BaseTrainer
from tianshou.utils import BaseLogger, LazyLogger

import torch
from torch import nn
import numpy as np
import gymnasium as gym

from .stats import CoreTrainingStats, EpNStepCollectStats

TArrLike = Union[np.ndarray, torch.Tensor, Batch, None]


class RandomActBatchProtocol(BatchProtocol, Protocol):
    """A BatchProtocol containing an action.

    Obtained by using a random policy, and usually used for offline training (see the docs for RandomPolicy).
    """

    act: TArr


class LatentObsActNextBatchProtocol(BatchProtocol, Protocol):
    """A BatchProtocol containing a latent observation, an action, and the latent observation after it.

    Usually used by the intrinsic module and obtained from the Collector.
    """

    latent_obs: torch.Tensor
    act: TArr
    latent_obs_next: torch.Tensor
    obs_next: TArr | BatchProtocol
    done: np.ndarray


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


class KBBatchProtocol(BatchProtocol, Protocol):
    """A BatchProtocol containing an observation, an action and additional entries needed by the knowledge base."""

    obs: TArr | BatchProtocol
    act: TArr
    rew: np.ndarray
    traj_id: int


RB = TypeVar("RB", bound=ReplayBuffer)


class GoalReplayBufferProtocol(Protocol[RB]):
    _reserved_keys: tuple
    _input_keys: tuple

    def __getitem__(
        self, index: Union[slice, int, List[int], np.ndarray]
    ) -> GoalBatchProtocol: ...


class FastIntrinsicModuleProtocol(Protocol):
    def get_reward(self, batch: LatentObsActNextBatchProtocol) -> np.ndarray: ...
    def learn(self, data: GoalBatchProtocol, **kwargs: Any) -> TrainingStats: ...


class SlowIntrinsicModuleProtocol(Protocol):
    def rewrite_rewards_(self, indices: np.ndarray) -> None: ...


class SelfModelProtocol(Protocol):
    fast_intrinsic_module: FastIntrinsicModuleProtocol
    slow_intrinsic_module: SlowIntrinsicModuleProtocol

    @torch.no_grad()
    def select_goal(self, latent_obs: torch.Tensor) -> np.ndarray: ...

    @torch.no_grad
    def fast_intrinsic_reward(
        self, batch: LatentObsActNextBatchProtocol
    ) -> np.ndarray: ...

    @torch.no_grad()
    def slow_intrinsic_reward_(self, indices: np.ndarray) -> np.ndarray: ...

    def learn(self, batch: GoalBatchProtocol, **kwargs: Any) -> TrainingStats: ...


class EnvModelProtocol(Protocol):
    vae: nn.Module
    mdnrnn: nn.Module
    vae_trainer: "VAETrainer"  # type: ignore
    mdnrnn_trainer: "MDNRNNTrainer"  # type: ignore
    device: torch.device

    def learn(
        self, data: Union[GoalBatchProtocol, GoalReplayBufferProtocol], **kwargs: Any
    ) -> TrainingStats: ...


TW = TypeVar("TW", bound=TrainingStatsWrapper)


TS = TypeVar("TS", bound=CoreTrainingStats)
BP = TypeVar("BP", bound=BasePolicy[TS])


class CorePolicyProtocol(Protocol[BP]):
    self_model: SelfModelProtocol
    env_model: EnvModelProtocol
    obs_net: nn.Module
    action_space: gym.Space
    observation_space: Optional[gym.Space]
    action_scaling: bool = False
    action_bound_method: Optional[Literal["clip", "tanh"]] = "clip"
    lr_scheduler: Optional[TLearningRateScheduler] = None
    beta: float = 0.314

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

    def update(
        self,
        sample_size: int | None,
        buffer: GoalReplayBufferProtocol | None,
        **kwargs: Any,
    ) -> CoreTrainingStats: ...

    def post_process_fn(
        self,
        batch: GoalBatchProtocol,
        buffer: GoalReplayBufferProtocol,
        indices: np.ndarray,
    ) -> None: ...

    def combine_fast_reward_(self, batch: GoalBatchProtocol) -> None: ...

    def combine_slow_reward_(self, indices: np.ndarray) -> None: ...


class GoalCollectorProtocol(Protocol):
    policy: CorePolicyProtocol
    env: Union[gym.Env, gym.vector.VectorEnv]
    buffer: Optional[GoalReplayBufferProtocol] = None
    exploration_noise: bool = False

    @torch.no_grad()
    def _collect(
        self,
        n_step: Optional[int],
        n_episode: Optional[int],
        random: bool,
        render: Optional[float],
        gym_reset_kwargs: Optional[Dict[str, Any]],
    ) -> EpNStepCollectStats: ...

    @torch.no_grad()
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
    policy: CorePolicyProtocol
    max_epoch: int
    batch_size: int | None
    train_collector: GoalCollectorProtocol | None = None
    test_collector: GoalCollectorProtocol | None = None
    buffer: GoalReplayBufferProtocol | None = None
    step_per_epoch: int | None = None
    repeat_per_collect: int | None = None
    episode_per_test: int | None = None
    update_per_step: float = 1.0
    step_per_collect: int | None = None
    episode_per_collect: int | None = None
    train_fn: Callable[[int, int], None] | None = None
    test_fn: Callable[[int, int | None], None] | None = None
    stop_fn: Callable[[float], bool] | None = None
    save_best_fn: Callable[[BasePolicy], None] | None = None
    save_checkpoint_fn: Callable[[int, int, int], str] | None = None
    resume_from_log: bool = False
    reward_metric: Callable[[np.ndarray], np.ndarray] | None = None
    logger: BaseLogger = LazyLogger()
    verbose: bool = True
    show_progress: bool = True
    test_in_train: bool = True

    def __next__(self) -> EpochStats: ...

    def _collect_training_data(self) -> EpNStepCollectStats: ...

    def test_step(self) -> tuple[EpNStepCollectStats, bool]: ...
