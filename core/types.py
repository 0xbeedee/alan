from typing import Protocol, TypeVar
from tianshou.data.types import RolloutBatchProtocol
from tianshou.data import Batch

import torch
import numpy as np

from networks import NetHackObsNet

# this type should come in handy if I want to experiment with different observation net architectures
ObservationNet = NetHackObsNet

TArrLike = TypeVar("TArrLike", bound="np.ndarray | torch.Tensor | Batch | None")


class IntrinsicBatchProtocol(RolloutBatchProtocol, Protocol):
    """A RolloutBatchProtocol with an added intrinsic reward.

    For details, see https://tianshou.org/en/stable/_modules/tianshou/data/types.html.
    """

    int_rew: torch.Tensor


class GoalBatchProtocol(IntrinsicBatchProtocol, Protocol):
    """An IntrinsicBatchProtocol with latent goals.

    Usually obtained form sampling a GoalReplayBuffer."""

    latent_goal: torch.Tensor
    latent_goal_next: torch.Tensor


# TODO a new type for stats, perhaps?
