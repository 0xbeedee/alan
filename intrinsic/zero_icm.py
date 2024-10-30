from typing import Any
from core.types import GoalBatchProtocol, LatentObsActNextBatchProtocol

from tianshou.data import SequenceSummaryStats

import numpy as np

from intrinsic.icm import ICMTrainingStats
from .icm import ICM


class ZeroICM(ICM):
    """An ICM wrapper that returns 0 reward and skips the learning phase.

    This wrapper exists so that we can easily substitute it for ICM in a SelfModel, seeing as its API is precisely like ICM's.
    This allows us to easily zero out the fast intrinsic reward, thus better studying its effect on our agent.
    """

    def get_reward(self, batch: LatentObsActNextBatchProtocol) -> np.ndarray:
        return np.zeros_like(batch.act)

    def learn(self, batch: GoalBatchProtocol, **kwargs: Any) -> ICMTrainingStats:
        return ICMTrainingStats(
            icm_loss=SequenceSummaryStats.from_sequence([0.0]),
            icm_forward_loss=SequenceSummaryStats.from_sequence([0.0]),
            icm_inverse_loss=SequenceSummaryStats.from_sequence([0.0]),
        )
