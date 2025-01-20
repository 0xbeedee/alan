from typing import Any
from core.types import GoalBatchProtocol, LatentObsActNextBatchProtocol

from tianshou.data import SequenceSummaryStats
import numpy as np

from .bebold import BBold, BBoldTrainingStats


class ZeroBBold(BBold):
    """A BBold wrapper that returns 0 reward and skips the learning phase.

    This wrapper exists so that we can easily substitute it for BBold in a SelfModel, seeing as its API is precisely like BBold's.
    This allows us to easily zero out the fast intrinsic reward, thus better studying its effect on our agent.
    """

    def get_reward(self, batch: LatentObsActNextBatchProtocol) -> np.ndarray:
        return np.zeros_like(batch.act)

    def learn(self, batch: GoalBatchProtocol, **kwargs: Any) -> BBoldTrainingStats:
        return BBoldTrainingStats(
            bbold_loss=SequenceSummaryStats.from_sequence([0.0]),
        )
