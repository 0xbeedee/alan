from core.types import GoalReplayBufferProtocol
from tianshou.data import Batch
import numpy as np

from torch import nn

from .her import HER


class ZeroHER(HER):
    """A HER wrapper that does nothing.

    This class overrides the methods of HER to perform no operations, effectively disabling the HER functionality while maintaining the same interface.

    (Alhough a more representative name for it could be NoOpHER, we kept the name similar to ZeroICM, seeing as the purpose of this class is exacly the same: ablation studies.)
    """

    def __init__(
        self,
        obs_net: nn.Module,
        buffer: GoalReplayBufferProtocol,
        horizon: int,
        future_k: float = 8.0,
        epsilon: float = 0.001,
    ) -> None:
        super().__init__(obs_net, buffer, horizon, future_k, epsilon)

    def get_future_observation_(self, indices: np.ndarray) -> Batch:
        # return the next observations without any modifications
        return self.buf[indices].obs_next

    def rewrite_transitions_(self, future_achieved_goal: np.ndarray) -> None:
        pass

    def _compute_reward(
        self,
        desired_goal: np.ndarray,
        achieved_goal: np.ndarray,
        lead_dims: int = 2,
    ) -> np.ndarray:
        # return zero rewards
        lead_shape = desired_goal.shape[:lead_dims]
        return np.zeros(lead_shape, dtype=np.float32)
