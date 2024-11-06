from core.types import GoalReplayBufferProtocol
import numpy as np

from torch import nn

from .her import HER


class ZeroHER(HER):
    """A HER wrapper that does nothing.

    This class overrides the HER's methods to perform no operations, effectively disabling the HER functionality while maintaining the same interface.

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

    def rewrite_rewards_(self, indices: np.ndarray) -> None:
        pass
