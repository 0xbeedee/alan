from typing import Sequence
from core.types import ObsActNextBatchProtocol, ObservationNetProtocol

import torch
import numpy as np
import gymnasium as gym

from .icm import ICM


class DeltaICM(ICM):
    """An implementation of ICM that applies a form of delta encoding (https://en.wikipedia.org/wiki/Delta_encoding) to the intrinsic reward.

    We employ delta encoding because in difficult environments with a great deal of novelty (e.g., NetHack), the agent gets flooded with fast intrinsic rewards, ignoring the extrinsic ones.
    This implementation aims to ameliorate this issue by only rewarding the agent for differences in intrinsic rewards, attempting to mimic biological neural adaptation (https://en.wikipedia.org/wiki/Neural_adaptation).
    """

    def __init__(
        self,
        obs_net: ObservationNetProtocol,
        action_space: gym.Space,
        hidden_sizes: Sequence[int] = [256, 128, 64],
        beta: float = 0.2,
        eta: float = 0.07,
        device: torch.device = torch.device("cpu"),
        delta_scale: float = 0.1,
        smoothing: float = 0.95,
    ) -> None:
        super().__init__(
            obs_net=obs_net,
            action_space=action_space,
            hidden_sizes=hidden_sizes,
            beta=beta,
            eta=eta,
            device=device,
        )

        self.delta_scale = delta_scale
        self.smoothing = smoothing
        self.previous_intrinsic_reward = 0.0
        # exponential moving average for intrinsic reward
        self.ema_intrinsic = 0.0

    def get_reward(self, batch: ObsActNextBatchProtocol) -> np.ndarray:
        intrinsic_reward = super().get_reward(batch)

        self.ema_intrinsic = (
            self.smoothing * self.ema_intrinsic
            + (1 - self.smoothing) * intrinsic_reward
        )

        delta = intrinsic_reward - self.ema_intrinsic
        delta_intrinsic_reward = self.delta_scale * delta
        return delta_intrinsic_reward.astype(np.float32)
