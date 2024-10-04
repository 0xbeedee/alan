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
    ) -> None:
        super().__init__(
            obs_net=obs_net,
            action_space=action_space,
            hidden_sizes=hidden_sizes,
            beta=beta,
            eta=eta,
            device=device,
        )

        self.running_avg_intrinsic = 0.0
        self.n_intrinsic = 0

    def get_reward(self, batch: ObsActNextBatchProtocol) -> np.ndarray:
        # to each action corresponds a reward
        self.n_intrinsic += batch.act.size
        # use this scaling factor so that the runninng average increases in weight as we get more an more samples
        alpha = self._normalised_log(self.n_intrinsic)

        vanilla_intrew = ICM.get_reward(self, batch)

        self.running_avg_intrinsic = self.running_avg_intrinsic + (
            vanilla_intrew - self.running_avg_intrinsic
        ) / (self.n_intrinsic + 1)

        delta = vanilla_intrew - alpha * self.running_avg_intrinsic
        return delta.astype(np.float32)

    def _normalised_log(self, n: int, max_n: int = 1_000_000) -> np.float32:
        """Computes a normalised log (with base e), i.e., a log that returns values between 0 and 1."""
        # TODO max_n is set rather arbitrarily, can I do better?
        min_log = 0
        max_log = np.log(max_n, dtype=np.float32)

        # TODO is the logarithm the correct option, here?
        return (np.log(n, dtype=np.float32) - min_log) / (max_log - min_log)
