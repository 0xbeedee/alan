from dataclasses import dataclass
from typing import Sequence, Self, Any
from core.types import ObsActNextBatchProtocol, ObservationNetProtocol

import torch
from torch.nn import functional as F

import numpy as np
import gymnasium as gym

from tianshou.utils.net.discrete import IntrinsicCuriosityModule
from tianshou.policy.base import TrainingStats
from tianshou.data import to_torch


@dataclass(kw_only=True)
class ICMTrainingStats(TrainingStats):
    # unlike most TrainingStats subclasses in Tianshou, ICMTrainingStats inherits from TrainingStatsWrapper, so we needed to duplicate it here to fit the "standard" API
    icm_loss: float
    icm_forward_loss: float
    icm_inverse_loss: float


class ICM(IntrinsicCuriosityModule):
    """An implementation of the Intrinsic Curiosity Module introduced by Pathak et al. (https://arxiv.org/abs/1705.05363)."""

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
            feature_net=obs_net.to(device),
            feature_dim=obs_net.o_dim,
            action_dim=action_space.n,
            hidden_sizes=hidden_sizes,
            device=device,
        )
        self.feature_net = self.feature_net.to(device)

        # (feature_net parameters not included because it's trained separately)
        params = set(
            list(self.forward_model.parameters())
            + list(self.inverse_model.parameters())
        )
        self.optim = torch.optim.Adam(params, lr=1e-3)

        self.beta = beta
        self.eta = eta
        self.device = device

    def get_reward(self, batch: ObsActNextBatchProtocol) -> np.ndarray:
        # no need torch.no_grad() as SelfModel takes care of it
        forward_loss, _ = self.forward(batch)

        intrinsic_reward = forward_loss * self.eta
        return intrinsic_reward.cpu().numpy().astype(np.float32)

    def forward(self, batch: ObsActNextBatchProtocol) -> np.ndarray:
        batch_actions = to_torch(batch.act, dtype=torch.long, device=self.device)
        batch_obs = to_torch(batch.obs, device=self.device)
        batch_obs_next = to_torch(batch.obs_next, device=self.device)

        phi1, phi2 = self.feature_net(batch_obs), self.feature_net(batch_obs_next)
        phi2_hat = self._forward_dynamics(phi1, batch_actions)

        forward_loss = 0.5 * F.mse_loss(phi2_hat, phi2, reduction="none").sum(1)
        inverse_loss = self._inverse_dynamics(phi1, phi2, batch_actions)
        return forward_loss, inverse_loss

    def learn(self, batch: ObsActNextBatchProtocol, **kwargs: Any) -> ICMTrainingStats:
        """Train the forward and backward models."""
        forward_loss, inverse_loss = self.forward(batch)
        forward_loss, inverse_loss = forward_loss.mean(), inverse_loss.mean()

        self.optim.zero_grad()
        loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss
        loss.backward()
        self.optim.step()

        return ICMTrainingStats(
            icm_loss=loss.item(),
            icm_forward_loss=forward_loss.item(),
            icm_inverse_loss=inverse_loss.item(),
        )

    def _forward_dynamics(
        self, phi1: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        one_hot_actions = F.one_hot(actions, num_classes=self.action_dim)
        return self.forward_model(torch.cat([phi1, one_hot_actions], dim=1))

    def _inverse_dynamics(
        self, phi1: torch.Tensor, phi2: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        act_hat = self.inverse_model(torch.cat([phi1, phi2], dim=1))
        return F.cross_entropy(act_hat, actions)

    def to(self, device: torch.device) -> Self:
        self.device = device
        self.feature_net = self.feature_net.to(device)
        self.forward_model = self.forward_model.to(device)
        self.inverse_model = self.inverse_model.to(device)
        return super().to(device)
