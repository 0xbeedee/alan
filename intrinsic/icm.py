from typing import Sequence, Self
from core.types import ObservationNetProtocol, TArrLike

import torch
from torch.nn import functional as F
from torch import device

import numpy as np
import gymnasium as gym

from tianshou.utils.net.discrete import IntrinsicCuriosityModule as tsICM
from tianshou.data import to_torch, Batch


class ICM(tsICM):
    """An implementation of the Intrinsic Curiosity Module introduced by Pathak et al. (https://arxiv.org/abs/1705.05363)."""

    def __init__(
        self,
        obs_net: ObservationNetProtocol,
        action_space: gym.Space,
        hidden_sizes: Sequence[int] = [256, 128, 64],
        device: torch.device = torch.device("cpu"),
        eta: float = 0.07,
    ) -> None:
        super().__init__(
            obs_net.to(device), obs_net.o_dim, action_space.n, hidden_sizes, device
        )
        self.feature_net = self.feature_net.to(device)
        self.eta = eta

    def forward(
        self, obs: TArrLike, act: np.ndarray, obs_next: TArrLike, **kwargs
    ) -> np.ndarray:
        # no need for the no_grad() context manager => check SelfModel
        batch_actions = to_torch(act, dtype=torch.long, device=self.device)
        batch_obs = to_torch(Batch(obs), device=self.device)
        batch_obs_next = to_torch(Batch(obs), device=self.device)

        phi1, phi2 = self.feature_net(batch_obs), self.feature_net(batch_obs_next)
        phi2_hat = self._forward_dynamics(phi1, batch_actions)

        loss = 0.5 * F.mse_loss(phi2_hat, phi2, reduction="none").sum(1)
        intrinsic_reward = loss * self.eta
        # inverse_loss = self._inverse_dynamics(phi1, phi2, batch_actions)
        return intrinsic_reward.cpu().numpy().astype(np.float32)

    # TODO these must be trained somewhere
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
