from typing import Sequence

import torch
from torch.nn import functional as F
from torch import device
from torch import nn

from tianshou.utils.net.discrete import IntrinsicCuriosityModule as tsICM
from tianshou.data import to_torch


class ICM(tsICM):
    """https://arxiv.org/abs/1705.05363"""

    def __init__(
        self,
        obs_net: nn.Module,
        action_space_n: int,
        eta: float = 0.04,
        hidden_sizes: Sequence[int] = [256, 128, 64],
        device: str | device = "cpu",
    ) -> None:
        super().__init__(obs_net, obs_net.o_dim, action_space_n, hidden_sizes, device)

        self.eta = eta

    def forward(self, batch_obs_t, batch_actions, batch_obs_t1, **kwargs):
        phi1, phi2 = self.feature_net(batch_obs_t), self.feature_net(batch_obs_t1)

        batch_actions = to_torch(batch_actions, dtype=torch.long, device=self.device)
        phi2_hat = self.forward_model(
            torch.cat(
                [phi1, F.one_hot(batch_actions, num_classes=self.action_dim)], dim=1
            ),
        )

        mse_loss = 0.5 * F.mse_loss(phi2_hat, phi2, reduction="none").sum(1)
        act_hat = self.inverse_model(torch.cat([phi1, phi2], dim=1))

        return mse_loss * self.eta, act_hat
