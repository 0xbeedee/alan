from dataclasses import dataclass
from typing import Sequence, Any, Tuple
from core.types import (
    LatentObsActNextBatchProtocol,
    GoalBatchProtocol,
)

import torch
from torch.nn import functional as F

import numpy as np

from tianshou.utils.net.common import MLP
from tianshou.policy.base import TrainingStats
from tianshou.data import to_torch, SequenceSummaryStats


@dataclass(kw_only=True)
class ICMTrainingStats(TrainingStats):
    # unlike most TrainingStats subclasses in Tianshou, ICMTrainingStats inherits from TrainingStatsWrapper, so we needed to duplicate it here to fit the "standard" API
    icm_loss: SequenceSummaryStats
    icm_forward_loss: SequenceSummaryStats
    icm_inverse_loss: SequenceSummaryStats


class ICM:
    """An implementation of the Intrinsic Curiosity Module introduced by Pathak et al. (https://arxiv.org/abs/1705.05363)."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        batch_size: int,
        learning_rate: float = 1e-3,
        hidden_sizes: Sequence[int] = [256, 128, 64],
        beta: float = 0.2,
        eta: float = 0.07,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.forward_model = MLP(
            feature_dim + action_dim,
            output_dim=feature_dim,
            hidden_sizes=hidden_sizes,
            device=device,
        ).to(device)
        self.inverse_model = MLP(
            feature_dim * 2,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            device=device,
        ).to(device)

        params = set(
            list(self.forward_model.parameters())
            + list(self.inverse_model.parameters())
        )
        self.optim = torch.optim.Adam(params, lr=learning_rate)

        self.action_dim = action_dim
        self.batch_size = batch_size
        self.beta = beta
        self.eta = eta
        self.device = device

    def get_reward(self, batch: LatentObsActNextBatchProtocol) -> np.ndarray:
        # no need torch.no_grad() as SelfModel takes care of it
        forward_loss, _ = self._forward(batch)

        # clip the reward to be in the [0, 1] range
        # we do so to mainly to have the fast intrinsic reward play well with the slow intrinsic one
        intrinsic_reward = torch.clamp(forward_loss * self.eta, min=0.0, max=1.0)

        return intrinsic_reward.cpu().numpy().astype(np.float32)

    def learn(self, data: GoalBatchProtocol, **kwargs: Any) -> ICMTrainingStats:
        """Trains the forward and inverse models."""
        losses, forward_losses, inverse_losses = [], [], []
        for batch in data.split(self.batch_size, merge_last=True):
            forward_loss, inverse_loss = self._forward(batch)
            forward_loss, inverse_loss = forward_loss.mean(), inverse_loss.mean()

            self.optim.zero_grad()
            loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss
            loss.backward()
            self.optim.step()
            losses.append(loss.item())
            forward_losses.append(forward_loss.item())
            inverse_losses.append(inverse_loss.item())

        losses_summary = SequenceSummaryStats.from_sequence(losses)
        forward_losses_summary = SequenceSummaryStats.from_sequence(forward_losses)
        inverse_losses_summary = SequenceSummaryStats.from_sequence(inverse_losses)

        return ICMTrainingStats(
            icm_loss=losses_summary,
            icm_forward_loss=forward_losses_summary,
            icm_inverse_loss=inverse_losses_summary,
        )

    def _forward(
        self, batch: LatentObsActNextBatchProtocol
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_actions = to_torch(batch.act, dtype=torch.long, device=self.device)
        # the batch.obs and batch.obs_next have already been passed through the obs_net by the Collector
        phi1 = to_torch(batch.latent_obs, device=self.device)
        phi2 = to_torch(batch.latent_obs_next, device=self.device)

        phi2_hat = self._forward_dynamics(phi1, batch_actions)

        forward_loss = 0.5 * F.mse_loss(phi2_hat, phi2, reduction="none").sum(1)
        inverse_loss = self._inverse_dynamics(phi1, phi2, batch_actions)
        return forward_loss, inverse_loss

    def _forward_dynamics(
        self, phi1: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Predicts the feature representation (i.e., latent vector) of the next state, given the latent representation of the current state and the action."""
        one_hot_actions = F.one_hot(actions, num_classes=self.action_dim)
        return self.forward_model(torch.cat([phi1, one_hot_actions], dim=1))

    def _inverse_dynamics(
        self, phi1: torch.Tensor, phi2: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Predicts the action taken, given the feature representations of the currnt state and the next one."""
        act_hat = self.inverse_model(torch.cat([phi1, phi2], dim=1))
        return F.cross_entropy(act_hat, actions)
