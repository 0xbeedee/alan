from typing import Dict
from core.types import GoalBatchProtocol

from tianshou.data import SequenceSummaryStats

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from core.types import ObservationNetProtocol


class VAETrainer:
    """Trainer class for the VAE model.

    Adapted from https://github.com/ctallec/world-models/blob/master/trainvae.py."""

    def __init__(
        self,
        obs_net: ObservationNetProtocol,
        vae: nn.Module,
        batch_size: int,
        learning_rate: float = 1e-3,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        # all are moved to the correct device by the environment model
        self.obs_net = obs_net
        self.vae = vae

        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", factor=0.5, patience=5
        )

        self.batch_size = batch_size
        self.device = device

    def train(self, data: GoalBatchProtocol) -> SequenceSummaryStats:
        """Trains the VAE model for one epoch."""
        losses_summary = self._data_pass(data)
        self.scheduler.step(losses_summary.mean)
        return losses_summary

    def _data_pass(
        self,
        data: GoalBatchProtocol,
    ) -> SequenceSummaryStats:
        """Performs one pass through the data."""
        losses = []
        for batch in data.split(self.batch_size, merge_last=True):
            obs_batch = batch.obs
            inputs = self.obs_net(obs_batch).to(self.device)

            self.optimizer.zero_grad()
            loss = self._get_loss(inputs)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        return SequenceSummaryStats.from_sequence(losses)

    def _get_loss(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes the VAE loss."""
        recon_batch, mu, logsigma = self.vae(inputs)
        loss = self._mse_kld(recon_batch, inputs, mu, logsigma)
        return loss

    def _mse_kld(self, recon_x, x, mu, logsigma):
        MSE = F.mse_loss(recon_x, x, reduction="sum")
        # KL divergence
        KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
        return (MSE + KLD) / x.size(0)  # normalise by batch size
