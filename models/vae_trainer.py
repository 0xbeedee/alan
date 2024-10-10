from typing import Dict, Union
from core.types import GoalBatchProtocol, GoalReplayBufferProtocol

from tianshou.data import SequenceSummaryStats

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from core.types import ObservationNetProtocol
import numpy as np


class VAETrainer:
    """Trainer class for the VAE model.

    Adapted from https://github.com/ctallec/world-models/blob/master/trainvae.py."""

    def __init__(
        self,
        obs_net: ObservationNetProtocol,
        vae: nn.Module,
        batch_size: int = 5,
        learning_rate: float = 1e-3,
        device: torch.device = torch.device("cpu"),
    ):
        # all are moved to the correct device by the environment model
        self.obs_net = obs_net
        self.vae = vae

        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", factor=0.5, patience=5
        )

        self.batch_size = batch_size
        self.device = device

    def train(self, data: Union[GoalBatchProtocol, GoalReplayBufferProtocol]):
        """Trains the VAE model for one epoch."""
        losses_summary = self._data_pass(data)
        self.scheduler.step(losses_summary.mean)
        return losses_summary

    def _data_pass(
        self,
        data: Union[GoalBatchProtocol, GoalReplayBufferProtocol],
    ) -> SequenceSummaryStats:
        """Performs one pass through the data."""
        total_samples = len(data)
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size

        losses = []
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_samples)
            batch_indices = np.arange(start_idx, end_idx)
            batch = data[batch_indices]

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
