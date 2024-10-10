from typing import Dict, Union
from core.types import GoalBatchProtocol, GoalReplayBufferProtocol

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
        self._data_pass(data, train=True)
        # TODO possibly calculate val loss async? maybe at update() time or something? (same for mdnrnn)
        val_loss = self._data_pass(data, train=False)
        # TODO only update scheduler on test data
        self.scheduler.step(val_loss)

    def _data_pass(
        self,
        data: Union[GoalBatchProtocol, GoalReplayBufferProtocol],
        train: bool = True,
    ) -> float:
        """Performs one pass through the data."""
        self.vae.train() if train else self.vae.eval()

        total_samples = len(data)
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size

        cum_loss = 0.0
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_samples)
            batch_indices = np.arange(start_idx, end_idx)
            batch = data[batch_indices]

            obs_batch = batch.obs
            inputs = self.obs_net(obs_batch).to(self.device)

            if train:
                self.optimizer.zero_grad()
                loss = self._get_loss(inputs)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    loss = self._get_loss(inputs)

            batch_size_actual = inputs.size(0)
            cum_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual

        avg_loss = cum_loss / total_samples
        return avg_loss

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
