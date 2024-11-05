from typing import Dict, Tuple
from core.types import GoalBatchProtocol

from tianshou.data import SequenceSummaryStats

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, kl
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np


class NetHackVAETrainer:
    """Trainer class for the NetHack VAE model."""

    def __init__(
        self,
        vae: nn.Module,
        batch_size: int,
        learning_rate: float = 1e-3,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.vae = vae.to(device)

        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", factor=0.5, patience=5
        )

        self.batch_size = batch_size
        self.device = device

    def train(
        self, data: GoalBatchProtocol
    ) -> Tuple[SequenceSummaryStats, SequenceSummaryStats, SequenceSummaryStats]:
        """Trains the VAE model for one epoch."""
        losses_summary, recon_losses_summary, kl_losses_summary = self._data_pass(data)
        self.scheduler.step(losses_summary.mean)
        return losses_summary, recon_losses_summary, kl_losses_summary

    def _data_pass(
        self,
        data: GoalBatchProtocol,
    ) -> Tuple[SequenceSummaryStats, SequenceSummaryStats, SequenceSummaryStats]:
        """Performs one pass through the data."""
        losses, recon_losses, kl_losses = [], [], []
        for batch in data.split(self.batch_size, merge_last=True):
            self.optimizer.zero_grad()
            loss, recon_loss, kl_loss = self._get_loss(batch.obs)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())

        losses_summary = SequenceSummaryStats.from_sequence(losses)
        recon_losses_summary = SequenceSummaryStats.from_sequence(recon_losses)
        kl_losses_summary = SequenceSummaryStats.from_sequence(kl_losses)
        return losses_summary, recon_losses_summary, kl_losses_summary

    def _get_loss(self, inputs: Dict[str, np.ndarray]) -> torch.Tensor:
        """Computes the VAE loss."""
        reconstructions, z, dist = self.vae(inputs)
        loss = self._xentropy_mse_kld(
            reconstructions, inputs, z, dist, self.vae.encoder.crop
        )
        return loss

    def _xentropy_mse_kld(
        self,
        reconstructions: Dict[str, torch.Tensor],
        inputs: Dict[str, np.ndarray],
        z: torch.Tensor,
        dist: torch.distributions.Distribution,
        enc_crop: "Crop",  # type:ignore
        kl_weight: float = 1.0,
    ):
        """Computes the cross-entropy loss, the MSE loss and the KLD loss, depending on the various parts of the observation."""
        recon_loss = 0.0
        total_elements = 0

        # egocentric_view is not part of the vanilla env observations, so we need to compute its ground truth here (using the encoder's Crop instance)
        inputs["egocentric_view"] = enc_crop(
            torch.as_tensor(inputs["glyphs"], device=self.device),
            torch.as_tensor(inputs["blstats"][:, :2], device=self.device),
        )
        for key in self.vae.categorical_keys:
            if key in reconstructions:
                logits = reconstructions[key]  # (B, num_classes, H, W)
                target = torch.as_tensor(
                    inputs[key], device=self.device
                ).long()  # (B, H, W)
                loss = F.cross_entropy(logits, target, reduction="mean")
                recon_loss += loss
                total_elements += 1

        for key in self.vae.continuous_keys:
            if key in reconstructions:
                recon = reconstructions[key]
                target = torch.as_tensor(inputs[key], device=self.device).float()
                loss = F.mse_loss(recon, target, reduction="mean")
                recon_loss += loss
                total_elements += 1
        recon_loss /= total_elements

        # from https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/
        std_normal = MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device)
            .unsqueeze(0)
            .expand(z.shape[0], -1, -1),
        )
        kl_loss = kl.kl_divergence(dist, std_normal).mean()

        total_loss = recon_loss + kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss
