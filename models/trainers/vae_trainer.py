from typing import Tuple
from abc import abstractmethod

from tianshou.data import SequenceSummaryStats, Batch

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions import Distribution, MultivariateNormal, kl


class VAETrainer:
    """A generic trainer class for a VAE."""

    def __init__(
        self,
        vae: nn.Module,
        batch_size: int,
        learning_rate: float = 1e-3,
        kl_weight: float = 1.0,
        device: torch.device = torch.device("cpu"),
        use_finetuning: bool = False,
        freeze_envmodel: bool = False,
    ) -> None:
        self.vae = vae.to(device)
        self.batch_size = batch_size
        self.kl_weight = kl_weight
        self.device = device
        self.use_finetuning = use_finetuning
        self.freeze_envmodel = freeze_envmodel

        if self.use_finetuning:
            # TODO this is a rather naive approach, and possibly unfit for stochasstic envs (like NetHack)
            # reduce the learning rate to make new experience less influential
            self.optimizer = torch.optim.Adam(
                self.vae.parameters(), lr=learning_rate / 10
            )
        else:
            self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", factor=0.5, patience=5
        )

    @abstractmethod
    def _get_loss(
        self, inputs: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the total loss, reconstruction loss, and KL divergence loss."""

    def train(
        self, data: Batch
    ) -> Tuple[SequenceSummaryStats, SequenceSummaryStats, SequenceSummaryStats]:
        """Trains the VAE for one epoch."""
        if self.freeze_envmodel:
            # no training needed
            return None, None, None

        losses_summary, recon_losses_summary, kl_losses_summary = self._data_pass(data)
        self.scheduler.step(losses_summary.mean)
        return losses_summary, recon_losses_summary, kl_losses_summary

    def _data_pass(
        self,
        data: Batch,
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

    def _compute_kl_loss(self, dist: Distribution, z: torch.Tensor) -> torch.Tensor:
        """Computes the Kullback-Leibler divergence loss."""
        # from https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/
        std_normal = MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device)
            .unsqueeze(0)
            .expand(z.shape[0], -1, -1),
        )
        return kl.kl_divergence(dist, std_normal).mean()
