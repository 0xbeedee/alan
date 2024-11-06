from typing import Tuple

import torch
from torch import nn

from .vae_trainer import VAETrainer


class DiscreteVAETrainer(VAETrainer):
    """Trainer class for the Discrete VAE model."""

    def _get_loss(
        self,
        inputs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the VAE loss components."""
        reconstructions, z, dist = self.vae(inputs)

        recon_loss = nn.functional.mse_loss(
            reconstructions, inputs.view(-1, 1), reduction="mean"
        )
        kl_loss = self._compute_kl_loss(dist, z)
        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss
