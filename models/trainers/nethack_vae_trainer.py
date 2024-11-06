from typing import Dict

import torch
import torch.nn.functional as F

import numpy as np

from .vae_trainer import VAETrainer


class NetHackVAETrainer(VAETrainer):
    """Trainer class for the NetHack VAE model."""

    def _get_loss(self, inputs: Dict[str, np.ndarray]) -> torch.Tensor:
        """Computes the VAE loss."""
        reconstructions, z, dist = self.vae(inputs)
        total_loss, recon_loss, kl_loss = self._xentropy_mse_kld(
            reconstructions, inputs, z, dist, self.vae.encoder.crop
        )
        return total_loss, recon_loss, kl_loss

    def _xentropy_mse_kld(
        self,
        reconstructions: Dict[str, torch.Tensor],
        inputs: Dict[str, np.ndarray],
        z: torch.Tensor,
        dist: torch.distributions.Distribution,
        enc_crop: "Crop",  # type:ignore
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

        kl_loss = self._compute_kl_loss(dist, z)

        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss
