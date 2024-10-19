from typing import Dict
from core.types import GoalBatchProtocol

from tianshou.data import SequenceSummaryStats

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


class NetHackVAETrainer:
    """Trainer class for the NetHack VAE model."""

    def __init__(
        self,
        obs_net: nn.Module,
        vae: nn.Module,
        batch_size: int,
        learning_rate: float = 1e-3,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.obs_net = obs_net.to(device)
        self.vae = vae.to(device)

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
            self.optimizer.zero_grad()
            loss = self._get_loss(batch.obs)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        # TODO probably a good idea to have more granularity for these losses
        return SequenceSummaryStats.from_sequence(losses)

    def _get_loss(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes the VAE loss."""
        outputs = self.vae(inputs)
        reconstructions = outputs["reconstructions"]
        mu = outputs["mu"]
        logsigma = outputs["logsigma"]
        loss = self._xentropy_mse_kld(reconstructions, inputs, mu, logsigma)
        return loss

    def _xentropy_mse_kld(self, reconstructions, inputs, mu, logsigma, kl_weight=1.0):
        """Computes the cross-entropy loss, the MSE loss and the KLD loss, depending on the various parts of the observation."""
        recon_loss = 0.0
        total_elements = 0

        categorical_keys = [
            "glyphs",
            "chars",
            "colors",
            "specials",
            "inv_glyphs",
            "inv_letters",
            "inv_oclasses",
            "inv_strs",
            "message",
            "screen_descriptions",
            "tty_chars",
            "tty_colors",
        ]
        continuous_keys = ["blstats", "tty_cursor"]

        for key in categorical_keys:
            if key in reconstructions:
                logits = reconstructions[key]
                target = inputs[key].long().to(self.device)
                # Adjust dimensions if necessary
                if logits.dim() > target.dim():
                    # For cases where logits have extra dimensions (e.g., channels)
                    logits = logits.view(-1, logits.size(-1))
                    target = target.view(-1)
                loss = F.cross_entropy(logits, target, reduction="mean")
                recon_loss += loss
                total_elements += 1

        for key in continuous_keys:
            if key in reconstructions:
                recon = reconstructions[key]
                target = inputs[key].float().to(self.device)
                loss = F.mse_loss(recon, target, reduction="mean")
                recon_loss += loss
                total_elements += 1

        recon_loss /= total_elements
        kld_loss = -0.5 * torch.mean(1 + logsigma - mu.pow(2) - logsigma.exp())
        total_loss = recon_loss + kl_weight * kld_loss
        return total_loss
