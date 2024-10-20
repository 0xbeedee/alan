from typing import Dict, Tuple
from core.types import GoalBatchProtocol, GoalReplayBufferProtocol

from tianshou.data import SequenceSummaryStats

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tianshou.data import Batch

from .utils import gmm_loss


class MDNRNNTrainer:
    """Trainer class for the MDNRNN model.

    Adapted from https://github.com/ctallec/world-models/blob/master/trainmdrnn.py."""

    def __init__(
        self,
        mdnrnn: nn.Module,
        vae: nn.Module,
        batch_size: int,
        learning_rate: float = 1e-3,
        alpha: float = 0.9,
        device: torch.device = torch.device("cpu"),
    ):
        self.mdnrnn = mdnrnn.to(device)
        self.vae = vae.to(device)

        self.optimizer = torch.optim.RMSprop(
            self.mdnrnn.parameters(), lr=learning_rate, alpha=alpha
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", factor=0.5, patience=5
        )

        self.batch_size = batch_size
        self.device = device

    def train(self, data: GoalBatchProtocol | GoalReplayBufferProtocol):
        """Trains the MDNRNN model for one epoch."""
        # train for one epoch only because we expect to accumulate plenty of data over the agent's lifetime
        losses_summary, gmm_losses_summary, bce_losses_summary, mse_losses_summary = (
            self._data_pass(data)
        )
        # TODO if I want to control latent imagination by checking loss threshold, it's probably a good idea to get the loss on some evaluation set
        self.scheduler.step(losses_summary.mean)
        return (
            losses_summary,
            gmm_losses_summary,
            bce_losses_summary,
            mse_losses_summary,
        )

    def _data_pass(
        self,
        data: GoalBatchProtocol | GoalReplayBufferProtocol,
    ) -> float:
        """Performs one pass through the data."""
        losses, gmm_losses, bce_losses, mse_losses = [], [], [], []
        for batch in data.split(self.batch_size, merge_last=True):
            batch = batch.to_torch(device=self.device)
            latent_obs, latent_obs_next = self._to_latent(batch.obs, batch.obs)

            self.optimizer.zero_grad()
            loss_dict = self._get_loss(
                latent_obs, batch.act, batch.rew, batch.done, latent_obs_next
            )
            loss_dict["loss"].backward()
            losses.append(loss_dict["loss"].item())
            gmm_losses.append(loss_dict["gmm"].item())
            bce_losses.append(loss_dict["bce"].item())
            mse_losses.append(loss_dict["mse"].item())

        losses_summary = SequenceSummaryStats.from_sequence(losses)
        gmm_losses_summary = SequenceSummaryStats.from_sequence(gmm_losses)
        bce_losses_summary = SequenceSummaryStats.from_sequence(bce_losses)
        mse_losses_summary = SequenceSummaryStats.from_sequence(mse_losses)
        return (
            losses_summary,
            gmm_losses_summary,
            bce_losses_summary,
            mse_losses_summary,
        )

    @torch.no_grad()
    def _to_latent(
        self, obs: Batch, obs_next: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transforms observations to latent space using the VAE."""
        obs_mu, obs_logsigma = self.vae.encoder(obs)
        obs_next_mu, obs_next_logsigma = self.vae.encoder(obs_next)

        latent_obs = obs_mu + obs_logsigma.exp() * torch.randn_like(obs_mu)
        latent_obs_next = obs_next_mu + obs_next_logsigma.exp() * torch.randn_like(
            obs_next_mu
        )
        return latent_obs, latent_obs_next

    def _get_loss(
        self,
        latent_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminal: torch.Tensor,
        latent_obs_next: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Computes the losses for the MDNRNN model."""
        action = action.unsqueeze(1)
        mus, sigmas, logpi, rs, ds = self.mdnrnn(action, latent_obs)

        gmm = gmm_loss(latent_obs_next, mus, sigmas, logpi)
        bce = F.binary_cross_entropy_with_logits(ds, terminal.float())

        latent_size = latent_obs.shape[1]
        mse = F.mse_loss(rs, reward)
        scale = latent_size + 2

        loss = (gmm + bce + mse) / scale
        return {"gmm": gmm, "bce": bce, "mse": mse, "loss": loss}
