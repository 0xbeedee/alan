from typing import Dict, Tuple
from core.types import GoalBatchProtocol, GoalReplayBufferProtocol

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from core.types import ObservationNetProtocol
import numpy as np
from tianshou.data import Batch

from .utils import gmm_loss


class MDNRNNTrainer:
    """Trainer class for the MDNRNN model."""

    def __init__(
        self,
        obs_net: ObservationNetProtocol,
        mdnrnn: nn.Module,
        vae: nn.Module,
        include_reward: bool = False,
        batch_size: int = 5,
        learning_rate: float = 1e-3,
        alpha: float = 0.9,
        device: torch.device = torch.device("cpu"),
    ):
        # all are moved to the correct device by EnvModel
        self.obs_net = obs_net
        self.mdnrnn = mdnrnn
        self.vae = vae

        self.optimizer = torch.optim.RMSprop(
            self.mdnrnn.parameters(), lr=learning_rate, alpha=alpha
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", factor=0.5, patience=5
        )

        self.include_reward = include_reward
        self.batch_size = batch_size
        self.device = device

    def train(self, data: GoalBatchProtocol | GoalReplayBufferProtocol):
        """Trains the MDNRNN model."""
        # train for one epoch only because we expect to accumulate plenty of data over the agent's lifetime
        self._data_pass(data, train=True)
        val_loss = self._data_pass(data, train=False)
        # TODO only update the scheduler on test data
        self.scheduler.step(val_loss)

    def _data_pass(
        self,
        data: GoalBatchProtocol | GoalReplayBufferProtocol,
        train: bool = True,
    ) -> float:
        """Performs one pass through the data."""
        if train:
            self.mdnrnn.train()
        else:
            self.mdnrnn.eval()

        cum_loss = 0.0
        cum_gmm = 0.0
        cum_bce = 0.0
        cum_mse = 0.0
        total_samples = 0

        total_samples = len(data)
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size

        # TODO this works for testing case, not for the learn() case, in which we only get ONE batch
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_samples)

            batch_indices = np.arange(start_idx, end_idx)
            batch = data[batch_indices]

            obs_batch = batch.obs
            obs_next_batch = batch.obs_next
            act_batch = torch.as_tensor(batch.act, device=self.device)
            rew_batch = torch.as_tensor(batch.rew, device=self.device)
            done_batch = torch.as_tensor(batch.done, device=self.device)

            # Process observations to latent space
            latent_obs, latent_obs_next = self._to_latent(obs_batch, obs_next_batch)

            # Proceed with the rest of the code
            if train:
                self.optimizer.zero_grad()
                losses = self._get_loss(
                    latent_obs, act_batch, rew_batch, done_batch, latent_obs_next
                )
                losses["loss"].backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    losses = self._get_loss(
                        latent_obs, act_batch, rew_batch, done_batch, latent_obs_next
                    )

            batch_size_actual = act_batch.size(0)
            cum_loss += losses["loss"].item() * batch_size_actual
            cum_gmm += losses["gmm"].item() * batch_size_actual
            cum_bce += losses["bce"].item() * batch_size_actual
            cum_mse += (
                losses["mse"].item() * batch_size_actual
                if isinstance(losses["mse"], torch.Tensor)
                else losses["mse"] * batch_size_actual
            )
            total_samples += batch_size_actual

        avg_loss = cum_loss / total_samples
        return avg_loss

    @torch.no_grad()
    def _to_latent(
        self, obs: Batch, obs_next: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transforms observations to latent space using the VAE."""
        obs_mu, obs_logsigma = self.vae.encoder(self.obs_net(obs))
        obs_next_mu, obs_next_logsigma = self.vae.encoder(self.obs_net(obs_next))

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
        if self.include_reward:
            mse = F.mse_loss(rs, reward)
            scale = latent_size + 2
        else:
            mse = torch.tensor(0.0, device=self.device)
            scale = latent_size + 1

        loss = (gmm + bce + mse) / scale
        return {"gmm": gmm, "bce": bce, "mse": mse, "loss": loss}
