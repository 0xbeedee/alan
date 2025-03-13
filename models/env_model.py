from typing import Any
from core.types import (
    GoalBatchProtocol,
    GoalReplayBufferProtocol,
)

from dataclasses import dataclass

from tianshou.policy.base import TrainingStats
from tianshou.data import SequenceSummaryStats

import torch
from torch import nn


@dataclass(kw_only=True)
class EnvModelStats(TrainingStats):
    vae_loss: SequenceSummaryStats
    vae_recon_loss: SequenceSummaryStats
    vae_kl_loss: SequenceSummaryStats

    mdnrnn_loss: SequenceSummaryStats
    mdnrnn_gmm_loss: SequenceSummaryStats
    mdnrnn_bce_loss: SequenceSummaryStats
    mdnrnn_mse_loss: SequenceSummaryStats


class EnvModel:
    def __init__(
        self,
        vae: nn.Module,
        mdnrnn: nn.Module,
        vae_trainer: "VAETrainer",  # type: ignore
        mdnrnn_trainer: "MDNRNNTrainer",  # type: ignore
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.vae = vae.to(device)
        self.mdnrnn = mdnrnn.to(device)
        self.vae_trainer = vae_trainer
        self.mdnrnn_trainer = mdnrnn_trainer
        self.device = device

    def learn(
        self, data: GoalBatchProtocol | GoalReplayBufferProtocol, **kwargs: Any
    ) -> TrainingStats:
        vae_losses_summary, vae_recon_losses_summary, vae_kl_losses_summary = (
            self.vae_trainer.train(data)
        )
        losses_summary, gmm_losses_summary, bce_losses_summary, mse_losses_summary = (
            self.mdnrnn_trainer.train(data)
        )

        return EnvModelStats(
            vae_loss=vae_losses_summary,
            vae_recon_loss=vae_recon_losses_summary,
            vae_kl_loss=vae_kl_losses_summary,
            mdnrnn_loss=losses_summary,
            mdnrnn_gmm_loss=gmm_losses_summary,
            mdnrnn_bce_loss=bce_losses_summary,
            mdnrnn_mse_loss=mse_losses_summary,
        )

    def evaluate(
        self, data: GoalBatchProtocol | GoalReplayBufferProtocol, **kwargs: Any
    ) -> TrainingStats:
        """Evaluate the environment model on data without updating weights."""
        losses, recon_losses, kl_losses = [], [], []
        gmm_losses, bce_losses, mse_losses = [], [], []

        with torch.no_grad():
            # evaluate VAE
            for batch in data.split(self.vae_trainer.batch_size, merge_last=True):
                loss, recon_loss, kl_loss = self.vae_trainer._get_loss(batch.obs)
                losses.append(loss.item())
                recon_losses.append(recon_loss.item())
                kl_losses.append(kl_loss.item())

            # evaluate MDNRNN
            for batch in data.split(self.mdnrnn_trainer.batch_size, merge_last=True):
                latent_obs, *_ = self.vae.encoder(batch.obs)
                latent_obs_next, *_ = self.vae.encoder(batch.obs_next)

                loss_dict = self.mdnrnn_trainer._get_loss(
                    latent_obs, batch.act, batch.rew, batch.done, latent_obs_next
                )

                gmm_losses.append(loss_dict["gmm"].item())
                bce_losses.append(loss_dict["bce"].item())
                mse_losses.append(loss_dict["mse"].item())
                losses.append(loss_dict["loss"].item())

        vae_losses_summary = SequenceSummaryStats.from_sequence(
            losses[: len(recon_losses)]
        )
        vae_recon_losses_summary = SequenceSummaryStats.from_sequence(recon_losses)
        vae_kl_losses_summary = SequenceSummaryStats.from_sequence(kl_losses)

        mdnrnn_losses_summary = SequenceSummaryStats.from_sequence(
            losses[len(recon_losses) :]
        )
        gmm_losses_summary = SequenceSummaryStats.from_sequence(gmm_losses)
        bce_losses_summary = SequenceSummaryStats.from_sequence(bce_losses)
        mse_losses_summary = SequenceSummaryStats.from_sequence(mse_losses)

        return EnvModelStats(
            vae_loss=vae_losses_summary,
            vae_recon_loss=vae_recon_losses_summary,
            vae_kl_loss=vae_kl_losses_summary,
            mdnrnn_loss=mdnrnn_losses_summary,
            mdnrnn_gmm_loss=gmm_losses_summary,
            mdnrnn_bce_loss=bce_losses_summary,
            mdnrnn_mse_loss=mse_losses_summary,
        )
