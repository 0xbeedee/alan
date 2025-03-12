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
    # the GMM overwhelms the display of the other losses
    # mdnrnn_gmm_loss: SequenceSummaryStats
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
        losses_summary, _, bce_losses_summary, mse_losses_summary = (
            self.mdnrnn_trainer.train(data)
        )

        return EnvModelStats(
            vae_loss=vae_losses_summary,
            vae_recon_loss=vae_recon_losses_summary,
            vae_kl_loss=vae_kl_losses_summary,
            mdnrnn_loss=losses_summary,
            # mdnrnn_gmm_loss=gmm_losses_summary,
            mdnrnn_bce_loss=bce_losses_summary,
            mdnrnn_mse_loss=mse_losses_summary,
        )
