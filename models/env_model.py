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
    mdnrnn_loss: SequenceSummaryStats
    mdnrnn_gmm_loss: SequenceSummaryStats
    mdnrnn_bce_loss: SequenceSummaryStats
    mdnrnn_mse_loss: SequenceSummaryStats


class EnvModel:
    def __init__(
        self,
        vae_trainer: nn.Module,
        mdnrnn_trainer: nn.Module,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.vae_trainer = vae_trainer.to(device)
        self.mdnrnn_trainer = mdnrnn_trainer.to(device)
        self.device = device

    def learn(
        self, data: GoalBatchProtocol | GoalReplayBufferProtocol, **kwargs: Any
    ) -> TrainingStats:
        vae_losses_summary = self.vae_trainer.train(data)
        losses_summary, gmm_losses_summary, bce_losses_summary, mse_losses_summary = (
            self.mdnrnn_trainer.train(data)
        )

        return EnvModelStats(
            vae_loss=vae_losses_summary,
            mdnrnn_loss=losses_summary,
            mdnrnn_gmm_loss=gmm_losses_summary,
            mdnrnn_bce_loss=bce_losses_summary,
            mdnrnn_mse_loss=mse_losses_summary,
        )
