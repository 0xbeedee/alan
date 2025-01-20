from dataclasses import dataclass
from typing import Sequence, Any, Tuple
from core.types import (
    LatentObsActNextBatchProtocol,
    GoalBatchProtocol,
)

import torch
import numpy as np
from collections import defaultdict

from tianshou.utils.net.common import MLP
from tianshou.policy.base import TrainingStats
from tianshou.data import SequenceSummaryStats


@dataclass(kw_only=True)
class BBoldTrainingStats(TrainingStats):
    bbold_loss: SequenceSummaryStats


class BBold:
    """An implementation of BeBold as introduced by Zhang et al. (https://arxiv.org/abs/2012.08621).

    We used the code at https://github.com/tianjunz/NovelD/blob/master/src/algos/bebold.py as reference.
    We keep the API exactly the same as the one for ICM for compatibility purposes.
    """

    def __init__(
        self,
        feature_dim: int,
        # keep n_actions to avoid API changes (even though we don't use it)
        n_actions: int,
        batch_size: int,
        learning_rate: float = 1e-3,
        hidden_sizes: Sequence[int] = [[256, 128, 128, 64], [256, 128, 64]],
        beta: float = 0.2,
        eta: float = 0.07,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        if len(hidden_sizes) == 2:
            # different hidden sizes for random and predictor nets
            # (in general, the predictor net should be more powerful than the random net, as in https://github.com/tianjunz/NovelD/blob/master/src/models.py)
            hidden_sizes_predictor, hidden_sizes_rand = hidden_sizes[0], hidden_sizes[1]
        else:
            hidden_sizes_predictor = hidden_sizes_rand = hidden_sizes

        self.predictor_net = MLP(
            feature_dim,
            output_dim=feature_dim // 2,
            hidden_sizes=hidden_sizes_predictor,
            device=device,
        )
        self.random_target_net = MLP(
            feature_dim,
            output_dim=feature_dim // 2,
            hidden_sizes=hidden_sizes_rand,
            device=device,
        )

        params = set(
            list(self.random_target_net.parameters())
            + list(self.predictor_net.parameters())
        )
        self.optim = torch.optim.Adam(params, lr=learning_rate)

        # a list of dicts to maintain episodic state counts across all the environments
        self.ep_state_count = None
        self.batch_size = batch_size
        self.beta = beta
        self.eta = eta
        self.device = device

    def get_reward(self, batch: LatentObsActNextBatchProtocol) -> np.ndarray:
        if self.ep_state_count is None:
            # need to initialise the episodic state counts
            # (do it here: batch is the only object with correct dims)
            self.ep_state_count = [defaultdict(int) for _ in range(len(batch))]

        # no need torch.no_grad() as SelfModel takes care of it
        random_emb, predicted_emb, random_emb_next, predicted_emb_next = self._forward(
            batch
        )

        count_indicator = np.zeros_like(batch.obs_next.obs, dtype=np.float32)
        for i, obs_next in enumerate(batch.obs_next.obs):
            if obs_next in self.ep_state_count[i]:
                # old observation, increment the counter
                self.ep_state_count[i][obs_next] += 1
            else:
                # new observation, initialise counter to 1...
                self.ep_state_count[i][obs_next] = 1
                # ...and do not zero out the intrinsic reward
                count_indicator[i] = 1.0

        # these norms approximate inverse state counts
        int_rew_next = torch.norm(predicted_emb_next - random_emb_next, p=2, dim=1)
        int_rew = torch.norm(predicted_emb - random_emb, p=2, dim=1)

        # clip the reward so that it remain in the [0, 1] range
        int_rew = torch.clamp(int_rew_next - self.beta * int_rew, min=0.0, max=1.0)

        # view it as a numpy float32 for consistency with Tianshou and MPS
        intrinsic_reward = int_rew.cpu().numpy().astype(np.float32)
        return intrinsic_reward * count_indicator

    def learn(self, data: GoalBatchProtocol, **kwargs: Any) -> BBoldTrainingStats:
        """Trains the predictor and random target networks."""
        losses = []
        for batch in data.split(self.batch_size, merge_last=True):
            _, _, random_emb_next, predicted_emb_next = self._forward(batch)
            loss = torch.norm(predicted_emb_next - random_emb_next, p=2, dim=1).mean()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            losses.append(loss.item())

        losses_summary = SequenceSummaryStats.from_sequence(losses)
        return BBoldTrainingStats(
            bbold_loss=losses_summary,
        )

    def _forward(
        self, batch: LatentObsActNextBatchProtocol
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        phi1 = torch.as_tensor(batch.latent_obs, device=self.device)
        phi2 = torch.as_tensor(batch.latent_obs_next, device=self.device)

        random_emb = self.random_target_net(phi1)
        predicted_emb = self.predictor_net(phi1)
        random_emb_next = self.random_target_net(phi2)
        predicted_emb_next = self.predictor_net(phi2)

        return random_emb, predicted_emb, random_emb_next, predicted_emb_next
