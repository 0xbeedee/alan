from typing import Dict, Optional

import torch
from torch import nn

from .observation_net import NetHackObsNet


class NetHackObsGoalNet(nn.Module):
    """A network combining observations and goals.

    We use the two-stream architecture from the UVFA paper (https://proceedings.mlr.press/v37/schaul15.html).
    """

    def __init__(self, obs_net: NetHackObsNet, goal_net: NetHackObsNet) -> None:
        super().__init__()

        self.obs_net = obs_net
        self.goal_net = goal_net

    def forward(
        self,
        batch_obs: Dict[str, torch.Tensor],
        batch_goal: Dict[str, torch.Tensor],
        state: Optional[torch.Tensor] = None,
        info: Dict = {},
    ):
        obs_out = self.obs_net(batch_obs)
        goal_out = self.goal_net(batch_goal)
        logits = torch.cat((obs_out, goal_out), dim=1)
        return logits, state
