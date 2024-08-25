from typing import Dict, Optional

from torch import nn
import torch

from gymnasium import Space

from .observation_net import NetHackObsNet


class SimpleNetHackActor(nn.Module):
    def __init__(self, obs_net: NetHackObsNet, action_space: Space):
        super().__init__()

        self.obs_net = obs_net
        self.n_actions = action_space.n
        self.final_layer = nn.Linear(self.obs_net.o_dim, self.n_actions)

    def forward(
        self,
        batch_obs: Dict[str, torch.Tensor],
        state: Optional[torch.Tensor] = None,
        info: Dict = {},
    ):
        obs_out = self.obs_net(batch_obs)
        logits = self.final_layer(obs_out)
        return logits, state


class SimpleNetHackCritic(nn.Module):
    def __init__(self, obs_net: NetHackObsNet):
        super().__init__()

        self.obs_net = obs_net
        self.final_layer = nn.Linear(self.obs_net.o_dim, 1)

    def forward(
        self,
        batch_obs: Dict[str, torch.Tensor],
        state: Optional[torch.Tensor] = None,
        info: Dict = {},
    ):
        obs_out = self.obs_net(batch_obs)
        v_s = self.final_layer(obs_out)
        return v_s
