from typing import Dict, Optional
from tianshou.data.batch import BatchProtocol, TArr
from .observation_net import NetHackObsNet
from gymnasium import Space

# TODO typing imports should be first in ALL files

from torch import nn
import torch


class SimpleNetHackActor(nn.Module):
    def __init__(self, obs_net: NetHackObsNet, action_space: Space):
        super().__init__()

        self.obs_net = obs_net
        self.n_actions = action_space.n
        self.final_layer = nn.Linear(self.obs_net.o_dim, self.n_actions)

    def forward(
        self,
        # TODO some other types need changing too...
        batch_obs: TArr | BatchProtocol,
        state: Optional[torch.Tensor] = None,
        info: Dict = {},
    ):
        obs_out = self.obs_net(batch_obs)
        logits = self.final_layer(obs_out)
        return logits, state


class GoalBasedNetHackActor(nn.Module):
    def __init__(self, obs_goal_dim: int, action_space: Space):
        super().__init__()

        self.n_actions = action_space.n

        # we use a two-stream architecture (https://proceedings.mlr.press/v37/schaul15.html)
        hidden_dim = obs_goal_dim // 3  # obs dim == goal dim
        # mu == micro
        self.obs_munet = nn.Sequential(nn.Linear(obs_goal_dim, hidden_dim), nn.ReLU())
        self.goal_munet = nn.Sequential(nn.Linear(obs_goal_dim, hidden_dim), nn.ReLU())

        self.final_layer = nn.Linear(hidden_dim + hidden_dim, self.n_actions)

    def forward(
        self,
        batch_latent_obs: torch.Tensor,
        batch_latent_goal: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        info: Dict = {},
    ):
        obss = self.obs_munet(batch_latent_obs)
        goals = self.goal_munet(batch_latent_goal)
        logits = self.final_layer(torch.cat((obss, goals), dim=1))
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


class GoalBasedNetHackCritic(nn.Module):
    def __init__(self, obs_goal_dim: int):
        super().__init__()

        hidden_dim = obs_goal_dim // 3
        self.obs_munet = nn.Sequential(nn.Linear(obs_goal_dim, hidden_dim), nn.ReLU())
        self.goal_munet = nn.Sequential(nn.Linear(obs_goal_dim, hidden_dim), nn.ReLU())

        self.final_layer = nn.Linear(hidden_dim + hidden_dim, 1)

    def forward(
        self,
        batch_latent_obs: torch.Tensor,
        batch_latent_goal: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        info: Dict = {},
    ):
        obss = self.obs_munet(batch_latent_obs)
        goals = self.goal_munet(batch_latent_goal)
        v_s = self.final_layer(torch.cat((obss, goals), dim=1))
        return v_s
