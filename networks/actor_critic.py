from typing import Dict, Optional
from tianshou.data.types import ObsBatchProtocol
from core.types import GoalBatchProtocol, ObservationNet

import gymnasium as gym

from torch import nn
import torch


class SimpleNetHackActor(nn.Module):
    def __init__(self, obs_net: ObservationNet, action_space: gym.Space):
        super().__init__()

        self.obs_net = obs_net
        self.n_actions = action_space.n
        self.final_layer = nn.Linear(self.obs_net.o_dim, self.n_actions)

    def forward(
        self,
        batch_obs: ObsBatchProtocol,
        state: Optional[torch.Tensor] = None,
        info: Dict = {},
    ):
        obs_out = self.obs_net(batch_obs)
        logits = self.final_layer(obs_out)
        return logits, state


class GoalNetHackActor(SimpleNetHackActor):
    def __init__(self, obs_net: ObservationNet, action_space: gym.Space):
        super().__init__(obs_net, action_space)

        # we use a two-stream architecture (https://proceedings.mlr.press/v37/schaul15.html)
        hidden_dim = obs_net.o_dim // 3  # obs dim == goal dim
        # mu == micro
        self.obs_munet = nn.Sequential(nn.Linear(obs_net.o_dim, hidden_dim), nn.ReLU())
        self.goal_munet = nn.Sequential(nn.Linear(obs_net.o_dim, hidden_dim), nn.ReLU())

        self.final_layer = nn.Linear(hidden_dim + hidden_dim, self.n_actions)

    def forward(
        self,
        batch_obs_goal: GoalBatchProtocol,
        state: Optional[torch.Tensor] = None,
        info: Dict = {},
    ):
        batch_obs = {k: v for k, v in batch_obs_goal.items() if k != "latent_goal"}

        obs_out = self.obs_net(batch_obs)
        obss = self.obs_munet(obs_out)
        goals = self.goal_munet(batch_obs_goal.latent_goal)
        logits = self.final_layer(torch.cat((obss, goals), dim=1))
        return logits, state


class SimpleNetHackCritic(nn.Module):
    def __init__(self, obs_net: ObservationNet):
        super().__init__()

        self.obs_net = obs_net
        self.final_layer = nn.Linear(self.obs_net.o_dim, 1)

    def forward(
        self,
        batch_obs: ObsBatchProtocol,
        state: Optional[torch.Tensor] = None,
        info: Dict = {},
    ):
        obs_out = self.obs_net(batch_obs)
        v_s = self.final_layer(obs_out)
        return v_s


class GoalNetHackCritic(SimpleNetHackCritic):
    def __init__(self, obs_net: ObservationNet):
        super().__init__(obs_net)

        hidden_dim = obs_net.o_dim // 3
        self.obs_munet = nn.Sequential(nn.Linear(obs_net.o_dim, hidden_dim), nn.ReLU())
        self.goal_munet = nn.Sequential(nn.Linear(obs_net.o_dim, hidden_dim), nn.ReLU())

        self.final_layer = nn.Linear(hidden_dim + hidden_dim, 1)

    def forward(
        self,
        batch_obs_goal: GoalBatchProtocol,
        state: Optional[torch.Tensor] = None,
        info: Dict = {},
    ):
        batch_obs = {k: v for k, v in batch_obs_goal.items() if k != "latent_goal"}
        obs_out = self.obs_net(batch_obs)
        obss = self.obs_munet(obs_out)
        goals = self.goal_munet(batch_obs_goal.latent_goal)
        v_s = self.final_layer(torch.cat((obss, goals), dim=1))
        return v_s
