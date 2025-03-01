from typing import Dict, Optional, Self
from core.types import GoalBatchProtocol

from tianshou.data import Batch
import gymnasium as gym
from torch import nn
import torch


class GoalActor(nn.Module):
    def __init__(
        self,
        obs_net: nn.Module,
        state_dim: int,
        action_space: gym.Space,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.obs_net = obs_net.to(device)
        self.state_dim = state_dim
        self.n_actions = action_space.n
        self.device = device

        hidden_dim = obs_net.o_dim // 3
        self.obs_munet = nn.Sequential(
            nn.Linear(obs_net.o_dim, hidden_dim), nn.ReLU()
        ).to(device)
        self.goal_munet = nn.Sequential(
            nn.Linear(obs_net.o_dim, hidden_dim), nn.ReLU()
        ).to(device)

        self.final_layer = nn.Linear(
            hidden_dim + hidden_dim + self.state_dim, self.n_actions
        ).to(device)

    def forward(
        self,
        batch_obs_goal: GoalBatchProtocol,
        state: Optional[torch.Tensor] = None,
        info: Dict = {},
    ):
        batch_obs = Batch(
            {k: v for k, v in batch_obs_goal.items() if k != "latent_goal"}
        )
        obs_out = self.obs_net(batch_obs)
        obss = self.obs_munet(obs_out)
        goals = self.goal_munet(
            torch.as_tensor(
                batch_obs_goal["latent_goal"], dtype=torch.float32, device=self.device
            )
        )
        if state is None:
            # the first policy.forward() call has a None state
            state = torch.zeros(obs_out.shape[0], self.state_dim, device=self.device)
        logits = self.final_layer(torch.cat((obss, goals, state), dim=1))
        return logits, state

    def to(self, device: torch.device) -> Self:
        self.device = device
        self.obs_net = self.obs_net.to(device)
        self.final_layer = self.final_layer.to(device)
        self.obs_munet = self.obs_munet.to(device)
        self.goal_munet = self.goal_munet.to(device)
        return super().to(device)


class GoalRainbowActor(GoalActor):
    def __init__(
        self, obs_net, state_dim, action_space, num_atoms=51, device=torch.device("cpu")
    ):
        super().__init__(obs_net, state_dim, action_space, device)
        self.num_atoms = num_atoms

    def forward(
        self,
        batch_obs_goal: GoalBatchProtocol,
        state: Optional[torch.Tensor] = None,
        info: Dict = {},
    ):
        logits, state = super().forward(batch_obs_goal, state, info)
        # reshape the logits to match the shape expected by C51
        # TODO this does not work!!! I think I might need to simply add a further layer for this expansion (or ovewrite the final layer)
        return logits.view(-1, self.n_actions, self.num_atoms)


class GoalCritic(nn.Module):
    def __init__(
        self,
        obs_net: nn.Module,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.obs_net = obs_net.to(device)
        self.device = device

        hidden_dim = obs_net.o_dim // 3
        self.obs_munet = nn.Sequential(
            nn.Linear(obs_net.o_dim, hidden_dim), nn.ReLU()
        ).to(device)
        self.goal_munet = nn.Sequential(
            nn.Linear(obs_net.o_dim, hidden_dim), nn.ReLU()
        ).to(device)
        self.final_layer = nn.Linear(hidden_dim + hidden_dim, 1).to(device)

    def forward(
        self,
        batch_obs_goal: GoalBatchProtocol,
        info: Dict = {},
    ):
        batch_obs = {k: v for k, v in batch_obs_goal.items() if k != "latent_goal"}
        obs_out = self.obs_net(batch_obs)
        obss = self.obs_munet(obs_out)
        goals = self.goal_munet(
            torch.as_tensor(
                batch_obs_goal["latent_goal"], dtype=torch.float32, device=self.device
            )
        )
        v_s = self.final_layer(torch.cat((obss, goals), dim=1))
        return v_s

    def to(self, device: torch.device) -> Self:
        self.device = device
        self.obs_net = self.obs_net.to(device)
        self.obs_munet = self.obs_munet.to(device)
        self.goal_munet = self.goal_munet.to(device)
        self.final_layer = self.final_layer.to(device)
        return super().to(device)
