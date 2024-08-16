from torch import nn

from .observation_net import NetHackObsNet


class SimpleNetHackActor(NetHackObsNet):
    def __init__(
        self,
        observation_space,
        action_space,
        embedding_dim=32,
        crop_dim=9,
        num_layers=5,
    ):
        super().__init__(observation_space, embedding_dim, crop_dim, num_layers)
        self.n_actions = action_space.n

        self.fc = nn.Sequential(self.fc, nn.Linear(self.h_dim // 2, self.n_actions))

    def forward(self, batch_obs, state=None, info={}):
        logits = super().forward(batch_obs)
        return logits, state


class SimpleNetHackCritic(NetHackObsNet):
    def __init__(
        self,
        observation_space,
        embedding_dim=32,
        crop_dim=9,
        num_layers=5,
    ):
        super().__init__(observation_space, embedding_dim, crop_dim, num_layers)

        self.fc = nn.Sequential(self.fc, nn.Linear(self.h_dim // 2, 1))

    def forward(self, batch_obs, state=None, info={}):
        return super().forward(batch_obs)
