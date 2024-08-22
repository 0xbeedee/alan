from torch import nn


class SimpleNetHackActor(nn.Module):
    def __init__(
        self,
        obs_net,
        observation_space,
        action_space,
        embedding_dim=32,
        crop_dim=9,
        num_layers=5,
    ):
        super().__init__()

        self.obs_net = obs_net
        self.obs_net.__init__(observation_space, embedding_dim, crop_dim, num_layers)

        self.n_actions = action_space.n
        self.final_layer = nn.Linear(self.obs_net.o_dim, self.n_actions)

    def forward(self, batch_obs, state=None, info={}):
        obs_out = self.obs_net.forward(batch_obs)
        logits = self.final_layer(obs_out)
        return logits, state


class SimpleNetHackCritic(nn.Module):
    def __init__(
        self,
        obs_net,
        observation_space,
        embedding_dim=32,
        crop_dim=9,
        num_layers=5,
    ):
        super().__init__()

        self.obs_net = obs_net
        self.obs_net.__init__(observation_space, embedding_dim, crop_dim, num_layers)

        self.final_layer = nn.Linear(self.obs_net.o_dim, 1)

    def forward(self, batch_obs, state=None, info={}):
        obs_out = self.obs_net.forward(batch_obs)
        v_s = self.final_layer(obs_out)
        return v_s
