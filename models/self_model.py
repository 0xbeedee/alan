from gymnasium import Space
from networks import NetHackObsNet

from torch import nn

from tianshou.data import Batch


class SelfModel:
    def __init__(
        self, obs_net: NetHackObsNet, action_space: Space, intrinsic_module: nn.Module
    ) -> None:
        self.intrinsic_module = intrinsic_module(obs_net, action_space)

    def __call__(self, batch: Batch, sleep: bool = False):
        # TODO do something with sleep
        return self.intrinsic_module.forward(batch)
