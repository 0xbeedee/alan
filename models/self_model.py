class SelfModel:
    def __init__(self, obs_net, action_space, intrinsic_module) -> None:
        self.intrinsic_module = intrinsic_module(obs_net, action_space.n)

    def __call__(self, batch, sleep=False):
        i_reward, _ = self.intrinsic_module.forward(batch)

        return i_reward
