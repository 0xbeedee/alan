import torch


class ResettingEnvironment:
    """Turns a Gym environment into something that can be step()ed indefinitely."""

    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None

        for attr, value in vars(gym_env).items():
            if not attr.startswith("__"):  # avoid copying special attributes
                setattr(self, attr, value)

    def reset(self, seed=None, options=None):
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        obs, reset_info = self.gym_env.reset(seed=seed, options=options)
        return obs, reset_info

    def step(self, action):
        observation, reward, done, truncated, info = self.gym_env.step(action.item())
        self.episode_step += 1
        self.episode_return += reward
        if done:
            observation, _ = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        return observation, reward, done, truncated, info

    def render(self):
        self.gym_env.render()

    def close(self):
        self.gym_env.close()
