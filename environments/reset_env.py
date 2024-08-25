import gymnasium as gym
import torch


class ResettingEnvironment(gym.Wrapper):
    """Turns a Gymnasium environment into something that can be step()ed indefinitely."""

    def __init__(self, gym_env):
        super().__init__(gym_env)
        self.episode_return = None
        self.episode_step = None

    def reset(self, seed=None, options=None):
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        observation, info = self.env.reset(seed=seed, options=options)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action.item())
        self.episode_step += 1
        self.episode_return += reward

        if terminated or truncated:
            observation, info = self.env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        return observation, reward, terminated, truncated, info
