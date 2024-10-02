import gymnasium as gym


class DictObservation(gym.Wrapper):
    """Turns simple observations (gym.spaces.Discrete) into dictionaries so that I can use the environment with my goal-aware pipeline."""

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.observation_space = gym.spaces.Dict({"obs": env.observation_space})

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return {"obs": observation}, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return {"obs": observation}, reward, terminated, truncated, info
