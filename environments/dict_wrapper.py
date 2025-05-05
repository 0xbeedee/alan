import gymnasium as gym
from gymnasium.spaces import Discrete, Dict as DictSpace
from typing import Any, Tuple, Dict, SupportsFloat


class DictObservation(gym.Wrapper):
    """Turns simple observations (gym.spaces.Discrete) into dictionaries so that I can use the environment with my goal-aware pipeline."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, Discrete)
        self.observation_space = DictSpace({"obs": env.observation_space})

    def reset(self, **kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        return {"obs": observation}, info

    def step(
        self, action: Any
    ) -> Tuple[Dict[str, Any], SupportsFloat, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return {"obs": observation}, reward, terminated, truncated, info
