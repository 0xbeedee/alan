import gymnasium as gym
from typing import Dict, Any, Tuple, SupportsFloat


class AutoReset(gym.Wrapper):
    """Wraps a Gymnasium environment to automatically reset it when an episode ends (terminated or truncated)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[Any, Dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        return observation, info

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            # auto-reset the environment
            observation, info = self.env.reset()

        return observation, reward, terminated, truncated, info
