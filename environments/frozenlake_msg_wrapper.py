import gymnasium as gym
from gymnasium.spaces import Dict as DictSpace, Text as TextSpace
from typing import Any, Tuple, Dict, SupportsFloat


class AddMessageFrozenLake(gym.Wrapper):
    """Adds a simple NetHack-inspired message to FrozenLake environments to allow fast prototyping with ALAN's language component."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # the env must be wrapped in DictObservation!
        assert isinstance(env.observation_space, DictSpace)
        self.observation_space = DictSpace(
            {
                **env.observation_space.spaces,
                "message": TextSpace(max_length=128, charset="ascii"),
            }
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs["message"] = ""
        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[Dict[str, Any], SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        message = ""
        if terminated:
            if reward > 0:
                message = "Congratulations, you've reached the goal!"
            else:
                message = "Wrong step, you fell in a hole."
        obs["message"] = message

        return obs, reward, terminated, truncated, info
