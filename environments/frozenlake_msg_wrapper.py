import gymnasium as gym


class AddMessageFrozenLake(gym.Wrapper):
    """Adds a simple NetHack-inspired message to FrozenLake environments to allow fast prototyping with ALAN's language component."""

    def __init__(self, env):
        super().__init__(env)
        # the env must be wrapped in DictObservation!
        assert isinstance(env.observation_space, gym.spaces.Dict)
        self.observation_space = gym.spaces.Dict(
            {
                **env.observation_space.spaces,
                "message": gym.spaces.Text(max_length=128, charset="ascii"),
            }
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs["message"] = ""
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        message = ""
        if terminated:
            if reward > 0:
                message = "Congratulations, you've reached the goal!"
            else:
                message = "Wrong step, you fell in a hole."
        obs["message"] = message

        return obs, reward, terminated, truncated, info
