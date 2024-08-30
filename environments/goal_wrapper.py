import gymnasium as gym


class GoalBased(gym.Wrapper):
    # we do not need an actual GoalEnv (https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/a35b1c1fa669428bf640a2c7101e66eb1627ac3a/gym_robotics/core.py), it is sufficient to adapt the observation space and pass the compute_reward function directly to the HERReplayBuffer
    def __init__(self, gym_env):
        super().__init__(gym_env)

        goal_keys = ("observation", "achieved_goal", "desired_goal")
        # the goal space is a subset (possibly improper) of the state space
        self.observation_space = gym.spaces.Dict(
            {key: self.env.observation_space for key in goal_keys}
        )

    def reset(self, seed=None, options=None):
        nle_obs, info = self.env.reset(seed=seed, options=options)
        goal_based_obs = _wrap_nle_observation(nle_obs)
        return goal_based_obs, info

    def step(self, action):
        nle_obs, info = self.env.step(action)
        goal_based_obs = _wrap_nle_observation(nle_obs)
        return goal_based_obs, info


def _wrap_nle_observation(nle_obs):
    goal_based_obs = {}
    goal_based_obs["observation"] = nle_obs
    # the goal space is a subset of the state space
    goal_based_obs["achieved_goal"] = 1337
    goal_based_obs["desired_goal"] = 1337
    return goal_based_obs
