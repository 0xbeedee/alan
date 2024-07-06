import gymnasium as gym


def get_possible_minigrids():
    env_specs = list(gym.envs.registry.values())
    minigrid_envs = [env_spec.id for env_spec in env_specs if "MiniGrid" in env_spec.id]
    return minigrid_envs
