import random

import gymnasium as gym


def generate_grid():
    return 0


def random_gen(grid_options):
    """The simplest possible procedural generation: return environments at random."""
    return gym.make(random.choice(grid_options))
