import random

import gymnasium as gym
from megagrid.utils import get_possible_minigrids


def random_gen(seed=None):
    """The simplest possible procedural generation: return environments at random."""
    grid_options = get_possible_minigrids()
    return gym.make(random.choice(grid_options))
