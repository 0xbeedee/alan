import gymnasium as gym
from megagrid_env import MegaGridEnv

import unittest


class TestMegaGridEnv(unittest.TestCase):

    def setUp(self):
        # Create an instance of the environment
        self.env = MegaGridEnv(selection_strategy="random", render_mode="human")

    def test_initialization(self):
        # Test that the environment initializes correctly
        self.assertIsInstance(self.env, MegaGridEnv)
        self.assertIn(self.env.render_mode, self.env.metadata["render_modes"])
        self.assertIsNotNone(self.env.grid)

    def test_reset(self):
        # Test that the environment resets correctly
        observation, info = self.env.reset()
        self.assertIsNotNone(observation)
        self.assertIsInstance(info, dict)
        self.assertEqual(self.env.num_envs_seen, 0)

    def test_step(self):
        # Test that the environment steps correctly
        self.env.reset()
        observation, reward, done, truncated, info = self.env.step(
            self.env.action_space.sample()
        )
        self.assertIsNotNone(observation)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_close(self):
        # Test that the environment closes correctly
        self.env.close()
        self.assertIsNone(self.env.window)
        self.assertIsNone(self.env.clock)

    def test_generate_grid(self):
        # Test that the grid generation function works
        grid = self.env._generate_grid()
        self.assertIsNotNone(grid)
        self.assertIsInstance(grid, gym.Env)


if __name__ == "__main__":
    unittest.main()
