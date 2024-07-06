import pygame

from megagrid.procedural_gen import *


class MegagridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
            self,
            grid=None,
            selection_strategy="random",
            seed=None,
            render_mode=None,
    ):
        self.selection_strategy = selection_strategy
        self.seed = seed
        self.render_mode = render_mode

        if grid is None:
            self.grid = self._generate_grid()
        else:
            self.grid = grid

        self.observation_space = self.grid.observation_space
        self.action_space = self.grid.action_space

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed

        self.grid = self._generate_grid()
        observation, info = self.grid.reset()

        if self.render_mode == "human":
            self.grid.render()

        return observation, info

    def step(self, action):
        observation, reward, done, truncated, info = self.grid.step(action)

        if done or truncated:
            # if we're done with the current environment, immediately start the next
            self.grid = self._generate_grid()
            self.observation_space = self.grid.observation_space
            self.action_space = self.grid.action_space

        if self.render_mode == "human":
            self.grid.render()

        # TODO might want to add a terminating condition for megagrid (else we run forever!)
        # TODO an initial idea could be max_steps number
        return observation, reward, done, truncated, info

    def render(self):
        return self.grid.render()

    def close(self):
        self.grid.close()
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    def _generate_grid(self):
        gen_function_name = f"{self.selection_strategy}_gen"
        gen_function = globals().get(gen_function_name)
        if gen_function is None:
            raise ValueError(
                f"Grid generation function '{gen_function_name}' not found!"
            )
        return gen_function(seed=self.seed)
