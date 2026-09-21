"""Task 1: Custom Gymnasium environment.

Complete a small stochastic maze environment for CS 272 PA2
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register


class MyEnv(gym.Env):
    """Navigate a 10x10 maze from the top entrance to the bottom exit"""
    # Fixed maze is terrain: 0 = open cell (a walkable floor tile), 1 = wall
    GRID = (
        "0011111000",
        "1101111101",
        "1100011110",
        "1111011110",
        "1111000110",
        "1111110110",
        "1111000100",
        "1111011100",
        "1111000001",
        "1111111100"
    )
    HEIGHT = len(GRID)
    WIDTH = len(GRID[0])
    START = (0, 0) # Maze entrance
    GOAL = (9, 8) # Maze exit

    # Action numbers are stable for the agent
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    # Map each action to its change in (row, column) position:
    ACTION_CHANGE = {
        UP: (-1, 0),        # row −1, column unchanged
        RIGHT: (0, 1),      # row unchanged, column +1
        DOWN: (1, 0),       # row +1, column unchanged
        LEFT: (0, -1),      # row unchanged, column -1
    }

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None):
        # TODO: describe your world here -- the map, the pieces, the constants.

        # There are 100 possible cell observations, including cells behind walls
        self.observation_space = spaces.Discrete(self.HEIGHT * self.HEIGHT)
        self.action_space = spaces.Discrete(4) # possible actions: U, R, D, L

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

    def reset(self, seed: int | None = None, options: dict | None = None):
        # This line seeds self.np_random. Without it, seeding does not work and
        # the reproducibility test fails.
        super().reset(seed=seed)

        # TODO: put the world back to its starting state.

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        # TODO: apply the action, with noise drawn from self.np_random.
        #
        # Return terminated=True when the episode genuinely ends -- goal reached,
        # agent died, game over. Leave truncated as False and let the TimeLimit
        # wrapper from register() handle running out of time. The agent treats
        # the two differently, and so should you.

        raise NotImplementedError

    def render(self):
        """Return a readable picture of the current state, as a string."""
        if self.render_mode != "ansi":
            return None
        # TODO: draw it. You need this for the sample episode in your report.
        raise NotImplementedError

    def close(self):
        pass


# TODO: name your environment. The id must start with "cs272/" and end with a
# version, and max_episode_steps must be large enough that a competent agent can
# finish but small enough that a lost one gives up.
register(
    id="cs272/MyEnv-v0",
    entry_point="myenv:MyEnv",
    max_episode_steps=300,
)
