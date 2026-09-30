
# Isis Martinez, Ramya Nayak
# CS 272 - Reinforcement Learning
# Assignment 2
# September 29, 2026

# ----------------------------------------------------

"""Task 1: Custom Gymnasium environment.

Complete a small stochastic maze environment for CS 272 PA2
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register


class MyEnv(gym.Env):
    """Navigate a 10x10 maze from a random entrance to a random exit"""    # Fixed maze is terrain: 0 = open cell (a walkable floor tile), 1 = wall
    GRID = (
        "0100010001",
        "0101010101",
        "0101010101",
        "0101010101",
        "0101010101",
        "0101010101",
        "0101010101",
        "0101010101",
        "0101010101",
        "0001000101"
    )
    HEIGHT = len(GRID)
    WIDTH = len(GRID[0])

    # Action numbers are stable for the agent
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    # Map each action to its change in (row, column) position:
    ACTION_CHANGE = {
        UP: (-1, 0),                        # row −1, column unchanged
        RIGHT: (0, 1),                      # row unchanged, column +1
        DOWN: (1, 0),                       # row +1, column unchanged
        LEFT: (0, -1),                      # row unchanged, column -1
    }

    SLIP_PROBABILITY = 0.20                 # make environment stochastic
    STEP_REWARD = -0.10                     # disincentivize extra steps
    WALL_REWARD = -0.35                     # disincentivize hitting wall
    GOAL_REWARD = 10.0                      # big reward for reaching goal

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None):
        # Precompute all open floor cells in the maze for random selection
        self.open_cells = [
            (r, c) for r in range(self.HEIGHT) for c in range(self.WIDTH) if self.GRID[r][c] == '0'
        ]

        # There are 100 possible cell observations, including cells behind walls
        self.observation_space = spaces.Discrete(self.HEIGHT * self.WIDTH)  # size of maze
        self.action_space = spaces.Discrete(4)                  # possible actions: U, R, D, L

        # validate render mode
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        
        self.render_mode = render_mode
        self._agent_pos = None                      # Agent position initialized in reset
        self._goal_pos = None                       # Goal position initialized in reset
        self._steps = 0                             # count of steps taken
        self._last_action: int | None = None        # action agent requested: either int or None
        self._last_true_action: int | None = None   # direction used after stochastic noise

    def reset(self, seed: int | None = None, options: dict | None = None):
        # This line seeds self.np_random. Without it, seeding does not work and
        # the reproducibility test fails.
        super().reset(seed=seed)

        # Randomly choose two UNIQUE positions from available open cells
        start_idx, goal_idx = self.np_random.choice(len(self.open_cells), size=2, replace=False)

        # put the world back to its starting state.
        self._agent_pos = self.open_cells[start_idx]
        self._goal_pos = self.open_cells[goal_idx]
        self._steps = 0
        self._last_action = None
        self._last_true_action = None

        return self._get_obs(), self._get_info()
    
    def step(self, action: int):
        """Run one timestep of the environment's dynamics using the agent's action.

        Args:
            action (int): The chosen action by the agent. Must be in [0, 3].

        Returns:
            tuple:
                - observation (int): The agent's current position encoded as an int.
                - reward (float): The reward earned from the action.
                - terminated (bool): True if the agent reaches the goal, False otherwise.
                - truncated (bool): Always False for this environment (handled by TimeLimit).
                - info (dict): Additional debugging info including slip tracking.
        """

        # validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Action is {action}: needs to be integer in [0,3], ")

        action = int(action)
        self._last_action = action

        # apply stochastic portion to flip direction perpendicularly
        true_action, slip = self._get_actual_action(action)
        self._last_true_action = true_action

        old_position = self._agent_pos
        self._agent_pos = self._move(true_action)
        self._steps += 1

        moved = self._agent_pos != old_position
        goal_reached = self._agent_pos == self._goal_pos

        reward = self._get_reward(moved, goal_reached)
        terminated, truncated = self._get_end_status(goal_reached)

        info = self._get_info()
        info["slipped"] = slip
        info["moved"] = moved

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        """Return a readable picture of the current state, as a string."""
        if self.render_mode != "ansi":
            return None

        symbols = {"0": ".", "1": "#"}                          # visual display for maze: open space is ".", wall is "#"
        rows = []

        for r in range(self.HEIGHT):
            row_chars = []
            for c in range(self.WIDTH):
                pos = (r, c)
                if pos == self._agent_pos:
                    row_chars.append("A")                       # agent position shows up as "A"
                elif pos == self._goal_pos:
                    row_chars.append("G")                       # goal position shows up as "G"
                else:
                    row_chars.append(symbols[self.GRID[r][c]])
            rows.append("".join(row_chars))                     # string for this row

        maze_str = "\n".join(rows)                              # stack rows into a maze block
        legend = f"\n[ Legend: Agent (A) at {self._agent_pos} | Goal (G) at {self._goal_pos} ]"

        return maze_str + legend

    def close(self):
        pass                                                    # this env has no external resources to release


    # helper functions
    def _position_encoded(self, position: tuple[int, int]) -> int:
        """Encode (row, column) as one integer in [0, 99]."""
        row, column = position
        return row * self.WIDTH + column

    def _get_obs(self) -> int:
        """Return encoded int for curr position"""
        return self._position_encoded(self._agent_pos)

    def _get_info(self) -> dict[str, object]:
        """Return debugging information about the current state."""
        return {
            "position": self._agent_pos,
            "steps": self._steps,
            "last_action": self._last_action,
            "last_actual_action": self._last_true_action,
        }

    def _is_open(self, position: tuple[int, int]) -> bool:
        """Return whether a position is inside the maze and open."""
        row, col = position
        return (
            0 <= row and row < self.HEIGHT and 0 <= col and col < self.WIDTH and self.GRID[row][col] != '1'
        )

    def _move(self, action: int) -> tuple[int, int]:
        row, col = self._agent_pos
        change_row, change_col = self.ACTION_CHANGE[action]
        candidate = (row + change_row, col + change_col)
        if self._is_open(candidate):
            return candidate
        else:
            return self._agent_pos

    def _get_actual_action(self, action: int) -> tuple[int, bool]:
        """Stochastic portion: Apply transition noise and return actual action
        and slip status. If slip true, push agent perpendicular direction
        from requested direction. Each direction are 10% likely to occur"""
        slip = bool(self.np_random.random() < self.SLIP_PROBABILITY)

        if not slip:
            return action, False
        
        flip_perpendicular = (
            (self.UP, self.DOWN)
            if action in (self.RIGHT, self.LEFT)
            else (self.RIGHT, self.LEFT)
        )

        true_action = int(self.np_random.choice(flip_perpendicular))
        return true_action, True

    def _get_reward(self, moved: bool, reached_goal: bool) -> float:
        """Calculate the reward for the transition."""
        if reached_goal:
            return self.GOAL_REWARD
        if moved:
            return self.STEP_REWARD
        return self.WALL_REWARD

    def _get_end_status(self, reached_goal: bool) -> tuple[bool, bool]:
        """Return terminated and truncated flags."""
        return reached_goal, False


# Named environment. The id must start with "cs272/" and end with a version, and
# max_episode_steps must be large enough that a competent agent can finish but
# small enough that a lost one gives up.
ENV_ID = "cs272/Maze-v0"
register(
    id=ENV_ID,
    entry_point="myenv:MyEnv",
    max_episode_steps=300,
)
