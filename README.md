
# CS 272 PA2: Maze Environment

This project implements a custom stochastic Gymnasium environment for Task 1 of
CS 272 PA2. Task 2 will add a model-free SARSA(λ) agent with eligibility traces.

## Project Status

### Task 1: In progress

Implemented or planned:

- 10×10 maze layout
- Tabular observations and actions
- Stochastic perpendicular movement
- Reward structure
- Episode termination at the goal
- Gymnasium environment registration
- Seeded resets
- ANSI rendering

Still to complete:

- Finish and test `render()`
- Pass Gymnasium's `check_env`
- Add complete environment tests
- Compare random play with a trained tabular agent
- Finish documentation after testing

## Environment Story

The agent is a traveler navigating a 10×10 maze. It begins at the entrance
along the top edge and must reach the exit along the bottom edge.

The maze contains open floor cells and walls. The agent can move up, right,
down, or left. Movement is stochastic because occasional gusts push the agent
in a perpendicular direction.

## Maze Layout

The maze uses the following representation:

- `0` = open cell
- `1` = wall

```text
0011111000
1101111101
1100011110
1111011110
1111000110
1111110110
1111000100
1111011100
1111000001
1111111100
```

The agent starts at 
```text
(0,0)
```

Tuesday 5:41 PM
pa2-gym.pdf
PDF

I have this assignment. Task 1 is based on the custom environment creation. Task 2 will be based on the lecture on Wednesday and next Monday. I want to start implementing the environment now. Help me setup the repo first and then help me with the environment implementation. I included notes on the code here and I want your help thinking about what to consider:
"""Task 1: your own custom Gymnasium environment.

Design the world yourself. The requirements it has to meet are in the assignment
readme.

Delete this docstring and describe your own world instead.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

class MyEnv(gym.Env):
   """TODO: one line on what this world is and what the agent is trying to do."""

*   *metadata = {"render_modes": ["ansi"], "render_fps": 4}

   def __init__(self, render_mode: str | None = None):
       # TODO: describe your world here -- the map, the pieces, the constants.
Small maze to start → make it more complicated if time allows?
Shape - rectangular or square
Size - 10x10
Start - top
End - bottom
Pieces - 1 agent character
Constants - the maze setup

*       *# TODO: set the two spaces. Both must be Discrete.
Observation space →
***Action space → up, down, left, right ***

*       *self.observation_space = # TODO
*       *self.action_space = # TODO

*       *if render_mode is not None and render_mode not in self.metadata["render_modes"]:
           raise ValueError(f"unsupported render_mode: {render_mode}")
       self.render_mode = render_mode

   def reset(self, seed: int | None = None, options: dict | None = None):
       # This line seeds self.np_random. Without it, seeding does not work and
       # the reproducibility test fails.
       super().reset(seed=seed)

       # TODO: put the world back to its starting state.

*       *return self._get_obs(), self._get_info()

   def step(self, action: int):
       # TODO: apply the action, with noise drawn from self.np_random.
*       *#
       # Return terminated=True when the episode genuinely ends -- goal reached,
       # agent died, game over. Leave truncated as False and let the TimeLimit
       # wrapper from register() handle running out of time. The agent treats
       # the two differently, and so should you.

       raise NotImplementedError

   def render(self):
       """Return a readable picture of the current state, as a string."""
*       *if self.render_mode != "ansi":
           return None
       # TODO: draw it. You need this for the sample episode in your report.
*       *raise NotImplementedError

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

Your IDE sees observation_space as the general Space type, which does not guarantee an .n attribute.

Add a type check first:

from gymnasium import spaces

Then:

assert isinstance(env.observation_space, spaces.Discrete)
assert isinstance(env.action_space, spaces.Discrete)

assert env.observation_space.n == 100
assert env.action_space.n == 4

Discrete spaces have the .n attribute; general Gymnasium Space objects do not.

PS C:\Users\terra\PycharmProjects\cs272_ReinforcementLearning\pa2-gym> python -m pytest -q

======================================================= ERRORS ========================================================
___________________________________________ ERROR collecting test_myenv.py ____________________________________________
ImportError while importing test module 'C:\Users\terra\PycharmProjects\cs272_ReinforcementLearning\pa2-gym\test_myenv.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3824.0_x64__qbz5n2kfra8p0\Lib\importlib_init_.py:88: in import_module
return _bootstrap.gcd_import(name[level:], package, level)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
test_myenv.py:8: in <module>
from myenv import ENV_ID, MyEnv
myenv.py:14: in <module>
from polars.testing.parametric import column
......\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\polars\testing\parametric_init.py:8: in <module>
raise ModuleNotFoundError(msg)
E ModuleNotFoundError: polars.testing.parametric requires the 'hypothesis' module
E Please install it using the command: pip install hypothesis
=============================================== short test summary info ===============================================
ERROR test_myenv.py
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
1 error in 5.83s

"""Tests for the custom Gymnasium maze environment."""

import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env
from gymnasium import spaces

from myenv import ENV_ID, MyEnv


def test_env_checker():
    """The environment follows the Gymnasium API."""
    env = MyEnv()
    check_env(env)
    env.close()


def test_spaces_and_initial_state():
    """The spaces and reset state have the expected values."""
    env = MyEnv()

    observation, info = env.reset(seed=272)

    assert isinstance(env.observation_space, spaces.Discrete)
    assert isinstance(env.action_space, spaces.Discrete)

    assert env.observation_space.n == 100
    assert env.action_space.n == 4

    assert observation == 0
    assert info["position"] == MyEnv.START

    env.close()


def test_registration_and_rendering():
    """The registered environment can be created and rendered."""
    env = gym.make(ENV_ID, render_mode="ansi")

    observation, _ = env.reset(seed=272)
    rendered = env.render()

    assert observation == 0
    assert isinstance(rendered, str)
    assert "A" in rendered
    assert "G" in rendered
    assert "#" in rendered

    env.close()


def test_seeded_trajectories_match():
    """The same seed and actions produce the same trajectory."""
    actions = [1, 1, 2, 2, 1, 0, 3, 2, 1, 1]

    def collect_trajectory():
        env = MyEnv()
        observation, _ = env.reset(seed=12345)
        trajectory = []

        for action in actions:
            result = env.step(action)
            next_observation, reward, terminated, truncated, info = result

            trajectory.append(
                (
                    observation,
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info["position"],
                    info["slipped"],
                )
            )

            observation = next_observation

            if terminated or truncated:
                break

        env.close()
        return trajectory

    assert collect_trajectory() == collect_trajectory()


def test_environment_is_stochastic():
    """The same action can produce different outcomes."""
    env = MyEnv()
    outcomes = set()

    for seed in range(50):
        env.reset(seed=seed)
        observation, reward, _, _, _ = env.step(MyEnv.RIGHT)
        outcomes.add((observation, reward))

    env.close()

    assert len(outcomes) > 1


def test_invalid_action_raises_error():
    """Actions outside the action space are rejected."""
    env = MyEnv()
    env.reset(seed=272)

    with pytest.raises(ValueError):
        env.step(4)

    env.close()


def test_goal_terminates_episode():
    """Reaching the goal returns the goal reward and terminates."""
    env = MyEnv()
    env.SLIP_PROBABILITY = 0.0
    env.reset(seed=272)

    # Place the agent immediately left of the goal for a focused test.
    env._agent_pos = (9, 7)

    observation, reward, terminated, truncated, _ = env.step(MyEnv.RIGHT)

    assert observation == 98
    assert reward == MyEnv.GOAL_REWARD
    assert terminated is True
    assert truncated is False

    env.close()


def test_time_limit_truncates_episode():
    """The Gymnasium TimeLimit wrapper truncates long episodes."""
    env = gym.make(ENV_ID)
    env.unwrapped.SLIP_PROBABILITY = 0.0
    env.reset(seed=272)

    for _ in range(299):
        _, _, terminated, truncated, _ = env.step(MyEnv.LEFT)
        assert terminated is False
        assert truncated is False

    _, _, terminated, truncated, _ = env.step(MyEnv.LEFT)

    assert terminated is False
    assert truncated is True

    env.close()

The error is caused by an unnecessary import in myenv.py:

from polars.testing.parametric import column

You do not need Polars, Hypothesis, Tkinter, Matplotlib, or Requests for this environment. Replace all imports at the top of myenv.py with:

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

You can remove:

from idlelib.autocomplete import TRY_A
from os import truncate
from tkinter.constants import RIGHT
import numpy as np
from matplotlib.testing.widgets import click_and_drag
from polars.testing.parametric import column
from requests.utils import rewind_body

Do not install Hypothesis; the test file is not the problem. Then rerun:

python -m pytest -q

After the import issue is fixed, your next likely fixes are:

super().__init__()

inside __init__, and:

slip = bool(self.np_random.random() < self.SLIP_PROBABILITY)

instead of calling self.np_random() directly.

Help me update my readme for what I have done so far:
this is what I have:

CS272-26fa-pa2: Build a World, Then Learn It
Project Goals
Learn how to build a custom environment with Gymnasium
Learn how to code a model-free control agent with eligibility traces, SARSA(λ)

In PA1 the casino was a black box you could only poke at. This time you are on the other side of the wall. You will build the world yourself, write down its rules so that someone else can understand them, and then hand it to a learning agent that knows nothing about what you built.

Task 1: Your Own Gymnasium Environment

Design and implement your own environment in myenv.py. It must follow the Gymnasium API.

Requirements

Your environment must satisfy all of the following.

It is a real Gymnasium environment. It subclasses gymnasium.Env, implements reset, step, render, and close, and is registered so that gym.make("cs272/YourEnv-v0") works. It must pass gymnasium.utils.env_checker.check_env.
It is tabular. observation_space is Discrete(n) with 20 <= n <= 500, and action_space is Discrete(m) with 2 <= m <= 10. Keep it small!
It is stochastic. The same action in the same state must not always produce the same outcome. A deterministic puzzle is a search problem, not an interesting MDP. All randomness must come from self.np_random, never from random or bare numpy.random.
Episodes end. At least one state sets terminated=True. Pass max_episode_steps to register() so that a lost agent is truncated instead of looping forever. Do not set truncated by hand — let the TimeLimit wrapper do it.
Seeding is honored. reset(seed=k) followed by the same actions must produce exactly the same trajectory, every time, in a fresh process.
It renders. Support render_mode="ansi" and return a human-readable string. You will need it for the sample run in your report.
It is learnable, and worth learning. A tabular agent must reach a clearly better return than a uniform-random agent within 5,000 episodes. If random play already does as well as your trained agent, the environment is too easy or the reward is not telling the agent anything.
It is yours. It must not be a built-in Gymnasium environment with the serial numbers filed off. FrozenLake, CliffWalking, Taxi and Blackjack with renamed variables will receive no credit for Task 1.
Environment Documentation

Your repo must include a README.md documenting the environment, in the style of the Gymnasium environment pages. The documentation is graded. It must cover:

The story: what the world is and what the agent is trying to do
Observation space, including exactly how you encode the state into one integer
Action space
Reward structure
Starting state, termination, and truncation conditions
The transition noise: what is random, and with what probability
Arguments to __init__ and the registered environment id

A reader who has never seen your code should be able to reimplement your environment from this document alone.

Task 2: SARSA(λ)

Implement SARSA(λ) with eligibility traces in myagent.py. The agent gets the environment in its constructor and must read its sizes off env.observation_space.n and env.action_space.n.

Your agent must not know anything about your environment. It may not import myenv, index into a grid, or special-case a state number. It will be run against a different environment than yours during grading, and anything that assumes your own world will fail there.

Tabular SARSA(λ) with eligibility traces

One-step SARSA updates only $Q(s_t, a_t)$ with $\delta_t$. But $\delta_t$ is also informative about the pairs visited before $t$, since they led here. The eligibility trace decides how much of $\delta_t$ each past pair receives.

You can read the textbook section here and watch the legendary lecture by David Silver here (53:38 to 1:19:50).

Input:  step size alpha in (0, 1], discount gamma in [0, 1],
        trace decay lambda in [0, 1], exploration epsilon
Init:   Q(s, a) <- 1        for all s in S, a in A
        (Q(terminal, .) = 0 and is never updated)

for each episode:
    E(s, a) <- 0            for all s, a          # traces are episodic
    S <- env.reset()
    A <- eps_greedy(Q, S)                         # ties broken at random

    repeat for each step of the episode:
        R, S', terminated, truncated <- env.step(A)
        A' <- eps_greedy(Q, S')

        if terminated:                            # no next state exists
            delta <- R - Q(S, A)
        else:                                     # incl. truncated: S' is real
            delta <- R + gamma * Q(S', A') - Q(S, A)

        E(S, A) <- E(S, A) + 1                    # accumulating

        for all s in S, a in A:
            Q(s, a) <- Q(s, a) + alpha * delta * E(s, a)
            E(s, a) <- gamma * lambda * E(s, a)

        S <- S'
        A <- A'
    until terminated or truncated
The λ Sweep

This is the part that shows you understood what λ does.

Train your agent for λ ∈ {0, 0.3, 0.6, 0.9, 1.0}, with at least 5 seeds per λ, and produce:

A learning-curve plot. Mean return per episode against episode number, one line per λ, with the spread across seeds shown (a band or error bars). Smooth the curves over a window of ~100 episodes or they will be unreadable.
A table giving, for each λ: the number of episodes to first reach a target return of your choosing, and the mean final return. State your threshold.

Then answer, in your report: why does λ change the picture on your environment the way it does? Relate it to how far your reward sits from the decisions that earn it. "λ=0.9 was fastest" is an observation, not an answer.

Two free correctness checks

Use both of these before you start debugging anything harder.

λ=0 must reduce to ordinary one-step SARSA. If it does not, your trace update is wrong.
Getting Started

The template is in the course GitHub repo: cs272pub-26fa/pa2_template.

pip install -r requirements.txt

Use myagent.py and myenv.py to implement your agent and environment.

Rules
This assignment is a PAIR project assignment. Find a partner and sign up on People tab on Canvas (PA2 Pair). You must not show to or share your code at any time with other groups.
Both members must commit to the repo, under their own GitHub accounts, with commit messages that say what was changed. The commit history is how you show that each member made a meaningful contribution to the code; a repo where one member did all the committing, or where the whole project lands in a single commit, will be treated as one where the other member did not contribute.
You are free to restructure the template (add helper functions, split files) as long as the required file names, the class name SarsaLambdaAgent, and its constructor signature stay as given, since the grading harness constructs it directly.
Your agent will be run against an environment you have not seen. It must not import or reference myenv.
Everything must be reproducible. Seed your runs and say in the report what seeds you used. A learning curve that cannot be regenerated is not a result.
Submission Guide
Push to a public GitHub repo (You may use one of the group member's repo.):
myenv.py — your environment
myagent.py — your SARSA(λ) agent
myrunner.py — training, the λ sweep, and the plot
env.md — the environment documentation described in Task 1
Submit to Canvas a PDF containing:
A clickable link to the repo (make sure it is public)
The rendered map or state diagram of your environment, and a one-paragraph description of it
One sample episode of the trained greedy policy, using your ansi renderer, with the return
The λ sweep plot and table
Your answer to the "why does λ do that here" question
Code quality, readability, and documentation are a part of the evaluation criteria.
AI Policy
Submitting or relying on AI-generated work that the student cannot explain, justify, or reproduce is not permitted and may result in a failing grade.
You may be asked by the instructor to explain any randomly selected portion of your code at any time after the assignment is submitted. Failure to respond within a reasonable timeframe will be considered academic plagiarism, regardless of whether AI tools were actually used, and will be reported immediately with an associated grade penalty.

This is the work I did:

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

    SLIP_PROBABILITY = 0.20         # make environment stochastic
    STEP_REWARD = -0.10             # disincentivize extra steps
    WALL_REWARD = -0.35             # disincentivize hitting wall
    GOAL_REWARD = 10.0              # big reward for reaching goal

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None):
        # TODO: describe your world here -- the map, the pieces, the constants.
        ### check if global constants above are okay

        ### Possibly add checks for the maze?

        # There are 100 possible cell observations, including cells behind walls
        self.observation_space = spaces.Discrete(self.HEIGHT * self.HEIGHT)
        self.action_space = spaces.Discrete(4) # possible actions: U, R, D, L

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self._agent_pos = self.START # agent position initialized at start
        self._steps = 0 # Count of steps taken
        self._last_action: int | None = None # action agent requested: either int or None
        self._last_true_action: int | None = None # direction used after stochastic noise

    def _position_encoded(self, position: tuple[int, int]) -> int:
        """Encode (row, column) as one integer in [0, 99]."""
        row, column = position
        return row * self.WIDTH + column

    def _get_obs(self) ->int:
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
            0 <= self.HEIGHT and 0 <= self.WIDTH and self.GRID[row][col] != 1
        )


    def reset(self, seed: int | None = None, options: dict | None = None):
        # This line seeds self.np_random. Without it, seeding does not work and
        # the reproducibility test fails.
        super().reset(seed=seed)

        # put the world back to its starting state.
        self._agent_pos = self.START
        self._steps = 0
        self._last_action = None
        self._last_true_action = None

        return self._get_obs(), self._get_info()

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


    def step(self, action: int):
        """ ### add documentation later"""
        # TODO: apply the action, with noise drawn from self.np_random.
        #
        # Return terminated=True when the episode genuinely ends -- goal reached,
        # agent died, game over. Leave truncated as False and let the TimeLimit
        # wrapper from register() handle running out of time. The agent treats
        # the two differently, and so should you.
        if not self.action_space.contains(action):
            raise ValueError(f"Action is {action}: needs to be integer in [0,3], ")

        action = int(action)
        self._last_action = action

        # Apply stochastic portion to flip direction perpendicularly
        true_action, slip = self._get_actual_action(action)
        self._last_true_action = true_action

        old_position = self._agent_pos
        self._agent_pos = self._move(true_action)
        self._steps += 1

        moved = self._agent_pos != old_position
        goal_reached = self._agent_pos == self.GOAL

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
        # TODO: draw it. You need this for the sample episode in your report.
        raise NotImplementedError

    def close(self):
        pass


# Named environment. The id must start with "cs272/" and end with a version, and
# max_episode_steps must be large enough that a competent agent can finish but
# small enough that a lost one gives up.
ENV_ID = "cs272/Maze-v0"
register(
    id=ENV_ID,
    entry_point="myenv:MyEnv",
    max_episode_steps=300,
)


Your current README mostly repeats the assignment. Replace it with a project-specific README that documents your maze and clearly identifies unfinished work.

# CS 272 PA2: Maze Environment

This project implements a custom stochastic Gymnasium environment for Task 1 of
CS 272 PA2. Task 2 will add a model-free SARSA(λ) agent with eligibility traces.

## Project Status

### Task 1: In progress

Implemented or planned:

- 10×10 maze layout
- Tabular observations and actions
- Stochastic perpendicular movement
- Reward structure
- Episode termination at the goal
- Gymnasium environment registration
- Seeded resets
- ANSI rendering

Still to complete:

- Finish and test `render()`
- Pass Gymnasium's `check_env`
- Add complete environment tests
- Compare random play with a trained tabular agent
- Finish documentation after testing

## Environment Story

The agent is a traveler navigating a 10×10 maze. It begins at the entrance
along the top edge and must reach the exit along the bottom edge.

The maze contains open floor cells and walls. The agent can move up, right,
down, or left. Movement is stochastic because occasional gusts push the agent
in a perpendicular direction.

## Maze Layout

The maze uses the following representation:

- `0` = open cell
- `1` = wall

```text
0011111000
1101111101
1100011110
1111011110
1111000110
1111110110
1111000100
1111011100
1111000001
1111111100

The agent starts at: `(0, 0)`

The goal is located at: `(9, 8)`


### Environment ID

The registered Gymnasium environment is: `cs272/Maze-v0`

Example:
```text
import gymnasium as gym
import myenv

env = gym.make("cs272/Maze-v0", render_mode="ansi")
```

### Constructor
The environment constructor is:

`MyEnv(render_mode: str | None = None)`

The supported render mode is: "ansi"

When ANSI rendering is enabled, render() returns a human-readable text
representation of the maze.


### Observation Space

The observation space is: `Discrete(100)`

The agent's `(row, column)` position is encoded into one integer:

`observation = row * width + column`

Because the maze is 10x10:

`observation = row * 10 + column`

For example:
| Position |	Observation |
| :--- | :---: |
| (0, 0) | 0 |
| (0, 1)	| 1 |
| (1, 0)	| 10 |
| (9, 8)	| 98 |

The observation identifies the agent's current cell. Wall cells are included
in the observation space but cannot be entered.

###  Action Space

The action space is: `Discrete(4)`

| Action |	Meaning | Position change |
| :--- | :---: | :---: |
0 |	Up |	(-1, 0) |
1 |	Right |	(0, 1) |
2 |	Down |	(1, 0) |
3 |	Left |	(0, -1) |

If an action would move the agent outside the maze or into a wall, the agent
remains in its current position.

### Transition Noise

The environment is stochastic.

With probability 0.80, the requested action is used.

With probability 0.20, a perpendicular gust occurs:

An up or down action may become left or right.
A left or right action may become up or down.
Each perpendicular direction has probability 0.10.

All random choices are made using Gymnasium's seeded random generator,
self.np_random.

### Rewards

The reward structure is:

| Event |	Reward | 
| :--- | :---: | 
| Reach the goal |	+10.0 |
| Move to an open non-goal cell |	-0.10 |
| Attempt to move into a wall or outside the maze |	-0.35 |

The step penalty encourages the agent to find a short route. The wall penalty
discourages repeatedly attempting invalid movements.

### Episode Behavior

Each call to `reset()`:

* Places the agent at (0, 0)
* Resets the step counter
* Clears the previous action information
* Applies the provided random seed

The episode terminates when the agent reaches `(9, 8)`.

The environment itself returns:

`truncated = False`

The registered Gymnasium TimeLimit wrapper ends an episode after 300 steps
by setting:

`truncated = True`

### Testing

Install the project dependencies:

`python -m pip install -r requirements.txt`

Run the test suite:

`python -m pytest -q`

The tests check:

* Gymnasium API compliance
* Observation and action spaces
* Environment registration
* ANSI rendering
* Reproducibility with seeds
* Stochastic transitions
* Invalid-action handling
* Goal termination
* Time-limit truncation


# Original README.md: 
# CS272-26fa-pa2: Build a World, Then Learn It

## Project Goals
- Learn how to build a custom environment with Gymnasium
- Learn how to code a model-free control agent with eligibility traces, SARSA(λ)

In PA1 the casino was a black box you could only poke at. This time you are on the other side of the wall. You will build the world yourself, write down its rules so that someone else can understand them, and then hand it to a learning agent that knows nothing about what you built.

---

## Task 1: Your Own Gymnasium Environment

Design and implement your own environment in ```myenv.py```. It must follow the [Gymnasium API](https://gymnasium.farama.org/api/env/). 

### Requirements

Your environment must satisfy all of the following.

1. **It is a real Gymnasium environment.** It subclasses ```gymnasium.Env```, implements ```reset```, ```step```, ```render```, and ```close```, and is registered so that ```gym.make("cs272/YourEnv-v0")``` works. It must pass ```gymnasium.utils.env_checker.check_env```.
2. **It is tabular.** ```observation_space``` is ```Discrete(n)``` with ```20 <= n <= 500```, and ```action_space``` is ```Discrete(m)``` with ```2 <= m <= 10```. Keep it small!
3. **It is stochastic.** The same action in the same state must not always produce the same outcome. A deterministic puzzle is a search problem, not an interesting MDP. All randomness must come from ```self.np_random```, never from ```random``` or bare ```numpy.random```.
4. **Episodes end.** At least one state sets ```terminated=True```. Pass ```max_episode_steps``` to ```register()``` so that a lost agent is truncated instead of looping forever. Do not set ```truncated``` by hand — let the ```TimeLimit``` wrapper do it.
5. **Seeding is honored.** ```reset(seed=k)``` followed by the same actions must produce exactly the same trajectory, every time, in a fresh process.
6. **It renders.** Support ```render_mode="ansi"``` and return a human-readable string. You will need it for the sample run in your report.
7. **It is learnable, and worth learning.** A tabular agent must reach a clearly better return than a uniform-random agent within 5,000 episodes. If random play already does as well as your trained agent, the environment is too easy or the reward is not telling the agent anything.
8. **It is yours.** It must not be a built-in Gymnasium environment with the serial numbers filed off. FrozenLake, CliffWalking, Taxi and Blackjack with renamed variables will receive no credit for Task 1.

### Environment Documentation

Your repo must include a ```README.md``` documenting the environment, in the style of the [Gymnasium environment pages](https://gymnasium.farama.org/environments/toy_text/frozen_lake/). **The documentation is graded.** It must cover:

- The story: what the world is and what the agent is trying to do
- Observation space, including exactly how you encode the state into one integer
- Action space
- Reward structure
- Starting state, termination, and truncation conditions
- The transition noise: what is random, and with what probability
- Arguments to ```__init__``` and the registered environment id

A reader who has never seen your code should be able to reimplement your environment from this document alone.

---

## Task 2: SARSA(λ)

Implement SARSA(λ) with eligibility traces in ```myagent.py```. The agent gets the environment in its constructor and must read its sizes off ```env.observation_space.n``` and ```env.action_space.n```.

**Your agent must not know anything about your environment.** It may not import ```myenv```, index into a grid, or special-case a state number. It will be run against a different environment than yours during grading, and anything that assumes your own world will fail there.

### Tabular SARSA(λ) with eligibility traces

One-step SARSA updates only $Q(s_t, a_t)$ with $\delta_t$. But $\delta_t$ is also informative about the pairs visited *before* $t$, since they led here. The eligibility trace decides how much of $\delta_t$ each past pair receives.

You can read the textbook section [here](http://incompleteideas.net/book/first/ebook/node77.html) and watch the legendary lecture by David Silver [here](https://youtu.be/0g4j2k_Ggc4?si=gue-KJgJfpLs_-h_&t=3216) **(53:38 to 1:19:50)**.


```
Input:  step size alpha in (0, 1], discount gamma in [0, 1],
        trace decay lambda in [0, 1], exploration epsilon
Init:   Q(s, a) <- 1        for all s in S, a in A
        (Q(terminal, .) = 0 and is never updated)

for each episode:
    E(s, a) <- 0            for all s, a          # traces are episodic
    S <- env.reset()
    A <- eps_greedy(Q, S)                         # ties broken at random

    repeat for each step of the episode:
        R, S', terminated, truncated <- env.step(A)
        A' <- eps_greedy(Q, S')

        if terminated:                            # no next state exists
            delta <- R - Q(S, A)
        else:                                     # incl. truncated: S' is real
            delta <- R + gamma * Q(S', A') - Q(S, A)

        E(S, A) <- E(S, A) + 1                    # accumulating

        for all s in S, a in A:
            Q(s, a) <- Q(s, a) + alpha * delta * E(s, a)
            E(s, a) <- gamma * lambda * E(s, a)

        S <- S'
        A <- A'
    until terminated or truncated
```


### The λ Sweep

This is the part that shows you understood what λ does.

Train your agent for **λ ∈ {0, 0.3, 0.6, 0.9, 1.0}**, with at least **5 seeds per λ**, and produce:

1. **A learning-curve plot.** Mean return per episode against episode number, one line per λ, with the spread across seeds shown (a band or error bars). Smooth the curves over a window of ~100 episodes or they will be unreadable.
2. **A table** giving, for each λ: the number of episodes to first reach a target return of your choosing, and the mean final return. State your threshold.

Then answer, in your report: **why** does λ change the picture on your environment the way it does? Relate it to how far your reward sits from the decisions that earn it. "λ=0.9 was fastest" is an observation, not an answer.

### Two free correctness checks

Use both of these before you start debugging anything harder.

- **λ=0 must reduce to ordinary one-step SARSA.** If it does not, your trace update is wrong.

---

## Getting Started

The template is in the course GitHub repo: [cs272pub-26fa/pa2_template](https://github.com/sjsu-interconnect/cs272pub-26fa/tree/main/pa2_template).

```bash
pip install -r requirements.txt
```

Use ```myagent.py``` and ```myenv.py``` to implement your agent and environment.

---

## Rules
- This assignment is a **PAIR** project assignment. Find a partner and sign up on People tab on Canvas (PA2 Pair). You must not show to or share your code at any time with other groups.
- Both members must commit to the repo, under their own GitHub accounts, with commit messages that say what was changed. The commit history is how you show that each member made a meaningful contribution to the code; a repo where one member did all the committing, or where the whole project lands in a single commit, will be treated as one where the other member did not contribute.
- You are free to restructure the template (add helper functions, split files) as long as the required file names, the class name ```SarsaLambdaAgent```, and its constructor signature stay as given, since the grading harness constructs it directly.
- Your agent will be run against an environment you have not seen. It must not import or reference ```myenv```.
- Everything must be reproducible. Seed your runs and say in the report what seeds you used. A learning curve that cannot be regenerated is not a result.

## Submission Guide
- Push to a **public GitHub repo** (You may use one of the group member's repo.):
  - ```myenv.py``` — your environment
  - ```myagent.py``` — your SARSA(λ) agent
  - ```myrunner.py``` — training, the λ sweep, and the plot
  - ```env.md``` — the environment documentation described in Task 1
- Submit to **Canvas** a PDF containing:
  - A **clickable** link to the repo (make sure it is public)
  - The rendered map or state diagram of your environment, and a one-paragraph description of it
  - One sample episode of the trained greedy policy, using your ```ansi``` renderer, with the return
  - The λ sweep plot and table
  - Your answer to the "why does λ do that here" question
- Code quality, readability, and documentation are a part of the evaluation criteria.

## AI Policy
- Submitting or relying on AI-generated work that the student cannot explain, justify, or reproduce is not permitted and may result in a failing grade.
- You may be asked by the instructor to explain any randomly selected portion of your code at any time after the assignment is submitted. Failure to respond within a reasonable timeframe will be considered academic plagiarism, regardless of whether AI tools were actually used, and will be reported immediately with an associated grade penalty.
