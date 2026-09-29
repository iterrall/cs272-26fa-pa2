
# CS 272 PA2: Maze Environment

This project implements a custom stochastic Gymnasium environment for Task 1 of
CS 272 PA2. Task 2 will add a model-free SARSA(λ) agent with eligibility traces.

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

The goal is located at: text```(9, 8)```


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

