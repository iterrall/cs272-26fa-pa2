# `cs272/Maze-v0` Environment Specification

## Story and Objective

The agent is a traveler navigating a fixed 10x10 maze. Each reset samples two distinct open cells: a random start and a random goal. The agent must reach the sampled goal while avoiding unnecessary movement and wall collisions. The environment is stochastic because a perpendicular gust can replace the requested direction.

## Full Map

`0` means an open/walkable cell. `1` means a wall.

```text
0100010001
0101010101
0101010101
0101010101
0101010101
0101010101
0101010101
0101010101
0101010101
0001000101
```

## State Encoding

The observation space is `Discrete(100)`. A position `(row, column)` is encoded as `state = row * 10 + column`.

For example, `(9, 8)` encodes to state `98`.

## Actions

| Action | Meaning | Position change |
|---:|---|---|
| 0 | Up | `(-1, 0)` |
| 1 | Right | `(0, 1)` |
| 2 | Down | `(1, 0)` |
| 3 | Left | `(0, -1)` |

An attempted move into a wall or outside the maze leaves the agent in the same position.

## Rewards

| Event | Reward |
|---|---:|
| Goal reached | `+10.0` |
| Open non-goal cell reached | `-0.10` |
| Wall/out-of-bounds attempt | `-0.35` |

## Start and Goal Positions

On every `reset()`, two different open cells are sampled without replacement. The first becomes the start position and the second becomes the goal position. Start and goal therefore vary by episode while the maze terrain stays fixed.

## Transition Probabilities

The requested action is used with probability `0.80`.

With probability `0.20`, it is replaced by one of the two perpendicular directions, each with probability `0.10`.

All random choices use `self.np_random`, so seeded episodes are reproducible.

## Rendering

Supported `render_mode`:

```text
"ansi"
```

ANSI rendering returns a text maze with `A` = agent, `G` = goal, `.` = open cell, and `#` = wall. A legend also reports the agent and goal positions.

## Episode Limits and Registration

Registered environment ID:

```text
cs272/Maze-v0
```

Registration applies:

```text
max_episode_steps = 300
```

The base `MyEnv` keeps `truncated = False`. Gymnasium's `TimeLimit` wrapper handles the 300-step truncation.

## Files

- `myenv.py` — Task 1 environment
- `myagent.py` — Task 2 SARSA(lambda) agent and RandomAgent baseline
- `myrunner.py` — five-seed lambda sweep, plots, summary table, and sample greedy run
- `test_myenv.py` — environment and agent behavior tests
- `visualize.py` — simple ANSI environment demonstration
