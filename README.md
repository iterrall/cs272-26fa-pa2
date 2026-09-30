
# CS 272 PA2: Maze Environment and SARSA(lambda) by Ramya Nayak and Isis Martinez

This project implements a custom stochastic Gymnasium maze environment for Task 
1 and a model-free SARSA(lambda) agent with eligibility traces for Task 2. The 
environment is a 10x10 maze with `0` representing open floor and `1` 
representing a wall. The agent receives a positive reward only when it reaches 
the goal, so learning depends on discovering useful multi-step paths through 
delayed reward.

## Status
Task 1 environment registration, ANSI rendering, documentation, and tests are 
implemented. Task 2 SARSA(lambda), the RandomAgent baseline, the five-seed 
lambda sweep, and report-material generation are implemented in `myagent.py` 
and `myrunner.py`.


## Environment Story

The agent is a traveler navigating a 10x10 maze. At each reset, the environment 
randomly selects two distinct open cells: one start cell for the agent and one 
goal cell representing the exit. The objective is to reach the goal while 
minimizing unnecessary movement and wall collisions. Movement is stochastic 
because a gust can replace the requested action with a perpendicular direction.

## Maze Layout

The fixed maze terrain is:

- `0` = open/walkable cell
- `1` = wall

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

There are 54 open cells. The start and goal are sampled from these open cells 
without replacement on every call to `reset()`.



* **Start Position:** Chosen dynamically at random on `reset()` from the pool of valid `0` tiles.
* **Goal Position:** Chosen dynamically at random on `reset()` from the pool of valid `0` tiles.

## Registered Environment ID

The registered Gymnasium environment is `cs272/Maze-v0`.

```python
import gymnasium as gym
import myenv

env = gym.make("cs272/Maze-v0", render_mode="ansi")
```

The registration applies `max_episode_steps=300`. The base environment always 
returns `truncated=False`; Gymnasium's `TimeLimit` wrapper changes this to 
`truncated=True` when the 300-step limit is reached.

## Constructor and Rendering

The constructor is:

```python
MyEnv(render_mode: str | None = None)
```

The supported rendering mode is `"ansi"`. With ANSI rendering enabled, 
`render()` returns the 10x10 maze as text plus a legend showing the agent 
(`A`) and goal (`G`) positions. Open cells are shown as `.`, and walls are shown as `#`.


### Observation Space

The observation space is `Discrete(100)`. The agent's `(row, column)` position 
is encoded as:

```text
state = row * 10 + column
```

Examples:

| Position | State |
|---|---:|
| `(0, 0)` | 0 |
| `(0, 1)` | 1 |
| `(1, 0)` | 10 |
| `(9, 8)` | 98 |

Wall cells are included in the 100-state observation space, but `_move()` never 
enters a wall.


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

### Transition Probabilities
The environment is stochastic:

- Probability `0.80`: the requested action is used.
- Probability `0.20`: a perpendicular gust occurs.
- Under the gust, each of the two perpendicular directions has probability `0.10`.

All random choices use Gymnasium's seeded `self.np_random` generator, so a fixed 
seed and action sequence is reproducible.

## Rewards

| Event | Reward |
|---|---:|
| Reach the goal | `+10.0` |
| Move to an open non-goal cell | `-0.10` |
| Attempt to move into a wall or outside the maze | `-0.35` |

The step penalty encourages shorter routes, while the wall penalty discourages repeated invalid movement.

## Start and Goal Positions

The current implementation does not fix the start at `(0,0)` or the goal at `(9,8)`. Instead, each `reset()` samples a unique start and goal from the open cells. This makes the environment a family of navigation episodes over the same fixed maze terrain.

The encoded state of `(9,8)` is `98`, which is used in tests and examples when that cell is intentionally assigned as the goal.

## Episode Termination and Time Limit

The base environment returns `terminated=True` when the agent reaches the sampled goal. The base environment always returns `truncated=False`.

The registered Gymnasium `TimeLimit` wrapper uses `max_episode_steps=300` and produces `truncated=True` when 300 steps are reached without termination.

## Task 2: SARSA(lambda)

`myagent.py` implements tabular SARSA(lambda) using the state and action sizes obtained from the supplied Gymnasium environment. The Q table starts at `1.0` by default. Terminal states are not used for bootstrapping. With `lambda=0`, the eligibility trace is zeroed after the current update, giving ordinary one-step SARSA behavior.

`best_run()` resets the environment, disables exploration, records `(state, action, reward)` for each step, and stops at either termination or truncation. `calc_return()` computes either the undiscounted sum of rewards or the discounted return using `gamma**timestep`.

## Runner and Lambda Sweep

Run:

```bash
python myrunner.py
```

The runner trains:

- a RandomAgent baseline for 5,000 episodes;
- SARSA(lambda) for `lambda ∈ {0, 0.3, 0.6, 0.9, 1.0}`;
- five seeds per lambda: `11, 22, 33, 44, 55`.

Every episode return is recorded. The runner computes the mean learning curve across seeds, standard deviation across seeds, a 100-episode moving average, the first episode reaching the selected target return of `7.0`, mean final return over the last 100 episodes, and a sample greedy-policy episode rendered in ANSI text.

Generated report materials are written to `results/`:

- `episode_returns.csv`
- `lambda_learning_curves.png`
- `lambda_summary.md`
- `lambda_summary.csv`
- `sample_greedy_run.md`

## Testing

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the tests:

```bash
python -m pytest -q
```

The tests cover Gymnasium API compliance, observation/action spaces, registration, ANSI rendering, reproducibility, stochastic transitions, invalid actions, goal termination, TimeLimit truncation, goal reachability through valid movement, `calc_return()`, and the lambda=0 one-step SARSA behavior.
