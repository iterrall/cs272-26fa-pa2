"""Tests for the custom Gymnasium maze environment and SARSA(lambda)."""

from collections import deque

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from myagent import SarsaLambdaAgent
from myenv import ENV_ID, MyEnv


def test_env_checker():
    """The environment follows the Gymnasium API."""
    env = gym.make(ENV_ID)
    check_env(env.unwrapped)
    env.close()


def test_spaces_and_initial_state():
    """The spaces and a seeded reset have expected valid values."""
    env = MyEnv()
    observation, info = env.reset(seed=272)

    assert isinstance(env.observation_space, spaces.Discrete)
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.observation_space.n == 100
    assert env.action_space.n == 4
    assert observation == env._position_encoded(info["position"])
    assert info["position"] in env.open_cells

    env.close()


def test_registration_and_rendering():
    """The registered environment can be created and rendered."""
    env = gym.make(ENV_ID, render_mode="ansi")
    observation, _ = env.reset(seed=272)
    rendered = env.render()

    assert env.spec.id == ENV_ID
    assert observation in range(100)
    assert isinstance(rendered, str)
    assert "A" in rendered
    assert "G" in rendered
    assert "#" in rendered
    assert "Agent" in rendered
    assert "Goal" in rendered

    env.close()


def test_seeded_trajectories_match():
    """The same seed and actions produce the same trajectory."""
    actions = [1, 1, 2, 2, 1, 0, 3, 2, 1, 1]

    def collect_trajectory():
        env = MyEnv()
        observation, _ = env.reset(seed=12345)
        trajectory = []

        for action in actions:
            next_observation, reward, terminated, truncated, info = env.step(action)
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
    """The same requested action can produce different outcomes under different seeds."""
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
    env._agent_pos = (9, 7)
    env._goal_pos = (9, 8)

    observation, reward, terminated, truncated, _ = env.step(MyEnv.RIGHT)

    assert observation == 98
    assert reward == MyEnv.GOAL_REWARD
    assert terminated is True
    assert truncated is False

    env.close()


def test_time_limit_truncates_after_300_steps():
    """The registered TimeLimit wrapper, not MyEnv, produces truncation."""
    env = gym.make(ENV_ID)
    base = env.unwrapped
    base.SLIP_PROBABILITY = 0.0
    env.reset(seed=272)
    base._agent_pos = (0, 0)
    base._goal_pos = (9, 8)

    terminated = False
    truncated = False
    for _ in range(300):
        _, _, terminated, truncated, _ = env.step(MyEnv.LEFT)
        if terminated or truncated:
            break

    assert terminated is False
    assert truncated is True
    assert base._steps == 300
    env.close()


def test_goal_reachable_through_valid_movement():
    """The goal is reachable from a valid open start using only legal moves."""
    env = MyEnv()
    env.SLIP_PROBABILITY = 0.0
    env.reset(seed=272)
    env._agent_pos = (0, 0)
    env._goal_pos = (9, 8)

    action_by_delta = {
        (-1, 0): MyEnv.UP,
        (1, 0): MyEnv.DOWN,
        (0, -1): MyEnv.LEFT,
        (0, 1): MyEnv.RIGHT,
    }

    queue = deque([((0, 0), [])])
    visited = {(0, 0)}
    path = None

    while queue:
        position, candidate_path = queue.popleft()
        if position == (9, 8):
            path = candidate_path
            break

        row, col = position
        for (dr, dc), action in action_by_delta.items():
            nxt = (row + dr, col + dc)
            if nxt not in visited and env._is_open(nxt):
                visited.add(nxt)
                queue.append((nxt, candidate_path + [action]))

    assert path is not None

    terminated = False
    for action in path:
        _, _, terminated, truncated, _ = env.step(action)
        assert truncated is False
        if terminated:
            break

    assert env._agent_pos == (9, 8)
    assert terminated is True
    env.close()


def test_lambda_zero_matches_one_step_sarsa():
    """At lambda=0, only the current state-action receives an update."""

    class TinyDeterministicEnv:
        """Minimal three-state environment used to verify the SARSA update."""

        def __init__(self):
            self.observation_space = spaces.Discrete(3)
            self.action_space = spaces.Discrete(2)
            self.state = 0

        def reset(self, seed=None):
            self.state = 0
            return 0, {}

        def step(self, action):
            if self.state == 0:
                self.state = 1
                return 1, 0.0, False, False, {}
            self.state = 2
            return 2, 10.0, True, False, {}

        def close(self):
            return None

    env = TinyDeterministicEnv()
    agent = SarsaLambdaAgent(
        env,
        gamma=0.9,
        alpha=0.5,
        eps=0.0,
        lam=0.0,
        total_epi=1,
        init_val=1.0,
        seed=7,
    )

    agent.q[0] = np.array([1.0, 2.0])
    agent.q[1] = np.array([2.0, 1.0])
    agent.learn()

    # One-step SARSA calculations:
    # Q(0,1) = 2 + 0.5 * (0 + 0.9*2 - 2) = 1.9
    # Q(1,0) = 2 + 0.5 * (10 - 2) = 6.0
    assert agent.q[0, 1] == pytest.approx(1.9)
    assert agent.q[1, 0] == pytest.approx(6.0)

    # The terminal row is zeroed and is never a bootstrap source.
    assert np.all(agent.q[2] == 0.0)


def test_calc_return():
    """calc_return implements both undiscounted and discounted returns."""
    env = MyEnv()
    agent = SarsaLambdaAgent(env, gamma=0.9, total_epi=1, seed=1)
    episode = [(0, 1, 1.0), (1, 2, 2.0), (2, 0, -1.0)]

    assert agent.calc_return(episode) == pytest.approx(2.0)
    assert agent.calc_return(episode, discounted=True) == pytest.approx(
        1.0 + 0.9 * 2.0 + 0.9**2 * -1.0
    )
    env.close()
