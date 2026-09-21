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


