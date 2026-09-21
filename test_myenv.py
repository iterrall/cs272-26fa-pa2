import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from myenv import MyEnv, ENV_ID

def test_env_checker():
    check_env(MyEnv())

def test_registration_andansi_render():
    env = gym.make(ENV_ID, render_mode="ansi")
    observation, _ = env.reset(seed=263)
    assert observation == 0
    rendered = env.render()
    assert isinstance(rendered, str)
    assert "A" in rendered and "G" in rendered
    env.close()