
# Isis Martinez, Ramya Nayak
# CS 272 - Reinforcement Learning
# Assignment 2
# September 29, 2026

# ----------------------------------------------------

# visualize maze in the terminal

import gymnasium as gym
import myenv

env = gym.make(myenv.ENV_ID, render_mode="ansi")
env.reset(seed=272)
print(env.render())

# hardcoded value for testing before the agent is implemented
for action in [1, 1, 2, 2, 1]:
    obs, reward, terminated, truncated, info = env.step(action)
    print()
    print(env.render())
    print(f"reward={reward}, terminated={terminated}, info={info}")
    if terminated or truncated:
        break

env.close()