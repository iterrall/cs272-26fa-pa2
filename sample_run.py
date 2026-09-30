# Isis Martinez, Ramya Nayak
# CS 272 - Reinforcement Learning
# Assignment 2
# September 29, 2026

import gymnasium as gym

import myenv
from myagent import SarsaLambdaAgent


def main():
    # Create the environment with ANSI rendering.
    env = gym.make(
        myenv.ENV_ID,
        render_mode="ansi",
    )

    # Train the SARSA(lambda) agent.
    agent = SarsaLambdaAgent(
        env=env,
        gamma=0.99,
        alpha=0.05,
        eps=0.1,
        lam=0.9,
        total_epi=5000,
        init_val=1.0,
        seed=272,
    )

    print("Training agent...")
    agent.learn()

    # Run one greedy episode using the learned Q-table.
    episode, terminated = agent.best_run(max_steps=300)

    print()
    print("Sample Greedy Policy Episode")
    print(f"Reached terminal goal: {terminated}")

    # Replay greedy episode so that we can print ANSI rendering after every action.
    env.reset(seed=agent.seed)

    print(env.render())

    for step_number, (_, action, reward) in enumerate(episode, start=1):
        _, _, replay_terminated, replay_truncated, info = env.step(action)

        print()
        print(f"Step {step_number}: action={action}, reward={reward}")
        print(env.render())

        if replay_terminated or replay_truncated:
            break

    # Calculate the return from the saved episode.
    episode_return = agent.calc_return(
        episode,
        discounted=False,
    )

    discounted_return = agent.calc_return(
        episode,
        discounted=True,
    )

    print(f"Undiscounted return: {episode_return:.2f}")
    print(f"Discounted return:   {discounted_return:.2f}")
    print(f"Steps:               {len(episode)}")
    print(f"Reached goal:        {terminated}")

    env.close()


if __name__ == "__main__":
    main()