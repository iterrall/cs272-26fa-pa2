"""Training runner and report-material generator for CS 272 PA2."""

from __future__ import annotations

import csv
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import myenv
from myagent import RandomAgent, SarsaLambdaAgent

TOTAL_EPISODES = 5_000
LAMBDAS = (0.0, 0.3, 0.6, 0.9, 1.0)
SEEDS = (11, 22, 33, 44, 55)
SMOOTH_WINDOW = 100
TARGET_RETURN = 7.0
OUTPUT_DIR = Path("results")


def smooth(values: list[float], window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Return a trailing moving average with approximately ``window`` episodes."""
    arr = np.asarray(values, dtype=float)
    if arr.size < window:
        return arr.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="valid")


def first_reach(values: list[float], target: float) -> int | None:
    """Return the first 1-based episode reaching target, or None."""
    for index, value in enumerate(values, start=1):
        if value >= target:
            return index
    return None


def train_sweep() -> dict[float, dict[int, list[float]]]:
    """Train every lambda/seed pair and store every episode return."""
    sweep: dict[float, dict[int, list[float]]] = {}
    for lam in LAMBDAS:
        sweep[lam] = {}
        for seed in SEEDS:
            env = gym.make(myenv.ENV_ID)
            agent = SarsaLambdaAgent(
                env,
                lam=lam,
                total_epi=TOTAL_EPISODES,
                init_val=1.0,
                seed=seed,
            )
            sweep[lam][seed] = agent.learn()
            env.close()
    return sweep


def train_random_baseline() -> list[float]:
    """Train the RandomAgent baseline for 5,000 episodes."""
    env = gym.make(myenv.ENV_ID)
    # Seed the environment once before RandomAgent.learn(); its repeated
    # unseeded resets then continue deterministically from that RNG stream.
    env.reset(seed=SEEDS[0])
    agent = RandomAgent(
        env,
        total_epi=TOTAL_EPISODES,
        init_val=1.0,
        seed=SEEDS[0],
    )
    returns = agent.learn()
    env.close()
    return returns


def save_episode_csv(
    sweep: dict[float, dict[int, list[float]]], random_returns: list[float]
) -> None:
    """Save every training return for every lambda/seed to CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "episode_returns.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "lambda", "seed", "episode", "return"])
        for lam in LAMBDAS:
            for seed in SEEDS:
                for episode, value in enumerate(sweep[lam][seed], start=1):
                    writer.writerow(["sarsa_lambda", lam, seed, episode, value])
        for episode, value in enumerate(random_returns, start=1):
            writer.writerow(["random", "", SEEDS[0], episode, value])


def make_learning_curve_plot(
    sweep: dict[float, dict[int, list[float]]], random_returns: list[float]
) -> Path:
    """Save smoothed lambda curves with across-seed standard-deviation bands."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(10, 6))
    axis = figure.add_subplot(1, 1, 1)

    for lam in LAMBDAS:
        matrix = np.asarray([sweep[lam][seed] for seed in SEEDS], dtype=float)
        smoothed = np.asarray([smooth(row) for row in matrix])
        mean_curve = smoothed.mean(axis=0)
        std_curve = smoothed.std(axis=0)
        episodes = np.arange(1, len(mean_curve) + 1) + SMOOTH_WINDOW - 1
        axis.plot(episodes, mean_curve, label=f"lambda={lam}")
        axis.fill_between(
            episodes,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.15,
        )

    random_smoothed = smooth(random_returns)
    random_x = np.arange(1, len(random_smoothed) + 1) + SMOOTH_WINDOW - 1
    axis.plot(random_x, random_smoothed, linestyle="--", label="RandomAgent")
    axis.axhline(TARGET_RETURN, linestyle=":", label=f"target={TARGET_RETURN:.1f}")

    axis.set_title("SARSA(lambda) Learning Curves Across Five Seeds")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Undiscounted return (100-episode moving average)")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()

    path = OUTPUT_DIR / "lambda_learning_curves.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def make_summary_table(sweep: dict[float, dict[int, list[float]]]) -> Path:
    """Write the lambda summary table and machine-readable CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Lambda Sweep Summary",
        "",
        f"Target return: **{TARGET_RETURN:.1f}**",
        "",
        "| lambda | Mean episodes to first reach target | Seeds reaching target | Mean final return (last 100) | Std. dev. | Target |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    csv_rows = []

    for lam in LAMBDAS:
        first_hits = [first_reach(sweep[lam][seed], TARGET_RETURN) for seed in SEEDS]
        hit_values = [value for value in first_hits if value is not None]
        final_values = [float(np.mean(sweep[lam][seed][-100:])) for seed in SEEDS]

        mean_hit = float(np.mean(hit_values)) if hit_values else None
        mean_final = float(np.mean(final_values))
        std_final = float(np.std(final_values))
        hit_display = f"{mean_hit:.1f}" if mean_hit is not None else "not reached"

        lines.append(
            f"| {lam:.1f} | {hit_display} | {len(hit_values)}/{len(SEEDS)} | {mean_final:.3f} | {std_final:.3f} | {TARGET_RETURN:.1f} |"
        )
        csv_rows.append(
            [
                lam,
                "" if mean_hit is None else mean_hit,
                len(hit_values),
                mean_final,
                std_final,
                TARGET_RETURN,
            ]
        )

    markdown_path = OUTPUT_DIR / "lambda_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n")

    with (OUTPUT_DIR / "lambda_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "lambda",
                "mean_episodes_to_target",
                "seeds_reaching_target",
                "mean_final_return_last_100",
                "std_final_return_last_100",
                "target_return",
            ]
        )
        writer.writerows(csv_rows)

    return markdown_path


def make_sample_greedy_run() -> Path:
    """Train lambda=0.9 and save one greedy episode with ANSI renderings."""
    env = gym.make(myenv.ENV_ID, render_mode="ansi")
    agent = SarsaLambdaAgent(
        env,
        lam=0.9,
        total_epi=TOTAL_EPISODES,
        init_val=1.0,
        seed=SEEDS[0],
    )
    agent.learn()
    episode, terminated = agent.best_run()
    total_return = agent.calc_return(episode)
    env.close()

    # Recreate the same seeded episode and replay the recorded greedy actions so
    # the saved frames correspond to the trajectory returned by best_run().
    replay_env = gym.make(myenv.ENV_ID, render_mode="ansi")
    replay_env.reset(seed=SEEDS[0])
    frames = [replay_env.render()]
    for _, action, _ in episode:
        replay_env.step(action)
        frames.append(replay_env.render())
        if len(frames) == len(episode) + 1:
            break
    replay_env.close()

    lines = [
        "# Sample Greedy-Policy Episode",
        "",
        "Policy: SARSA(lambda), lambda=0.9",
        f"Seed: {SEEDS[0]}",
        f"Reached terminal goal: {terminated}",
        f"Undiscounted return: {total_return:.3f}",
        f"Number of steps: {len(episode)}",
        "",
    ]
    for step_number, frame in enumerate(frames):
        lines.extend([f"## Step {step_number}", "", "```text", frame, "```", ""])

    output_path = OUTPUT_DIR / "sample_greedy_run.md"
    output_path.write_text("\n".join(lines))
    return output_path


def main() -> None:
    """Run the full experiment and generate report materials."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Training RandomAgent baseline (5,000 episodes)...")
    random_returns = train_random_baseline()
    print(f"Random mean return: {np.mean(random_returns):.3f}")

    print("Training lambda sweep (5 lambdas x 5 seeds x 5,000 episodes)...")
    sweep = train_sweep()
    save_episode_csv(sweep, random_returns)
    plot_path = make_learning_curve_plot(sweep, random_returns)
    summary_path = make_summary_table(sweep)
    sample_path = make_sample_greedy_run()

    print(f"Saved {plot_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {sample_path}")
    print(f"Saved {OUTPUT_DIR / 'episode_returns.csv'}")


if __name__ == "__main__":
    main()
