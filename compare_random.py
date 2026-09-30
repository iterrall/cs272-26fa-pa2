import pandas as pd

CSV_FILE = "results/episode_returns.csv"

df = pd.read_csv(CSV_FILE)

# RandomAgent: mean return over the final 100 episodes
# for each seed

random_df = df[df["kind"] == "random"].copy()

random_final = (
    random_df
    .sort_values(["seed", "episode"])
    .groupby("seed")
    .tail(100)
)

random_by_seed = (
    random_final
    .groupby("seed")["return"]
    .mean()
)

random_mean = random_by_seed.mean()
random_std = random_by_seed.std()

print("RandomAgent")
print("-----------")
print("Mean final return (last 100):", round(random_mean, 3))
print("Std. dev. across seeds:", round(random_std, 3))
print("Per-seed final means:")
print(random_by_seed.round(3))
print()


# SARSA(lambda): mean final return over last 100
# for each lambda

sarsa_df = df[df["kind"] == "sarsa_lambda"].copy()

sarsa_final = (
    sarsa_df
    .sort_values(["lambda", "seed", "episode"])
    .groupby(["lambda", "seed"])
    .tail(100)
)

sarsa_by_seed = (
    sarsa_final
    .groupby(["lambda", "seed"])["return"]
    .mean()
)

sarsa_summary = (
    sarsa_by_seed
    .groupby("lambda")
    .agg(["mean", "std"])
)

print("SARSA(lambda)")
print(sarsa_summary.round(3))
print()


# Comparison using the lambda with the highest
# mean final return

best_lambda = sarsa_summary["mean"].idxmax()
best_sarsa_mean = sarsa_summary.loc[best_lambda, "mean"]

print("Comparison")
print(f"SARSA(lambda={best_lambda}) final mean: {best_sarsa_mean:.3f}")
print(f"RandomAgent final mean:               {random_mean:.3f}")
print(f"Difference:                            {best_sarsa_mean - random_mean:.3f}")