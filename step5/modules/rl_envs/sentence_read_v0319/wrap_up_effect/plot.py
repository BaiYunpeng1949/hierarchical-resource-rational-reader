import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------

HUMAN_PATH = "data/wrap_up_human.csv"
SIM_PATH = "data/wrap_up_simulation.csv"

OUTPUT_DIR = "figures"
OUTPUT_FIG = os.path.join(OUTPUT_DIR, "wrap_up_regression_plot.png")
OUTPUT_STATS = os.path.join(OUTPUT_DIR, "wrap_up_regression_stats.txt")


# -----------------------------
# Helper function
# -----------------------------

def fit_and_report_regression(df, x_col, y_col, label):
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    # Linear fit: y = slope * x + intercept
    slope, intercept = np.polyfit(x, y, 1)

    # Predicted values
    y_pred = slope * x + intercept

    # Goodness-of-fit
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    # Pearson correlation
    r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan

    # Sample size
    n = len(df)

    stats_text = (
        f"{label}\n"
        f"n = {n}\n"
        f"slope = {slope:.6f}\n"
        f"intercept = {intercept:.6f}\n"
        f"R^2 = {r_squared:.6f}\n"
        f"r = {r:.6f}\n"
        f"equation: y = {slope:.6f} * x + {intercept:.6f}\n"
    )

    return {
        "label": label,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "r": r,
        "n": n,
        "y_pred": y_pred,
        "stats_text": stats_text,
    }


# -----------------------------
# Load data
# -----------------------------

human = pd.read_csv(HUMAN_PATH)
sim = pd.read_csv(SIM_PATH)


# -----------------------------
# Prepare regression
# -----------------------------

human_stats = fit_and_report_regression(
    human,
    "difficulty_zscore",
    "avg_regression_probability",
    "Human"
)

sim_stats = fit_and_report_regression(
    sim,
    "difficulty_zscore",
    "avg_regression_probability",
    "Simulation"
)

# x values for smooth regression lines
x_vals = np.linspace(
    min(human["difficulty_zscore"].min(), sim["difficulty_zscore"].min()),
    max(human["difficulty_zscore"].max(), sim["difficulty_zscore"].max()),
    200
)

human_line = human_stats["slope"] * x_vals + human_stats["intercept"]
sim_line = sim_stats["slope"] * x_vals + sim_stats["intercept"]


# -----------------------------
# Plot
# -----------------------------

plt.figure(figsize=(6, 4))

# scatter points
plt.scatter(
    human["difficulty_zscore"],
    human["avg_regression_probability"],
    alpha=0.6,
    label="Human"
)

plt.scatter(
    sim["difficulty_zscore"],
    sim["avg_regression_probability"],
    alpha=0.6,
    label="Simulation"
)

# regression lines
plt.plot(x_vals, human_line, linewidth=2)
plt.plot(x_vals, sim_line, linewidth=2)

plt.xlabel("Sentence Difficulty (z-score)")
plt.ylabel("Boundary Regression Probability")
plt.title("Wrap-up Effect: Regression Analysis")
plt.legend()
plt.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(OUTPUT_FIG, dpi=300)
plt.show()


# -----------------------------
# Save regression stats
# -----------------------------

with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
    f.write("Wrap-up Effect Regression Statistics\n")
    f.write("=" * 40 + "\n\n")
    f.write(human_stats["stats_text"])
    f.write("\n")
    f.write(sim_stats["stats_text"])


# -----------------------------
# Print summary
# -----------------------------

print("Figure saved to:", OUTPUT_FIG)
print("Regression stats saved to:", OUTPUT_STATS)
print()
print(human_stats["stats_text"])
print(sim_stats["stats_text"])
