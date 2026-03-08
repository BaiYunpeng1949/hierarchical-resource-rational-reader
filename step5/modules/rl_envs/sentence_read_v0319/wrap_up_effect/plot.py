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


# -----------------------------
# Load data
# -----------------------------

human = pd.read_csv(HUMAN_PATH)
sim = pd.read_csv(SIM_PATH)


# -----------------------------
# Prepare regression
# -----------------------------

# Human regression
human_coef = np.polyfit(
    human["difficulty_zscore"],
    human["avg_regression_probability"],
    1
)

# Simulation regression
sim_coef = np.polyfit(
    sim["difficulty_zscore"],
    sim["avg_regression_probability"],
    1
)

# x values for smooth regression lines
x_vals = np.linspace(
    min(human["difficulty_zscore"].min(), sim["difficulty_zscore"].min()),
    max(human["difficulty_zscore"].max(), sim["difficulty_zscore"].max()),
    200
)

human_line = human_coef[0] * x_vals + human_coef[1]
sim_line = sim_coef[0] * x_vals + sim_coef[1]


# -----------------------------
# Plot
# -----------------------------

plt.figure(figsize=(6,4))

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

print("Figure saved to:", OUTPUT_FIG)