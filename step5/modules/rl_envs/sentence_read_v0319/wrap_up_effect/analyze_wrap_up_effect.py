import os
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------

DIFFICULTY_PATH = "data/sentence_difficulty.csv"

HUMAN_PATH = "data/human/all_words_regression_and_skip_probabilities.csv"
SIM_PATH = "data/simulation/all_words_regression_and_skip_probabilities.csv"

OUTPUT_DIR = "data"
OUTPUT_HUMAN = os.path.join(OUTPUT_DIR, "wrap_up_human.csv")
OUTPUT_SIM = os.path.join(OUTPUT_DIR, "wrap_up_simulation.csv")

# boundary size
K = 2   # last k words


# -----------------------------
# Helper function
# -----------------------------

def compute_boundary_regression(csv_path, difficulty_df, k):

    df = pd.read_csv(csv_path)

    # ensure correct sorting
    df = df.sort_values(["sentence_id", "word_id"])

    results = []

    grouped = df.groupby("sentence_id")

    for sentence_id, group in grouped:

        group = group.reset_index(drop=True)

        n = len(group)

        # select last k words
        boundary = group.iloc[max(0, n - k):n]

        avg_reg = boundary["regression_probability"].mean()

        sentence_row = difficulty_df[difficulty_df["index"] == sentence_id]

        if len(sentence_row) == 0:
            continue

        sentence = sentence_row["sentence"].values[0]
        difficulty_z = sentence_row["difficulty_zscore"].values[0]

        results.append({
            "sentence_id": sentence_id,
            "sentence": sentence,
            "difficulty_zscore": difficulty_z,
            "k": k,
            "avg_regression_probability": avg_reg
        })

    return pd.DataFrame(results)


# -----------------------------
# Main
# -----------------------------

print("Loading sentence difficulty...")
difficulty_df = pd.read_csv(DIFFICULTY_PATH)

print("Processing human data...")
human_df = compute_boundary_regression(HUMAN_PATH, difficulty_df, K)

print("Processing simulation data...")
sim_df = compute_boundary_regression(SIM_PATH, difficulty_df, K)

os.makedirs(OUTPUT_DIR, exist_ok=True)

human_df.to_csv(OUTPUT_HUMAN, index=False)
sim_df.to_csv(OUTPUT_SIM, index=False)

print("Saved results:")
print(OUTPUT_HUMAN)
print(OUTPUT_SIM)