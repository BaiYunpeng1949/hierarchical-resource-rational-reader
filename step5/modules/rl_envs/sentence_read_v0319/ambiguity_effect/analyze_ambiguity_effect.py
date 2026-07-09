import os
import pandas as pd
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
AMBIGUITY_PATH = "data/sentence_ambiguity.csv"

HUMAN_PATH = "data/human/all_words_regression_and_skip_probabilities.csv"
SIM_PATH = "data/simulation/all_words_regression_and_skip_probabilities.csv"

OUTPUT_DIR = "data"

# raw sentence-level outputs
OUTPUT_HUMAN_RAW = os.path.join(OUTPUT_DIR, "ambiguity_effect_human.csv")
OUTPUT_SIM_RAW = os.path.join(OUTPUT_DIR, "ambiguity_effect_simulation.csv")

# binned outputs (for plotting)
OUTPUT_HUMAN_BINNED = os.path.join(OUTPUT_DIR, "ambiguity_effect_human_binned.csv")
OUTPUT_SIM_BINNED = os.path.join(OUTPUT_DIR, "ambiguity_effect_simulation_binned.csv")

# expected columns
SENTENCE_COL = "sentence_id"
WORD_ID_COL = "word_id"
REG_COL = "regression_probability"
SKIP_COL = "skip_probability"

# number of bins
N_BINS = 10


# -----------------------------
# Step 1: sentence-level effects
# -----------------------------
def compute_sentence_level_effects(csv_path: str, ambiguity_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each sentence:
    - compute whole-sentence average regression probability
    - compute whole-sentence average skip probability
    - attach sentence text and ambiguity score
    """
    df = pd.read_csv(csv_path)

    if WORD_ID_COL in df.columns:
        df = df.sort_values([SENTENCE_COL, WORD_ID_COL])

    results = []

    grouped = df.groupby(SENTENCE_COL)

    for sentence_id, group in grouped:
        group = group.reset_index(drop=True)

        avg_reg = group[REG_COL].mean() if REG_COL in group.columns else np.nan
        avg_skip = group[SKIP_COL].mean() if SKIP_COL in group.columns else np.nan

        meta = ambiguity_df[ambiguity_df["index"] == sentence_id]
        if len(meta) == 0:
            continue

        results.append({
            "sentence_id": sentence_id,
            "sentence": meta["sentence"].values[0],
            "ambiguity": meta["ambiguity"].values[0],
            "ambiguity_zscore": meta["ambiguity_zscore"].values[0],
            "ambiguity_norm": meta["ambiguity_norm"].values[0],
            "avg_regression_probability": avg_reg,
            "avg_skip_probability": avg_skip,
            "num_words": len(group),
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("sentence_id").reset_index(drop=True)
    return result_df


# -----------------------------
# Step 2: binning
# -----------------------------
def bin_effects(df: pd.DataFrame, x_col: str = "ambiguity_zscore", n_bins: int = 10) -> pd.DataFrame:
    """
    Partition x-axis into bins, then compute mean values within each bin.
    """
    work_df = df.copy()

    # qcut = quantile bins, approximately equal number of sentences per bin
    work_df["bin"] = pd.qcut(
        work_df[x_col],
        q=n_bins,
        labels=False,
        duplicates="drop"
    )

    binned = (
        work_df.groupby("bin", as_index=False)
        .agg({
            x_col: "mean",
            "ambiguity": "mean",
            "ambiguity_norm": "mean",
            "avg_regression_probability": "mean",
            "avg_skip_probability": "mean",
            "sentence_id": "count",
        })
        .rename(columns={"sentence_id": "n_sentences"})
    )

    return binned


# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading sentence ambiguity...")
    ambiguity_df = pd.read_csv(AMBIGUITY_PATH)

    print("Processing human raw sentence-level data...")
    human_raw = compute_sentence_level_effects(HUMAN_PATH, ambiguity_df)

    print("Processing simulation raw sentence-level data...")
    sim_raw = compute_sentence_level_effects(SIM_PATH, ambiguity_df)

    print(f"Binning data into {N_BINS} partitions...")
    human_binned = bin_effects(human_raw, x_col="ambiguity_zscore", n_bins=N_BINS)
    sim_binned = bin_effects(sim_raw, x_col="ambiguity_zscore", n_bins=N_BINS)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # save raw
    human_raw.to_csv(OUTPUT_HUMAN_RAW, index=False)
    sim_raw.to_csv(OUTPUT_SIM_RAW, index=False)

    # save binned
    human_binned.to_csv(OUTPUT_HUMAN_BINNED, index=False)
    sim_binned.to_csv(OUTPUT_SIM_BINNED, index=False)

    print("Saved raw sentence-level files:")
    print(OUTPUT_HUMAN_RAW)
    print(OUTPUT_SIM_RAW)

    print("\nSaved binned files for plotting:")
    print(OUTPUT_HUMAN_BINNED)
    print(OUTPUT_SIM_BINNED)

    print("\nHuman binned preview:")
    print(human_binned.head())

    print("\nSimulation binned preview:")
    print(sim_binned.head())


if __name__ == "__main__":
    main()