import os
import pandas as pd
import numpy as np
from glob import glob

# -----------------------------
# Configuration
# -----------------------------
# Sentence-level ambiguity file
AMBIGUITY_PATH = "assets/sentence_ambiguity.csv"

# Human data
HUMAN_PATH = "human_data/all_words_regression_and_skip_probabilities.csv"
HUMAN_OUTPUT_RAW = "human_data/ambiguity_effect_human.csv"
HUMAN_OUTPUT_BINNED = "human_data/ambiguity_effect_human_binned.csv"

# Simulation root
SIM_ROOT = "simulation_data"

# Expected file names inside each simulation subfolder
SIM_INPUT_FILENAME = "all_words_regression_and_skip_probabilities.csv"
SIM_OUTPUT_RAW_FILENAME = "ambiguity_effect_simulation.csv"
SIM_OUTPUT_BINNED_FILENAME = "ambiguity_effect_simulation_binned.csv"

# Expected columns
SENTENCE_COL = "sentence_id"
WORD_ID_COL = "word_id"
REG_COL = "regression_probability"
SKIP_COL = "skip_probability"

# Number of bins
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
    Partition x-axis into quantile bins, then compute mean values within each bin.
    """
    work_df = df.copy()

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
        .rename(columns={
            x_col: "ambiguity_zscore",
            "sentence_id": "n_sentences"
        })
    )

    return binned


# -----------------------------
# Save helper
# -----------------------------
def process_one_file(input_csv: str, ambiguity_df: pd.DataFrame, output_raw: str, output_binned: str):
    """
    Process one human/simulation csv and save both raw and binned ambiguity-effect files.
    """
    raw_df = compute_sentence_level_effects(input_csv, ambiguity_df)
    binned_df = bin_effects(raw_df, x_col="ambiguity_zscore", n_bins=N_BINS)

    out_dir = os.path.dirname(output_raw)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    raw_df.to_csv(output_raw, index=False)
    binned_df.to_csv(output_binned, index=False)

    print(f"  Saved raw:    {output_raw}")
    print(f"  Saved binned: {output_binned}")


# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading sentence ambiguity...")
    ambiguity_df = pd.read_csv(AMBIGUITY_PATH)

    # ---- human ----
    print("\nProcessing human data...")
    if not os.path.exists(HUMAN_PATH):
        raise FileNotFoundError(f"Human file not found: {HUMAN_PATH}")

    process_one_file(
        input_csv=HUMAN_PATH,
        ambiguity_df=ambiguity_df,
        output_raw=HUMAN_OUTPUT_RAW,
        output_binned=HUMAN_OUTPUT_BINNED,
    )

    # ---- simulations ----
    print("\nProcessing simulation subfolders...")

    if not os.path.isdir(SIM_ROOT):
        raise FileNotFoundError(f"Simulation root folder not found: {SIM_ROOT}")

    sim_subfolders = sorted([
        p for p in glob(os.path.join(SIM_ROOT, "*"))
        if os.path.isdir(p)
    ])

    if len(sim_subfolders) == 0:
        print(f"No simulation subfolders found in {SIM_ROOT}")
        return

    for subfolder in sim_subfolders:
        sim_input = os.path.join(subfolder, SIM_INPUT_FILENAME)

        if not os.path.exists(sim_input):
            print(f"Skipping {subfolder} (missing {SIM_INPUT_FILENAME})")
            continue

        sim_output_raw = os.path.join(subfolder, SIM_OUTPUT_RAW_FILENAME)
        sim_output_binned = os.path.join(subfolder, SIM_OUTPUT_BINNED_FILENAME)

        print(f"\nProcessing folder: {subfolder}")
        process_one_file(
            input_csv=sim_input,
            ambiguity_df=ambiguity_df,
            output_raw=sim_output_raw,
            output_binned=sim_output_binned,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()