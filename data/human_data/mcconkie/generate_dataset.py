import pandas as pd
from pathlib import Path

# =========================================================
# Config
# =========================================================
INPUT_CSV = "digitilized_data/proportion_of_fixations_vs_letter_positions.csv"
OUTPUT_CSV = "mcconkie.csv"

# =========================================================
# Load raw digitized CSV
# =========================================================
df = pd.read_csv(INPUT_CSV)

# The digitized file is arranged as paired columns:
# [length 3 X, length 3 Y, length 4 X, length 4 Y, ...]
# with the first row containing X/Y labels.
results = []

cols = list(df.columns)

for i in range(0, len(cols), 2):
    x_col = cols[i]
    y_col = cols[i + 1]

    # Parse word length from column name like "length 3 dataset"
    # -> 3
    try:
        word_length = int(str(x_col).split()[1])
    except Exception as e:
        raise ValueError(f"Could not parse word length from column name: {x_col}") from e

    # Extract the two columns
    sub = df[[x_col, y_col]].copy()
    sub.columns = ["x_raw", "y_raw"]

    # Remove non-numeric rows such as the first row: X / Y
    sub["x_raw"] = pd.to_numeric(sub["x_raw"], errors="coerce")
    sub["y_raw"] = pd.to_numeric(sub["y_raw"], errors="coerce")
    sub = sub.dropna(subset=["x_raw", "y_raw"]).copy()

    # Round x values to nearest integer letter index
    # examples: 1.02 -> 1, 3.04 -> 3
    sub["letter_number"] = sub["x_raw"].round().astype(int)

    # Keep requested output columns
    sub["word_length"] = word_length
    sub["proportion_of_fixation"] = sub["y_raw"]

    sub = sub[["word_length", "letter_number", "proportion_of_fixation"]]

    results.append(sub)

# Concatenate all word lengths
out = pd.concat(results, ignore_index=True)

# Optional: sort nicely
out = out.sort_values(["word_length", "letter_number"]).reset_index(drop=True)

# Save
out.to_csv(OUTPUT_CSV, index=False)

print(f"Saved cleaned dataset to: {OUTPUT_CSV}")
print(out.head(20))