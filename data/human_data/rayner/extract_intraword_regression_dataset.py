import pandas as pd
import numpy as np

INPUT_BOTH = "rayner_regressions_both.csv"
INPUT_INTER = "rayner_regressions_interword_only.csv"
OUTPUT_INTRA = "rayner_regressions_intraword_only.csv"

VALUE_COL = "probability_of_regression"
EPS = 1e-15

df_both = pd.read_csv(INPUT_BOTH)
df_inter = pd.read_csv(INPUT_INTER)

required = {"word_length", "letter_number", VALUE_COL}
if not required.issubset(df_both.columns):
    raise ValueError(f"{INPUT_BOTH} missing required columns: {required}")
if not required.issubset(df_inter.columns):
    raise ValueError(f"{INPUT_INTER} missing required columns: {required}")

df = pd.merge(
    df_both,
    df_inter,
    on=["word_length", "letter_number"],
    how="outer",
    suffixes=("_both", "_inter")
).fillna(0.0)

df["intraword_raw"] = df[f"{VALUE_COL}_both"] - df[f"{VALUE_COL}_inter"]
df["intraword_clipped"] = df["intraword_raw"].clip(lower=0.0)

normalized_groups = []

for word_length, group in df.groupby("word_length", sort=True):
    group = group.copy()
    total = group["intraword_clipped"].sum()

    if total <= EPS:
        group[VALUE_COL] = 0.0
    else:
        group[VALUE_COL] = group["intraword_clipped"] / total

        corrected_sum = group[VALUE_COL].sum()
        diff = 1.0 - corrected_sum

        # absorb residual numerical error into the last bin
        if abs(diff) > 0:
            last_idx = group.index[-1]
            group.loc[last_idx, VALUE_COL] += diff

    normalized_groups.append(group)

df_out = pd.concat(normalized_groups, ignore_index=True)
df_out = df_out[["word_length", "letter_number", VALUE_COL]].sort_values(
    ["word_length", "letter_number"]
).reset_index(drop=True)

df_out.to_csv(OUTPUT_INTRA, index=False)

print(f"Saved: {OUTPUT_INTRA}")
print("\nCheck sums by word length:")
print(df_out.groupby("word_length")[VALUE_COL].sum())