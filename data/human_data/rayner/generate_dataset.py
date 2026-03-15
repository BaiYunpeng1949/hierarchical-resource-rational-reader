import os
import re
import pandas as pd

INPUT_DIR = "digitilized_data"

OUT_FORWARD_MULTIPLE = "rayner_forward_fixations_multiple.csv"
OUT_FORWARD_SINGLE = "rayner_forward_fixations_single_only.csv"
OUT_REG_BOTH = "rayner_regressions_both.csv"
OUT_REG_INTER = "rayner_regressions_interword_only.csv"


def infer_word_length(filename: str) -> int:
    m = re.search(r"length(\d+)", filename)
    if not m:
        raise ValueError(f"Cannot infer word length from filename: {filename}")
    return int(m.group(1))


def clean_value_column(series: pd.Series) -> pd.Series:
    """
    Convert a value column to numeric and drop non-numeric rows like 'Y'.
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna().reset_index(drop=True)
    return s


def extract_two_series(filepath: str):
    """
    Each file actually has 4 columns:
      [series_name_1, values_1, series_name_2, values_2]

    The first row is usually: Label, Y, Label, Y
    """
    df = pd.read_csv(filepath)
    cols = list(df.columns)

    if len(cols) < 4:
        raise ValueError(
            f"Expected 4 columns in {os.path.basename(filepath)}, got {len(cols)}: {cols}"
        )

    series_name_1 = str(cols[0]).strip().lower()
    value_col_1 = cols[1]

    series_name_2 = str(cols[2]).strip().lower()
    value_col_2 = cols[3]

    values_1 = clean_value_column(df[value_col_1])
    values_2 = clean_value_column(df[value_col_2])

    return [
        (series_name_1, values_1),
        (series_name_2, values_2),
    ]


def to_long_df(word_length: int, values: pd.Series, value_name: str) -> pd.DataFrame:
    """
    Letter positions should NOT start from 1.
    They should either start from 0, or the 'Y' row should be removed.
    We remove the 'Y' row and start at 0.
    """
    return pd.DataFrame({
        "word_length": word_length,
        "letter_number": range(len(values)),
        value_name: values.values
    })


forward_multiple_all = []
forward_single_all = []
reg_both_all = []
reg_inter_all = []


for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".csv"):
        continue

    path = os.path.join(INPUT_DIR, fname)
    word_length = infer_word_length(fname)
    series_list = extract_two_series(path)

    for series_name, values in series_list:
        name = series_name.lower()

        # Forward fixation files
        if "fixation" in fname.lower():
            if "one" in name or "single" in name:
                forward_single_all.append(
                    to_long_df(word_length, values, "proportion_of_fixation")
                )
            elif "more" in name or "multiple" in name:
                forward_multiple_all.append(
                    to_long_df(word_length, values, "proportion_of_fixation")
                )
            else:
                raise ValueError(
                    f"Could not classify fixation series '{series_name}' in file {fname}"
                )

        # Regression files
        elif "regression" in fname.lower():
            if "interword" in name:
                reg_inter_all.append(
                    to_long_df(word_length, values, "probability_of_regression")
                )
            elif "both" in name:
                reg_both_all.append(
                    to_long_df(word_length, values, "probability_of_regression")
                )
            else:
                raise ValueError(
                    f"Could not classify regression series '{series_name}' in file {fname}"
                )


# Merge and save
if forward_multiple_all:
    df_forward_multiple = pd.concat(forward_multiple_all, ignore_index=True) \
        .sort_values(["word_length", "letter_number"]) \
        .reset_index(drop=True)
    df_forward_multiple.to_csv(OUT_FORWARD_MULTIPLE, index=False)

if forward_single_all:
    df_forward_single = pd.concat(forward_single_all, ignore_index=True) \
        .sort_values(["word_length", "letter_number"]) \
        .reset_index(drop=True)
    df_forward_single.to_csv(OUT_FORWARD_SINGLE, index=False)

if reg_both_all:
    df_reg_both = pd.concat(reg_both_all, ignore_index=True) \
        .sort_values(["word_length", "letter_number"]) \
        .reset_index(drop=True)
    df_reg_both.to_csv(OUT_REG_BOTH, index=False)

if reg_inter_all:
    df_reg_inter = pd.concat(reg_inter_all, ignore_index=True) \
        .sort_values(["word_length", "letter_number"]) \
        .reset_index(drop=True)
    df_reg_inter.to_csv(OUT_REG_INTER, index=False)

print("Generated files:")
for f in [OUT_FORWARD_MULTIPLE, OUT_FORWARD_SINGLE, OUT_REG_BOTH, OUT_REG_INTER]:
    if os.path.exists(f):
        print(" -", f)