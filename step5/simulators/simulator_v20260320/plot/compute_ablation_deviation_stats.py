"""
Compute model--human deviation statistics for ablation/baseline models.

Inputs:
    assets/aggregated_panel_metrics.json
    assets/aggregated_panel_metrics_baseline.json

Output:
    ablation_model_human_deviation_stats.csv

This compares each simulation/baseline model against human data across:
    5 metrics x 3 time constraints = 15 condition means.

Metrics:
    - raw MAE / RMSE
    - standardized MAE / RMSE
    - raw Pearson correlation
    - standardized pattern correlation

Standardized errors are computed by dividing each model--human difference by
the corresponding human SD for that metric and time condition.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------

AGG_MAIN_PATH = Path("assets/aggregated_panel_metrics.json")
AGG_BASE_PATH = Path("assets/aggregated_panel_metrics_baseline.json")
OUT_CSV = Path("ablation_model_human_deviation_stats.csv")


# ---------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------

METRICS = [
    ("reading_speed", "Reading speed"),
    ("skip_rate", "Skip rate"),
    ("regression_rate", "Regression rate"),
    ("mcq_accuracy", "MCQ accuracy"),
    ("free_recall_score", "Free recall"),
]

PRETTY_MODEL_NAMES = {
    "full_hierarchical_model": "Full hierarchical model",
    "full_memory": "Unlimited-memory model",
    "sentence_reader_gamma_0dot2": r"Myopic sentence reader ($\gamma=0.2$)",
    "sentence_reader_gamma_0dot6": r"Myopic sentence reader ($\gamma=0.6$)",
    "text_reader_gamma_0dot2": r"Myopic text reader ($\gamma=0.2$)",
    "text_reader_gamma_0dot6": r"Myopic text reader ($\gamma=0.6$)",
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def safe_corr(x, y):
    """
    Pearson correlation with protection against zero-variance vectors.
    Returns np.nan when correlation is undefined.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return np.nan

    if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def get_human_vectors(data_main, conditions):
    """
    Extract human means and human SDs across all metric x condition cells.

    Returns:
        human_means: array of length 15
        human_sds: array of length 15
        labels: list of strings describing each cell
    """
    human_means = []
    human_sds = []
    labels = []

    for metric_key, metric_label in METRICS:
        for condition in conditions:
            cell = data_main["human"][metric_key][condition]
            human_means.append(float(cell["mean"]))
            human_sds.append(float(cell["std"]))
            labels.append(f"{metric_key}_{condition}")

    return np.array(human_means), np.array(human_sds), labels


def get_model_vector_from_main(data_main, conditions, model_key="simulation"):
    """
    Extract the full model vector from aggregated_panel_metrics.json.
    """
    vals = []

    for metric_key, _ in METRICS:
        for condition in conditions:
            vals.append(float(data_main[model_key][metric_key][condition]["mean"]))

    return np.array(vals)


def get_model_vector_from_baseline(data_base, conditions, variant_key):
    """
    Extract a baseline model vector from aggregated_panel_metrics_baseline.json.
    """
    vals = []

    for metric_key, _ in METRICS:
        for condition in conditions:
            vals.append(float(data_base["baselines"][variant_key][metric_key][condition]["mean"]))

    return np.array(vals)


def compute_deviation_stats(model_name, model_vector, human_vector, human_sds):
    """
    Compute raw and standardized deviation statistics.
    """
    raw_error = model_vector - human_vector

    raw_mae = float(np.mean(np.abs(raw_error)))
    raw_rmse = float(np.sqrt(np.mean(raw_error ** 2)))
    raw_r = safe_corr(human_vector, model_vector)

    # Avoid division by zero in standardization.
    # If any human SD is zero, replace it with NaN so it is ignored.
    human_sds_safe = np.where(np.asarray(human_sds) > 0, human_sds, np.nan)

    human_z = human_vector / human_sds_safe
    model_z = model_vector / human_sds_safe
    standardized_error = raw_error / human_sds_safe

    std_mae = float(np.nanmean(np.abs(standardized_error)))
    std_rmse = float(np.sqrt(np.nanmean(standardized_error ** 2)))
    std_r = safe_corr(human_z[~np.isnan(human_z) & ~np.isnan(model_z)],
                      model_z[~np.isnan(human_z) & ~np.isnan(model_z)])

    return {
        "model": model_name,
        "raw_mae": raw_mae,
        "raw_rmse": raw_rmse,
        "raw_pattern_r": raw_r,
        "standardized_mae": std_mae,
        "standardized_rmse": std_rmse,
        "standardized_pattern_r": std_r,
    }


def main():
    data_main = json.loads(AGG_MAIN_PATH.read_text(encoding="utf-8"))
    data_base = json.loads(AGG_BASE_PATH.read_text(encoding="utf-8"))

    conditions = data_main["conditions"]

    human_vector, human_sds, labels = get_human_vectors(data_main, conditions)

    rows = []

    # 1. Full hierarchical model from main aggregate file
    full_model_vector = get_model_vector_from_main(
        data_main,
        conditions,
        model_key="simulation"
    )

    rows.append(
        compute_deviation_stats(
            model_name=PRETTY_MODEL_NAMES["full_hierarchical_model"],
            model_vector=full_model_vector,
            human_vector=human_vector,
            human_sds=human_sds,
        )
    )

    # 2. Baseline / ablation models
    for variant_key in data_base["meta"]["variants"]:
        model_vector = get_model_vector_from_baseline(
            data_base,
            conditions,
            variant_key
        )

        rows.append(
            compute_deviation_stats(
                model_name=PRETTY_MODEL_NAMES.get(variant_key, variant_key),
                model_vector=model_vector,
                human_vector=human_vector,
                human_sds=human_sds,
            )
        )

    df = pd.DataFrame(rows)

    # Order columns
    df = df[
        [
            "model",
            "raw_mae",
            "raw_rmse",
            "raw_pattern_r",
            "standardized_mae",
            "standardized_rmse",
            "standardized_pattern_r",
        ]
    ]

    # Sort by standardized RMSE: lower = closer to human
    df = df.sort_values("standardized_rmse", ascending=True)

    df.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}")
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
