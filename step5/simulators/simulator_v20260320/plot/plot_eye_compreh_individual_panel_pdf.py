#!/usr/bin/env python3
"""
Plot individual 1x1 panels (one per metric) from aggregated_panel_metrics.json.

- Each metric is saved as its own PDF figure.
- All figures have the same panel size (PANEL_AX_WIDTH_IN x PANEL_AX_HEIGHT_IN).
- No legends are drawn (color coding is implicit: blue=Human, green=Simulation).
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math
import numpy as np
import pandas as pd
from scipy import stats

# ==== House style (tweak here) ====
HUMAN_COLOR = "#1f77b4"
SIM_COLOR   = "#2ca02c"

FONT_SIZE_BASE = 12
TICK_SIZE      = 12

BAR_GROUP_WIDTH = 0.80
BAR_CAPSIZE     = 3
BAR_LINEWIDTH   = 1.0

# ---- Per-axes size (inches) ----
PANEL_AX_WIDTH_IN   = 3
PANEL_AX_HEIGHT_IN  = 3


def _set_fonts():
    plt.rcParams.update({'font.size': FONT_SIZE_BASE})
    plt.rc('xtick', labelsize=TICK_SIZE)
    plt.rc('ytick', labelsize=TICK_SIZE)


def _style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.set_title("")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def _bar_group(ax, centers, means_h, stds_h, means_s, stds_s, ylabel=None, xlabels=None):
    """
    Draw side-by-side bars for human vs simulation.
    Returns the two bar containers (for possible future legend use, though we omit legends here).
    """
    bar_w = BAR_GROUP_WIDTH / 2.0
    offsets_h = [c - BAR_GROUP_WIDTH / 2 + 0.5 * bar_w for c in centers]
    offsets_s = [c - BAR_GROUP_WIDTH / 2 + 1.5 * bar_w for c in centers]

    rects_h = ax.bar(
        offsets_h, means_h, bar_w,
        yerr=stds_h, capsize=BAR_CAPSIZE,
        edgecolor=HUMAN_COLOR, linewidth=BAR_LINEWIDTH, color=HUMAN_COLOR,
        label="Human",
    )
    rects_s = ax.bar(
        offsets_s, means_s, bar_w,
        yerr=stds_s, capsize=BAR_CAPSIZE,
        edgecolor=SIM_COLOR, linewidth=BAR_LINEWIDTH, color=SIM_COLOR,
        label="Simulation",
    )

    if ylabel:
        ax.set_ylabel(ylabel)

    _style_axes(ax)
    if xlabels:
        ax.set_xticks(centers)
        ax.set_xticklabels(xlabels)
    else:
        ax.set_xticks([])

    return rects_h, rects_s


def format_p(p):
    if pd.isna(p):
        return "N/A"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def linear_trend_from_summary(means, stds, ns):
    """
    Summary-statistics linear trend test across three equally spaced conditions:
    30 s, 60 s, 90 s.

    This uses the linear contrast [-1, 0, 1].
    It is valid only if the n values correspond to independent observations
    behind each condition mean.
    """

    if any(n <= 1 for n in ns):
        return None

    k = 3
    N = sum(ns)
    df = N - k

    if df <= 0:
        return None

    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    ns = np.asarray(ns, dtype=float)

    contrast = np.asarray([-1.0, 0.0, 1.0])

    # Pooled within-condition variance
    mse = np.sum((ns - 1) * stds ** 2) / df

    # Linear contrast: 90 s minus 30 s
    L = np.sum(contrast * means)
    se_L = math.sqrt(mse * np.sum((contrast ** 2) / ns))

    t_value = L / se_L
    p_value = 2 * stats.t.sf(abs(t_value), df)

    # Slope per 30-s step: because contrast [-1,0,1], beta = L / 2
    beta = L / 2.0
    se_beta = se_L / 2.0
    tcrit = stats.t.ppf(0.975, df)
    ci_low = beta - tcrit * se_beta
    ci_high = beta + tcrit * se_beta

    return {
        "beta_per_30s_step": beta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t": t_value,
        "df": int(df),
        "p": p_value,
        "p_formatted": format_p(p_value),
    }


def compute_model_fit(data, metrics, conditions):
    """
    Compare human and simulation condition means across all plotted values:
    5 metrics x 3 time conditions = 15 values.

    Because the metrics have different units, this also reports correlation.
    MAE/RMSE are computed on raw scales, so interpret them descriptively.
    """

    human_vals = []
    sim_vals = []

    for metric_key, _ in metrics:
        for condition in conditions:
            human_vals.append(data["human"][metric_key][condition]["mean"])
            sim_vals.append(data["simulation"][metric_key][condition]["mean"])

    human_vals = np.asarray(human_vals, dtype=float)
    sim_vals = np.asarray(sim_vals, dtype=float)

    errors = sim_vals - human_vals

    return {
        "analysis": "human_simulation_fit_all_15_means",
        "dataset": "Human vs. simulation",
        "outcome": "All plotted metrics",
        "test": "descriptive model fit",
        "n": len(human_vals),
        "beta_per_30s_step": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "t": np.nan,
        "df": np.nan,
        "p": np.nan,
        "p_formatted": "N/A",
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "r": float(np.corrcoef(human_vals, sim_vals)[0, 1]),
    }


def compute_time_pressure_stats(data, metrics, conditions, human_eye_n_override=None):
    rows = []

    for dataset in ["human", "simulation"]:
        for metric_key, ylabel in metrics:
            means = [data[dataset][metric_key][c]["mean"] for c in conditions]
            stds = [data[dataset][metric_key][c]["std"] for c in conditions]
            ns = [data[dataset][metric_key][c]["n"] for c in conditions]

            # Your aggregate file currently has n = 1 for human eye metrics.
            # If the SDs are participant-level SDs, set human_eye_n_override=39.
            if (
                dataset == "human"
                and metric_key in ["reading_speed", "skip_rate", "regression_rate"]
                and human_eye_n_override is not None
            ):
                ns = [human_eye_n_override] * len(conditions)

            result = linear_trend_from_summary(means, stds, ns)

            if result is None:
                rows.append({
                    "analysis": "linear_time_trend",
                    "dataset": dataset,
                    "outcome": metric_key,
                    "test": "not computed; n <= 1 in aggregated file",
                    "n": "/".join(map(str, ns)),
                    "mean_30s": means[0],
                    "mean_60s": means[1],
                    "mean_90s": means[2],
                    "beta_per_30s_step": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "t": np.nan,
                    "df": np.nan,
                    "p": np.nan,
                    "p_formatted": "N/A",
                    "mae": np.nan,
                    "rmse": np.nan,
                    "r": np.nan,
                })
            else:
                rows.append({
                    "analysis": "linear_time_trend",
                    "dataset": dataset,
                    "outcome": metric_key,
                    "test": "summary-statistics linear trend test",
                    "n": "/".join(map(str, ns)),
                    "mean_30s": means[0],
                    "mean_60s": means[1],
                    "mean_90s": means[2],
                    **result,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "r": np.nan,
                })

    rows.append(compute_model_fit(data, metrics, conditions))

    return pd.DataFrame(rows)


def main():
    data = json.loads(Path("assets/aggregated_panel_metrics.json").read_text(encoding="utf-8"))
    conditions = data["conditions"]
    pretty_labels = [c.replace("s", " s") for c in conditions]  
    metrics = [
        ("reading_speed", "Reading speed (WPM)"),
        ("skip_rate", "Skip rate"),
        ("regression_rate", "Regression rate"),
        ("mcq_accuracy", "MCQ accuracy"),
        ("free_recall_score", "Free recall"),
    ]

    _set_fonts()

    centers = list(range(len(conditions)))

    # Create one figure per metric, all with identical size
    for metric_key, ylabel in metrics:
        fig = plt.figure(figsize=(PANEL_AX_WIDTH_IN, PANEL_AX_HEIGHT_IN))
        ax = fig.add_subplot(1, 1, 1)

        h_means = [data["human"][metric_key][c]["mean"] for c in conditions]
        h_stds  = [data["human"][metric_key][c]["std"]  for c in conditions]
        s_means = [data["simulation"][metric_key][c]["mean"] for c in conditions]
        s_stds  = [data["simulation"][metric_key][c]["std"]  for c in conditions]

        _bar_group(ax, centers, h_means, h_stds, s_means, s_stds,
                   ylabel=ylabel, xlabels=pretty_labels)

        # === X-axis label control for consistent axes size ===
        if metric_key == "reading_speed":
            ax.set_xlabel("Time Constraints", fontsize=FONT_SIZE_BASE)
        else:
            # Reserve identical x-label space without showing text
            ax.set_xlabel(" ", fontsize=FONT_SIZE_BASE)

        # No legend: color mapping is implicit (blue=Human, green=Simulation)

        out = Path(f"eye_comp_{metric_key}.pdf")
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print("Saved:", out)
    
    # === Inferential / summary statistics ===
    # Use 39 only if the human eye-movement SDs are participant-level SDs.
    stats_df = compute_time_pressure_stats(
        data,
        metrics,
        conditions,
        human_eye_n_override=39
    )

    stats_out = Path("time_pressure_inferential_stats.csv")
    stats_df.to_csv(stats_out, index=False)
    print("Saved:", stats_out)


if __name__ == "__main__":
    main()
