#!/usr/bin/env python3
# Plot five separate dot+mean±SD panels from aggregated_panel_metrics.json
# Also writes statistical summaries from the same point-wise values.
# Keeps the original house style, axes size, colours, font sizes, and output format.

import json
from pathlib import Path
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from scipy import stats

# ==== House style (kept from original) ====
HUMAN_COLOR = "#1f77b4"
SIM_COLOR   = "#2ca02c"

FONT_SIZE_BASE = 14
TICK_SIZE      = 12
LEGEND_SIZE    = 14

BAR_GROUP_WIDTH = 0.80
BAR_CAPSIZE     = 3
BAR_LINEWIDTH   = 1.0

# ---- Per-axes size (inches), kept from original ----
PANEL_AX_WIDTH_IN   = 3
PANEL_AX_HEIGHT_IN  = 3

# ---- Point/summary style ----
POINT_SIZE = 20
POINT_ALPHA = 0.30
MEAN_MARKER_SIZE = 8
JITTER_WIDTH = 0.12
RNG_SEED = 7


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


def _load_aggregated_json():
    """
    Prefer the original project path, but allow running this script from a
    directory where aggregated_panel_metrics.json is directly present.
    """
    candidates = [
        Path("assets/aggregated_panel_metrics.json"),
        Path("aggregated_panel_metrics.json"),
        Path(__file__).with_name("aggregated_panel_metrics.json"),
    ]
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError("Could not find aggregated_panel_metrics.json")


def _group_x_positions(centers):
    """
    Keep the same grouping logic as the original bar plot:
    two side-by-side groups under each time condition.
    """
    group_w = BAR_GROUP_WIDTH
    half_w = group_w / 2.0
    x_h = np.array([c - group_w/2 + 0.5*half_w for c in centers], dtype=float)
    x_s = np.array([c - group_w/2 + 1.5*half_w for c in centers], dtype=float)
    return x_h, x_s


def _jittered_x(x, n, width, rng):
    if n <= 1:
        return np.array([x], dtype=float)
    return x + rng.uniform(-width, width, size=n)


def _clean_values(block):
    arr = np.array(block.get("values", []), dtype=float)
    return arr[np.isfinite(arr)]


def _dot_mean_sd_group(ax, centers, metric_key, data, ylabel=None, xlabels=None):
    rng = np.random.default_rng(RNG_SEED)
    x_h, x_s = _group_x_positions(centers)

    # Draw individual points first: filled colour points
    for i, cond in enumerate(data["conditions"]):
        h_block = data["human"][metric_key][cond]
        s_block = data["simulation"][metric_key][cond]

        h_values = _clean_values(h_block)
        s_values = _clean_values(s_block)

        if h_values.size:
            ax.scatter(
                _jittered_x(x_h[i], h_values.size, JITTER_WIDTH, rng),
                h_values,
                s=POINT_SIZE,
                facecolors=HUMAN_COLOR,
                edgecolors=HUMAN_COLOR,
                linewidths=0.5,
                alpha=POINT_ALPHA,
                zorder=2,
            )

        if s_values.size:
            ax.scatter(
                _jittered_x(x_s[i], s_values.size, JITTER_WIDTH, rng),
                s_values,
                s=POINT_SIZE,
                facecolors=SIM_COLOR,
                edgecolors=SIM_COLOR,
                linewidths=0.5,
                alpha=POINT_ALPHA,
                zorder=2,
            )

    # Draw mean ± SD on top
    h_means = [data["human"][metric_key][c]["mean"] for c in data["conditions"]]
    h_stds  = [data["human"][metric_key][c]["std"]  for c in data["conditions"]]
    s_means = [data["simulation"][metric_key][c]["mean"] for c in data["conditions"]]
    s_stds  = [data["simulation"][metric_key][c]["std"]  for c in data["conditions"]]

    ax.errorbar(
        x_h, h_means, yerr=h_stds,
        fmt="o",
        markersize=MEAN_MARKER_SIZE,
        markerfacecolor=HUMAN_COLOR,
        markeredgecolor=HUMAN_COLOR,
        markeredgewidth=BAR_LINEWIDTH,
        ecolor=HUMAN_COLOR,
        elinewidth=BAR_LINEWIDTH,
        capsize=BAR_CAPSIZE,
        label="Human",
        zorder=4,
    )

    ax.errorbar(
        x_s, s_means, yerr=s_stds,
        fmt="o",
        markersize=MEAN_MARKER_SIZE,
        markerfacecolor=SIM_COLOR,
        markeredgecolor=SIM_COLOR,
        markeredgewidth=BAR_LINEWIDTH,
        ecolor=SIM_COLOR,
        elinewidth=BAR_LINEWIDTH,
        capsize=BAR_CAPSIZE,
        label="Simulation",
        zorder=4,
    )

    if ylabel:
        ax.set_ylabel(ylabel)

    _style_axes(ax)

    if xlabels:
        ax.set_xticks(centers)
        ax.set_xticklabels(xlabels)
    else:
        ax.set_xticks([])

    # Keep enough side margin for jittered side-by-side points
    ax.set_xlim(min(centers) - 0.55, max(centers) + 0.55)


def _make_legend(out_name="eye_comp_legend.pdf"):
    fig = plt.figure(figsize=(PANEL_AX_WIDTH_IN, 1.05))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="-",
            color=HUMAN_COLOR,
            markerfacecolor=HUMAN_COLOR,
            markeredgecolor=HUMAN_COLOR,
            markeredgewidth=BAR_LINEWIDTH,
            markersize=MEAN_MARKER_SIZE,
            linewidth=BAR_LINEWIDTH,
            label="Human"
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="-",
            color=SIM_COLOR,
            markerfacecolor=SIM_COLOR,
            markeredgecolor=SIM_COLOR,
            markeredgewidth=BAR_LINEWIDTH,
            markersize=MEAN_MARKER_SIZE,
            linewidth=BAR_LINEWIDTH,
            label="Simulation"
        ),
    ]

    ax.legend(
        handles=handles,
        loc="center",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0.9,
        borderpad=0.4,
        labelspacing=0.3,
        fontsize=LEGEND_SIZE,
    )

    fig.savefig(out_name, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _eta_squared(groups):
    clean = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    all_values = np.concatenate(clean)
    grand_mean = np.mean(all_values)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in clean)
    ss_total = np.sum((all_values - grand_mean) ** 2)
    return float(ss_between / ss_total) if ss_total > 0 else float("nan")


def _cohens_d_independent(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1 = np.var(a, ddof=1)
    s2 = np.var(b, ddof=1)
    sp = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else float("nan")


def _write_statistics(data, metrics, out_name="time_pressure_inferential_stats.csv"):
    """
    Write descriptive statistics, one-way ANOVA across time conditions within
    each source, and pairwise Welch t-tests between time conditions.

    All tests use the same point-wise values that are plotted.
    """
    conditions = data["conditions"]
    rows = []

    for metric_key, ylabel, _ in metrics:
        for source in ["human", "simulation"]:
            groups = []
            for cond in conditions:
                values = _clean_values(data[source][metric_key][cond])
                groups.append(values)
                rows.append({
                    "section": "descriptive",
                    "metric": metric_key,
                    "metric_label": ylabel,
                    "source": source,
                    "condition": cond,
                    "comparison": "",
                    "n": int(values.size),
                    "mean": float(np.mean(values)) if values.size else float("nan"),
                    "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "sem": float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0,
                    "test": "",
                    "df1": "",
                    "df2": "",
                    "statistic": "",
                    "p": "",
                    "effect_size": "",
                    "effect_size_name": "",
                })

            # One-way ANOVA across the three time conditions
            if len(groups) >= 2 and all(len(g) > 1 for g in groups):
                f_stat, p_val = stats.f_oneway(*groups)
                df1 = len(groups) - 1
                df2 = sum(len(g) for g in groups) - len(groups)
                eta2 = _eta_squared(groups)
                rows.append({
                    "section": "inferential",
                    "metric": metric_key,
                    "metric_label": ylabel,
                    "source": source,
                    "condition": "",
                    "comparison": "30s_vs_60s_vs_90s",
                    "n": int(sum(len(g) for g in groups)),
                    "mean": "",
                    "sd": "",
                    "sem": "",
                    "test": "one_way_anova",
                    "df1": int(df1),
                    "df2": int(df2),
                    "statistic": float(f_stat),
                    "p": float(p_val),
                    "effect_size": eta2,
                    "effect_size_name": "eta_squared",
                })

            # Pairwise Welch t-tests between time conditions
            for i in range(len(conditions)):
                for j in range(i + 1, len(conditions)):
                    a = groups[i]
                    b = groups[j]
                    if len(a) > 1 and len(b) > 1:
                        t_res = stats.ttest_ind(a, b, equal_var=False)
                        d = _cohens_d_independent(a, b)
                        rows.append({
                            "section": "inferential",
                            "metric": metric_key,
                            "metric_label": ylabel,
                            "source": source,
                            "condition": "",
                            "comparison": f"{conditions[i]}_vs_{conditions[j]}",
                            "n": int(len(a) + len(b)),
                            "mean": "",
                            "sd": "",
                            "sem": "",
                            "test": "welch_t_test",
                            "df1": "",
                            "df2": float(t_res.df) if hasattr(t_res, "df") else "",
                            "statistic": float(t_res.statistic),
                            "p": float(t_res.pvalue),
                            "effect_size": d,
                            "effect_size_name": "cohens_d",
                        })

    fieldnames = [
        "section", "metric", "metric_label", "source", "condition",
        "comparison", "n", "mean", "sd", "sem",
        "test", "df1", "df2", "statistic", "p",
        "effect_size", "effect_size_name"
    ]

    with open(out_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    data = _load_aggregated_json()
    conditions = data["conditions"]
    pretty_conditions = [c.replace("s", " s") for c in conditions]

    metrics = [
        ("reading_speed", "Reading speed (WPM)", "eye_comp_reading_speed.pdf"),
        ("skip_rate", "Skip rate", "eye_comp_skip_rate.pdf"),
        ("regression_rate", "Regression rate", "eye_comp_regression_rate.pdf"),
        ("mcq_accuracy", "MCQ accuracy", "eye_comp_mcq_accuracy.pdf"),
        ("free_recall_score", "Free recall", "eye_comp_free_recall_score.pdf"),
    ]

    _set_fonts()
    centers = list(range(len(conditions)))

    for metric_key, ylabel, out_name in metrics:
        fig = plt.figure(figsize=(PANEL_AX_WIDTH_IN, PANEL_AX_HEIGHT_IN))
        ax = fig.add_subplot(1, 1, 1)

        _dot_mean_sd_group(
            ax,
            centers=centers,
            metric_key=metric_key,
            data=data,
            ylabel=ylabel,
            xlabels=pretty_conditions,
        )

        fig.savefig(out_name, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print("Saved:", out_name)

    _make_legend("eye_comp_legend.pdf")
    print("Saved:", "eye_comp_legend.pdf")

    _write_statistics(data, metrics, "time_pressure_inferential_stats.csv")
    print("Saved:", "time_pressure_inferential_stats.csv")


if __name__ == "__main__":
    main()
