#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_eye_comp_comparison_baselines_clipped_pdf.py

Same as the original script, but exports a single PDF figure:
    comparison_panel_baselines_clipped.pdf
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

# ================== House style ==================
HUMAN_COLOR = "#1f77b4"     # blue
SIM_COLOR   = "#2ca02c"     # green

BASELINE_FILL   = "#bdbdbd" # grey fill
BASELINE_EDGE   = SIM_COLOR # green edge
BASELINE_HATCHES = ["/", "\\", ".", "xx", "oo", "++", "--"]  # cycle textures

# For bounded metrics, draw a tiny bar when mean==0 so the outline/hatch is visible
MIN_VISIBLE_BAR = 0.005   # in [0,1] units; set None to disable

FONT_SIZE_BASE = 12
TICK_SIZE      = 12
LEGEND_SIZE    = 12

BAR_GROUP_WIDTH = 0.80
BAR_LINEWIDTH   = 1.2
BAR_CAPSIZE     = 3

# Per-axes size (inches)
AX_W_IN = 5.0
AX_H_IN = 3.0

# Absolute gaps (inches)
H_GAP_IN = 0.8   # horizontal gap between columns
V_GAP_IN = 0.8   # vertical gap between rows

# Legend target panel (we'll put legend in bottom-right blank slot)
LEGEND_LOC = "upper left"
LEGEND_BBOX = (0.0, 1.0)    # anchor inside the legend panel

# Bounded-metric handling
CLIP_BOUNDED_ERR = True
BOUNDED_METRICS = {"skip_rate", "regression_rate", "mcq_accuracy", "free_recall_score"}

# Files
AGG_MAIN_PATH = Path("assets/aggregated_panel_metrics.json")
AGG_BASE_PATH = Path("assets/aggregated_panel_metrics_baseline.json")
OUT_PDF       = Path("comparison_panel_baselines_clipped.pdf")

# ================== Helpers ==================
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

def bounded_yerr(means, stds, lo=0.0, hi=1.0):
    """Asymmetric yerr clipped to [lo, hi]."""
    m = np.array(means, dtype=float)
    s = np.array(stds, dtype=float)
    lower = np.minimum(s, m - lo)   # don't extend below lo
    upper = np.minimum(s, hi - m)   # don't extend above hi
    lower = np.clip(lower, 0, None)
    upper = np.clip(upper, 0, None)
    return [lower, upper]

def _prepare_handles_for_legend(baseline_labels, hatch_cycle):
    import matplotlib.patches as mpatches
    h_human = mpatches.Patch(facecolor=HUMAN_COLOR, edgecolor=HUMAN_COLOR, label="Human")
    h_sim   = mpatches.Patch(facecolor=SIM_COLOR,   edgecolor=SIM_COLOR, label="Simulation")
    baseline_handles = []
    for i, lab in enumerate(baseline_labels):
        hatch = hatch_cycle[i % len(hatch_cycle)]
        baseline_handles.append(
            mpatches.Patch(facecolor=BASELINE_FILL, edgecolor=BASELINE_EDGE,
                           hatch=hatch, linewidth=BAR_LINEWIDTH, label=lab)
        )
    return [h_human, h_sim] + baseline_handles

def _bar_cluster(ax, centers, series, colors, edges, hatches,
                 ylab=None, xlabels=None, metric_key=None):
    n_series = len(series)
    bar_w = BAR_GROUP_WIDTH / n_series

    is_bounded = (metric_key in BOUNDED_METRICS)

    for i, y in enumerate(series):
        offs = [c - BAR_GROUP_WIDTH/2 + (i + 0.5)*bar_w for c in centers]

        if is_bounded and CLIP_BOUNDED_ERR:
            yerr = bounded_yerr(y["mean"], y.get("std", [0]*len(centers)), lo=0.0, hi=1.0)
        else:
            yerr = y.get("std")

        vals = np.array(y["mean"], dtype=float)
        if is_bounded and MIN_VISIBLE_BAR is not None:
            vals_plot = np.where(vals == 0.0, MIN_VISIBLE_BAR, vals)
        else:
            vals_plot = vals

        ax.bar(
            offs, vals_plot, bar_w,
            yerr=yerr, capsize=BAR_CAPSIZE,
            facecolor=colors[i], edgecolor=edges[i],
            linewidth=BAR_LINEWIDTH, hatch=hatches[i]
        )

    if ylab:
        ax.set_ylabel(ylab)

    _style_axes(ax)
    if is_bounded:
        ax.set_ylim(0.0, 1.0)

    if xlabels:
        ax.set_xticks(centers)
        ax.set_xticklabels(xlabels)
    else:
        ax.set_xticks([])

def _extract_metric(d, group, metric_key, conditions):
    return {
        "mean": [d[group][metric_key][c]["mean"] for c in conditions],
        "std":  [d[group][metric_key][c]["std"]  for c in conditions],
    }

# ================== Main ==================
def main():
    data_main = json.loads(AGG_MAIN_PATH.read_text(encoding="utf-8"))
    data_base = json.loads(AGG_BASE_PATH.read_text(encoding="utf-8"))

    conditions = data_main["conditions"]
    centers    = list(range(len(conditions)))

    row1 = [
        ("reading_speed", "Reading Speed (WPM)"),
        ("skip_rate", "Skip Rate"),
        ("regression_rate", "Regression Rate"),
    ]
    row2 = [
        ("mcq_accuracy", "MCQ Accuracy"),
        ("free_recall_score", "Free Recall"),
    ]

    variants = data_base["meta"]["variants"]
    pretty = {
        "full_memory": "Sim with unlimited memory",
        "text_reader_gamma_0dot2": "Sim myopic text reader (\u03B3=0.2)",
        "text_reader_gamma_0dot6": "Sim myopic text reader (\u03B3=0.6)",
        "sentence_reader_gamma_0dot2": "Sim myopic sentence reader (\u03B3=0.2)",
        "sentence_reader_gamma_0dot6": "Sim myopic sentence reader (\u03B3=0.6)",
    }
    baseline_labels = [pretty.get(v, v.replace("_", " ")) for v in variants]

    # Figure with absolute spacer columns/rows
    fig_w = 3 * AX_W_IN + 2 * H_GAP_IN
    fig_h = 2 * AX_H_IN + 1 * V_GAP_IN
    fig = plt.figure(figsize=(fig_w, fig_h))

    widths  = [AX_W_IN, H_GAP_IN, AX_W_IN, H_GAP_IN, AX_W_IN]
    heights = [AX_H_IN, V_GAP_IN, AX_H_IN]
    gs = gridspec.GridSpec(
        nrows=3,
        ncols=5,
        figure=fig,
        width_ratios=widths,
        height_ratios=heights,
        wspace=0.0,
        hspace=0.0,
    )

    # Row 1 axes
    ax_speed = fig.add_subplot(gs[0, 0]); fig.add_subplot(gs[0, 1]).axis("off")
    ax_skip  = fig.add_subplot(gs[0, 2]); fig.add_subplot(gs[0, 3]).axis("off")
    ax_regr  = fig.add_subplot(gs[0, 4])
    # Row 2 axes
    ax_mcq   = fig.add_subplot(gs[2, 0]); fig.add_subplot(gs[2, 1]).axis("off")
    ax_free  = fig.add_subplot(gs[2, 2]); fig.add_subplot(gs[2, 3]).axis("off")
    ax_legend= fig.add_subplot(gs[2, 4]); ax_legend.set_axis_off(); ax_legend.set_facecolor("none")

    def series_for(metric_key):
        series = []
        colors = []
        edges  = []
        hatches = []

        # Human
        series.append(_extract_metric(data_main, "human", metric_key, conditions))
        colors.append(HUMAN_COLOR); edges.append(HUMAN_COLOR); hatches.append(None)

        # Main model
        series.append(_extract_metric(data_main, "simulation", metric_key, conditions))
        colors.append(SIM_COLOR); edges.append(SIM_COLOR); hatches.append(None)

        # Baselines
        for i, v in enumerate(variants):
            s = {
                "mean": [data_base["baselines"][v][metric_key][c]["mean"] for c in conditions],
                "std":  [data_base["baselines"][v][metric_key][c]["std"]  for c in conditions],
            }
            series.append(s)
            colors.append(BASELINE_FILL)
            edges.append(BASELINE_EDGE)
            hatches.append(BASELINE_HATCHES[i % len(BASELINE_HATCHES)])

        return series, colors, edges, hatches

    # Plot row 1
    for ax, (metric, ylab) in zip([ax_speed, ax_skip, ax_regr], row1):
        series, colors, edges, hatches = series_for(metric)
        _bar_cluster(ax, centers, series, colors, edges, hatches,
                     ylab=ylab, xlabels=conditions, metric_key=metric)

    # Plot row 2
    for ax, (metric, ylab) in zip([ax_mcq, ax_free], row2):
        series, colors, edges, hatches = series_for(metric)
        _bar_cluster(ax, centers, series, colors, edges, hatches,
                     ylab=ylab, xlabels=conditions, metric_key=metric)

    # Legend in bottom-right blank panel
    handles = _prepare_handles_for_legend(baseline_labels, BASELINE_HATCHES)
    labels  = [h.get_label() for h in handles]
    ax_legend.legend(
        handles, labels,
        loc=LEGEND_LOC, bbox_to_anchor=LEGEND_BBOX,
        frameon=True, facecolor="white", edgecolor="black",
        framealpha=0.9, fontsize=LEGEND_SIZE,
        borderpad=0.4, labelspacing=0.3,
    )

    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {OUT_PDF}")

if __name__ == "__main__":
    _set_fonts()
    main()
