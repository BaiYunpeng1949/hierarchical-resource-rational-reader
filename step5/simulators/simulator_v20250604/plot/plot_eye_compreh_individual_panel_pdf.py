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


def main():
    data = json.loads(Path("assets/aggregated_panel_metrics.json").read_text(encoding="utf-8"))
    conditions = data["conditions"]
    metrics = [
        ("reading_speed", "Reading Speed (WPM)"),
        ("skip_rate", "Skip Rate"),
        ("regression_rate", "Regression Rate"),
        ("mcq_accuracy", "MCQ Accuracy"),
        ("free_recall_score", "Free Recall"),
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
                   ylabel=ylabel, xlabels=conditions)

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


if __name__ == "__main__":
    main()
