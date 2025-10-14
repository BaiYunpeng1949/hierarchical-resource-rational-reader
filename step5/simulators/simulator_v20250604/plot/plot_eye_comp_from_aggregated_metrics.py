#!/usr/bin/env python3
# Plot 1x5 panel from aggregated_panel_metrics.json with absolute gaps + easy legend control
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

# ==== House style (tweak here) ====
HUMAN_COLOR = "#1f77b4"
SIM_COLOR   = "#2ca02c"

FONT_SIZE_BASE = 14
TICK_SIZE      = 12
LEGEND_SIZE    = 14

BAR_GROUP_WIDTH = 0.80
BAR_CAPSIZE     = 3
BAR_LINEWIDTH   = 1.0

# ---- Per-axes size (inches) ----
PANEL_AX_WIDTH_IN   = 3
PANEL_AX_HEIGHT_IN  = 3

# ---- Absolute gaps (inches) between panels ----
# 4 gaps for 5 panels; these DO NOT shrink the panels
GAP_INCHES = 1

# ---- Legend placement ----
LEGEND_SUBPLOT_INDEX = 0
LEGEND_LOC = "lower left"          # bottom-left corner of the legend box
LEGEND_BBOX_TO_ANCHOR = (0.02, 0.02)  # x,y in axes coords

# ---- Legend box style ----
LEGEND_FRAME_ON  = True
LEGEND_FACECOLOR = "white"
LEGEND_EDGEcolor = "black"   # or "none" to hide the border
LEGEND_ALPHA     = 0.9       # 0=transparent, 1=opaque
LEGEND_BORDERPAD = 0.4       # inner padding
LEGEND_LABELSP   = 0.3       # spacing between entries


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
    bar_w = BAR_GROUP_WIDTH / 2.0
    offsets_h = [c - BAR_GROUP_WIDTH/2 + 0.5*bar_w for c in centers]
    offsets_s = [c - BAR_GROUP_WIDTH/2 + 1.5*bar_w for c in centers]

    rects_h = ax.bar(offsets_h, means_h, bar_w,
                     yerr=stds_h, capsize=BAR_CAPSIZE,
                     edgecolor=HUMAN_COLOR, linewidth=BAR_LINEWIDTH, color=HUMAN_COLOR,
                     label="Human")
    rects_s = ax.bar(offsets_s, means_s, bar_w,
                     yerr=stds_s, capsize=BAR_CAPSIZE,
                     edgecolor=SIM_COLOR, linewidth=BAR_LINEWIDTH, color=SIM_COLOR,
                     label="Simulation")

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
        ("reading_speed", "Reading speed (WPM)"),
        ("skip_rate", "Skip rate"),
        ("regression_rate", "Regression rate"),
        ("mcq_accuracy", "MCQ accuracy"),
        ("free_recall_score", "Free recall"),
    ]

    _set_fonts()

    # Figure width = 5*axes_width + 4*gaps
    fig_w = 5*PANEL_AX_WIDTH_IN + 4*GAP_INCHES
    fig_h = PANEL_AX_HEIGHT_IN
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Build a 1x9 GridSpec: [ax gap ax gap ax gap ax gap ax]
    widths = []
    slots = []
    for i in range(5):
        widths.append(PANEL_AX_WIDTH_IN)
        slots.append("ax")
        if i < 4:
            widths.append(GAP_INCHES)
            slots.append("gap")

    gs = gridspec.GridSpec(
        nrows=1, ncols=len(widths), figure=fig,
        width_ratios=widths, wspace=0.0
    )

    axes = []
    col = 0
    for s in slots:
        if s == "ax":
            axes.append(fig.add_subplot(gs[0, col]))
        else:
            ax_gap = fig.add_subplot(gs[0, col])
            ax_gap.axis("off")
        col += 1

    centers = list(range(len(conditions)))
    legend_handles = None

    for i, (metric_key, ylabel) in enumerate(metrics):
        ax = axes[i]
        h_means = [data["human"][metric_key][c]["mean"] for c in conditions]
        h_stds  = [data["human"][metric_key][c]["std"]  for c in conditions]
        s_means = [data["simulation"][metric_key][c]["mean"] for c in conditions]
        s_stds  = [data["simulation"][metric_key][c]["std"]  for c in conditions]

        rects_h, rects_s = _bar_group(ax, centers, h_means, h_stds, s_means, s_stds,
                                      ylabel=ylabel, xlabels=conditions)
        if legend_handles is None:
            legend_handles = [rects_h[0], rects_s[0]]

    # Legend in chosen subplot, positioned by constants above
    axL = axes[LEGEND_SUBPLOT_INDEX]
    axL.legend(
        legend_handles, ["Human", "Simulation"],
        loc=LEGEND_LOC,
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        frameon=LEGEND_FRAME_ON,
        facecolor=LEGEND_FACECOLOR,
        edgecolor=LEGEND_EDGEcolor,
        framealpha=LEGEND_ALPHA,
        borderpad=LEGEND_BORDERPAD,
        labelspacing=LEGEND_LABELSP,
        fontsize=LEGEND_SIZE,
    )

    out = Path("eye_comp_panel_from_aggregated_abs_gap.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("Saved:", out)

if __name__ == "__main__":
    main()
