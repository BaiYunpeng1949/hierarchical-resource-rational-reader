#!/usr/bin/env python3
"""
Parameter inference for LTM activation thresholds + unified panel plotting.

This script:
1) Sweeps (high_threshold, low_threshold) over grids to best-fit human targets.
2) Produces a single figure with TWO subplots side-by-side ("same panel"):
   - Left: grouped bar chart (Human vs Simulation) for 4 conditions.
   - Right: binned scatter (shown as hollow dots) + linear regression + 95% CI
            of Proportion Regressed vs Initial Appraisal Score (from raw sim JSON).

All visual styling follows the shared house style used elsewhere.

Outputs under --out_dir:
- grid_results.csv
- best_summary.txt
- best_pair.json
- panel_best_params_and_regression.png
- regression_stats_scatter.txt   (intercept, slope, R^2, n for the scatter regression)
"""

import argparse
import json
import os
import sys
import importlib.util
from dataclasses import dataclass, asdict
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec


# =========================
# House style (tweak here)
# =========================
HUMAN_COLOR = "#1f77b4"   # blue for human
SIM_COLOR   = "#2ca02c"   # green for simulation
CI_ALPHA    = 0.5         # confidence band alpha

# Line/marker styles for scatter/regression
LINE_WIDTH        = 2.0
REGRESSION_DASHED = True     # dashed regression line by default
REG_LINESTYLE     = "--" if REGRESSION_DASHED else "-"
SHOW_SCATTER      = True     # show *binned* dots (averaged)
SCATTER_SIZE      = 36       # matplotlib scatter size (points^2)
SCATTER_EDGEWIDTH = 1.0

# Font/size controls (adjust once here)
FONT_SIZE_BASE = 14
TICK_SIZE      = 12
LEGEND_SIZE    = 14

# Tick granularity (set to None to auto, or an int for max # major ticks)
MAX_X_TICKS = 6
MAX_Y_TICKS = 6

# ---- Binning controls for scatter ----
# Equal-width bins on appraisal score [0,1]; bins with zero count are dropped.
BIN_COUNT_SCATTER = 20

# ---- Per-axes sizing controls ----
# Keep consistent axes sizes for side-by-side panels
PANEL_AX_WIDTH_IN   = 5.0
PANEL_AX_HEIGHT_IN  = 3.0
SUBPLOT_WSPACE      = 0.05   # keep tiny; we’ll use an absolute spacer column
# Absolute widths for the middle area
LEGEND_WIDTH_IN     = 3.0   # width reserved for legend
LEGEND_SPACER_IN    = 0.1   # tiny spacer between legend and right plot
# Legend placement (in the dedicated legend column)
LEGEND_LOC         = "middle left"
LEGEND_ANCHOR_X    = 0.8    # 0 = left edge of legend column
LEGEND_ANCHOR_Y    = 1.0   # raise/lower legend: 0 (bottom) … 1 (top)

# Legend placement
LEGEND_LOC = "best"  # e.g., "best", "upper left", etc.

# ---------------- config (defaults) ----------------
DEFAULT_INPUT = "./assets/organized_example_propositions_v0527.json"
DEFAULT_OUT   = "./parameter_inference/ltm_threshold_grid"
DEFAULT_SIM_JSON = None  # set via --sim_json to raw_sim_results.json if you want the right subplot

# Human targets (override via CLI if needed)
DEFAULT_HUMAN = dict(
    highcoh_high=0.484,  # Fully coherent, high-knowledge
    highcoh_low =0.381,  # Fully coherent, low-knowledge
    lowcoh_high =0.417,  # Minimally coherent, high-knowledge
    lowcoh_low  =0.291,  # Minimally coherent, low-knowledge
)

# ---------------- helper: import calculate_proportional_recall ----------------
def import_calc_module(path_hint: str | None = None):
    """
    Import calculate_proportional_recall.py by module name or a provided path.
    Must provide functions:
      - load_propositions(json_path)
      - calculate_proportional_recall(propositions, high_threshold, low_threshold)
        -> tuple (fully_high, fully_low, minimal_high, minimal_low)
    """
    # 1) try normal import
    try:
        import calculate_proportional_recall as calc
        if hasattr(calc, "load_propositions") and hasattr(calc, "calculate_proportional_recall"):
            return calc
    except Exception:
        pass
    # 2) try explicit path(s)
    candidates = []
    if path_hint:
        candidates.append(path_hint)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "calculate_proportional_recall.py"))
    candidates.append(os.path.join(os.getcwd(), "calculate_proportional_recall.py"))
    for p in candidates:
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("calc_mod", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["calc_mod"] = mod
            spec.loader.exec_module(mod)  # type: ignore
            if hasattr(mod, "load_propositions") and hasattr(mod, "calculate_proportional_recall"):
                return mod
    raise ImportError(
        "Could not import 'calculate_proportional_recall'. "
        "Place it next to this script or pass --calc_path /path/to/calculate_proportional_recall.py"
    )

# ---------------- data + loss ----------------
@dataclass
class HumanTargets:
    highcoh_high: float
    highcoh_low: float
    lowcoh_high: float
    lowcoh_low: float

def frange(start: float, stop: float, step: float):
    """Inclusive float range."""
    if step <= 0:
        raise ValueError("step must be > 0")
    n = int(round((stop - start) / step))
    vals = [start + i * step for i in range(max(n, 0) + 1)]
    if abs(vals[-1] - stop) > 1e-9:
        vals.append(stop)
    return vals

def sse(a: float, b: float) -> float:
    return float((a - b) ** 2)

def mae(a: float, b: float) -> float:
    return float(abs(a - b))

def evaluate_pair(calc_mod, propositions, hi, lo, human, use_sse=False):
    """
    Returns (sim_four, loss, per_component_errors)
      sim_four = (fully_high, fully_low, minimal_high, minimal_low)
    """
    fch, fcl, mch, mcl = calc_mod.calculate_proportional_recall(
        propositions, high_threshold=hi, low_threshold=lo
    )
    err = sse if use_sse else mae
    comp = {
        "err_fully_high": err(fch, human.highcoh_high),
        "err_fully_low" : err(fcl, human.highcoh_low),
        "err_min_high"  : err(mch, human.lowcoh_high),
        "err_min_low"   : err(mcl, human.lowcoh_low),
    }
    loss = sum(comp.values())
    return (fch, fcl, mch, mcl), loss, comp

# ---------------- regression utils (scatter) ----------------
def linregress_basic(x, y):
    """Return (a, b, r2, sigma2, n) for y = a + b x"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan, n
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    Sxx = np.sum((x - x_mean)**2)
    Sxy = np.sum((x - x_mean)*(y - y_mean))
    b = Sxy / (Sxx if Sxx != 0 else 1e-12)
    a = y_mean - b * x_mean
    y_hat = a + b * x
    # R^2
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y_mean)**2) if np.any(y != y_mean) else 0.0
    r2 = 1.0 - ss_res / (ss_tot if ss_tot != 0 else 1e-12)
    # Residual variance
    dof = max(n - 2, 1)
    sigma2 = ss_res / dof
    return a, b, r2, sigma2, n

def regress_and_ci(x, y, x_smooth=None):
    """
    Linear regression y = a + b x with 95% CI for mean prediction.
    Returns x_line, y_hat, y_low, y_high, (a,b,r2,n)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    a, b, r2, sigma2, n = linregress_basic(x, y)

    if np.isnan(a):
        order = np.argsort(x)
        return x[order], y[order], None, None, (a, b, r2, len(x))

    if x_smooth is None:
        x_line = np.linspace(np.min(x), np.max(x), 200)
    else:
        x_line = np.asarray(x_smooth, dtype=float)

    y_hat_line = a + b * x_line

    # 95% CI for the mean prediction
    x_mean = np.mean(x)
    Sxx = np.sum((x - x_mean)**2)
    tcrit = 1.96
    with np.errstate(divide='ignore', invalid='ignore'):
        se_mean = np.sqrt(sigma2 * (1.0/len(x) + (x_line - x_mean)**2 / (Sxx if Sxx != 0 else 1e-12)))
    y_low = y_hat_line - tcrit * se_mean
    y_high = y_hat_line + tcrit * se_mean

    return x_line, y_hat_line, y_low, y_high, (a, b, r2, n)

def style_axes(ax):
    """Shared style: remove top/right spines, no grid/title, tick controls."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.set_title("")
    if MAX_X_TICKS is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=MAX_X_TICKS, prune=None))
    if MAX_Y_TICKS is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=MAX_Y_TICKS, prune=None))

def bin_proportion_regressed(all_appraisals, regressed_appraisals, n_bins=BIN_COUNT_SCATTER):
    """Return (bin_centers, proportions) over [0,1] with equal-width bins; drops zero-count bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    all_counts, _ = np.histogram(all_appraisals, bins=bins)
    regress_counts, _ = np.histogram(regressed_appraisals, bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2.0

    mask = all_counts > 0
    centers = centers[mask]
    props = np.zeros_like(centers, dtype=float)
    props[:] = (regress_counts[mask] / all_counts[mask])
    return centers, props

def load_appraisals_from_json(json_path):
    """Extract all valid initial appraisals and those that were regressed, from raw_sim_results.json"""
    with open(json_path, "r") as f:
        data = json.load(f)

    all_appraisals = []
    regressed_appraisals = []

    for episode in data:
        init_appraisals = episode.get("init_sentence_appraisal_scores_distribution", [])
        valid_appraisals = [score for score in init_appraisals if isinstance(score, (int, float)) and score >= 0]
        all_appraisals.extend(valid_appraisals)

        for step in episode.get("step_wise_log", []):
            if step.get("is_regress"):
                idx = step.get("actual_reading_sentence_index")
                if isinstance(idx, int) and 0 <= idx < len(init_appraisals):
                    score = init_appraisals[idx]
                    if isinstance(score, (int, float)) and score >= 0:
                        regressed_appraisals.append(score)

    return np.asarray(all_appraisals, dtype=float), np.asarray(regressed_appraisals, dtype=float)

# ---------------- plotting (bar + scatter) ----------------

# def plot_panel(human: 'HumanTargets', sim_four, scatter_x, scatter_y, out_png: str, out_stats_txt: str):
#     """
#     Two subplots:
#       Left: grouped bar chart (Human vs Sim) across coherence × knowledge.
#       Right: binned dots + regression + 95% CI.
#     """
#     plt.rcParams.update({'font.size': FONT_SIZE_BASE})
#     plt.rc('xtick', labelsize=TICK_SIZE)
#     plt.rc('ytick', labelsize=TICK_SIZE)

#     # fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(PANEL_AX_WIDTH_IN*2, PANEL_AX_HEIGHT_IN))
#     # fig.subplots_adjust(wspace=SUBPLOT_WSPACE)

#     # Figure width = left panel + GAP + right panel
#     fig = plt.figure(figsize=(PANEL_AX_WIDTH_IN*2 + LEGEND_WIDTH_IN + LEGEND_SPACER_IN, PANEL_AX_HEIGHT_IN))

#     # 3-column layout: [bar] [gap] [scatter]
#     gs = gridspec.GridSpec(
#         ncols=4, nrows=1, figure=fig,
#         width_ratios=[PANEL_AX_WIDTH_IN, LEGEND_WIDTH_IN, LEGEND_SPACER_IN, PANEL_AX_WIDTH_IN],
#         wspace=SUBPLOT_WSPACE
#     )

#     ax_bar     = fig.add_subplot(gs[0, 0])
#     ax_legend  = fig.add_subplot(gs[0, 1])  # dedicated legend column
#     ax_pad     = fig.add_subplot(gs[0, 2])  # tiny spacer to keep legend off the right plot
#     ax_scatter = fig.add_subplot(gs[0, 3])

#     # Hide the middle axes visuals
#     ax_legend.axis("off")
#     ax_pad.axis("off")

#     # --- Left: grouped bar ---
#     fch, fcl, mch, mcl = sim_four
#     Hh, Hl, Mh, Ml = human.highcoh_high, human.highcoh_low, human.lowcoh_high, human.lowcoh_low

#     bar_width = 0.18
#     x_groups = np.arange(2)  # 0: High coherence, 1: Low coherence
#     r1 = x_groups
#     r2 = r1 + bar_width
#     r3 = r2 + bar_width
#     r4 = r3 + bar_width

#     ax_bar.bar(r1, [Hh, Mh], width=bar_width, label="Human (High-K)", color=HUMAN_COLOR, hatch="/")
#     ax_bar.bar(r2, [fch, mch], width=bar_width, label="Sim (High-K)",   color=SIM_COLOR,   hatch="/")
#     ax_bar.bar(r3, [Hl, Ml], width=bar_width, label="Human (Low-K)",  color=HUMAN_COLOR, hatch=".")
#     ax_bar.bar(r4, [fcl, mcl], width=bar_width, label="Sim (Low-K)",    color=SIM_COLOR,   hatch=".")

#     # Style first, then set custom ticks so the locator doesn’t override them
#     style_axes(ax_bar)
#     ax_bar.set_xlabel("Text Coherence Level")
#     ax_bar.set_ylabel("Proportional Recall")

#     # Put labels *under* the two clusters (center = r1 + 1.5*bar_width), remove tick marks
#     group_centers = r1 + 1.5 * bar_width
#     ax_bar.set_xticks(group_centers)
#     ax_bar.set_xticklabels(["High Coherence", "Low Coherence"])
#     ax_bar.tick_params(axis="x", length=0, labelsize=12)

#     # Keep legend handles to place a figure-level legend in the gap
#     handles, labels = ax_bar.get_legend_handles_labels()
#     bylabel = dict(zip(labels, handles))

#     # place legend anchored to the *left* of the legend column
#     # ax_legend.legend(bylabel.values(), bylabel.keys(), loc="center left", frameon=False, fontsize=LEGEND_SIZE)
#     # ax_gap.legend(handles, labels, loc="center", frameon=False, fontsize=LEGEND_SIZE)
#     ax_legend.legend(
#         bylabel.values(), bylabel.keys(),
#         loc=LEGEND_LOC,
#         bbox_to_anchor=(LEGEND_ANCHOR_X, LEGEND_ANCHOR_Y),  # <— raise via Y
#         frameon=False,
#         fontsize=LEGEND_SIZE
#     )

#     # --- Right: scatter (binned) + regression + CI ---
#     scatter_stats = None
#     if scatter_x is not None and scatter_y is not None and scatter_x.size > 0 and scatter_y.size > 0:
#         x_line, y_hat, y_low, y_high, stats = regress_and_ci(scatter_x, scatter_y)
#         a, b, r2, n = stats
#         scatter_stats = (a, b, r2, n)

#         if y_low is not None and y_high is not None:
#             ax_scatter.fill_between(x_line, y_low, y_high, color=SIM_COLOR, alpha=CI_ALPHA)

#         ax_scatter.plot(x_line, y_hat, REG_LINESTYLE, linewidth=LINE_WIDTH, color=SIM_COLOR, label="Simulation")
#         if SHOW_SCATTER:
#             ax_scatter.scatter(scatter_x, scatter_y, s=SCATTER_SIZE, facecolor='none',
#                                edgecolor=SIM_COLOR, linewidth=SCATTER_EDGEWIDTH)
#                             #    , label="Simulation (binned)")

#     ax_scatter.set_xlabel("Initial Appraisal Score")
#     ax_scatter.set_ylabel("Proportion Regressed")
#     style_axes(ax_scatter)
#     ax_scatter.legend(loc=LEGEND_LOC, fontsize=LEGEND_SIZE, frameon=False)

#     # # Figure-level legend centered in the *gap* between subplots
#     # pos_left  = ax_bar.get_position()
#     # pos_right = ax_scatter.get_position()
#     # mid_x = (pos_left.x1 + pos_right.x0) / 2.0
#     # # Put it slightly above the vertical center to reduce any overlap
#     # mid_y = 0.5 * (pos_left.y0 + pos_left.y1) + 0.1
#     # fig.legend(handles, labels, loc='center', bbox_to_anchor=(mid_x, mid_y),
#     #            frameon=False, fontsize=LEGEND_SIZE)

#     # --- Write stats file (bar data + scatter regression) ---
#     with open(out_stats_txt, "w", encoding="utf-8") as f:
#         f.write("section\tcoherence\tknowledge\tseries\tvalue\n")
#         f.write(f"bar\tHigh\tHigh-K\tHuman\t{Hh:.6f}\n")
#         f.write(f"bar\tHigh\tHigh-K\tSim\t{fch:.6f}\n")
#         f.write(f"bar\tHigh\tLow-K\tHuman\t{Hl:.6f}\n")
#         f.write(f"bar\tHigh\tLow-K\tSim\t{fcl:.6f}\n")
#         f.write(f"bar\tLow\tHigh-K\tHuman\t{Mh:.6f}\n")
#         f.write(f"bar\tLow\tHigh-K\tSim\t{mch:.6f}\n")
#         f.write(f"bar\tLow\tLow-K\tHuman\t{Ml:.6f}\n")
#         f.write(f"bar\tLow\tLow-K\tSim\t{mcl:.6f}\n")
#         f.write("section\tseries\tintercept\tslope\tr2\tn\n")
#         if scatter_stats is not None:
#             a, b, r2, n = scatter_stats
#             f.write(f"scatter_regression\tsimulation\t{a:.6f}\t{b:.6f}\t{r2:.6f}\t{n}\n")
#         else:
#             f.write("scatter_regression\tsimulation\tNA\tNA\tNA\t0\n")

#     fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
#     plt.close(fig)

# === Figure size constants (you can tune later) ===
BAR_FIG_WIDTH  = 3
BAR_FIG_HEIGHT = 3
SCATTER_FIG_WIDTH  = 3
SCATTER_FIG_HEIGHT = 3

# === Font / size constants (you can tune freely) ===
AX_LABEL_SIZE   = 12
AX_TICK_SIZE    = 12
AX_TEXT_SIZE    = 12        # for annotations like High-K / Low-K

def plot_panel(human: 'HumanTargets', sim_four, scatter_x, scatter_y,
               out_png: str, out_stats_txt: str):
    """
    Modified plotting:
        - No combined PNG panel
        - No legends
        - Outputs two PDFs:
            * <base>_bar.pdf
            * <base>_scatter.pdf
    """

    # Unpack values as before
    fch, fcl, mch, mcl = sim_four
    Hh, Hl, Mh, Ml = (human.highcoh_high,
                      human.highcoh_low,
                      human.lowcoh_high,
                      human.lowcoh_low)

    # ========= BAR FIGURE (PDF) =========
    fig_bar = plt.figure(figsize=(BAR_FIG_WIDTH, BAR_FIG_HEIGHT))
    ax = fig_bar.add_subplot(1, 1, 1)

    bar_width = 0.18
    x_groups = np.arange(2)
    r1 = x_groups
    r2 = r1 + bar_width
    r3 = r2 + bar_width
    r4 = r3 + bar_width

    # Bars
    ax.bar(r1, [Hh, Mh], width=bar_width, color=HUMAN_COLOR, hatch="/")
    ax.bar(r2, [fch, mch], width=bar_width, color=SIM_COLOR, hatch="/")
    ax.bar(r3, [Hl, Ml], width=bar_width, color=HUMAN_COLOR, hatch=".")
    ax.bar(r4, [fcl, mcl], width=bar_width, color=SIM_COLOR, hatch=".")

    # Axes styling
    style_axes(ax)
    ax.set_xlabel("Text Coherence Level", fontsize=AX_LABEL_SIZE)
    ax.set_ylabel("Proportional Recall", fontsize=AX_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=AX_TICK_SIZE)

    group_centers = r1 + 1.5 * bar_width
    ax.set_xticks(group_centers)
    ax.set_xticklabels(["High", "Low"])
    ax.tick_params(axis="x", length=0)

    # ---- NEW LEGEND (bottom-center, solid colors only) ----
    human_patch = mpatches.Patch(color=HUMAN_COLOR, label="Human")
    sim_patch   = mpatches.Patch(color=SIM_COLOR,   label="Simulation")
    ax.legend(handles=[human_patch, sim_patch],
          loc="lower center",
          bbox_to_anchor=(0.5, 0.10),   # moves the legend below the axes
          frameon=True,                  # white box
          facecolor="white",
          ncol=1,
          fontsize=AX_TICK_SIZE)
    
        # ======= High-K / Low-K text annotations =======

    # Find max height among the 4 bars
    ymax = max(Hh, fch, Hl, fcl, Mh, mch, Ml, mcl)

    # A bit of vertical padding above tallest bar
    y_text_high = fch + 0.01   # tune as you like
    y_text_low = fcl + 0.01   # tune as you like

    # x-positions: midpoint of each 2-bar cluster
    x_highk = 0.5 * (r1[0] + r2[0])
    x_lowk  = 0.55 * (r3[0] + r4[0])

    ax.text(x_highk, y_text_high, "High-K",
            ha="center", va="bottom",
            fontsize=AX_TEXT_SIZE)

    ax.text(x_lowk,  y_text_low, "Low-K",
            ha="center", va="bottom",
            fontsize=AX_TEXT_SIZE)

    # Expand y-limit so text is not clipped
    ax.set_ylim(0, max(y_text_high, y_text_low) + 0.05)

    # SAVE BAR PDF
    base, _ = os.path.splitext(out_png)
    bar_pdf = f"{base}_bar.pdf"
    fig_bar.savefig(bar_pdf, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig_bar)

    # ========= SCATTER FIGURE (PDF) =========

    fig_sc = plt.figure(figsize=(SCATTER_FIG_WIDTH, SCATTER_FIG_HEIGHT))
    ax_sc = fig_sc.add_subplot(1, 1, 1)

    # Regression line + scatter
    if scatter_x is not None and scatter_y is not None and len(scatter_x) > 0:
        x_line, y_hat, y_low, y_high, stats = regress_and_ci(scatter_x, scatter_y)
        if y_low is not None:
            ax_sc.fill_between(x_line, y_low, y_high, color=SIM_COLOR, alpha=0.2)
        ax_sc.plot(x_line, y_hat, color=SIM_COLOR, linewidth=2)
        if SHOW_SCATTER:
            ax_sc.scatter(scatter_x, scatter_y, s=SCATTER_SIZE,
                          facecolor='none', edgecolor=SIM_COLOR)

    ax_sc.set_xlabel("Initial Appraisal Score", fontsize=AX_LABEL_SIZE)
    ax_sc.set_ylabel("Proportion Recalled", fontsize=AX_LABEL_SIZE)
    style_axes(ax_sc)
    ax_sc.tick_params(axis="both", labelsize=AX_TICK_SIZE)

    # Save plots
    scatter_pdf = f"{base}_scatter.pdf"
    fig_sc.savefig(scatter_pdf, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig_sc)

    # Write stats unchanged
    with open(out_stats_txt, "w") as f:
        f.write("section\tcoherence\tknowledge\tseries\tvalue\n")
        f.write(f"bar\tHigh\tHigh-K\tHuman\t{Hh}\n")
        f.write(f"bar\tHigh\tHigh-K\tSim\t{fch}\n")
        f.write(f"bar\tHigh\tLow-K\tHuman\t{Hl}\n")
        f.write(f"bar\tHigh\tLow-K\tSim\t{fcl}\n")
        f.write(f"bar\tLow\tHigh-K\tHuman\t{Mh}\n")
        f.write(f"bar\tLow\tHigh-K\tSim\t{mch}\n")
        f.write(f"bar\tLow\tLow-K\tHuman\t{Ml}\n")
        f.write(f"bar\tLow\tLow-K\tSim\t{mcl}\n")


# ---------------- main ----------------

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Infer thresholds and plot bar + scatter (binned regression) in one figure.")
    ap.add_argument("--input_json", type=str, default=DEFAULT_INPUT, help="Path to organized propositions JSON.")
    ap.add_argument("--calc_path", type=str, default=None, help="Optional explicit path to calculate_proportional_recall.py")
    ap.add_argument("--high_range", type=float, nargs=3, default=[0.0, 1.0, 0.05], metavar=("START","END","STEP"))
    ap.add_argument("--low_range",  type=float, nargs=3, default=[0.0, 1.0, 0.05], metavar=("START","END","STEP"))
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT, help="Output directory for results.")
    ap.add_argument("--sse", action="store_true", help="Use SSE instead of MAE (default is MAE)."
    )
    # Human targets
    ap.add_argument("--human_highcoh_high", type=float, default=DEFAULT_HUMAN["highcoh_high"])
    ap.add_argument("--human_highcoh_low",  type=float, default=DEFAULT_HUMAN["highcoh_low"])
    ap.add_argument("--human_lowcoh_high",  type=float, default=DEFAULT_HUMAN["lowcoh_high"])
    ap.add_argument("--human_lowcoh_low",   type=float, default=DEFAULT_HUMAN["lowcoh_low"])

    # Optional JSON with raw sim logs to compute appraisal->regression scatter
    ap.add_argument("--sim_json", type=str, default=DEFAULT_SIM_JSON,
                    help="Path to raw_sim_results.json to build the right subplot (binned proportion regressed).")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Import & load propositions
    calc_mod = import_calc_module(args.calc_path)
    propositions = calc_mod.load_propositions(args.input_json)

    human = HumanTargets(
        highcoh_high=args.human_highcoh_high,
        highcoh_low=args.human_highcoh_low,
        lowcoh_high=args.human_lowcoh_high,
        lowcoh_low=args.human_lowcoh_low,
    )

    his = frange(*args.high_range)
    los = frange(*args.low_range)

    rows = []
    best = None  # (loss, idx, sim_four)

    for hi in his:
        for lo in los:
            sim_four, loss, comp = evaluate_pair(calc_mod, propositions, hi, lo, human, use_sse=args.sse)
            row = {
                "high_threshold": hi,
                "low_threshold": lo,
                "sim_fully_high": sim_four[0],
                "sim_fully_low":  sim_four[1],
                "sim_min_high":   sim_four[2],
                "sim_min_low":    sim_four[3],
                **comp,
                "loss_total": loss,
            }
            rows.append(row)
            if best is None or loss < best[0]:
                best = (loss, len(rows)-1, sim_four)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "grid_results.csv")
    df.to_csv(csv_path, index=False)

    assert best is not None
    best_row = rows[best[1]]
    best_sim_four = best[2]

    # write summaries
    best_txt = os.path.join(args.out_dir, "best_summary.txt")
    with open(best_txt, "w", encoding="utf-8") as f:
        f.write(
            "Best thresholds (objective = {})\n".format("SSE" if args.sse else "MAE")
            + f"High threshold: {best_row['high_threshold']:.3f}\n"
            + f"Low  threshold: {best_row['low_threshold']:.3f}\n\n"
            + "Simulated means:\n"
            + f"  Fully Coherent - High knowledge: {best_row['sim_fully_high']:.3f}\n"
            + f"  Fully Coherent - Low  knowledge: {best_row['sim_fully_low']:.3f}\n"
            + f"  Minimal Coherent - High knowledge: {best_row['sim_min_high']:.3f}\n"
            + f"  Minimal Coherent - Low  knowledge: {best_row['sim_min_low']:.3f}\n\n"
            + "Human targets:\n"
            + f"  Fully Coherent - High knowledge: {human.highcoh_high:.3f}\n"
            + f"  Fully Coherent - Low  knowledge: {human.highcoh_low:.3f}\n"
            + f"  Minimal Coherent - High knowledge: {human.lowcoh_high:.3f}\n"
            + f"  Minimal Coherent - Low  knowledge: {human.lowcoh_low:.3f}\n\n"
            + "Per-component errors:\n"
            + f"  err_fully_high: {best_row['err_fully_high']:.6f}\n"
            + f"  err_fully_low : {best_row['err_fully_low']:.6f}\n"
            + f"  err_min_high  : {best_row['err_min_high']:.6f}\n"
            + f"  err_min_low   : {best_row['err_min_low']:.6f}\n\n"
            + f"Total loss: {best_row['loss_total']:.6f}\n"
        )

    best_json = os.path.join(args.out_dir, "best_pair.json")
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(
            {"objective": ("SSE" if args.sse else "MAE"),
             "human": asdict(human),
             "best": best_row},
            f, indent=2
        )

    # Build right subplot data from sim JSON (if provided)
    scatter_centers = None
    scatter_props = None
    if args.sim_json:
        if os.path.exists(args.sim_json):
            all_app, reg_app = load_appraisals_from_json(args.sim_json)
            centers, props = bin_proportion_regressed(all_app, reg_app, n_bins=BIN_COUNT_SCATTER)
            scatter_centers, scatter_props = centers, props
        else:
            print(f"[WARN] --sim_json not found: {args.sim_json}. Right subplot will be empty.")

    # Plot combined panel
    out_png = os.path.join(args.out_dir, "panel_best_params_and_regression.png")
    out_stats = os.path.join(args.out_dir, "plot_stats.txt")
    plot_panel(human, best_sim_four, scatter_centers, scatter_props, out_png, out_stats)

    print(f"\nSaved grid CSV: {csv_path}")
    print(f"Saved best summary: {best_txt}")
    print(f"Saved best json: {best_json}")
    print(f"Saved panel figure: {out_png}")
    print(f"Saved scatter regression stats: {out_stats}")
    print("\nBest thresholds: high={:.3f}, low={:.3f}, loss={:.6f}".format(
        best_row['high_threshold'], best_row['low_threshold'], best_row['loss_total'])
    )

if __name__ == "__main__":
    main()
