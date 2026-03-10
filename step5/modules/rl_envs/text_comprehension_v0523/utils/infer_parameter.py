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

def extract_sentence_level_records_from_json(json_path):
    """
    From raw_sim_results.json, extract one record per sentence instance:
      - appraisal score
      - whether that sentence was ever regressed to

    Returns a DataFrame with columns:
      appraisal, regressed
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []

    for episode in data:
        init_appraisals = episode.get("init_sentence_appraisal_scores_distribution", [])
        if not isinstance(init_appraisals, list):
            continue

        # collect sentence indices that received at least one regression
        regressed_sentence_indices = set()
        for step in episode.get("step_wise_log", []):
            if step.get("is_regress"):
                idx = step.get("actual_reading_sentence_index")
                if isinstance(idx, int) and 0 <= idx < len(init_appraisals):
                    score = init_appraisals[idx]
                    if isinstance(score, (int, float)) and score >= 0:
                        regressed_sentence_indices.add(idx)

        # one row per valid sentence appraisal
        for idx, score in enumerate(init_appraisals):
            if isinstance(score, (int, float)) and score >= 0:
                rows.append({
                    "appraisal": float(score),
                    "regressed": int(idx in regressed_sentence_indices),
                })

    return pd.DataFrame(rows)


def classify_ambiguity_from_appraisal(df, split_method="median", threshold=0.5):
    """
    Convert continuous appraisal into binary ambiguity labels.

    split_method:
      - 'median': ambiguous = appraisal < median, unambiguous = appraisal >= median
      - 'fixed' : ambiguous = appraisal < threshold, unambiguous = appraisal >= threshold
    """
    df = df.copy()

    if df.empty:
        df["ambiguity"] = pd.Series(dtype=str)
        return df, np.nan

    if split_method == "median":
        split_value = float(df["appraisal"].median())
    elif split_method == "fixed":
        split_value = float(threshold)
    else:
        raise ValueError(f"Unknown split_method: {split_method}")

    df["ambiguity"] = np.where(
        df["appraisal"] < split_value,
        "Ambiguous",
        "Unambiguous"
    )

    return df, split_value


def compute_regression_probabilities_binary(df):
    """
    Compute P(regression | ambiguity class).

    Returns:
      {
        "Ambiguous": {"prob": ..., "n_total": ..., "n_regressed": ...},
        "Unambiguous": {"prob": ..., "n_total": ..., "n_regressed": ...},
      }
    """
    out = {}

    for label in ["Ambiguous", "Unambiguous"]:
        sub = df[df["ambiguity"] == label]
        n_total = int(len(sub))
        n_regressed = int(sub["regressed"].sum())
        prob = (n_regressed / n_total) if n_total > 0 else np.nan

        out[label] = {
            "prob": float(prob),
            "n_total": n_total,
            "n_regressed": n_regressed,
        }

    return out


def load_aligned_binary_probs_from_json(json_path, split_method="median", threshold=0.5):
    """
    Full pipeline:
      raw sim json -> sentence records -> binary ambiguity -> regression probabilities
    """
    df = extract_sentence_level_records_from_json(json_path)
    df, split_value = classify_ambiguity_from_appraisal(
        df,
        split_method=split_method,
        threshold=threshold
    )
    probs = compute_regression_probabilities_binary(df)
    return df, probs, split_value


# === Figure size constants (you can tune later) ===
BAR_FIG_WIDTH  = 3
BAR_FIG_HEIGHT = 3
SCATTER_FIG_WIDTH  = 3
SCATTER_FIG_HEIGHT = 3

# === Font / size constants (you can tune freely) ===
AX_LABEL_SIZE   = 12
AX_TICK_SIZE    = 12
AX_TEXT_SIZE    = 12        # for annotations like High-K / Low-K


# def plot_panel(human: 'HumanTargets', sim_four, scatter_x, scatter_y,
#                out_png: str, out_stats_txt: str):

#     import numpy as np
#     import matplotlib.pyplot as plt

#     # -----------------------------
#     # Human benchmark (Staub & Clifton 2006)
#     # -----------------------------
#     human_x = np.array([0, 1], dtype=float)
#     human_y = np.array([0.19, 0.068], dtype=float)

#     fig = plt.figure(figsize=(SCATTER_FIG_WIDTH, SCATTER_FIG_HEIGHT))
#     gs = fig.add_gridspec(
#         nrows=2,
#         ncols=1,
#         height_ratios=[0.9, 1.1],
#         hspace=0.30
#     )

#     ax_top = fig.add_subplot(gs[0, 0])
#     ax_bot = fig.add_subplot(gs[1, 0])

#     # =============================
#     # Top panel — Human benchmark
#     # =============================
#     ax_top.plot(
#         human_x, human_y,
#         color=HUMAN_COLOR,
#         linewidth=LINE_WIDTH,
#         linestyle=REG_LINESTYLE
#     )

#     ax_top.scatter(
#         human_x, human_y,
#         s=SCATTER_SIZE,
#         facecolor="none",
#         edgecolor=HUMAN_COLOR,
#         linewidth=SCATTER_EDGEWIDTH,
#         zorder=3
#     )

#     style_axes(ax_top)

#     ax_top.set_ylabel("Regression\nprobability", fontsize=AX_LABEL_SIZE, linespacing=0.9)
#     ax_top.set_xticks([0, 1])
#     ax_top.set_xticklabels(["Ambiguous", "Unambiguous"], fontsize=AX_TICK_SIZE)

#     ax_top.set_xlim(-0.15, 1.15)
#     ax_top.set_ylim(0.0, 0.22)

#     # sparse Y ticks with two decimals
#     ax_top.set_yticks([0.00, 0.10, 0.20])
#     ax_top.set_yticklabels(["0.00", "0.10", "0.20"], fontsize=AX_TICK_SIZE, linespacing=0.9)

#     ax_top.tick_params(axis="x", labelsize=AX_TICK_SIZE)

#     # =============================
#     # Bottom panel — Simulation
#     # =============================
#     if scatter_x is not None and scatter_y is not None and len(scatter_x) > 0:

#         x_line, y_hat, y_low, y_high, stats = regress_and_ci(scatter_x, scatter_y)

#         if y_low is not None:
#             ax_bot.fill_between(
#                 x_line, y_low, y_high,
#                 color=SIM_COLOR, alpha=0.2
#             )

#         ax_bot.plot(
#             x_line, y_hat,
#             color=SIM_COLOR,
#             linewidth=LINE_WIDTH,
#             linestyle=REG_LINESTYLE
#         )

#         if SHOW_SCATTER:
#             ax_bot.scatter(
#                 scatter_x, scatter_y,
#                 s=SCATTER_SIZE,
#                 facecolor="none",
#                 edgecolor=SIM_COLOR,
#                 linewidth=SCATTER_EDGEWIDTH,
#                 zorder=3
#             )
#     else:
#         stats = (np.nan, np.nan, np.nan, 0)

#     style_axes(ax_bot)

#     ax_bot.set_xlabel("Initial appraisal score", fontsize=AX_LABEL_SIZE)
#     ax_bot.set_ylabel("Rereading\nprobability", fontsize=AX_LABEL_SIZE)

#     ax_bot.set_xlim(0.0, 1.0)

#     # sparse X ticks (5)
#     ax_bot.set_xticks([0.0, 0.30, 0.60, 0.90])

#     # fixed Y ticks for rereading probability
#     ax_bot.set_yticks([0.00, 0.35, 0.70])
#     ax_bot.set_yticklabels(["0.00", "0.35", "0.70"], fontsize=AX_TICK_SIZE)

#     ax_bot.tick_params(axis="x", labelsize=AX_TICK_SIZE)

#     # -----------------------------
#     # Save
#     # -----------------------------
#     base, _ = os.path.splitext(out_png)

#     stacked_pdf = f"{base}_stacked_human_simulation.pdf"
#     stacked_png = f"{base}_stacked_human_simulation.png"

#     fig.savefig(stacked_pdf, dpi=300, bbox_inches="tight", pad_inches=0.05)
#     fig.savefig(stacked_png, dpi=300, bbox_inches="tight", pad_inches=0.05)

#     plt.close(fig)

#     # -----------------------------
#     # Save stats
#     # -----------------------------
#     with open(out_stats_txt, "w") as f:

#         f.write("section\tseries\tx\ty\n")

#         f.write(f"human\tambiguous\t0\t{human_y[0]}\n")
#         f.write(f"human\tunambiguous\t1\t{human_y[1]}\n")

#         if scatter_x is not None and scatter_y is not None and len(scatter_x) > 0:

#             for x, y in zip(scatter_x, scatter_y):
#                 f.write(f"simulation_binned\tscatter\t{x}\t{y}\n")

#             a, b, r2, n = stats

#             f.write("\n")
#             f.write(f"simulation_regression_intercept\t{a}\n")
#             f.write(f"simulation_regression_slope\t{b}\n")
#             f.write(f"simulation_regression_r2\t{r2}\n")
#             f.write(f"simulation_regression_n\t{n}\n")


def plot_panel(human_probs, sim_probs, out_png: str, out_stats_txt: str):
    """
    Unified aligned plot:
      x-axis: Ambiguous / Unambiguous
      y-axis: Regression probability
      blue: human
      green: simulation
    """

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.array([0, 1], dtype=float)
    x_human = x - 0.03
    x_sim   = x + 0.03

    y_human = np.array([
        human_probs["Ambiguous"],
        human_probs["Unambiguous"]
    ], dtype=float)

    y_sim = np.array([
        sim_probs["Ambiguous"]["prob"],
        sim_probs["Unambiguous"]["prob"]
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(BAR_FIG_WIDTH, BAR_FIG_HEIGHT))

    # Human
    ax.plot(
        x_human, y_human,
        color=HUMAN_COLOR,
        linewidth=LINE_WIDTH,
        linestyle=REG_LINESTYLE
    )
    ax.scatter(
        x_human, y_human,
        s=SCATTER_SIZE,
        facecolor="none",
        edgecolor=HUMAN_COLOR,
        linewidth=SCATTER_EDGEWIDTH,
        zorder=3,
        label="Human"
    )

    # Simulation
    ax.plot(
        x_sim, y_sim,
        color=SIM_COLOR,
        linewidth=LINE_WIDTH,
        linestyle=REG_LINESTYLE
    )
    ax.scatter(
        x_sim, y_sim,
        s=SCATTER_SIZE,
        facecolor="none",
        edgecolor=SIM_COLOR,
        linewidth=SCATTER_EDGEWIDTH,
        zorder=3,
        label="Simulation"
    )

    style_axes(ax)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Ambiguous", "Unambiguous"], fontsize=AX_TICK_SIZE)

    ax.set_ylabel(
        "Regression\nprobability",
        fontsize=AX_LABEL_SIZE,
        labelpad=2,
        linespacing=0.9
    )

    ax.set_xlim(-0.2, 1.2)

    ymax = np.nanmax(np.concatenate([y_human, y_sim]))
    ylim_top = max(0.20, np.ceil((ymax + 0.02) / 0.05) * 0.05)
    ax.set_ylim(0.00, ylim_top)

    yticks = np.linspace(0.00, ylim_top, 3)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.2f}" for t in yticks], fontsize=AX_TICK_SIZE)

    ax.tick_params(axis="x", labelsize=AX_TICK_SIZE)

    ax.legend(
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="upper right",
        handlelength=1.6
    )

    base, _ = os.path.splitext(out_png)
    aligned_pdf = f"{base}_aligned_regression_probability.pdf"
    aligned_png = f"{base}_aligned_regression_probability.png"

    fig.savefig(aligned_pdf, dpi=300, pad_inches=0.02)
    fig.savefig(aligned_png, dpi=300, pad_inches=0.02)
    plt.close(fig)

    with open(out_stats_txt, "w") as f:
        f.write("source\tcondition\tprobability\n")
        f.write(f"human\tAmbiguous\t{y_human[0]:.6f}\n")
        f.write(f"human\tUnambiguous\t{y_human[1]:.6f}\n")
        f.write(f"simulation\tAmbiguous\t{y_sim[0]:.6f}\n")
        f.write(f"simulation\tUnambiguous\t{y_sim[1]:.6f}\n")

        f.write("\n")
        f.write(f"simulation_n_total_ambiguous\t{sim_probs['Ambiguous']['n_total']}\n")
        f.write(f"simulation_n_regressed_ambiguous\t{sim_probs['Ambiguous']['n_regressed']}\n")
        f.write(f"simulation_n_total_unambiguous\t{sim_probs['Unambiguous']['n_total']}\n")
        f.write(f"simulation_n_regressed_unambiguous\t{sim_probs['Unambiguous']['n_regressed']}\n")


# ---------------- main ----------------

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Infer thresholds and plot bar + scatter (binned regression) in one figure.")
    ap.add_argument("--input_json", type=str, default=DEFAULT_INPUT, help="Path to organized propositions JSON.")
    ap.add_argument("--calc_path", type=str, default=None, help="Optional explicit path to calculate_proportional_recall.py")
    ap.add_argument("--high_range", type=float, nargs=3, default=[0.0, 1.0, 0.05], metavar=("START","END","STEP"))
    ap.add_argument("--low_range",  type=float, nargs=3, default=[0.0, 1.0, 0.05], metavar=("START","END","STEP"))
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT, help="Output directory for results.")
    ap.add_argument("--sse", action="store_true", help="Use SSE instead of MAE (default is MAE).")
    # Human targets
    ap.add_argument("--human_highcoh_high", type=float, default=DEFAULT_HUMAN["highcoh_high"])
    ap.add_argument("--human_highcoh_low",  type=float, default=DEFAULT_HUMAN["highcoh_low"])
    ap.add_argument("--human_lowcoh_high",  type=float, default=DEFAULT_HUMAN["lowcoh_high"])
    ap.add_argument("--human_lowcoh_low",   type=float, default=DEFAULT_HUMAN["lowcoh_low"])
    ap.add_argument(
        "--appraisal_split_method",
        type=str,
        default="median",
        choices=["median", "fixed"],
        help="How to binarize appraisal into Ambiguous / Unambiguous."
    )
    ap.add_argument(
        "--appraisal_threshold",
        type=float,
        default=0.5,
        help="Threshold used when --appraisal_split_method fixed."
    )

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

        # # write the extra section separately
        # f.write("\nAligned regression comparison:\n")
        # f.write("  Human ambiguous regression prob    : 0.190\n")
        # f.write("  Human unambiguous regression prob  : 0.068\n")
        # f.write(f"  Simulation ambiguous regression prob   : {sim_probs['Ambiguous']['prob']:.6f}\n")
        # f.write(f"  Simulation unambiguous regression prob : {sim_probs['Unambiguous']['prob']:.6f}\n")
        # f.write(f"  Appraisal split method: {args.appraisal_split_method}\n")
        # f.write(f"  Appraisal split value : {split_value:.6f}\n")

    best_json = os.path.join(args.out_dir, "best_pair.json")
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(
            {"objective": ("SSE" if args.sse else "MAE"),
             "human": asdict(human),
             "best": best_row},
            f, indent=2
        )

    # # Build right subplot data from sim JSON (if provided)
    # scatter_centers = None
    # scatter_props = None
    # if args.sim_json:
    #     if os.path.exists(args.sim_json):
    #         all_app, reg_app = load_appraisals_from_json(args.sim_json)
    #         centers, props = bin_proportion_regressed(all_app, reg_app, n_bins=BIN_COUNT_SCATTER)
    #         scatter_centers, scatter_props = centers, props
    #     else:
    #         print(f"[WARN] --sim_json not found: {args.sim_json}. Right subplot will be empty.")
    
    # Build aligned binary regression probabilities from sim JSON
    sim_probs = {
        "Ambiguous": {"prob": np.nan, "n_total": 0, "n_regressed": 0},
        "Unambiguous": {"prob": np.nan, "n_total": 0, "n_regressed": 0},
    }
    split_value = np.nan

    if args.sim_json:
        if os.path.exists(args.sim_json):
            sim_df, sim_probs, split_value = load_aligned_binary_probs_from_json(
                args.sim_json,
                split_method=args.appraisal_split_method,
                threshold=args.appraisal_threshold
            )
        else:
            print(f"[WARN] --sim_json not found: {args.sim_json}. Aligned regression plot will use NaNs.")


    # # Plot combined panel
    # out_png = os.path.join(args.out_dir, "panel_best_params_and_regression.png")
    # out_stats = os.path.join(args.out_dir, "plot_stats.txt")
    # plot_panel(human, best_sim_four, scatter_centers, scatter_props, out_png, out_stats)

    human_regression_probs = {
        "Ambiguous": 0.19,
        "Unambiguous": 0.068,
    }

    out_png = os.path.join(args.out_dir, "panel_best_params_and_regression.png")
    out_stats = os.path.join(args.out_dir, "plot_stats.txt")
    plot_panel(human_regression_probs, sim_probs, out_png, out_stats)

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
