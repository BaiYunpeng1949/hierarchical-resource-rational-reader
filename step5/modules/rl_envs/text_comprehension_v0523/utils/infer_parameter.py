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
from scipy.stats import chi2_contingency
from math import log, exp, sqrt
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
PANEL_AX_WIDTH_IN   = 3.0
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


def classify_ambiguity_from_ranges(
    df,
    ambiguous_min=0.60,
    ambiguous_max=0.68,
    unambiguous_min=0.70,
    unambiguous_max=0.78,
):
    """
    Classify sentence instances using calibrated appraisal ranges.

    Rules:
      - Ambiguous    if ambiguous_min <= appraisal <= ambiguous_max
      - Unambiguous  if unambiguous_min <= appraisal <= unambiguous_max
      - Exclude      otherwise
    """
    df = df.copy()

    labels = []
    for score in df["appraisal"]:
        if ambiguous_min <= score <= ambiguous_max:
            labels.append("Ambiguous")
        elif unambiguous_min <= score <= unambiguous_max:
            labels.append("Unambiguous")
        else:
            labels.append("Exclude")

    df["ambiguity"] = labels
    return df


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
    Compute P(regression | ambiguity class), excluding rows with ambiguity == 'Exclude'.

    Returns:
      {
        "Ambiguous": {
            "prob": ...,
            "n_total": ...,
            "n_regressed": ...,
            "std": ...,
            "se": ...
        },
        "Unambiguous": {...},
        "Excluded": {"n_total": ...}
      }
    """
    out = {}

    for label in ["Ambiguous", "Unambiguous"]:
        sub = df[df["ambiguity"] == label]
        n_total = int(len(sub))
        n_regressed = int(sub["regressed"].sum())

        if n_total > 0:
            values = sub["regressed"].astype(float).to_numpy()
            prob = float(values.mean())
            std = float(np.std(values, ddof=1)) if n_total > 1 else 0.0
            se = float(std / np.sqrt(n_total)) if n_total > 0 else np.nan
        else:
            prob = np.nan
            std = np.nan
            se = np.nan

        out[label] = {
            "prob": prob,
            "n_total": n_total,
            "n_regressed": n_regressed,
            "std": std,
            "se": se,
        }

    out["Excluded"] = {
        "n_total": int((df["ambiguity"] == "Exclude").sum())
    }

    return out

def format_p(p):
    """Nature-style p-value formatting."""
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def compute_binary_contrast_stats(sim_probs):
    """
    Test whether regression probability differs between ambiguous and unambiguous
    simulated sentence instances.

    Uses a 2 x 2 chi-square test and reports odds ratio with 95% CI.
    """

    a = sim_probs["Ambiguous"]["n_regressed"]
    b = sim_probs["Ambiguous"]["n_total"] - sim_probs["Ambiguous"]["n_regressed"]
    c = sim_probs["Unambiguous"]["n_regressed"]
    d = sim_probs["Unambiguous"]["n_total"] - sim_probs["Unambiguous"]["n_regressed"]

    table = np.array([[a, b], [c, d]], dtype=float)

    chi2, p, dof, expected = chi2_contingency(table, correction=False)

    # Haldane-Anscombe correction protects against zero cells
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = log((aa * dd) / (bb * cc))
    se_log_or = sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)

    ci_low = exp(log_or - 1.96 * se_log_or)
    ci_high = exp(log_or + 1.96 * se_log_or)
    odds_ratio = exp(log_or)

    return {
        "analysis": "simulation_ambiguous_vs_unambiguous_regression",
        "test": "chi-square test of independence",
        "n_ambiguous": int(sim_probs["Ambiguous"]["n_total"]),
        "n_regressed_ambiguous": int(sim_probs["Ambiguous"]["n_regressed"]),
        "mean_ambiguous": sim_probs["Ambiguous"]["prob"],
        "n_unambiguous": int(sim_probs["Unambiguous"]["n_total"]),
        "n_regressed_unambiguous": int(sim_probs["Unambiguous"]["n_regressed"]),
        "mean_unambiguous": sim_probs["Unambiguous"]["prob"],
        "chi2": chi2,
        "df": int(dof),
        "p": p,
        "p_formatted": format_p(p),
        "odds_ratio": odds_ratio,
        "or_95ci_low": ci_low,
        "or_95ci_high": ci_high,
    }


def compute_text_recall_fit_stats(human, best_row):
    """
    Descriptive model-fit statistics for the four text-comprehension condition means.
    No p value is computed here because the human data are condition-level targets.
    """

    human_vals = np.array([
        human.highcoh_high,
        human.highcoh_low,
        human.lowcoh_high,
        human.lowcoh_low,
    ], dtype=float)

    sim_vals = np.array([
        best_row["sim_fully_high"],
        best_row["sim_fully_low"],
        best_row["sim_min_high"],
        best_row["sim_min_low"],
    ], dtype=float)

    errors = sim_vals - human_vals
    mae_value = float(np.mean(np.abs(errors)))
    rmse_value = float(np.sqrt(np.mean(errors ** 2)))
    r_value = float(np.corrcoef(human_vals, sim_vals)[0, 1])

    return {
        "analysis": "text_comprehension_four_condition_fit",
        "test": "descriptive model fit; no inferential p value",
        "n_conditions": 4,
        "mae": mae_value,
        "rmse": rmse_value,
        "correlation_r": r_value,
        "p": np.nan,
        "p_formatted": "not applicable",
    }

def load_aligned_binary_probs_from_json_with_ranges(
    json_path,
    ambiguous_min=0.60,
    ambiguous_max=0.68,
    unambiguous_min=0.70,
    unambiguous_max=0.78,
):
    """
    Full pipeline using calibrated appraisal ranges:
      raw sim json -> sentence records -> range-based ambiguity classes -> regression probabilities
    """
    df = extract_sentence_level_records_from_json(json_path)

    df = classify_ambiguity_from_ranges(
        df,
        ambiguous_min=ambiguous_min,
        ambiguous_max=ambiguous_max,
        unambiguous_min=unambiguous_min,
        unambiguous_max=unambiguous_max,
    )

    probs = compute_regression_probabilities_binary(df)
    return df, probs



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



def plot_text_recall_bar_pdf(human: 'HumanTargets', sim_four, out_png: str):
    """
    Output the original text-level proportional-recall bar PDF, but with the
    legend and High-K / Low-K annotations moved out into a separate legend-only
    figure. All other bar-plot settings are kept unchanged.
    """

    # Unpack values as in the original plotting code
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

    # Bars (unchanged)
    ax.bar(r1, [Hh, Mh], width=bar_width, color=HUMAN_COLOR, hatch='/')
    ax.bar(r2, [fch, mch], width=bar_width, color=SIM_COLOR, hatch='/')
    ax.bar(r3, [Hl, Ml], width=bar_width, color=HUMAN_COLOR, hatch='oo')
    ax.bar(r4, [fcl, mcl], width=bar_width, color=SIM_COLOR, hatch='oo')

    # Axes styling (unchanged)
    style_axes(ax)
    ax.set_xlabel('Text coherence level', fontsize=AX_LABEL_SIZE)
    ax.set_ylabel('Proportional recall', fontsize=AX_LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=AX_TICK_SIZE)

    group_centers = r1 + 1.5 * bar_width
    ax.set_xticks(group_centers)
    ax.set_xticklabels(['High', 'Low'])
    ax.tick_params(axis='x', length=0)

    # Keep the original y-limit logic so plot scaling stays unchanged.
    x_highk_0 = 0.5 * (r1[0] + r2[0])
    x_lowk_0  = 0.525 * (r3[0] + r4[0])
    x_highk_1 = 0.5 * (r1[1] + r2[1])
    x_lowk_1  = 0.51 * (r3[1] + r4[1])

    y_high_0 = max(Hh, fch) + 0.01
    y_low_0  = max(Hl, fcl) + 0.01
    y_high_1 = max(Mh, mch) + 0.01
    y_low_1  = max(Ml, mcl) + 0.01
    y_top = max(y_high_0, y_low_0, y_high_1, y_low_1)
    ax.set_ylim(0, y_top + 0.05)

    base, _ = os.path.splitext(out_png)
    bar_pdf = f"{base}_bar.pdf"
    fig_bar.savefig(bar_pdf, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_bar)

    # ========= SEPARATE LEGEND FIGURE (PDF) =========
    # Same visual encodings as before:
    #   - colour: Human vs Simulation
    #   - hatch : High-K vs Low-K
    fig_leg = plt.figure(figsize=(2.6, 1.2))
    ax_leg = fig_leg.add_subplot(1, 1, 1)
    ax_leg.axis('off')

    legend_handles = [
        mpatches.Patch(
            facecolor=HUMAN_COLOR,
            edgecolor='black',
            hatch='/',
            label='Human, High-K'
        ),
        mpatches.Patch(
            facecolor=SIM_COLOR,
            edgecolor='black',
            hatch='/',
            label='Simulation, High-K'
        ),
        mpatches.Patch(
            facecolor=HUMAN_COLOR,
            edgecolor='black',
            hatch='oo',
            label='Human, Low-K'
        ),
        mpatches.Patch(
            facecolor=SIM_COLOR,
            edgecolor='black',
            hatch='oo',
            label='Simulation, Low-K'
        ),
    ]

    ax_leg.legend(
        handles=legend_handles,
        loc='center',
        frameon=True,
        facecolor='white',
        ncol=1,
        fontsize=AX_TICK_SIZE,
        handlelength=2.2,      # wider patch
        handleheight=1.2,      # taller patch
        handletextpad=0.8,
        borderpad=0.5,
        labelspacing=0.5
    )

    legend_pdf = f"{base}_bar_legend.pdf"
    fig_leg.savefig(legend_pdf, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_leg)

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

    # Human data from Staub & Clifton (2006), spillover region
    # Means are in percent in the paper; convert to proportions here
    human_mean_percent = {
        "Ambiguous": 19.0,   # No-either S-coordination
        "Unambiguous": 6.8,  # Either S-coordination
    }
    human_se_percent = {
        "Ambiguous": 4.5,
        "Unambiguous": 2.8,
    }
    human_n = 24  # participants reported in the paper
    human_sd_percent = {
        k: v * np.sqrt(human_n) for k, v in human_se_percent.items()
    }

    y_human = np.array([
        human_mean_percent["Ambiguous"] / 100.0,
        human_mean_percent["Unambiguous"] / 100.0
    ], dtype=float)

    y_sim = np.array([
        sim_probs["Ambiguous"]["prob"],
        sim_probs["Unambiguous"]["prob"]
    ], dtype=float)

    # fig, ax = plt.subplots(figsize=(BAR_FIG_WIDTH, BAR_FIG_HEIGHT))
    fig, ax = plt.subplots(figsize=(PANEL_AX_WIDTH_IN, PANEL_AX_HEIGHT_IN))
    fig.subplots_adjust(left=0.24, bottom=0.16, right=0.98, top=0.98)

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

    ax.set_xticks([0, 0.875])
    ax.set_xticklabels(["Ambiguous", "Unambiguous"], fontsize=AX_TICK_SIZE)

    ax.set_ylabel(
        "Regression probability",
        fontsize=AX_LABEL_SIZE,
        labelpad=2,
        linespacing=0.9
    )

    ax.set_xlim(-0.2, 1.2)

    # ymax = np.nanmax(np.concatenate([y_human, y_sim]))
    # ylim_top = max(0.20, np.ceil((ymax + 0.02) / 0.05) * 0.05)
    # ax.set_ylim(0.00, ylim_top)

    ax.set_ylim(0.00, 0.40)
    ax.set_yticks([0.00, 0.20, 0.40])
    ax.set_yticklabels(["0.00", "0.20", "0.40"], fontsize=AX_TICK_SIZE)

    ax.tick_params(axis="x", labelsize=AX_TICK_SIZE)

    base, _ = os.path.splitext(out_png)
    aligned_pdf = f"{base}_aligned_regression_probability.pdf"
    aligned_png = f"{base}_aligned_regression_probability.png"

    fig.savefig(aligned_pdf, dpi=300, pad_inches=0.02)
    fig.savefig(aligned_png, dpi=300, pad_inches=0.02)
    plt.close(fig)

    with open(out_stats_txt, "w") as f:
        f.write("source\tcondition\tmean_probability\tmean_percent\tse_percent\tsd_percent\tn\n")

        # Human: from Staub & Clifton (2006)
        f.write(
            f"human\tAmbiguous\t"
            f"{human_mean_percent['Ambiguous']/100.0:.6f}\t"
            f"{human_mean_percent['Ambiguous']:.3f}\t"
            f"{human_se_percent['Ambiguous']:.3f}\t"
            f"{human_sd_percent['Ambiguous']:.3f}\t"
            f"{human_n}\n"
        )
        f.write(
            f"human\tUnambiguous\t"
            f"{human_mean_percent['Unambiguous']/100.0:.6f}\t"
            f"{human_mean_percent['Unambiguous']:.3f}\t"
            f"{human_se_percent['Unambiguous']:.3f}\t"
            f"{human_sd_percent['Unambiguous']:.3f}\t"
            f"{human_n}\n"
        )

        # Simulation
        f.write(
            f"simulation\tAmbiguous\t"
            f"{sim_probs['Ambiguous']['prob']:.6f}\t"
            f"{sim_probs['Ambiguous']['prob']*100:.3f}\t"
            f"{sim_probs['Ambiguous']['se']*100:.3f}\t"
            f"{sim_probs['Ambiguous']['std']*100:.3f}\t"
            f"{sim_probs['Ambiguous']['n_total']}\n"
        )
        f.write(
            f"simulation\tUnambiguous\t"
            f"{sim_probs['Unambiguous']['prob']:.6f}\t"
            f"{sim_probs['Unambiguous']['prob']*100:.3f}\t"
            f"{sim_probs['Unambiguous']['se']*100:.3f}\t"
            f"{sim_probs['Unambiguous']['std']*100:.3f}\t"
            f"{sim_probs['Unambiguous']['n_total']}\n"
        )

        f.write("\n")
        f.write("# Notes\n")
        f.write("# Human SE values are reported directly in Staub & Clifton (2006), Table 1.\n")
        f.write("# Human SD values are estimated as SE * sqrt(n), using n = 24 participants.\n")
        f.write("# Simulation SD/SE are computed across sentence instances within each ambiguity class.\n")
        f.write("# Simulation probability = n_regressed / n_total.\n")

        f.write("\n")
        f.write(f"simulation_n_regressed_ambiguous\t{sim_probs['Ambiguous']['n_regressed']}\n")
        f.write(f"simulation_n_regressed_unambiguous\t{sim_probs['Unambiguous']['n_regressed']}\n")
        f.write(f"simulation_n_excluded\t{sim_probs['Excluded']['n_total']}\n")
        f.write("ambiguous_range_min\t0.620\n")
        f.write("ambiguous_range_max\t0.680\n")
        f.write("unambiguous_range_min\t0.705\n")
        f.write("unambiguous_range_max\t0.780\n")


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

    best_json = os.path.join(args.out_dir, "best_pair.json")
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(
            {"objective": ("SSE" if args.sse else "MAE"),
             "human": asdict(human),
             "best": best_row},
            f, indent=2
        )
    
    # Build aligned binary regression probabilities from sim JSON using calibrated ranges
    sim_probs = {
        "Ambiguous": {"prob": np.nan, "n_total": 0, "n_regressed": 0, "std": np.nan, "se": np.nan},
        "Unambiguous": {"prob": np.nan, "n_total": 0, "n_regressed": 0, "std": np.nan, "se": np.nan},
        "Excluded": {"n_total": 0},
    }

    # calibrated from human stimuli
    ambiguous_min = 0.6
    ambiguous_max = 0.68
    unambiguous_min = 0.7
    unambiguous_max = 0.78

    if args.sim_json:
        if os.path.exists(args.sim_json):
            sim_df, sim_probs = load_aligned_binary_probs_from_json_with_ranges(
                args.sim_json,
                ambiguous_min=ambiguous_min,
                ambiguous_max=ambiguous_max,
                unambiguous_min=unambiguous_min,
                unambiguous_max=unambiguous_max,
            )
        else:
            print(f"[WARN] --sim_json not found: {args.sim_json}. Aligned regression plot will use NaNs.")

    human_regression_probs = {
        "Ambiguous": 0.19,
        "Unambiguous": 0.068,
    }

    out_png = os.path.join(args.out_dir, "panel_best_params_and_regression.png")
    out_stats = os.path.join(args.out_dir, "plot_stats.txt")
    plot_text_recall_bar_pdf(human, best_sim_four, out_png)
    plot_panel(human_regression_probs, sim_probs, out_png, out_stats)

    
    inferential_rows = []

    # 1. Descriptive fit for four text-comprehension condition means
    inferential_rows.append(compute_text_recall_fit_stats(human, best_row))

    # 2. Inferential test for simulated ambiguous vs unambiguous regression probability
    if (
        sim_probs["Ambiguous"]["n_total"] > 0
        and sim_probs["Unambiguous"]["n_total"] > 0
    ):
        inferential_rows.append(compute_binary_contrast_stats(sim_probs))

    inferential_df = pd.DataFrame(inferential_rows)
    inferential_csv = os.path.join(args.out_dir, "inferential_stats.csv")
    inferential_df.to_csv(inferential_csv, index=False)

    print(f"\nSaved grid CSV: {csv_path}")
    print(f"Saved best summary: {best_txt}")
    print(f"Saved best json: {best_json}")
    print(f"Saved panel figure: {out_png}")
    print(f"Saved scatter regression stats: {out_stats}")
    print("\nBest thresholds: high={:.3f}, low={:.3f}, loss={:.6f}".format(
        best_row['high_threshold'], best_row['low_threshold'], best_row['loss_total'])
    )
    print(f"Saved inferential stats: {inferential_csv}")

if __name__ == "__main__":
    main()