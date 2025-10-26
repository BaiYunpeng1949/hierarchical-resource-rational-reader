#!/usr/bin/env python3
"""
Plot Vibert-style eye-movement effects (simulation data) using the *core* metrics.

Reads:
  analyzed_fixation_metrics_core.json

Writes:
  effects_french_simulation_core.png  (multi-panel if no --single/--all-single)
  OR a set of per-metric 3x3 figures when using --single or --all-single

Panels (left->right) for multi:
  Fixation duration (ms)
  Saccade amp (px)
  Saccades/s
  Mean gaze vel (px/s)
  % fix content word
  % time in fixation

Formatting decisions (per user):
  - X ticks strictly "30s, 60s, 90s" (categorical)
  - No panel titles; only y-axis labels
  - Use green for simulation
  - Figure size per panel = 3x3 inches (constants below)
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# -------------------- Styling constants (edit here) --------------------
GREEN = "#1b9e77"   # simulation color
FIG_W = 3           # inches
FIG_H = 3           # inches
FONT_SIZE_YLABEL = 14
TICK_SIZE = 12
MARKER_SIZE = 4
LINE_WIDTH  = 2
CAP_SIZE    = 2

# Metric spec: (mean_key, std_key, y_label, is_percent, shortname)
METRICS = [
    ("fixation_duration_mean_ms", "fixation_duration_std_ms", "Fixation duration (ms)", False, "fix_dur"),
    ("saccade_amplitude_mean_px", "saccade_amplitude_std_px", "Saccade amp (px)", False, "sacc_amp"),
    ("saccade_rate_hz_mean",      "saccade_rate_hz_std",      "Saccades/s", False, "sacc_rate"),
    ("gaze_velocity_px_s_mean",   "gaze_velocity_px_s_std",   "Mean gaze vel (px/s)", False, "gaze_vel"),
    ("percent_fixated_content_words_mean", "percent_fixated_content_words_std", "% fix content word", True, "pct_fix_content"),
    # ("percent_time_in_fixation_mean",      "percent_time_in_fixation_std",      "% time in fixation", True, "pct_time_fix"),
]

def order_conditions(conds):
    pref = ["30s", "60s", "90s"]
    if all(p in conds for p in pref):
        return pref
    return list(conds)

def style_axes(ax, ylabel):
    # No top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Only y-axis label (no title)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_YLABEL)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)

def plot_one(ax, labels, ys, es, ylabel, color=GREEN):
    x = list(range(len(labels)))
    ax.errorbar(x, ys, yerr=es, fmt='o-', color=color, ms=MARKER_SIZE,
                lw=LINE_WIDTH, capsize=CAP_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)  # strictly "30s, 60s, 90s"
    style_axes(ax, ylabel)

def render_single(fig_path, cond_labels, series, ylabel, is_percent):
    ys, es = series
    if is_percent:
        ys = [v * 100.0 for v in ys]
        es = [e * 100.0 for e in es]
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    plot_one(ax, cond_labels, ys, es, ylabel)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("analyzed_fixation_metrics_core.json"))
    ap.add_argument("--out",   type=Path, default=Path("effects_french_simulation_core.png"))
    ap.add_argument("--single", choices=[m[4] for m in METRICS], help="render only one metric to OUT (3x3)")
    ap.add_argument("--all-single", action="store_true", help="render all six metrics as separate 3x3 PNGs (OUT used as prefix)")
    args = ap.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    conds = order_conditions(list(data["by_condition"].keys()))
    cond_labels = conds  # show exactly "30s, 60s, 90s"

    # Build series for each metric
    series = {}
    for k_mean, k_std, ylab, is_percent, short in METRICS:
        ys = [data["by_condition"][c][k_mean] for c in conds]
        es = [data["by_condition"][c].get(k_std, 0.0) for c in conds]
        series[short] = (ys, es, ylab, is_percent)

    # Option A: single metric figure (3x3)
    if args.single:
        ys, es, ylab, is_percent = series[args.single]
        render_single(args.out, cond_labels, (ys, es), ylab, is_percent)
        print("Saved:", args.out)
        return

    # Option B: all single figures (prefix + shortname)
    if args.all_single:
        prefix = str(args.out)
        if prefix.lower().endswith(".png"):
            prefix = prefix[:-4]
        for short in series:
            ys, es, ylab, is_percent = series[short]
            path = Path(f"{prefix}_{short}.png")
            render_single(path, cond_labels, (ys, es), ylab, is_percent)
            print("Saved:", path)
        return

    # Option C: multi-panel figure (each panel uses the new axis styling)
    n = len(METRICS)
    fig_w = FIG_W * n
    fig_h = FIG_H
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), sharex=False)

    # matplotlib returns a single Axes (not a list) when n == 1
    if n == 1:
        axes = [axes]

    for ax, (k_mean, k_std, ylab, is_percent, short) in zip(axes, METRICS):
        ys, es, _, _ = series[short]
        if is_percent:
            ys = [v * 100.0 for v in ys]
            es = [e * 100.0 for e in es]
        plot_one(ax, cond_labels, ys, es, ylab)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print("Saved:", args.out)

if __name__ == "__main__":

    """
    python plot_french_corpus_effects.py --input assets/analyzed_by_episode_fixation_metrics.json --out french_corpus_effects_panel.png
    """

    main()
