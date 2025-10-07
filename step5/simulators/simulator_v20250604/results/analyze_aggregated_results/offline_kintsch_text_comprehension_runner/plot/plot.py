#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot.py - Compare human vs three simulation variants (v3, v1, v2b)

Usage:
  python plot.py \
    --human_mcq path/to/human_mcq_acc_metrics.json \
    --human_fr  path/to/human_free_recall_metrics.json \
    --v3        path/to/comprehension_metrics_v3.json \
    --v1        path/to/comprehension_metrics_v1.json \
    --v2b       path/to/comprehension_metrics_v2b.json \
    --out       path/to/output.png \
    [--title "Comprehension (MCQ & Free Recall)"]

Style choices per request:
- Figure size & font sizes tuned for compactness
- Legend placed INSIDE the figure (on FR panel)
- Figure-level title right-aligned
- Human = blue; V3 = green; V1/V2b = grey bars with GREEN edges & hatches
- Two subplots in one figure (left: MCQ, right: Free Recall). No linking lines.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt

# ---------- Global style ----------
plt.rcParams.update({
    "figure.figsize": (10, 4.1),
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

TIME_BINS = ("30s", "60s", "90s")

# Keys expected in JSON files
# Human files (two separate files):
HUMAN_MCQ_MEAN_KEY = "mcq_mean_by_time"
HUMAN_MCQ_STD_KEY  = "mcq_std_by_time"
HUMAN_FR_MEAN_KEY  = "fr_mean_by_time"
HUMAN_FR_STD_KEY   = "fr_std_by_time"

# Simulation files (each contains both MCQ + FR):
SIM_MCQ_MEAN_KEY = "mcq_accuracy_by_time"
SIM_MCQ_STD_KEY  = "mcq_accuracy_std_by_time"        # optional
SIM_FR_MEAN_KEY  = "fr_mean_by_time"
SIM_FR_STD_KEY   = "fr_std_by_time"         # optional


def _get_means_stds(d: Dict, mean_key: str, std_key: str) -> Tuple[Dict[str,float], Dict[str,float]]:
    means = d.get(mean_key, {})
    stds  = d.get(std_key, {})
    # fallback: if stds missing, fill zeros for present means
    if not stds:
        stds = {k: 0.0 for k in means.keys()}
    return means, stds


def load_human(human_mcq_path: Path, human_fr_path: Path):
    with open(human_mcq_path, "r", encoding="utf-8") as f:
        mcq_json = json.load(f)
    with open(human_fr_path, "r", encoding="utf-8") as f:
        fr_json = json.load(f)

    mcq_mean, mcq_std = _get_means_stds(mcq_json, HUMAN_MCQ_MEAN_KEY, HUMAN_MCQ_STD_KEY)
    fr_mean,  fr_std  = _get_means_stds(fr_json,    HUMAN_FR_MEAN_KEY,  HUMAN_FR_STD_KEY)

    for t in TIME_BINS:
        if t not in mcq_mean:
            raise KeyError(f"Human MCQ missing mean for {t}")
        if t not in fr_mean:
            raise KeyError(f"Human FR missing mean for {t}")
        # ensure std entries exist
        mcq_std.setdefault(t, 0.0)
        fr_std.setdefault(t, 0.0)

    return mcq_mean, mcq_std, fr_mean, fr_std


def load_sim(sim_path: Path):
    with open(sim_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    mcq_mean = d.get(SIM_MCQ_MEAN_KEY, {})
    fr_mean  = d.get(SIM_FR_MEAN_KEY,  {})
    mcq_std  = d.get(SIM_MCQ_STD_KEY,  {})
    fr_std   = d.get(SIM_FR_STD_KEY,   {})

    if not mcq_mean or not fr_mean:
        raise KeyError(f"{sim_path.name}: missing required mean keys")

    for t in TIME_BINS:
        if t not in mcq_mean:
            raise KeyError(f"{sim_path.name}: MCQ mean missing {t}")
        if t not in fr_mean:
            raise KeyError(f"{sim_path.name}: FR mean missing {t}")
        mcq_std.setdefault(t, 0.0)
        fr_std.setdefault(t, 0.0)

    return mcq_mean, mcq_std, fr_mean, fr_std


def annotate_bars(ax, rects, values):
    """Place numeric annotations above bars."""
    for rect, val in zip(rects, values):
        height = rect.get_height()
        ax.annotate(f"{val:.2f}",
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # offset in points
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)


def plot(human, v3, v1, v2b, out_path: Path, fig_title: str):
    # palette
    # blue  = "#1f77b4"  # human
    # green = "#2ca02c"  # v3 + edge/hatch for baselines
    grey  = "#7f7f7f"  # baseline fill
    blue = "blue"
    green = "green"
    grey = "grey"

    # face/edge/hatch per series
    series_style = {
        "human": dict(color=blue,  edgecolor=blue,  hatch=None,   linewidth=1.0),
        "v3":    dict(color=green, edgecolor=green, hatch=None,   linewidth=1.0),
        "v1":    dict(color=grey,  edgecolor=green, hatch="//",   linewidth=1.2),
        "v2b":   dict(color=grey,  edgecolor=green, hatch="\\\\",  linewidth=1.2),
    }

    # Unpack dicts
    h_mcq_mean, h_mcq_std, h_fr_mean, h_fr_std = human
    v3_mcq_mean, v3_mcq_std, v3_fr_mean, v3_fr_std = v3
    v1_mcq_mean, v1_mcq_std, v1_fr_mean, v1_fr_std = v1
    v2_mcq_mean, v2_mcq_std, v2_fr_mean, v2_fr_std = v2b

    # Values in plotting order: Human, V3, V1, V2b
    groups = [
        ("Human", "human", h_mcq_mean, h_mcq_std, h_fr_mean, h_fr_std),
        ("Simulation",    "v3",    v3_mcq_mean, v3_mcq_std, v3_fr_mean, v3_fr_std),
        ("Sim with unlimited memory",    "v1",    v1_mcq_mean, v1_mcq_std, v1_fr_mean, v1_fr_std),
        ("Sim with local schema",   "v2b",   v2_mcq_mean, v2_mcq_std, v2_fr_mean, v2_fr_std),
    ]

    fig, axes = plt.subplots(1, 2, sharey=True)
    mcq_ax, fr_ax = axes

    def draw_panel(ax, pick_mean, pick_std, title):
        n_bins = len(TIME_BINS)
        n_groups = len(groups)
        x = list(range(n_bins))
        total_group_width = 0.80
        bar_w = total_group_width / n_groups
        
        for i, (label, key, mcq_m, mcq_s, fr_m, fr_s) in enumerate(groups):
            means = [pick_mean(mcq_m, fr_m)[t] for t in TIME_BINS]
            stds  = [pick_std(mcq_s, fr_s)[t]  for t in TIME_BINS]
            offsets = [xi - total_group_width/2 + (i + 0.5)*bar_w for xi in x]

            style = series_style[key]
            rects = ax.bar(offsets, means, bar_w,
                           label=label,
                           yerr=stds, capsize=3,
                           edgecolor=style["edgecolor"],
                           linewidth=style["linewidth"],
                           color=style["color"],
                           hatch=style["hatch"])
            annotate_bars(ax, rects, means)

        ax.set_xticks(x)
        ax.set_xticklabels(TIME_BINS)
        ax.set_ylim(0, 1.1)
        # ax.set_title(title, loc="left")
        ax.set_ylabel(title)
        ax.margins(x=0.02)

        # cleaner look
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Left: MCQ
    draw_panel(
        mcq_ax,
        pick_mean=lambda mcq_m, fr_m: mcq_m,
        pick_std =lambda mcq_s, fr_s: mcq_s,
        title="MCQ Accuracy"
    )

    # Right: Free Recall and legend inside this panel
    draw_panel(
        fr_ax,
        pick_mean=lambda mcq_m, fr_m: fr_m,
        pick_std =lambda mcq_s, fr_s: fr_s,
        title="Free Recall"
    )

    # Single legend INSIDE FR panel, compact
    handles, labels = fr_ax.get_legend_handles_labels()
    fr_ax.legend(handles[:4], labels[:4],
                 loc="upper right", bbox_to_anchor=(0.98, 1.1),
                 frameon=True, framealpha=0.85, ncol=2, borderaxespad=0.4,
                 handlelength=2, handletextpad=0.5, borderpad=0.4,)
                #  frameon=False, ncol=1, borderaxespad=0.5)

    # Right-aligned figure title
    if fig_title:
        fig.suptitle(fig_title, x=0.985, ha="right", y=0.995, fontsize=13, fontweight="bold")

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human_mcq", required=True, type=str)
    ap.add_argument("--human_fr",  required=True, type=str)
    ap.add_argument("--v3",        required=True, type=str)
    ap.add_argument("--v1",        required=True, type=str)
    ap.add_argument("--v2b",       required=True, type=str)
    ap.add_argument("--out",       required=True, type=str)
    ap.add_argument("--title",     required=False, type=str, default="Comprehension (MCQ & Free Recall)")
    args = ap.parse_args()

    human = load_human(Path(args.human_mcq), Path(args.human_fr))
    v3     = load_sim(Path(args.v3))
    v1     = load_sim(Path(args.v1))
    v2b    = load_sim(Path(args.v2b))

    plot(human, v3, v1, v2b, Path(args.out), args.title)


if __name__ == "__main__":
    main()
