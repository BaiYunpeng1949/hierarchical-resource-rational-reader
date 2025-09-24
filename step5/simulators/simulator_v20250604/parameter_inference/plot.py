#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List

# Reuse your analysis functions & plotting schema
from _analyze_data import (
    process_simulation_results_to_fixation_sequence,
    process_fixation_sequences_to_metrics,
    plot_metrics_comparison,
)

DEFAULT_FIG_NAME = "comparison_human_vs_sim.png"


def ensure_analyzed_metrics(combo_dir: Path) -> Path:
    """
    Make sure we have per-episode metrics ready in the combo folder.
    Returns path to analyzed_fixation_metrics.json
    """
    combo_dir = Path(combo_dir)
    sim_json = combo_dir / "all_simulation_results.json"
    fix_seq = combo_dir / "processed_fixation_sequences.json"
    per_ep = combo_dir / "analyzed_fixation_metrics.json"

    if not sim_json.exists():
        raise FileNotFoundError(f"Missing simulation JSON: {sim_json}")
    if not fix_seq.exists():
        process_simulation_results_to_fixation_sequence(str(sim_json), str(fix_seq))
    if not per_ep.exists():
        process_fixation_sequences_to_metrics(str(fix_seq), str(per_ep))
    return per_ep


def load_episode_metrics(per_ep_path: Path) -> List[Dict]:
    with open(per_ep_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Plot best parameter sets vs human using your analyze_data plotting schema.")
    parser.add_argument("--grid_dir", type=str, required=True,
                        help="GRID root folder (e.g., simulated_results/GRID_YYYYMMDD_HHMM)")
    parser.add_argument("--human", type=str, required=True,
                        help="Path to analyzed_human_metrics.json")
    parser.add_argument("--summary_csv", type=str, default=None,
                        help="Path to grid_inference_summary.csv; if omitted, auto-detect in grid_dir")
    parser.add_argument("--topk", type=int, default=3, help="Number of best runs to plot")
    parser.add_argument("--fig_name", type=str, default=DEFAULT_FIG_NAME,
                        help=f"Filename for saved figure (default: {DEFAULT_FIG_NAME})")
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)

    # Locate summary CSV (produced by infer_parameter(s).py)
    summary_csv = Path(args.summary_csv) if args.summary_csv else (grid_dir / "grid_inference_summary.csv")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}. Run infer_parameter.py first.")

    # Read and sort by loss_total
    import csv
    rows = []
    with open(summary_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row["loss_total"] = float(row.get("loss_total", "inf"))
            except Exception:
                row["loss_total"] = float("inf")
            rows.append(row)
    rows_sorted = sorted(rows, key=lambda x: (x["loss_total"], x.get("run_dir","")))

    # Take top-k
    top = rows_sorted[: args.topk]

    # For each best run, ensure metrics and plot using your function
    for i, row in enumerate(top, 1):
        run_dir = Path(row["run_dir"])
        per_ep_path = ensure_analyzed_metrics(run_dir)
        sim_episodes = load_episode_metrics(per_ep_path)

        # Output path: save inside each run folder, suffix with rank
        out_path = run_dir / (args.fig_name if args.topk == 1 else f"{Path(args.fig_name).stem}_rank{i}.png")

        # Call your plotting function to maintain the same visual style
        plot_metrics_comparison(args.human, sim_episodes, str(out_path))

        # Echo params into a small text file for quick check
        meta_path = run_dir / "metadata.json"
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            with open(run_dir / f"{Path(args.fig_name).stem}_rank{i}_params.txt", "w", encoding="utf-8") as f:
                f.write(json.dumps(meta.get("used_params", {}), indent=2))
        except Exception:
            pass

        print(f"[{i}] loss={row.get('loss_total')} -> saved figure: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
