#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Infer parameters by scoring each grid-run folder (with metadata.json + all_simulation_results.json)
against human metrics using SSE (or L1). Reuses analyze_data.py to generate episode-level metrics.

Usage:
  python infer_parameter.py \
    --grid_dir simulated_results/GRID_2025XXXX_HHMM \
    --human analyzed_human_metrics.json \
    --loss sse \
    --out_csv simulated_results/GRID_2025XXXX_HHMM/grid_inference_summary.csv \
    --topk 15
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

# ---- Import your pipeline utilities from analyze_data.py ----
# If your function names differ, just edit the import lines below.
from _analyze_data import (
    process_simulation_results_to_fixation_sequence,
    process_fixation_sequences_to_metrics,
)

# Time conditions & metric names used for loss
TIME_CONDS = ("30s", "60s", "90s")
METRICS = ("reading_speed", "skip_rate", "regression_rate")


def load_human_metrics(human_path: str) -> Dict[str, Dict[str, float]]:
    with open(human_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_condition_means(per_episode_metrics_path: str) -> Dict[str, Dict[str, float]]:
    """
    Aggregate episode-level metrics into condition means matching analyzed_human_metrics.json schema.
    Expects each episode entry to include: time_condition, reading_speed, skip_rate, regression_rate.
    """
    import numpy as np

    with open(per_episode_metrics_path, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    # Group episodes by time_condition
    by_cond: Dict[str, List[dict]] = {}
    for ep in episodes:
        cond = ep.get("time_condition")
        if cond is None:
            # Skip if not labeled
            continue
        by_cond.setdefault(cond, []).append(ep)

    out: Dict[str, Dict[str, float]] = {}
    for cond, eps in by_cond.items():
        if not eps:
            continue
        # Build arrays per metric
        arr = {m: np.array([e[m] for e in eps], dtype=float) for m in METRICS}
        out[cond] = {
            "reading_speed_mean": float(arr["reading_speed"].mean()),
            "reading_speed_std": float(arr["reading_speed"].std(ddof=0)),
            "skip_rate_mean": float(arr["skip_rate"].mean()),
            "skip_rate_std": float(arr["skip_rate"].std(ddof=0)),
            "regression_rate_mean": float(arr["regression_rate"].mean()),
            "regression_rate_std": float(arr["regression_rate"].std(ddof=0)),
            "num_episodes": int(len(eps)),
        }
    return out


def loss_between(sim_means: Dict[str, Dict[str, float]],
                 human_means: Dict[str, Dict[str, float]],
                 loss: str = "sse") -> Tuple[float, Dict[str, float]]:
    """
    Compute total loss across conditions & metrics.
    Returns (total_loss, breakdown_per_metric).
    loss in {'sse', 'l1'}  -> SSE: (h-s)^2; L1: |h-s|
    """
    total = 0.0
    per_metric = {m: 0.0 for m in METRICS}
    for cond in TIME_CONDS:
        if cond not in human_means or cond not in sim_means:
            continue
        for m in METRICS:
            h = human_means[cond][f"{m}_mean"]
            s = sim_means[cond][f"{m}_mean"]
            d = abs(h - s) if loss == "l1" else (h - s) ** 2
            total += d
            per_metric[m] += d
    return total, per_metric


def discover_combo_folders(grid_dir: str) -> List[Path]:
    """
    Return subfolders that contain both metadata.json and all_simulation_results.json.
    Works whether you pass the GRID_YYYYMMDD_HHMM folder or a parent of it.
    """
    base = Path(grid_dir)
    if not base.exists():
        raise FileNotFoundError(f"Grid directory not found: {grid_dir}")
    hits: List[Path] = []
    for p in base.rglob("*"):
        if p.is_dir():
            meta = p / "metadata.json"
            sim = p / "all_simulation_results.json"
            if meta.exists() and sim.exists():
                hits.append(p)
    return sorted(hits)


def ensure_processed_metrics(combo_dir: Path) -> Path:
    """
    Run analyze_data pipeline to create per-episode metrics JSON in the combo directory.
    Returns path to analyzed_fixation_metrics.json
    """
    combo_dir = Path(combo_dir)
    sim_json = combo_dir / "all_simulation_results.json"
    fix_seq = combo_dir / "processed_fixation_sequences.json"
    per_ep = combo_dir / "analyzed_fixation_metrics.json"

    # Only compute if missing (you can force regeneration by deleting the files)
    if not fix_seq.exists():
        process_simulation_results_to_fixation_sequence(str(sim_json), str(fix_seq))
    if not per_ep.exists():
        process_fixation_sequences_to_metrics(str(fix_seq), str(per_ep))
    return per_ep


def read_params_from_metadata(meta_path: Path) -> Dict[str, float]:
    """
    Extract commonly tuned params from metadata.json (flattened).
    Adjust keys here if your metadata structure differs.
    """
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    used = meta.get("used_params", {})
    rho = used.get("word_recognizer_params", {}).get("rho_inflation_percentage", None)
    w = used.get("sentence_reader_params", {}).get("w_skip_degradation_factor", None)
    cov = used.get("text_reader_params", {}).get("coverage_factor", None)
    return {
        "rho_inflation_percentage": rho,
        "w_skip_degradation_factor": w,
        "coverage_factor": cov,
    }


def write_csv(rows: List[Dict], out_csv: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Infer parameters by comparing grid runs to human data (SSE or L1)."
    )
    parser.add_argument(
        "--grid_dir",
        type=str,
        required=True,
        help="Path to GRID folder (e.g., simulated_results/GRID_YYYYMMDD_HHMM)",
    )
    parser.add_argument(
        "--human",
        type=str,
        required=True,
        help="Path to analyzed_human_metrics.json",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="sse",
        choices=["sse", "l1"],
        help="Loss type: sse (sum of squared errors) or l1 (sum of absolute errors)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Path to write summary CSV (default: grid_dir/grid_inference_summary.csv)",
    )
    parser.add_argument(
        "--topk", type=int, default=10, help="Show top-k results in the console"
    )
    args = parser.parse_args()

    human = load_human_metrics(args.human)
    combo_dirs = discover_combo_folders(args.grid_dir)

    rows: List[Dict] = []
    for combo in combo_dirs:
        try:
            # 1) Ensure we have per-episode metrics via your analysis pipeline
            per_ep_path = ensure_processed_metrics(combo)

            # 2) Aggregate per-condition means
            sim_means = compute_condition_means(per_ep_path)

            # 3) Compute loss vs human
            total_loss, per_metric = loss_between(sim_means, human, loss=args.loss)

            # 4) Read parameters
            params = read_params_from_metadata(combo / "metadata.json")

            # 5) Flatten per-condition means into columns for CSV
            flat = {}
            for cond in TIME_CONDS:
                if cond in sim_means:
                    for m in METRICS:
                        flat[f"{cond}_{m}"] = sim_means[cond][f"{m}_mean"]
                else:
                    for m in METRICS:
                        flat[f"{cond}_{m}"] = None

            row = {
                "run_dir": str(combo),
                **params,
                **flat,
                "loss_type": args.loss,
                "loss_total": total_loss,
                "loss_reading_speed": per_metric["reading_speed"],
                "loss_skip_rate": per_metric["skip_rate"],
                "loss_regression_rate": per_metric["regression_rate"],
            }
            rows.append(row)
        except Exception as e:
            rows.append(
                {
                    "run_dir": str(combo),
                    "error": str(e),
                    "loss_type": args.loss,
                    "loss_total": float("inf"),
                }
            )

    # Sort by total loss
    rows_sorted = sorted(
        rows, key=lambda r: (r.get("loss_total", float("inf")), str(r.get("run_dir", "")))
    )

    # Write CSV
    out_csv = args.out_csv or (Path(args.grid_dir) / "grid_inference_summary.csv")
    write_csv(rows_sorted, str(out_csv))

    # Print brief top-k
    print("\nTop results (lowest loss):")
    for r in rows_sorted[: args.topk]:
        print(
            f"loss={r.get('loss_total'):.6f} | "
            f"rho={r.get('rho_inflation_percentage')} | "
            f"w={r.get('w_skip_degradation_factor')} | "
            f"cov={r.get('coverage_factor')} | "
            f"dir={r.get('run_dir')}"
        )

    print(f"\nSaved summary: {out_csv}")


if __name__ == "__main__":
    main()
