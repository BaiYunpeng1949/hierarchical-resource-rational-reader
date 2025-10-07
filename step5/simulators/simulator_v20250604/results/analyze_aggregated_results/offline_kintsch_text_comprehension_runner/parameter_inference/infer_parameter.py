
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parameter inference for comprehension metrics (MCQ + Free Recall).

We minimize SSE on six mean values:
- MCQ accuracy at 30s, 60s, 90s
- Free recall mean at 30s, 60s, 90s

Inputs:
  --sims_dir: folder containing many simulation files named "comprehension_metrics_*.json"
  --human_mcq: path to human MCQ metrics JSON (expects "mcq_mean_by_time")
  --human_fr:  path to human Free Recall metrics JSON (expects "fr_mean_by_time")
Outputs (written to --out_dir, default = --sims_dir):
  - parameter_inference_summary.csv : one row per simulation file with per-cell SSE and total SSE
  - best_simulation.txt             : filename, SSE, and details for the best (lowest SSE) simulation
  - best_comparison.png             : bar chart comparing human vs best simulation across the 6 cells
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import csv

TIME_BINS = ("30s", "60s", "90s")
MCQ_KEY = "mcq_accuracy_by_time"
FR_KEY = "fr_mean_by_time"


def load_human(human_mcq_path: Path, human_fr_path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    with open(human_mcq_path, "r", encoding="utf-8") as f:
        mcq = json.load(f).get("mcq_mean_by_time", {})
    with open(human_fr_path, "r", encoding="utf-8") as f:
        fr = json.load(f).get("fr_mean_by_time", {})
    # Validate keys
    for t in TIME_BINS:
        if t not in mcq:
            raise KeyError(f"Human MCQ missing time bin: {t}")
        if t not in fr:
            raise KeyError(f"Human FR missing time bin: {t}")
    return mcq, fr


def load_sim(sim_path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    with open(sim_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mcq = data.get(MCQ_KEY, {})
    fr = data.get(FR_KEY, {})
    # Basic validation; allow missing but raise with informative error
    for t in TIME_BINS:
        if t not in mcq:
            raise KeyError(f"{sim_path.name}: missing {MCQ_KEY}['{t}']")
        if t not in fr:
            raise KeyError(f"{sim_path.name}: missing {FR_KEY}['{t}']")
    return mcq, fr


def sse(human_mcq: Dict[str,float], human_fr: Dict[str,float],
        sim_mcq: Dict[str,float], sim_fr: Dict[str,float]) -> Tuple[float, Dict[str, float]]:
    per_cell = {}
    total = 0.0
    for t in TIME_BINS:
        d_mcq = (sim_mcq[t] - human_mcq[t]) ** 2
        d_fr  = (sim_fr[t]  - human_fr[t])  ** 2
        per_cell[f"mcq_{t}"] = d_mcq
        per_cell[f"fr_{t}"] = d_fr
        total += d_mcq + d_fr
    return total, per_cell


def write_summary(rows: List[Dict], out_csv: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_best(human_mcq: Dict[str,float], human_fr: Dict[str,float],
              sim_mcq: Dict[str,float], sim_fr: Dict[str,float],
              out_png: Path) -> None:
    # Compose x labels in fixed order
    labels = [f"MCQ {t}" for t in TIME_BINS] + [f"FR {t}" for t in TIME_BINS]
    human_vals = [human_mcq[t] for t in TIME_BINS] + [human_fr[t] for t in TIME_BINS]
    sim_vals   = [sim_mcq[t]   for t in TIME_BINS] + [sim_fr[t]   for t in TIME_BINS]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = list(range(len(labels)))
    width = 0.38
    ax.bar([i - width/2 for i in x], human_vals, width, label="Human")
    ax.bar([i + width/2 for i in x], sim_vals,   width, label="Simulation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean (proportion)")
    ax.set_title("Comprehension: Human vs Simulation (best SSE)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sims_dir", type=str, required=True, help="Folder containing comprehension_metrics_*.json")
    p.add_argument("--human_mcq", type=str, required=True, help="Path to human_mcq_acc_metrics.json")
    p.add_argument("--human_fr", type=str, required=True, help="Path to human_free_recall_metrics.json")
    p.add_argument("--out_dir", type=str, default=None, help="Output directory (default = sims_dir)")
    args = p.parse_args()

    sims_dir = Path(args.sims_dir)
    out_dir = Path(args.out_dir) if args.out_dir else sims_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    human_mcq, human_fr = load_human(Path(args.human_mcq), Path(args.human_fr))

    sim_files = sorted(sims_dir.glob("comprehension_metrics_*.json"))
    if not sim_files:
        raise FileNotFoundError(f"No simulation files found in: {sims_dir}")

    rows = []
    best = None  # (total_sse, sim_path, sim_mcq, sim_fr, per_cell)
    for fp in sim_files:
        try:
            sim_mcq, sim_fr = load_sim(fp)
            total, cells = sse(human_mcq, human_fr, sim_mcq, sim_fr)
            row = {
                "filename": fp.name,
                "path": str(fp),
                "sse_total": total,
                **{f"sse_{k}": v for k,v in cells.items()},
                **{f"sim_mcq_{t}": sim_mcq[t] for t in TIME_BINS},
                **{f"sim_fr_{t}": sim_fr[t] for t in TIME_BINS},
            }
            rows.append(row)
            if best is None or total < best[0]:
                best = (total, fp, sim_mcq, sim_fr, cells)
        except Exception as e:
            rows.append({
                "filename": fp.name,
                "path": str(fp),
                "error": str(e),
                "sse_total": float("inf"),
            })

    # Write CSV
    out_csv = out_dir / "parameter_inference_summary.csv"
    write_summary(sorted(rows, key=lambda r: r.get("sse_total", float("inf"))), out_csv)

    # Log best
    if best is None:
        raise RuntimeError("Failed to evaluate any simulation files.")
    total, fp, sim_mcq, sim_fr, cells = best
    best_txt = out_dir / "best_simulation.txt"
    with open(best_txt, "w", encoding="utf-8") as f:
        f.write(f"BEST FILE: {fp.name}\n")
        f.write(f"FULL PATH: {fp}\n")
        f.write(f"SSE TOTAL: {total:.6f}\n")
        f.write("\nPer-cell SSE:\n")
        for k, v in cells.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write("\nSim means:\n")
        for t in TIME_BINS:
            f.write(f"  MCQ {t}: {sim_mcq[t]:.6f}\n")
        for t in TIME_BINS:
            f.write(f"  FR  {t}: {sim_fr[t]:.6f}\n")

    # Plot best comparison
    out_png = out_dir / "best_comparison.png"
    plot_best(human_mcq, human_fr, sim_mcq, sim_fr, out_png)

    print(f"Scored {len(sim_files)} files.")
    print(f"Best: {fp.name}  SSE={total:.6f}")
    print(f"Wrote summary: {out_csv}")
    print(f"Wrote best log: {best_txt}")
    print(f"Wrote plot: {out_png}")


if __name__ == "__main__":
    main()
