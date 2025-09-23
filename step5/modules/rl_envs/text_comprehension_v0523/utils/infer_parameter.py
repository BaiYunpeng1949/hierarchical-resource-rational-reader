#!/usr/bin/env python3
"""
Parameter inference for LTM activation thresholds (high & low) + plotting.

Sweeps two thresholds over user-specified ranges (e.g., 0..1 step 0.05),
computes proportional recall (Fully vs. Minimally coherent × High/Low knowledge),
selects the best pair against human means (SSE by default; MAE with --mae),
and draws a grouped bar chart for the **best thresholds**.

Outputs (under --out_dir):
- grid_results.csv
- best_summary.txt
- best_pair.json
- similarity_comparison.png   # human vs sim bars for best thresholds
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

# ---------------- config (defaults) ----------------
DEFAULT_INPUT = "./assets/organized_example_propositions_v0527.json"
DEFAULT_OUT   = "./parameter_inference/ltm_threshold_grid"

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

# def evaluate_pair(calc_mod, propositions, hi: float, lo: float, human: HumanTargets, use_mae: bool = False):
def evaluate_pair(calc_mod, propositions, hi, lo, human, use_sse=False):    
    """
    Returns (sim_four, loss, per_component_errors)
      sim_four = (fully_high, fully_low, minimal_high, minimal_low)
    """
    fch, fcl, mch, mcl = calc_mod.calculate_proportional_recall(
        propositions, high_threshold=hi, low_threshold=lo
    )
    # err = mae if use_mae else sse
    err = sse if use_sse else mae
    comp = {
        "err_fully_high": err(fch, human.highcoh_high),
        "err_fully_low" : err(fcl, human.highcoh_low),
        "err_min_high"  : err(mch, human.lowcoh_high),
        "err_min_low"   : err(mcl, human.lowcoh_low),
    }
    loss = sum(comp.values())
    return (fch, fcl, mch, mcl), loss, comp

# ---------------- plotting ----------------
def plot_similarity(human: HumanTargets, sim_four: tuple[float,float,float,float], out_png: str):
    """
    Create a grouped bar chart matching the style of the provided example:
      - X: High Coherence, Low Coherence
      - Bars: Human HighK, Sim HighK, Human LowK, Sim LowK
    """
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import numpy as np

    # unpack
    fch, fcl, mch, mcl = sim_four
    # human values
    Hh, Hl, Mh, Ml = human.highcoh_high, human.highcoh_low, human.lowcoh_high, human.lowcoh_low

    # order: high coherence group then low coherence group
    # each group: Human HighK, Sim HighK, Human LowK, Sim LowK
    barWidth = 0.15
    r1 = np.arange(2)
    r2 = r1 + barWidth
    r3 = r2 + barWidth
    r4 = r3 + barWidth

    plt.figure(figsize=(12,6))

    bars1 = plt.bar(r1, [Hh, Mh], width=barWidth, label="Human (High Knowledge)", color='blue', hatch="/")
    bars2 = plt.bar(r2, [fch, mch], width=barWidth, label="Sim (High Knowledge)", color='green', hatch="/")
    bars3 = plt.bar(r3, [Hl, Ml], width=barWidth, label="Human (Low Knowledge)", color='blue', hatch=".")
    bars4 = plt.bar(r4, [fcl, mcl], width=barWidth, label="Sim (Low Knowledge)", color='green', hatch=".")

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')


    for bs in (bars1, bars2, bars3, bars4):
        add_labels(bs)

    plt.xlabel("Coherence Level")
    plt.ylabel("Similarity Score")
    plt.title("Human vs Simulation Performance Comparison (Best Thresholds)")
    plt.xticks([r + barWidth*1.5 for r in range(2)], ['High Coherence', 'Low Coherence'])

    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Grid search high/low LTM thresholds to match human proportional recall (with plotting).")
    ap.add_argument("--input_json", type=str, default=DEFAULT_INPUT,
                    help="Path to organized propositions JSON.")
    ap.add_argument("--calc_path", type=str, default=None,
                    help="Optional explicit path to calculate_proportional_recall.py")
    ap.add_argument("--high_range", type=float, nargs=3, default=[0.0, 1.0, 0.05],
                    metavar=("START","END","STEP"),
                    help="Grid for HIGH threshold (inclusive). Default: 0 1 0.05")
    ap.add_argument("--low_range", type=float, nargs=3, default=[0.0, 1.0, 0.05],
                    metavar=("START","END","STEP"),
                    help="Grid for LOW threshold (inclusive). Default: 0 1 0.05")
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT,
                    help="Output directory for results.")
    # ap.add_argument("--mae", action="store_true", help="Use MAE instead of SSE.")
    ap.add_argument("--sse", action="store_true", help="Use SSE instead of MAE (default is MAE).")
    ap.add_argument("--no_plot", action="store_true", help="Disable plotting.")
    # Human targets
    ap.add_argument("--human_highcoh_high", type=float, default=DEFAULT_HUMAN["highcoh_high"])
    ap.add_argument("--human_highcoh_low",  type=float, default=DEFAULT_HUMAN["highcoh_low"])
    ap.add_argument("--human_lowcoh_high",  type=float, default=DEFAULT_HUMAN["lowcoh_high"])
    ap.add_argument("--human_lowcoh_low",   type=float, default=DEFAULT_HUMAN["lowcoh_low"])

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

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
            # sim_four, loss, comp = evaluate_pair(calc_mod, propositions, hi, lo, human, use_mae=args.mae)
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

    # Plot best comparison
    if not args.no_plot:
        out_png = os.path.join(args.out_dir, "similarity_comparison.png")
        plot_similarity(human, best_sim_four, out_png)

    print(f"\nSaved grid CSV: {csv_path}")
    print(f"Saved best summary: {best_txt}")
    print(f"Saved best json: {best_json}")
    if not args.no_plot:
        print(f"Saved plot: {out_png}")
    print("\nBest thresholds: high={:.3f}, low={:.3f}, loss={:.6f}".format(
        best_row['high_threshold'], best_row['low_threshold'], best_row['loss_total'])
    )

if __name__ == "__main__":
    main()
