#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bayesian optimization for simulator parameters:
  theta = (rho_inflation_percentage, w_skip_degradation_factor, coverage_factor)

It evaluates a candidate by:
  1) running a small single-batch simulation,
  2) computing episode metrics via _analyze_data.py,
  3) aggregating per-condition means, computing SSE/L1 to human,
  4) updating a tiny GP and selecting the next candidate via Expected Improvement.

No third-party deps beyond NumPy.

Example:
  python bayesian_inference.py \
    --human human_data/analyzed_human_metrics.json \
    --out_root parameter_inference/bayes_runs \
    --iters 40 --init 8 --cand 512 --xi 0.01 \
    --bounds_rho 0.10 0.30 --bounds_w 0.50 1.00 --bounds_cov 0.00 3.00 \
    --stimuli 0-8 --conds 30s,60s,90s --trials 1 \
    --loss sse \
    --warm_start_from parameter_inference/grid_inference_summary.csv
"""

import argparse
import json
import math
import os, sys
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from math import erf as _math_erf

# Project root = parent of this folder (which contains simulator.py, utils/, etc.)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ==== your code ====
# simulator is at project root
from simulator import ReaderAgent, run_batch_simulations
# analyze helpers are in the SAME folder as this file
from _analyze_data import (
    process_simulation_results_to_fixation_sequence,
    process_fixation_sequences_to_metrics,
)
# ===================

TIME_CONDS = ("30s", "60s", "90s")
METRICS = ("reading_speed", "skip_rate", "regression_rate")


# ----------------- helpers -----------------
def parse_stimuli(s: str) -> List[int]:
    s = s.strip()
    if "-" in s:
        a, b = [int(x) for x in s.split("-")]
        return list(range(a, b + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def parse_conds(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def load_human_metrics(human_path: str) -> Dict[str, Dict[str, float]]:
    with open(human_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_condition_means(per_episode_metrics_path: str) -> Dict[str, Dict[str, float]]:
    """Aggregate episode-level metrics into condition means (keys like reading_speed_mean)."""
    with open(per_episode_metrics_path, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    by_cond: Dict[str, List[dict]] = {}
    for ep in episodes:
        cond = ep.get("time_condition")
        if cond is None:
            continue
        by_cond.setdefault(cond, []).append(ep)

    out: Dict[str, Dict[str, float]] = {}
    for cond, eps in by_cond.items():
        if not eps:
            continue
        arr = {m: np.array([e[m] for e in eps], dtype=float) for m in METRICS}
        out[cond] = {
            "reading_speed_mean": float(arr["reading_speed"].mean()),
            "skip_rate_mean": float(arr["skip_rate"].mean()),
            "regression_rate_mean": float(arr["regression_rate"].mean()),
            "num_episodes": int(len(eps)),
        }
    return out


def loss_between(sim_means: Dict[str, Dict[str, float]],
                 human_means: Dict[str, Dict[str, float]],
                 loss: str = "sse") -> float:
    total = 0.0
    for cond in TIME_CONDS:
        if cond not in human_means or cond not in sim_means:
            continue
        for m in METRICS:
            h = human_means[cond][f"{m}_mean"]
            s = sim_means[cond][f"{m}_mean"]
            total += abs(h - s) if loss == "l1" else (h - s) ** 2
    return float(total)

def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """
    Φ(z) computed elementwise.
    Uses np.erf if available; otherwise falls back to math.erf via vectorize.
    """
    try:
        # Most NumPy builds have this; keep fast path when available.
        return 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
    except AttributeError:
        vec_erf = np.vectorize(lambda t: _math_erf(float(t)))
        return 0.5 * (1.0 + vec_erf(z / np.sqrt(2.0)))

# ------------- evaluation -------------
@dataclass
class EvalConfig:
    stimuli: List[int]
    conds: List[str]
    trials: int
    out_root: Path
    loss_type: str
    human_metrics: Dict[str, Dict[str, float]]


def evaluate_theta(theta: Tuple[float, float, float], cfg: EvalConfig) -> Tuple[float, Path]:
    """Run one evaluation: simulator → metrics → loss. Returns (loss, run_dir)."""
    rho, w, cov = theta
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"rho_{rho:.6f}__w_{w:.6f}__cov_{cov:.6f}"
    run_dir = cfg.out_root / f"{stamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) simulate a small batch (reusing your API)
    sim = ReaderAgent()
    run_batch_simulations(
        simulator=sim,
        stimulus_ids=cfg.stimuli,
        time_conditions=cfg.conds,
        num_trials=cfg.trials,
        word_recognizer_params={"rho_inflation_percentage": float(rho)},
        sentence_reader_params={"w_skip_degradation_factor": float(w)},
        text_reader_params={"coverage_factor": float(cov)},
        output_dir=str(run_dir),
        run_name=None,
        extra_metadata={"bayes_eval": True},
    )

    sim_json = run_dir / "all_simulation_results.json"
    fix_seq = run_dir / "processed_fixation_sequences.json"
    per_ep = run_dir / "analyzed_fixation_metrics.json"

    # 2) analyze
    process_simulation_results_to_fixation_sequence(str(sim_json), str(fix_seq))
    process_fixation_sequences_to_metrics(str(fix_seq), str(per_ep))
    sim_means = compute_condition_means(str(per_ep))

    # 3) loss
    L = loss_between(sim_means, cfg.human_metrics, loss=cfg.loss_type)

    # Save a tiny summary
    with open(run_dir / "bayes_eval.json", "w", encoding="utf-8") as f:
        json.dump({
            "theta": {"rho": rho, "w": w, "cov": cov},
            "loss": L,
            "stimuli": cfg.stimuli, "conds": cfg.conds, "trials": cfg.trials,
            "loss_type": cfg.loss_type,
        }, f, indent=2)
    return L, run_dir


# ------------- tiny GP (RBF) -------------
def rbf_kernel(X: np.ndarray, Y: np.ndarray, lengthscales: np.ndarray, variance: float) -> np.ndarray:
    Xs = X / lengthscales
    Ys = Y / lengthscales
    X2 = np.sum(Xs**2, axis=1, keepdims=True)
    Y2 = np.sum(Ys**2, axis=1, keepdims=True).T
    d2 = X2 + Y2 - 2.0 * (Xs @ Ys.T)
    return variance * np.exp(-0.5 * np.clip(d2, 0.0, None))


@dataclass
class GPState:
    X: np.ndarray   # [n,d] in [0,1]
    y: np.ndarray   # [n]
    Ls: np.ndarray  # [d]
    var: float
    noise: float


def gp_fit(X: np.ndarray, y: np.ndarray) -> GPState:
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    d = X.shape[1]
    Ls = np.maximum(0.2, np.std(X, axis=0) * 0.5 + 1e-6)  # heuristic
    var = float(np.var(y) + 1e-6) or 1.0
    noise = max(1e-6, 0.01 * (np.max(y) - np.min(y) + 1e-6))
    return GPState(X=X, y=y, Ls=Ls, var=var, noise=noise)


def gp_predict(gp: GPState, Xstar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    K = rbf_kernel(gp.X, gp.X, gp.Ls, gp.var) + gp.noise * np.eye(len(gp.X))
    Ks = rbf_kernel(gp.X, Xstar, gp.Ls, gp.var)
    Kss = rbf_kernel(Xstar, Xstar, gp.Ls, gp.var)
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, gp.y))
        mu = Ks.T @ alpha
        v = np.linalg.solve(L, Ks)
        var = np.diag(Kss) - np.sum(v * v, axis=0)
        var = np.maximum(var, 1e-12)
    except np.linalg.LinAlgError:
        Ki = np.linalg.pinv(K)
        mu = Ks.T @ (Ki @ gp.y)
        var = np.maximum(np.diag(Kss) - np.sum((Ks.T @ Ki) * Ks.T, axis=1), 1e-12)
    return mu, var


# ------------- acquisition (EI) -------------
def expected_improvement(mu: np.ndarray, var: np.ndarray, best_y: float, xi: float = 0.01) -> np.ndarray:
    """We minimize y. EI computed on improvement over best."""
    sigma = np.sqrt(np.maximum(var, 1e-12))
    imp = best_y - mu - xi
    Z = imp / sigma

    # standard normal pdf and cdf
    pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * Z**2)
    cdf = _norm_cdf(Z)

    ei = imp * cdf + sigma * pdf
    # No improvement when predicted mean already worse than best
    ei[imp < 0] = 0.0
    # And when uncertainty is ~zero
    ei[sigma < 1e-12] = 0.0
    return ei


# ------------- scaling -------------
def sample_candidates(n: int, d: int) -> np.ndarray:
    return np.random.rand(n, d)


def scale_to_unit(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return (x - lo) / (hi - lo + 1e-12)


def unscale_from_unit(z: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return lo + z * (hi - lo)


# ------------- warm start from CSV -------------
def warm_start_from_summary(csv_path: Path, bounds: np.ndarray, topk: int = 20) -> Tuple[List[List[float]], List[float]]:
    rows = []
    if not (csv_path and csv_path.exists()):
        return [], []
    import csv
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                loss = float(row.get("loss_total", "inf"))
                rho = float(row["rho_inflation_percentage"])
                w = float(row["w_skip_degradation_factor"])
                cov = float(row["coverage_factor"])
            except Exception:
                continue
            rows.append((loss, [rho, w, cov]))
    rows.sort(key=lambda t: t[0])
    rows = rows[:topk]
    X = [scale_to_unit(np.array(v, float), bounds).tolist() for _, v in rows]
    y = [float(l) for l, _ in rows]
    return X, y


# ------------- main loop -------------
def main():
    p = argparse.ArgumentParser(description="Bayesian optimization for simulator parameters (pure NumPy GP).")
    p.add_argument("--human", required=True, type=str, help="Path to analyzed_human_metrics.json")
    p.add_argument("--out_root", required=True, type=str, help="Folder to store bayes runs")
    p.add_argument("--iters", type=int, default=40, help="Total BO evaluations")
    p.add_argument("--init", type=int, default=8, help="Initial evaluations (warm-start rows count if provided)")
    p.add_argument("--loss", type=str, default="sse", choices=["sse", "l1"], help="Objective")
    p.add_argument("--stimuli", type=str, default="0-8", help='e.g. "0-8" or "0,2,5"')
    p.add_argument("--conds", type=str, default="30s,60s,90s", help='e.g. "30s,60s"')
    p.add_argument("--trials", type=int, default=1, help="Trials per (stimulus, condition)")
    p.add_argument("--bounds_rho", type=float, nargs=2, default=[0.10, 0.30])
    p.add_argument("--bounds_w",   type=float, nargs=2, default=[0.50, 1.00])
    p.add_argument("--bounds_cov", type=float, nargs=2, default=[0.00, 3.00])
    p.add_argument("--cand", type=int, default=512, help="Random candidates per EI step")
    p.add_argument("--xi", type=float, default=0.01, help="EI exploration constant")
    p.add_argument("--warm_start_from", type=str, default="", help="grid_inference_summary.csv to seed BO")
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    human = load_human_metrics(args.human)
    cfg = EvalConfig(
        stimuli=parse_stimuli(args.stimuli),
        conds=parse_conds(args.conds),
        trials=args.trials,
        out_root=out_root,
        loss_type=args.loss,
        human_metrics=human,
    )

    bounds = np.array([
        [args.bounds_rho[0], args.bounds_rho[1]],
        [args.bounds_w[0],   args.bounds_w[1]],
        [args.bounds_cov[0], args.bounds_cov[1]],
    ], dtype=float)
    d = 3

    # Warm start
    X, y = warm_start_from_summary(Path(args.warm_start_from) if args.warm_start_from else None,
                                   bounds, topk=max(0, args.init))

    # Add random initials to reach --init
    while len(X) < args.init:
        X.append(np.random.rand(d).tolist())
        y.append(np.nan)

    log_csv = out_root / "bayes_log.csv"
    if not log_csv.exists():
        with open(log_csv, "w", encoding="utf-8") as f:
            f.write("iter,rho,w,cov,loss,run_dir\n")

    # Evaluate any NaN warm-start rows
    for i in range(len(X)):
        if not (isinstance(y[i], float) and math.isfinite(y[i])):
            theta = unscale_from_unit(np.array(X[i]), bounds)
            L, run_dir = evaluate_theta(tuple(theta.tolist()), cfg)
            y[i] = L
            with open(log_csv, "a", encoding="utf-8") as f:
                f.write(f"{len([v for v in y if math.isfinite(v)])},{theta[0]},{theta[1]},{theta[2]},{L},{run_dir}\n")
            # >>> progress print for initials
            print(
                f"[init {sum(math.isfinite(v) for v in y)}/{args.init}] "
                f"loss={L:.6f}  theta=({theta[0]:.4f},{theta[1]:.4f},{theta[2]:.4f})",
                flush=True
            )

    best_idx = int(np.argmin(y))
    best_y = float(y[best_idx])
    best_theta = unscale_from_unit(np.array(X[best_idx]), bounds)

    with open(out_root / "best_so_far.txt", "w", encoding="utf-8") as f:
        json.dump({"loss": best_y,
                   "rho": float(best_theta[0]),
                   "w": float(best_theta[1]),
                   "cov": float(best_theta[2])}, f, indent=2)

    # Remaining BO iterations
    remaining = max(0, args.iters - len(y))
    for _ in range(remaining):
        gp = gp_fit(np.array(X, float), np.array(y, float))

        # Propose next via EI
        Z = sample_candidates(args.cand, d)              # [cand, 3] in [0,1]
        mu, var = gp_predict(gp, Z)
        ei = expected_improvement(mu, var, best_y, xi=args.xi)
        z_next = Z[int(np.argmax(ei))]
        theta_next = unscale_from_unit(z_next, bounds)

        # Evaluate
        L, run_dir = evaluate_theta(tuple(theta_next.tolist()), cfg)
        X.append(z_next.tolist())
        y.append(L)

        with open(log_csv, "a", encoding="utf-8") as f:
            f.write(f"{len(y)},{theta_next[0]},{theta_next[1]},{theta_next[2]},{L},{run_dir}\n")

        if L < best_y:
            best_y = float(L)
            best_theta = theta_next.copy()
            with open(out_root / "best_so_far.txt", "w", encoding="utf-8") as f:
                json.dump({"loss": best_y,
                           "rho": float(best_theta[0]),
                           "w": float(best_theta[1]),
                           "cov": float(best_theta[2])}, f, indent=2)

        print(f"[iter {len(y)}] loss={L:.6f}  best={best_y:.6f}  "
              f"theta=({theta_next[0]:.4f},{theta_next[1]:.4f},{theta_next[2]:.4f})")

    print("\nDone. See:")
    print(f"  {log_csv}")
    print(f"  {out_root/'best_so_far.txt'}")
    print("  Per-eval subfolders contain all_simulation_results.json + analyzed metrics.")

        # --- Auto-plot best run(s) ---
    try:
        from plot import ensure_analyzed_metrics, load_episode_metrics, plot_metrics_comparison
        human_path = args.human

        # read best_so_far
        best_info = json.load(open(out_root / "best_so_far.txt", "r", encoding="utf-8"))
        best_loss = float(best_info["loss"])
        best_theta = np.array([best_info["rho"], best_info["w"], best_info["cov"]], dtype=float)

        # try to find a matching run_dir in this session's log (within tolerance)
        import csv
        TOL = 1e-9
        best_run_dir = None
        with open(log_csv, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    loss_row = float(row["loss"])
                except Exception:
                    continue
                if abs(loss_row - best_loss) <= TOL:
                    cand_dir = Path(row["run_dir"])
                    if cand_dir.exists():
                        best_run_dir = cand_dir
                        break

        # If best came from warm start (no local folder), re-evaluate it now
        if best_run_dir is None:
            print("Best comes from warm start; re-evaluating locally for artifacts & plotting...", flush=True)
            # Re-evaluate to create a local folder consistent with current settings
            L2, run_dir2 = evaluate_theta(tuple(best_theta.tolist()), cfg)
            best_run_dir = run_dir2
            # log it
            with open(log_csv, "a", encoding="utf-8") as f:
                f.write(f"{len(y)+1},{best_theta[0]},{best_theta[1]},{best_theta[2]},{L2},{run_dir2}\n")
            # if it actually beats previous best (due to randomness), update best_so_far.txt
            if L2 < best_loss:
                best_loss = L2
                with open(out_root / "best_so_far.txt", "w", encoding="utf-8") as f:
                    json.dump({"loss": float(best_loss),
                               "rho": float(best_theta[0]),
                               "w": float(best_theta[1]),
                               "cov": float(best_theta[2])}, f, indent=2)

        # Ensure analyzed metrics and plot
        per_ep_path = ensure_analyzed_metrics(best_run_dir)
        sim_episodes = load_episode_metrics(per_ep_path)
        out_path = best_run_dir / "comparison_human_vs_sim_best.png"
        plot_metrics_comparison(human_path, sim_episodes, str(out_path))
        print(f"\nAuto-plotted best run at {out_path}")

    except Exception as e:
        print(f"Auto-plotting failed: {e}")



if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    main()
