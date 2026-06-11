#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_aggregated_panel_metrics_with_human_pointwise.py

Build aggregated panel metrics for human + simulation data, while retaining
point-wise values for plotting dots over bars.

Important design choice:
- Human bar means/stds are still read from the existing aggregate files:
  human_eye_movement_metrics.json, human_mcq_acc_metrics.json,
  human_free_recall_metrics.json.
- Human point-wise values are read from:
  human_eye_movement_participant_metrics_pointwise.json
  comprehension_scores_p1_to_p32.csv
- Simulation means/stds/values are built from the simulation raw metric files,
  as before.

Output:
- aggregated_panel_metrics.json
- aggregated_panel_metrics_simulation_pointwise.json
- aggregated_panel_metrics_human_pointwise.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

CONDITIONS = ["30s", "60s", "90s"]
METRICS = [
    "reading_speed",
    "skip_rate",
    "regression_rate",
    "mcq_accuracy",
    "free_recall_score",
]


def safe_mean(a: List[float]) -> float:
    arr = np.array(a, dtype=float)
    return float(np.mean(arr)) if arr.size else float("nan")


def safe_std(a: List[float]) -> float:
    arr = np.array(a, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_condition(x: Any) -> str:
    """Return condition labels as 30s/60s/90s."""
    if isinstance(x, str):
        x = x.strip()
        if x in CONDITIONS:
            return x
        if x.endswith("s") and x[:-1].isdigit():
            return x
        if x.isdigit():
            return f"{int(x)}s"
    if isinstance(x, (int, float)) and not pd.isna(x):
        return f"{int(x)}s"
    return str(x)


def add_aggregate_entry(mean, std, n, values=None, value_unit=None, aggregate_n=None):
    """
    Return one metric/condition entry.

    n is the point-wise n used for dots if values are supplied.
    aggregate_n keeps the original n from the aggregate files when different.
    """
    out = {
        "mean": mean,
        "std": std,
        "n": len(values) if values is not None else n,
    }
    if aggregate_n is not None:
        out["aggregate_n"] = aggregate_n
    if values is not None:
        out["values"] = values
        out["value_unit"] = value_unit or "pointwise"
    return out


def build_human_pointwise_rows(human_eye_pointwise: List[Dict], human_comp_csv: Path) -> List[Dict]:
    """
    Build human point-wise rows in the same long format as the simulation rows.

    Eye rows are already participant-level: one participant x condition row.
    Comprehension CSV is trial-level, so this function aggregates it to
    participant x condition. The final rows are the UNION of eye participants
    and comprehension participants, so comprehension participants are not
    dropped if they do not appear in the eye file.
    """
    comp_df = pd.read_csv(human_comp_csv)
    required = {"participant_index", "time_constraint", "MCQ Accuracy", "Free Recall Score"}
    missing = required - set(comp_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {human_comp_csv}: {sorted(missing)}")

    comp_df = comp_df.copy()
    comp_df["participant_id"] = comp_df["participant_index"].astype(int)
    comp_df["time_condition"] = comp_df["time_constraint"].apply(normalize_condition)

    comp_part = (
        comp_df.groupby(["participant_id", "time_condition"], as_index=False)
        .agg(
            mcq_accuracy=("MCQ Accuracy", "mean"),
            free_recall_score=("Free Recall Score", "mean"),
            n_comp_trials=("MCQ Accuracy", "size"),
        )
    )

    eye_lookup = {}
    for row in human_eye_pointwise:
        pid = int(row.get("participant_id"))
        cond = normalize_condition(row.get("time_condition"))
        eye_lookup[(pid, cond)] = row

    comp_lookup = {
        (int(r["participant_id"]), r["time_condition"]): r
        for _, r in comp_part.iterrows()
    }

    all_keys = sorted(
        set(eye_lookup.keys()) | set(comp_lookup.keys()),
        key=lambda x: (x[0], CONDITIONS.index(x[1]) if x[1] in CONDITIONS else 999)
    )

    rows = []
    for participant_id, cond in all_keys:
        eye = eye_lookup.get((participant_id, cond), {})
        comp = comp_lookup.get((participant_id, cond), {})
        rows.append({
            "source": "human",
            "participant_id": participant_id,
            "time_condition": cond,
            "total_time": int(cond.replace("s", "")) if cond in CONDITIONS else eye.get("total_time"),
            "reading_speed": eye.get("reading_speed"),
            "skip_rate": eye.get("skip_rate"),
            "regression_rate": eye.get("regression_rate"),
            "mcq_accuracy": None if len(comp) == 0 else float(comp["mcq_accuracy"]),
            "free_recall_score": None if len(comp) == 0 else float(comp["free_recall_score"]),
            "n_comp_trials": None if len(comp) == 0 else int(comp["n_comp_trials"]),
        })

    return rows

def values_from_rows(rows: List[Dict], metric: str, cond: str) -> List[float]:
    vals = []
    for r in rows:
        if normalize_condition(r.get("time_condition")) != cond:
            continue
        v = r.get(metric)
        if isinstance(v, (int, float)) and not pd.isna(v):
            vals.append(float(v))
    return vals


def build_human_block(h_eye: Dict, h_mcq: Dict, h_fr: Dict, human_pointwise_rows=None) -> Dict:
    """
    Human aggregate means/stds are read from the existing aggregate JSON files.
    If human_pointwise_rows are supplied, values are attached for plotting dots.
    """
    human_pointwise_rows = human_pointwise_rows or []
    human = {m: {} for m in METRICS}

    for cond in CONDITIONS:
        he = h_eye[cond]

        for metric, mean_key, std_key in [
            ("reading_speed", "reading_speed_mean", "reading_speed_std"),
            ("skip_rate", "skip_rate_mean", "skip_rate_std"),
            ("regression_rate", "regression_rate_mean", "regression_rate_std"),
        ]:
            vals = values_from_rows(human_pointwise_rows, metric, cond)
            human[metric][cond] = add_aggregate_entry(
                mean=he.get(mean_key),
                std=he.get(std_key),
                n=he.get("num_episodes"),
                values=vals if vals else None,
                value_unit="participant" if vals else None,
                aggregate_n=he.get("num_episodes") if vals else None,
            )

        vals = values_from_rows(human_pointwise_rows, "mcq_accuracy", cond)
        human["mcq_accuracy"][cond] = add_aggregate_entry(
            mean=h_mcq["mcq_mean_by_time"][cond],
            std=h_mcq["mcq_std_by_time"][cond],
            n=h_mcq.get("n_scored"),
            values=vals if vals else None,
            value_unit="participant_mean_across_trials" if vals else None,
            aggregate_n=h_mcq.get("n_scored") if vals else None,
        )

        vals = values_from_rows(human_pointwise_rows, "free_recall_score", cond)
        human["free_recall_score"][cond] = add_aggregate_entry(
            mean=h_fr["fr_mean_by_time"][cond],
            std=h_fr["fr_std_by_time"][cond],
            n=h_fr.get("n_scored"),
            values=vals if vals else None,
            value_unit="participant_mean_across_trials" if vals else None,
            aggregate_n=h_fr.get("n_scored") if vals else None,
        )

    return human


def build_simulation_block(s_eye_rows: List[Dict], s_comp: Dict) -> Dict:
    eye_buckets = {c: {k: [] for k in ["reading_speed", "skip_rate", "regression_rate"]} for c in CONDITIONS}
    for row in s_eye_rows:
        cond = normalize_condition(row.get("time_condition"))
        if cond in eye_buckets:
            for k in ["reading_speed", "skip_rate", "regression_rate"]:
                v = row.get(k)
                if isinstance(v, (int, float)):
                    eye_buckets[cond][k].append(float(v))

    comp_buckets = {c: {k: [] for k in ["mcq_accuracy", "free_recall_score"]} for c in CONDITIONS}
    for ep in s_comp.get("results", []):
        if not isinstance(ep, dict):
            continue
        cond = normalize_condition(ep.get("time_condition"))
        epi = ep.get("episodic_info", {}) or {}
        if cond in comp_buckets:
            for k in ["mcq_accuracy", "free_recall_score"]:
                v = epi.get(k)
                if isinstance(v, (int, float)):
                    comp_buckets[cond][k].append(float(v))

    sim = {m: {} for m in METRICS}
    for cond in CONDITIONS:
        for k in ["reading_speed", "skip_rate", "regression_rate"]:
            arr = eye_buckets[cond][k]
            sim[k][cond] = {"mean": safe_mean(arr), "std": safe_std(arr), "n": len(arr), "values": arr, "value_unit": "episode_stimulus"}
        for k in ["mcq_accuracy", "free_recall_score"]:
            arr = comp_buckets[cond][k]
            sim[k][cond] = {"mean": safe_mean(arr), "std": safe_std(arr), "n": len(arr), "values": arr, "value_unit": "episode_stimulus"}
    return sim


def build_simulation_pointwise_rows(s_eye_rows: List[Dict], s_comp: Dict) -> List[Dict]:
    comp_lookup = {}
    for ep in s_comp.get("results", []):
        if not isinstance(ep, dict):
            continue
        key = (ep.get("episode_index"), ep.get("stimulus_index"), normalize_condition(ep.get("time_condition")))
        epi = ep.get("episodic_info", {}) or {}
        comp_lookup[key] = {
            "mcq_accuracy": epi.get("mcq_accuracy"),
            "free_recall_score": epi.get("free_recall_score"),
            "n_mcq": epi.get("n_mcq"),
        }

    rows = []
    for row in s_eye_rows:
        cond = normalize_condition(row.get("time_condition"))
        key = (row.get("episode_index"), row.get("stimulus_index"), cond)
        comp = comp_lookup.get(key, {})
        rows.append({
            "source": "simulation",
            "episode_index": row.get("episode_index"),
            "stimulus_index": row.get("stimulus_index"),
            "time_condition": cond,
            "total_time": row.get("total_time"),
            "reading_speed": row.get("reading_speed"),
            "skip_rate": row.get("skip_rate"),
            "regression_rate": row.get("regression_rate"),
            "mcq_accuracy": comp.get("mcq_accuracy"),
            "free_recall_score": comp.get("free_recall_score"),
            "n_mcq": comp.get("n_mcq"),
        })
    return rows


def build_pointwise_checks(human: Dict, human_rows: List[Dict]) -> List[Dict]:
    checks = []
    for metric in METRICS:
        for cond in CONDITIONS:
            vals = values_from_rows(human_rows, metric, cond)
            if not vals:
                continue
            checks.append({
                "source": "human",
                "metric": metric,
                "time_condition": cond,
                "aggregate_mean": human[metric][cond]["mean"],
                "pointwise_mean": safe_mean(vals),
                "abs_difference": abs(float(human[metric][cond]["mean"]) - safe_mean(vals)),
                "n_values": len(vals),
                "value_unit": human[metric][cond].get("value_unit"),
            })
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human_eye", type=str, default="human_eye_movement_metrics.json")
    ap.add_argument("--human_mcq", type=str, default="human_mcq_acc_metrics.json")
    ap.add_argument("--human_fr", type=str, default="human_free_recall_metrics.json")
    ap.add_argument("--human_eye_pointwise", type=str, default="human_eye_movement_participant_metrics_pointwise.json")
    ap.add_argument("--human_comp_pointwise", type=str, default="comprehension_scores_p1_to_p32.csv")
    ap.add_argument("--sim_eye", type=str, default="simulation_eye_movement_metrics.json")
    ap.add_argument("--sim_comp", type=str, default="comprehension_results_20251006-150327.json")
    ap.add_argument("--out", type=str, default="aggregated_panel_metrics.json")
    args = ap.parse_args()

    p_h_eye = Path(args.human_eye)
    p_h_mcq = Path(args.human_mcq)
    p_h_fr = Path(args.human_fr)
    p_h_eye_pointwise = Path(args.human_eye_pointwise)
    p_h_comp_pointwise = Path(args.human_comp_pointwise)
    p_s_eye = Path(args.sim_eye)
    p_s_comp = Path(args.sim_comp)
    p_out = Path(args.out)

    h_eye = load_json(p_h_eye)
    h_mcq = load_json(p_h_mcq)
    h_fr = load_json(p_h_fr)
    h_eye_pointwise = load_json(p_h_eye_pointwise)
    s_eye = load_json(p_s_eye)
    s_comp = load_json(p_s_comp)

    human_pointwise = build_human_pointwise_rows(h_eye_pointwise, p_h_comp_pointwise)
    human = build_human_block(h_eye, h_mcq, h_fr, human_pointwise)
    simulation = build_simulation_block(s_eye, s_comp)
    simulation_pointwise = build_simulation_pointwise_rows(s_eye, s_comp)
    human_checks = build_pointwise_checks(human, human_pointwise)

    out = {
        "conditions": CONDITIONS,
        "human": human,
        "simulation": simulation,
        "pointwise": {
            "human": human_pointwise,
            "simulation": simulation_pointwise,
        },
        "meta": {
            "human_eye_file": p_h_eye.name,
            "human_mcq_file": p_h_mcq.name,
            "human_fr_file": p_h_fr.name,
            "human_eye_pointwise_file": p_h_eye_pointwise.name,
            "human_comp_pointwise_file": p_h_comp_pointwise.name,
            "sim_eye_file": p_s_eye.name,
            "sim_comp_file": p_s_comp.name,
            "std_ddof": 1,
            "human_aggregate_source": "aggregate JSON files",
            "human_pointwise_source": "participant-level eye JSON and participant-aggregated comprehension CSV",
            "human_pointwise_checks": human_checks,
        },
    }

    p_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote: {p_out.resolve()}")

    p_sim_pw = p_out.with_name(p_out.stem + "_simulation_pointwise.json")
    p_sim_pw.write_text(json.dumps(simulation_pointwise, indent=2), encoding="utf-8")
    print(f"Wrote: {p_sim_pw.resolve()}")

    p_hum_pw = p_out.with_name(p_out.stem + "_human_pointwise.json")
    p_hum_pw.write_text(json.dumps(human_pointwise, indent=2), encoding="utf-8")
    print(f"Wrote: {p_hum_pw.resolve()}")

    # Print warnings when point-wise values do not reproduce aggregate means.
    # This can happen if aggregate JSONs and the point-wise file use different
    # scoring definitions or different analysis units.
    large = [c for c in human_checks if c["abs_difference"] > 1e-6]
    if large:
        print("\n[CHECK] Some human point-wise means differ from aggregate JSON means:")
        for c in large:
            print(
                f"  {c['metric']} {c['time_condition']}: "
                f"aggregate={c['aggregate_mean']:.6f}, "
                f"pointwise={c['pointwise_mean']:.6f}, "
                f"diff={c['abs_difference']:.6f}, "
                f"n_values={c['n_values']}"
            )


if __name__ == "__main__":
    main()
