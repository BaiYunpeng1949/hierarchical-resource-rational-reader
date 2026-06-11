#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build aggregated_panel_metrics.json consistently from point-wise data.

Human:
- Eye metrics are aggregated directly from human_eye_movement_participant_metrics_pointwise.json
  at participant x time-condition level.
- Comprehension metrics are aggregated from comprehension_scores_p1_to_p32.csv by first
  averaging trials within participant x time-condition, then aggregating participant-level
  means across participants.

Simulation:
- Eye metrics are aggregated from simulation_eye_movement_metrics.json.
- Comprehension metrics are extracted from comprehension_results_*.json episodic_info.

The output keeps the same structure used by the plotting scripts:
  data[source][metric][condition] = {mean, std, n, values, value_unit}
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

CONDITIONS = ["30s", "60s", "90s"]
METRICS = [
    "reading_speed",
    "skip_rate",
    "regression_rate",
    "mcq_accuracy",
    "free_recall_score",
]
EYE_METRICS = ["reading_speed", "skip_rate", "regression_rate"]
COMP_METRICS = ["mcq_accuracy", "free_recall_score"]


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if x == "":
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.mean(arr)) if arr.size else float("nan")


def safe_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cond_label(x: Any) -> Optional[str]:
    """Normalize 30, '30', '30s', '30 s' to '30s'."""
    if x is None:
        return None
    s = str(x).strip().lower().replace(" ", "")
    if s.endswith("s"):
        s = s[:-1]
    try:
        n = int(float(s))
    except ValueError:
        return None
    label = f"{n}s"
    return label if label in CONDITIONS else None


def summarize(values: List[float], value_unit: str) -> Dict[str, Any]:
    return {
        "mean": safe_mean(values),
        "std": safe_std(values),
        "n": len(values),
        "values": values,
        "value_unit": value_unit,
    }


def build_human_eye_from_pointwise(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    buckets = {metric: {cond: [] for cond in CONDITIONS} for metric in EYE_METRICS}

    for row in rows:
        cond = cond_label(row.get("time_condition") or row.get("time_constraint") or row.get("total_time"))
        if cond not in CONDITIONS:
            continue
        for metric in EYE_METRICS:
            v = safe_float(row.get(metric))
            if v is not None:
                buckets[metric][cond].append(v)

    return {
        metric: {
            cond: summarize(buckets[metric][cond], "participant")
            for cond in CONDITIONS
        }
        for metric in EYE_METRICS
    }


def build_human_comp_from_csv(path: Path) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Return participant-level comprehension summaries and long-format pointwise rows.
    Each participant x condition value is the mean across that participant's trials.
    """
    trial_buckets: Dict[Tuple[int, str], Dict[str, List[float]]] = defaultdict(lambda: {m: [] for m in COMP_METRICS})

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid_raw = row.get("participant_index") or row.get("participant_id")
            try:
                pid = int(float(pid_raw))
            except (TypeError, ValueError):
                continue
            cond = cond_label(row.get("time_constraint") or row.get("time_condition") or row.get("total_time"))
            if cond not in CONDITIONS:
                continue

            mcq = safe_float(row.get("MCQ Accuracy") or row.get("mcq_accuracy"))
            fr = safe_float(row.get("Free Recall Score") or row.get("free_recall_score"))
            if mcq is not None:
                trial_buckets[(pid, cond)]["mcq_accuracy"].append(mcq)
            if fr is not None:
                trial_buckets[(pid, cond)]["free_recall_score"].append(fr)

    comp_values = {metric: {cond: [] for cond in CONDITIONS} for metric in COMP_METRICS}
    participant_rows = []

    for pid, cond in sorted(trial_buckets.keys()):
        item = trial_buckets[(pid, cond)]
        row = {
            "source": "human",
            "participant_id": pid,
            "time_condition": cond,
            "total_time": int(cond.replace("s", "")),
            "mcq_accuracy": None,
            "free_recall_score": None,
            "n_comp_trials": 0,
        }
        n_trials = 0
        for metric in COMP_METRICS:
            vals = item[metric]
            if vals:
                mean_val = safe_mean(vals)
                comp_values[metric][cond].append(mean_val)
                row[metric] = mean_val
                n_trials = max(n_trials, len(vals))
        row["n_comp_trials"] = n_trials
        participant_rows.append(row)

    human_comp = {
        metric: {
            cond: summarize(comp_values[metric][cond], "participant_mean_across_trials")
            for cond in CONDITIONS
        }
        for metric in COMP_METRICS
    }
    return human_comp, participant_rows


def build_simulation_blocks(s_eye_rows: List[Dict[str, Any]], s_comp: Dict[str, Any]):
    comp_lookup: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
    for ep in s_comp.get("results", []):
        if not isinstance(ep, dict):
            continue
        key = (ep.get("episode_index"), ep.get("stimulus_index"), cond_label(ep.get("time_condition")))
        epi = ep.get("episodic_info", {}) or {}
        comp_lookup[key] = {
            "mcq_accuracy": safe_float(epi.get("mcq_accuracy")),
            "free_recall_score": safe_float(epi.get("free_recall_score")),
            "n_mcq": epi.get("n_mcq"),
        }

    values = {metric: {cond: [] for cond in CONDITIONS} for metric in METRICS}
    pointwise_rows = []

    for row in s_eye_rows:
        cond = cond_label(row.get("time_condition") or row.get("total_time"))
        if cond not in CONDITIONS:
            continue
        key = (row.get("episode_index"), row.get("stimulus_index"), cond)
        comp = comp_lookup.get(key, {})
        out_row = {
            "source": "simulation",
            "episode_index": row.get("episode_index"),
            "stimulus_index": row.get("stimulus_index"),
            "time_condition": cond,
            "total_time": row.get("total_time"),
            "reading_speed": safe_float(row.get("reading_speed")),
            "skip_rate": safe_float(row.get("skip_rate")),
            "regression_rate": safe_float(row.get("regression_rate")),
            "mcq_accuracy": comp.get("mcq_accuracy"),
            "free_recall_score": comp.get("free_recall_score"),
            "n_mcq": comp.get("n_mcq"),
        }
        pointwise_rows.append(out_row)
        for metric in METRICS:
            v = out_row.get(metric)
            if v is not None:
                values[metric][cond].append(float(v))

    simulation = {
        metric: {
            cond: summarize(values[metric][cond], "episode_stimulus")
            for cond in CONDITIONS
        }
        for metric in METRICS
    }
    return simulation, pointwise_rows


def merge_human_pointwise(human_eye_rows: List[Dict[str, Any]], human_comp_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for row in human_comp_rows:
        key = (int(row["participant_id"]), row["time_condition"])
        lookup[key] = dict(row)

    # Start from comprehension rows so participants without eye metrics are retained.
    merged = {key: dict(row) for key, row in lookup.items()}

    for row in human_eye_rows:
        pid_raw = row.get("participant_id") or row.get("participant_index")
        try:
            pid = int(float(pid_raw))
        except (TypeError, ValueError):
            continue
        cond = cond_label(row.get("time_condition") or row.get("time_constraint") or row.get("total_time"))
        if cond not in CONDITIONS:
            continue
        key = (pid, cond)
        base = merged.setdefault(key, {
            "source": "human",
            "participant_id": pid,
            "time_condition": cond,
            "total_time": int(cond.replace("s", "")),
            "mcq_accuracy": None,
            "free_recall_score": None,
            "n_comp_trials": 0,
        })
        for metric in EYE_METRICS:
            base[metric] = safe_float(row.get(metric))

    # Ensure all fields exist and stable order.
    out = []
    for key in sorted(merged.keys()):
        row = merged[key]
        out.append({
            "source": "human",
            "participant_id": row.get("participant_id"),
            "time_condition": row.get("time_condition"),
            "total_time": row.get("total_time"),
            "reading_speed": row.get("reading_speed"),
            "skip_rate": row.get("skip_rate"),
            "regression_rate": row.get("regression_rate"),
            "mcq_accuracy": row.get("mcq_accuracy"),
            "free_recall_score": row.get("free_recall_score"),
            "n_comp_trials": row.get("n_comp_trials"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human_eye_pointwise", type=str, default="human_eye_movement_participant_metrics_pointwise.json")
    ap.add_argument("--human_comp_pointwise", type=str, default="comprehension_scores_p1_to_p32.csv")
    ap.add_argument("--sim_eye", type=str, default="simulation_eye_movement_metrics.json")
    ap.add_argument("--sim_comp", type=str, default="comprehension_results_20251006-150327.json")
    ap.add_argument("--out", type=str, default="aggregated_panel_metrics.json")
    args = ap.parse_args()

    p_h_eye = Path(args.human_eye_pointwise)
    p_h_comp = Path(args.human_comp_pointwise)
    p_s_eye = Path(args.sim_eye)
    p_s_comp = Path(args.sim_comp)
    p_out = Path(args.out)

    h_eye_rows = load_json(p_h_eye)
    s_eye_rows = load_json(p_s_eye)
    s_comp = load_json(p_s_comp)

    human_eye = build_human_eye_from_pointwise(h_eye_rows)
    human_comp, human_comp_rows = build_human_comp_from_csv(p_h_comp)
    human = {**human_eye, **human_comp}
    human_pointwise = merge_human_pointwise(h_eye_rows, human_comp_rows)

    simulation, simulation_pointwise = build_simulation_blocks(s_eye_rows, s_comp)

    out = {
        "conditions": CONDITIONS,
        "human": human,
        "simulation": simulation,
        "pointwise": {
            "human": human_pointwise,
            "simulation": simulation_pointwise,
        },
        "meta": {
            "human_eye_pointwise_file": p_h_eye.name,
            "human_comp_pointwise_file": p_h_comp.name,
            "sim_eye_file": p_s_eye.name,
            "sim_comp_file": p_s_comp.name,
            "std_ddof": 1,
            "human_aggregate_source": "pointwise files only",
            "simulation_aggregate_source": "episode/stimulus pointwise files",
        },
    }

    p_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote: {p_out.resolve()}")

    p_h_long = p_out.with_name(p_out.stem + "_human_pointwise.json")
    p_s_long = p_out.with_name(p_out.stem + "_simulation_pointwise.json")
    p_h_long.write_text(json.dumps(human_pointwise, indent=2), encoding="utf-8")
    p_s_long.write_text(json.dumps(simulation_pointwise, indent=2), encoding="utf-8")
    print(f"Wrote: {p_h_long.resolve()}")
    print(f"Wrote: {p_s_long.resolve()}")


if __name__ == "__main__":
    main()
