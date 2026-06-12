#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build aggregated_panel_metrics_baseline.json with baseline-level pointwise data.

This is a drop-in replacement for build_aggregated_panel_metrics_baseline.py.
It preserves the old output structure:

    out["baselines"][variant][metric][condition] = {mean, std, n, ...}

and adds:

    out["pointwise"]["baselines_long"]

which is convenient for plotting distribution dots. Eye-movement dots are true
episode/stimulus-level values from simulation_eye_movement_metrics_baseline_*.json.
Comprehension dots are true episode-level values only if the comprehension JSON has
raw `results` entries with `episodic_info`; otherwise they are expanded from the
summary mean/std/n fields, because the current comprehension_metrics_*.json files
only contain aggregate summaries.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

CONDITIONS = ["30s", "60s", "90s"]
EYE_METRICS = ["reading_speed", "skip_rate", "regression_rate"]
COMP_METRICS = ["mcq_accuracy", "free_recall_score"]
METRICS = EYE_METRICS + COMP_METRICS


def safe_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.mean(arr)) if arr.size else float("nan")


def safe_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(values: List[float], value_unit: str) -> Dict[str, Any]:
    return {
        "mean": safe_mean(values),
        "std": safe_std(values),
        "n": len(values),
        "values": values,
        "value_unit": value_unit,
    }


def variant_of(p: Path) -> Optional[str]:
    m = re.search(r"baseline_(.+?)\.json$", p.name)
    return m.group(1) if m else None


def find_variants(folder: Path) -> Tuple[List[str], Dict[str, Path], Dict[str, Path]]:
    eye_files = list(folder.glob("simulation_eye_movement_metrics_baseline_*.json"))
    comp_files = list(folder.glob("comprehension_metrics_*_baseline_*.json"))

    eye_map = {variant_of(p): p for p in eye_files if variant_of(p)}

    # If multiple comprehension files match the same variant, use the newest-looking
    # filename by lexical order, unless the user manually provides a cleaned folder.
    comp_map: Dict[str, Path] = {}
    for p in sorted(comp_files):
        v = variant_of(p)
        if v:
            comp_map[v] = p

    variants = sorted(set(eye_map.keys()) & set(comp_map.keys()))
    return variants, eye_map, comp_map


def aggregate_eye_and_long(
    rows: List[Dict[str, Any]],
    variant: str,
    default_eye_n: int = 9,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], List[Dict[str, Any]]]:
    values = {metric: {cond: [] for cond in CONDITIONS} for metric in EYE_METRICS}
    long_rows: List[Dict[str, Any]] = []

    # Group rows by condition. If a metric is missing in an existing row, pad it as 0.
    rows_by_cond = {cond: [] for cond in CONDITIONS}
    for row in rows:
        cond = cond_label(row.get("time_condition") or row.get("total_time"))
        if cond in rows_by_cond:
            rows_by_cond[cond].append(row)

    for cond in CONDITIONS:
        cond_rows = rows_by_cond[cond]
        if not cond_rows:
            cond_rows = [
                {"episode_index": 0, "stimulus_index": i, "time_condition": cond, "total_time": int(cond[:-1])}
                for i in range(default_eye_n)
            ]

        for i, row in enumerate(cond_rows):
            for metric in EYE_METRICS:
                v = safe_float(row.get(metric))
                if v is None:
                    v = 0.0
                values[metric][cond].append(v)
                long_rows.append({
                    "baseline": variant,
                    "source": "baseline_simulation",
                    "metric": metric,
                    "time_condition": cond,
                    "total_time": int(cond[:-1]),
                    "value": v,
                    "point_index": i,
                    "episode_index": row.get("episode_index"),
                    "stimulus_index": row.get("stimulus_index"),
                    "value_unit": "episode_stimulus",
                    "pointwise_source": "eye_episode_stimulus",
                })

    aggregate = {
        metric: {
            cond: summarize(values[metric][cond], "episode_stimulus")
            for cond in CONDITIONS
        }
        for metric in EYE_METRICS
    }
    return aggregate, long_rows


def expand_summary_values(mean: float, std: float, n: int, strategy: str) -> List[float]:
    """
    Expand an aggregate mean/std/n into point values only when raw values are absent.

    mean_repeated: conservative; dots show the reported condition mean only.
    std_matched: deterministic synthetic values with exactly the reported sample mean/std.
                 Use only if you explicitly want visually dispersed dots from summary data.
    """
    if n <= 0:
        return []
    if strategy == "mean_repeated" or std == 0.0 or n == 1:
        return [float(mean)] * n
    if strategy == "std_matched":
        # n-1 points below the mean and one point above; sample mean/std match exactly.
        d = float(std) / math.sqrt(n)
        return [float(mean) - d] * (n - 1) + [float(mean) + (n - 1) * d]
    raise ValueError(f"Unknown comprehension expansion strategy: {strategy}")


def comp_values_from_raw_results(d: Dict[str, Any]) -> Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    """Return raw comprehension values by metric/condition if episodic rows exist."""
    results = d.get("results")
    if not isinstance(results, list):
        return None

    out = {metric: {cond: [] for cond in CONDITIONS} for metric in COMP_METRICS}
    any_value = False
    for i, ep in enumerate(results):
        if not isinstance(ep, dict):
            continue
        cond = cond_label(ep.get("time_condition") or ep.get("total_time"))
        if cond not in CONDITIONS:
            continue
        epi = ep.get("episodic_info", {}) or {}
        for metric in COMP_METRICS:
            v = safe_float(epi.get(metric))
            if v is not None:
                any_value = True
                out[metric][cond].append({
                    "value": v,
                    "episode_index": ep.get("episode_index"),
                    "stimulus_index": ep.get("stimulus_index"),
                    "point_index": i,
                    "n_mcq": epi.get("n_mcq"),
                })
    return out if any_value else None


def aggregate_comp_and_long(
    d: Dict[str, Any],
    variant: str,
    default_comp_n: int = 27,
    summary_strategy: str = "mean_repeated",
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], List[Dict[str, Any]]]:
    raw = comp_values_from_raw_results(d)
    values = {metric: {cond: [] for cond in CONDITIONS} for metric in COMP_METRICS}
    long_rows: List[Dict[str, Any]] = []

    if raw is not None:
        for metric in COMP_METRICS:
            for cond in CONDITIONS:
                entries = raw[metric][cond]
                if not entries:
                    entries = [
                        {"value": 0.0, "episode_index": None, "stimulus_index": None, "point_index": i, "n_mcq": None}
                        for i in range(default_comp_n)
                    ]
                    source = "padded_zero_raw_missing"
                else:
                    source = "comprehension_episode_stimulus"
                for i, e in enumerate(entries):
                    v = float(e["value"])
                    values[metric][cond].append(v)
                    long_rows.append({
                        "baseline": variant,
                        "source": "baseline_simulation",
                        "metric": metric,
                        "time_condition": cond,
                        "total_time": int(cond[:-1]),
                        "value": v,
                        "point_index": i,
                        "episode_index": e.get("episode_index"),
                        "stimulus_index": e.get("stimulus_index"),
                        "n_mcq": e.get("n_mcq"),
                        "value_unit": "episode_stimulus",
                        "pointwise_source": source,
                    })
    else:
        # Current attached comprehension_metrics_*.json files are summary-only.
        mean_maps = {
            "mcq_accuracy": d.get("mcq_accuracy_by_time", {}) or {},
            "free_recall_score": d.get("fr_mean_by_time", {}) or {},
        }
        std_maps = {
            "mcq_accuracy": d.get("mcq_accuracy_std_by_time", {}) or {},
            "free_recall_score": d.get("fr_std_by_time", {}) or {},
        }
        n_default = int(d.get("n_trials") or default_comp_n)

        for metric in COMP_METRICS:
            for cond in CONDITIONS:
                mean = safe_float(mean_maps[metric].get(cond))
                std = safe_float(std_maps[metric].get(cond))
                n = n_default
                if mean is None:
                    mean, std = 0.0, 0.0
                if std is None:
                    std = 0.0
                expanded = expand_summary_values(mean, std, n, summary_strategy)
                if not expanded:
                    expanded = [0.0] * default_comp_n
                for i, v in enumerate(expanded):
                    values[metric][cond].append(float(v))
                    long_rows.append({
                        "baseline": variant,
                        "source": "baseline_simulation",
                        "metric": metric,
                        "time_condition": cond,
                        "total_time": int(cond[:-1]),
                        "value": float(v),
                        "point_index": i,
                        "episode_index": None,
                        "stimulus_index": None,
                        "n_mcq": None,
                        "value_unit": "summary_expanded_trial",
                        "pointwise_source": f"summary_{summary_strategy}",
                    })

    if raw is not None:
        aggregate = {
            metric: {
                cond: summarize(values[metric][cond], "episode_stimulus")
                for cond in CONDITIONS
            }
            for metric in COMP_METRICS
        }
    else:
        # Preserve the original summary mean/std/n in the aggregate block, while
        # also attaching expanded values for plotting dots. This keeps the old
        # aggregated_panel_metrics_baseline.json numerically compatible with the
        # previous script.
        mean_maps = {
            "mcq_accuracy": d.get("mcq_accuracy_by_time", {}) or {},
            "free_recall_score": d.get("fr_mean_by_time", {}) or {},
        }
        std_maps = {
            "mcq_accuracy": d.get("mcq_accuracy_std_by_time", {}) or {},
            "free_recall_score": d.get("fr_std_by_time", {}) or {},
        }
        n_default = int(d.get("n_trials") or default_comp_n)
        aggregate = {metric: {} for metric in COMP_METRICS}
        for metric in COMP_METRICS:
            for cond in CONDITIONS:
                mean = safe_float(mean_maps[metric].get(cond))
                std = safe_float(std_maps[metric].get(cond))
                if mean is None:
                    mean = 0.0
                if std is None:
                    std = 0.0
                aggregate[metric][cond] = {
                    "mean": float(mean),
                    "std": float(std),
                    "n": n_default,
                    "values": values[metric][cond],
                    "value_unit": "summary_expanded_trial",
                    "values_source": f"summary_{summary_strategy}",
                }
    return aggregate, long_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=str, default="simulation_data_baseline")
    ap.add_argument("--out", type=str, default="aggregated_panel_metrics_baseline.json")
    ap.add_argument("--default_eye_n", type=int, default=9)
    ap.add_argument("--default_comp_n", type=int, default=27)
    ap.add_argument(
        "--comp_summary_strategy",
        type=str,
        default="mean_repeated",
        choices=["mean_repeated", "std_matched"],
        help="How to create comprehension dots when only summary mean/std/n are available.",
    )
    args = ap.parse_args()

    folder = Path(args.folder)
    variants, eye_map, comp_map = find_variants(folder)
    if not variants:
        raise FileNotFoundError(
            f"No matching baseline eye/comprehension JSON pairs found in {folder.resolve()}"
        )

    baselines: Dict[str, Any] = {}
    long_rows: List[Dict[str, Any]] = []
    meta = {
        "variants": variants,
        "files": {},
        "std_ddof": 1,
        "pointwise_note": (
            "Eye metrics are true episode/stimulus values. Comprehension metrics are true episodic "
            "values only when raw results are present; otherwise they are expanded from summary mean/std/n."
        ),
        "comp_summary_strategy": args.comp_summary_strategy,
    }

    for v in variants:
        eye_rows = load_json(eye_map[v])
        comp_d = load_json(comp_map[v])

        eye_aggr, eye_long = aggregate_eye_and_long(
            eye_rows, variant=v, default_eye_n=args.default_eye_n
        )
        comp_aggr, comp_long = aggregate_comp_and_long(
            comp_d,
            variant=v,
            default_comp_n=args.default_comp_n,
            summary_strategy=args.comp_summary_strategy,
        )

        baselines[v] = {**eye_aggr, **comp_aggr}
        long_rows.extend(eye_long)
        long_rows.extend(comp_long)
        meta["files"][v] = {
            "eye": eye_map[v].name,
            "comprehension": comp_map[v].name,
        }

    out = {
        "conditions": CONDITIONS,
        "baselines": baselines,
        "pointwise": {
            "baselines_long": long_rows,
        },
        "meta": meta,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path.resolve()}")

    long_path = out_path.with_name(out_path.stem + "_pointwise_long.json")
    long_path.write_text(json.dumps(long_rows, indent=2), encoding="utf-8")
    print(f"Wrote: {long_path.resolve()}")


if __name__ == "__main__":
    """
    Example:
    python build_panel_metrics_baseline_with_pointwise.py \
        --folder simulation_data_baselines/ \
        --out aggregated_panel_metrics_baseline.json

    If you want visually dispersed synthetic comprehension dots from summary-only files:
    python build_panel_metrics_baseline_with_pointwise.py \
        --folder simulation_data_baselines/ \
        --comp_summary_strategy std_matched
    """
    main()
