#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, re
from pathlib import Path
import numpy as np

CONDITIONS = ["30s","60s","90s"]

def safe_mean(a):
    import numpy as np
    arr = np.array(a, dtype=float)
    return float(np.mean(arr)) if arr.size else float("nan")

def safe_std(a):
    import numpy as np
    arr = np.array(a, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))

def find_variants(folder: Path):
    eye_files = list(folder.glob("simulation_eye_movement_metrics_baseline_*.json"))
    comp_files = list(folder.glob("comprehension_metrics_*_baseline_*.json"))
    def variant_of(p: Path):
        m = re.search(r"baseline_(.+?)\.json$", p.name)
        return m.group(1) if m else None
    eye_map = {variant_of(p): p for p in eye_files if variant_of(p)}
    comp_map = {}
    for p in comp_files:
        v = variant_of(p)
        if v and v not in comp_map:
            comp_map[v] = p
    variants = sorted(set(eye_map.keys()) & set(comp_map.keys()))
    return variants, eye_map, comp_map

def aggregate_eye(rows):
    buckets = {c: {k: [] for k in ["reading_speed","skip_rate","regression_rate"]} for c in CONDITIONS}
    for row in rows:
        cond = row.get("time_condition")
        if cond in buckets:
            for k in ["reading_speed","skip_rate","regression_rate"]:
                v = row.get(k)
                if isinstance(v, (int, float)):
                    buckets[cond][k].append(float(v))
    out = {k: {} for k in ["reading_speed","skip_rate","regression_rate"]}
    for cond in CONDITIONS:
        for k in out.keys():
            arr = buckets[cond][k]
            out[k][cond] = {"mean": safe_mean(arr), "std": safe_std(arr), "n": len(arr)}
    return out

def parse_comprehension(d):
    mcq_m = d.get("mcq_accuracy_by_time", {})
    mcq_s = d.get("mcq_accuracy_std_by_time", {})
    fr_m  = d.get("fr_mean_by_time", {})
    fr_s  = d.get("fr_std_by_time", {})
    ntr   = d.get("n_trials", None)
    out = {"mcq_accuracy": {}, "free_recall_score": {}}
    for cond in CONDITIONS:
        out["mcq_accuracy"][cond] = {"mean": float(mcq_m.get(cond, float("nan"))),
                                     "std":  float(mcq_s.get(cond, 0.0)),
                                     "n":    ntr}
        out["free_recall_score"][cond] = {"mean": float(fr_m.get(cond, float("nan"))),
                                          "std":  float(fr_s.get(cond, 0.0)),
                                          "n":    ntr}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=str, default="simulation_data_baseline")
    ap.add_argument("--out", type=str, default="aggregated_panel_metrics_baseline.json")
    args = ap.parse_args()

    folder = Path(args.folder)
    variants, eye_map, comp_map = find_variants(folder)

    baselines = {}
    meta = {"variants": variants, "files": {}}
    for v in variants:
        eye_rows = json.loads(Path(eye_map[v]).read_text(encoding="utf-8"))
        comp_d   = json.loads(Path(comp_map[v]).read_text(encoding="utf-8"))
        eye_aggr = aggregate_eye(eye_rows)
        comp_aggr= parse_comprehension(comp_d)
        merged = {**eye_aggr, **comp_aggr}
        baselines[v] = merged
        meta["files"][v] = {"eye": Path(eye_map[v]).name, "comprehension": Path(comp_map[v]).name}

    out = {"conditions": CONDITIONS, "baselines": baselines, "meta": meta}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(str(Path(args.out).resolve()))

if __name__ == "__main__":
    main()
