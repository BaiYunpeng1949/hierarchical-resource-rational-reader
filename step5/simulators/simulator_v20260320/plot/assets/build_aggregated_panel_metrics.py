# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# build_aggregated_panel_metrics.py

# Merge human aggregated metrics (means/stds) with aggregated simulation metrics (means/stds)
# for five metrics across three time-pressure conditions (30s, 60s, 90s):

# - reading_speed        (WPM)
# - skip_rate
# - regression_rate
# - mcq_accuracy
# - free_recall_score

# Inputs (defaults can be overridden via CLI):
# - human_eye_movement_metrics.json
# - human_mcq_acc_metrics.json
# - human_free_recall_metrics.json
# - simulation_eye_movement_metrics.json
# - comprehension_results_YYYYMMDD-HHMMSS.json  (V3 results with episodic_info)

# Output:
# - aggregated_panel_metrics.json

# Usage: NOTE: use this!!!!!
# python build_aggregated_panel_metrics.py \
#   --human_eye human_data/human_eye_movement_metrics.json \
#   --human_mcq human_data/human_mcq_acc_metrics.json \
#   --human_fr  human_data/human_free_recall_metrics.json \
#   --sim_eye   simulation_data/simulation_eye_movement_metrics.json \
#   --sim_comp  simulation_data/comprehension_results_20251006-150327.json \
#   --out       aggregated_panel_metrics.json
# """

# import argparse
# import json
# from pathlib import Path
# from typing import Dict, List
# import numpy as np

# CONDITIONS = ["30s", "60s", "90s"]

# def safe_mean(a: List[float]) -> float:
#     arr = np.array(a, dtype=float)
#     return float(np.mean(arr)) if arr.size else float("nan")

# def safe_std(a: List[float]) -> float:
#     arr = np.array(a, dtype=float)
#     if arr.size <= 1:
#         return 0.0
#     return float(np.std(arr, ddof=1))

# def load_json(path: Path):
#     return json.loads(path.read_text(encoding="utf-8"))

# def build_human_block(h_eye: Dict, h_mcq: Dict, h_fr: Dict) -> Dict:
#     human = {m: {} for m in ["reading_speed","skip_rate","regression_rate","mcq_accuracy","free_recall_score"]}
#     for cond in CONDITIONS:
#         he = h_eye[cond]
#         # Eye movement means/stds are already aggregated in the human file
#         human["reading_speed"][cond] = {
#             "mean": he.get("reading_speed_mean"),
#             "std":  he.get("reading_speed_std"),
#             "n":    he.get("num_episodes")
#         }
#         human["skip_rate"][cond] = {
#             "mean": he.get("skip_rate_mean"),
#             "std":  he.get("skip_rate_std"),
#             "n":    he.get("num_episodes")
#         }
#         human["regression_rate"][cond] = {
#             "mean": he.get("regression_rate_mean"),
#             "std":  he.get("regression_rate_std"),
#             "n":    he.get("num_episodes")
#         }
#         # Comprehension (per-time-bin means/stds already given)
#         human["mcq_accuracy"][cond] = {
#             "mean": h_mcq["mcq_mean_by_time"][cond],
#             "std":  h_mcq["mcq_std_by_time"][cond],
#             "n":    h_mcq.get("n_scored")  # total N if per-condition not available
#         }
#         human["free_recall_score"][cond] = {
#             "mean": h_fr["fr_mean_by_time"][cond],
#             "std":  h_fr["fr_std_by_time"][cond],
#             "n":    h_fr.get("n_scored")
#         }
#     return human

# def build_simulation_block(s_eye_rows: List[Dict], s_comp: Dict) -> Dict:
#     # Aggregate per condition
#     eye_buckets = {c: {k: [] for k in ["reading_speed","skip_rate","regression_rate"]} for c in CONDITIONS}
#     for row in s_eye_rows:
#         cond = row.get("time_condition")
#         if cond in eye_buckets:
#             for k in ["reading_speed","skip_rate","regression_rate"]:
#                 v = row.get(k)
#                 if isinstance(v, (int, float)):
#                     eye_buckets[cond][k].append(float(v))

#     comp_buckets = {c: {k: [] for k in ["mcq_accuracy","free_recall_score"]} for c in CONDITIONS}
#     for ep in s_comp.get("results", []):
#         if not isinstance(ep, dict):
#             continue
#         cond = ep.get("time_condition")
#         epi  = ep.get("episodic_info", {}) or {}
#         if cond in comp_buckets:
#             for k in ["mcq_accuracy","free_recall_score"]:
#                 v = epi.get(k)
#                 if isinstance(v, (int, float)):
#                     comp_buckets[cond][k].append(float(v))

#     sim = {m: {} for m in ["reading_speed","skip_rate","regression_rate","mcq_accuracy","free_recall_score"]}
#     for cond in CONDITIONS:
#         # Eye
#         for k in ["reading_speed","skip_rate","regression_rate"]:
#             arr = eye_buckets[cond][k]
#             sim[k][cond] = {"mean": safe_mean(arr), "std": safe_std(arr), "n": len(arr)}
#         # Comprehension
#         for k in ["mcq_accuracy","free_recall_score"]:
#             arr = comp_buckets[cond][k]
#             sim[k][cond] = {"mean": safe_mean(arr), "std": safe_std(arr), "n": len(arr)}
#     return sim

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--human_eye", type=str, default="human_eye_movement_metrics.json")
#     ap.add_argument("--human_mcq", type=str, default="human_mcq_acc_metrics.json")
#     ap.add_argument("--human_fr",  type=str, default="human_free_recall_metrics.json")
#     ap.add_argument("--sim_eye",   type=str, default="simulation_eye_movement_metrics.json")
#     ap.add_argument("--sim_comp",  type=str, default="comprehension_results_20251006-150327.json")
#     ap.add_argument("--out",       type=str, default="aggregated_panel_metrics.json")
#     args = ap.parse_args()

#     p_h_eye = Path(args.human_eye)
#     p_h_mcq = Path(args.human_mcq)
#     p_h_fr  = Path(args.human_fr)
#     p_s_eye = Path(args.sim_eye)
#     p_s_comp= Path(args.sim_comp)
#     p_out   = Path(args.out)

#     h_eye = load_json(p_h_eye)
#     h_mcq = load_json(p_h_mcq)
#     h_fr  = load_json(p_h_fr)
#     s_eye = load_json(p_s_eye)
#     s_comp= load_json(p_s_comp)

#     human = build_human_block(h_eye, h_mcq, h_fr)
#     simulation = build_simulation_block(s_eye, s_comp)

#     out = {
#         "conditions": CONDITIONS,
#         "human": human,
#         "simulation": simulation,
#         "meta": {
#             "human_eye_file": p_h_eye.name,
#             "human_mcq_file": p_h_mcq.name,
#             "human_fr_file":  p_h_fr.name,
#             "sim_eye_file":   p_s_eye.name,
#             "sim_comp_file":  p_s_comp.name,
#             "std_ddof": 1
#         }
#     }

#     p_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
#     print(f"Wrote: {p_out.resolve()}")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_aggregated_panel_metrics.py

Merge human aggregated metrics (means/stds) with aggregated simulation metrics (means/stds)
for five metrics across three time-pressure conditions (30s, 60s, 90s):

- reading_speed        (WPM)
- skip_rate
- regression_rate
- mcq_accuracy
- free_recall_score

Inputs (defaults can be overridden via CLI):
- human_eye_movement_metrics.json
- human_mcq_acc_metrics.json
- human_free_recall_metrics.json
- simulation_eye_movement_metrics.json
- comprehension_results_YYYYMMDD-HHMMSS.json  (V3 results with episodic_info)

Output:
- aggregated_panel_metrics.json

Usage: NOTE: use this!!!!!
python build_aggregated_panel_metrics.py \
  --human_eye human_data/human_eye_movement_metrics.json \
  --human_mcq human_data/human_mcq_acc_metrics.json \
  --human_fr  human_data/human_free_recall_metrics.json \
  --sim_eye   simulation_data/simulation_eye_movement_metrics.json \
  --sim_comp  simulation_data/comprehension_results_20251006-150327.json \
  --out       aggregated_panel_metrics.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

CONDITIONS = ["30s", "60s", "90s"]

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

def build_human_block(h_eye: Dict, h_mcq: Dict, h_fr: Dict) -> Dict:
    human = {m: {} for m in ["reading_speed","skip_rate","regression_rate","mcq_accuracy","free_recall_score"]}
    for cond in CONDITIONS:
        he = h_eye[cond]
        # Eye movement means/stds are already aggregated in the human file
        human["reading_speed"][cond] = {
            "mean": he.get("reading_speed_mean"),
            "std":  he.get("reading_speed_std"),
            "n":    he.get("num_episodes")
        }
        human["skip_rate"][cond] = {
            "mean": he.get("skip_rate_mean"),
            "std":  he.get("skip_rate_std"),
            "n":    he.get("num_episodes")
        }
        human["regression_rate"][cond] = {
            "mean": he.get("regression_rate_mean"),
            "std":  he.get("regression_rate_std"),
            "n":    he.get("num_episodes")
        }
        # Comprehension (per-time-bin means/stds already given)
        human["mcq_accuracy"][cond] = {
            "mean": h_mcq["mcq_mean_by_time"][cond],
            "std":  h_mcq["mcq_std_by_time"][cond],
            "n":    h_mcq.get("n_scored")  # total N if per-condition not available
        }
        human["free_recall_score"][cond] = {
            "mean": h_fr["fr_mean_by_time"][cond],
            "std":  h_fr["fr_std_by_time"][cond],
            "n":    h_fr.get("n_scored")
        }
    return human

def build_simulation_block(s_eye_rows: List[Dict], s_comp: Dict) -> Dict:
    """
    Aggregate simulation metrics per condition, while also retaining point-wise
    values for plotting individual data points.

    Point-wise unit: one simulated episode/stimulus under one time condition.
    With the current simulation files this gives 27 total rows = 9 rows per
    time condition.
    """
    # Aggregate per condition
    eye_buckets = {c: {k: [] for k in ["reading_speed","skip_rate","regression_rate"]} for c in CONDITIONS}
    for row in s_eye_rows:
        cond = row.get("time_condition")
        if cond in eye_buckets:
            for k in ["reading_speed","skip_rate","regression_rate"]:
                v = row.get(k)
                if isinstance(v, (int, float)):
                    eye_buckets[cond][k].append(float(v))

    comp_buckets = {c: {k: [] for k in ["mcq_accuracy","free_recall_score"]} for c in CONDITIONS}
    for ep in s_comp.get("results", []):
        if not isinstance(ep, dict):
            continue
        cond = ep.get("time_condition")
        epi  = ep.get("episodic_info", {}) or {}
        if cond in comp_buckets:
            for k in ["mcq_accuracy","free_recall_score"]:
                v = epi.get(k)
                if isinstance(v, (int, float)):
                    comp_buckets[cond][k].append(float(v))

    sim = {m: {} for m in ["reading_speed","skip_rate","regression_rate","mcq_accuracy","free_recall_score"]}
    for cond in CONDITIONS:
        # Eye
        for k in ["reading_speed","skip_rate","regression_rate"]:
            arr = eye_buckets[cond][k]
            sim[k][cond] = {
                "mean": safe_mean(arr),
                "std": safe_std(arr),
                "n": len(arr),
                "values": arr,
            }
        # Comprehension
        for k in ["mcq_accuracy","free_recall_score"]:
            arr = comp_buckets[cond][k]
            sim[k][cond] = {
                "mean": safe_mean(arr),
                "std": safe_std(arr),
                "n": len(arr),
                "values": arr,
            }
    return sim


def build_simulation_pointwise_rows(s_eye_rows: List[Dict], s_comp: Dict) -> List[Dict]:
    """
    Build one row per simulated episode/stimulus/time condition with all five
    plotted metrics. This is useful for checking and later plotting overlaid
    data points.
    """
    comp_lookup = {}
    for ep in s_comp.get("results", []):
        if not isinstance(ep, dict):
            continue
        key = (ep.get("episode_index"), ep.get("stimulus_index"), ep.get("time_condition"))
        epi = ep.get("episodic_info", {}) or {}
        comp_lookup[key] = {
            "mcq_accuracy": epi.get("mcq_accuracy"),
            "free_recall_score": epi.get("free_recall_score"),
            "n_mcq": epi.get("n_mcq"),
        }

    rows = []
    for row in s_eye_rows:
        key = (row.get("episode_index"), row.get("stimulus_index"), row.get("time_condition"))
        comp = comp_lookup.get(key, {})
        rows.append({
            "source": "simulation",
            "episode_index": row.get("episode_index"),
            "stimulus_index": row.get("stimulus_index"),
            "time_condition": row.get("time_condition"),
            "total_time": row.get("total_time"),
            "reading_speed": row.get("reading_speed"),
            "skip_rate": row.get("skip_rate"),
            "regression_rate": row.get("regression_rate"),
            "mcq_accuracy": comp.get("mcq_accuracy"),
            "free_recall_score": comp.get("free_recall_score"),
            "n_mcq": comp.get("n_mcq"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human_eye", type=str, default="human_eye_movement_metrics.json")
    ap.add_argument("--human_mcq", type=str, default="human_mcq_acc_metrics.json")
    ap.add_argument("--human_fr",  type=str, default="human_free_recall_metrics.json")
    ap.add_argument("--sim_eye",   type=str, default="simulation_eye_movement_metrics.json")
    ap.add_argument("--sim_comp",  type=str, default="comprehension_results_20251006-150327.json")
    ap.add_argument("--out",       type=str, default="aggregated_panel_metrics.json")
    args = ap.parse_args()

    p_h_eye = Path(args.human_eye)
    p_h_mcq = Path(args.human_mcq)
    p_h_fr  = Path(args.human_fr)
    p_s_eye = Path(args.sim_eye)
    p_s_comp= Path(args.sim_comp)
    p_out   = Path(args.out)

    h_eye = load_json(p_h_eye)
    h_mcq = load_json(p_h_mcq)
    h_fr  = load_json(p_h_fr)
    s_eye = load_json(p_s_eye)
    s_comp= load_json(p_s_comp)

    human = build_human_block(h_eye, h_mcq, h_fr)
    simulation = build_simulation_block(s_eye, s_comp)
    simulation_pointwise = build_simulation_pointwise_rows(s_eye, s_comp)

    out = {
        "conditions": CONDITIONS,
        "human": human,
        "simulation": simulation,
        "pointwise": {
            "simulation": simulation_pointwise
        },
        "meta": {
            "human_eye_file": p_h_eye.name,
            "human_mcq_file": p_h_mcq.name,
            "human_fr_file":  p_h_fr.name,
            "sim_eye_file":   p_s_eye.name,
            "sim_comp_file":  p_s_comp.name,
            "std_ddof": 1
        }
    }

    p_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote: {p_out.resolve()}")

    # Also write a compact long-format simulation point-wise file next to the
    # aggregated JSON. This is optional but convenient for plotting/checking.
    p_pointwise = p_out.with_name(p_out.stem + "_simulation_pointwise.json")
    p_pointwise.write_text(json.dumps(simulation_pointwise, indent=2), encoding="utf-8")
    print(f"Wrote: {p_pointwise.resolve()}")

if __name__ == "__main__":
    main()
