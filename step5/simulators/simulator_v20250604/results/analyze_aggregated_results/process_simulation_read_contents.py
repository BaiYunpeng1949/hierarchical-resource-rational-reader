#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_get(d: Dict, key: str, default=None):
    v = d.get(key, default)
    return v if v is not None else default

def transform_episode(ep: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "episode_index": safe_get(ep, "episode_index"),
        "stimulus_index": safe_get(ep, "stimulus_index"),
        "time_condition": safe_get(ep, "time_condition"),
        "total_time": safe_get(ep, "total_time"),
        "text_reading_logs": []
    }

    tr_logs = ep.get("text_reading_logs", []) or []
    for tlog in tr_logs:
        # sentence summary may carry sampled_words_in_sentence
        ssum = tlog.get("sentence_reading_summary", {}) or {}
        sampled_words = ssum.get("sampled_words_in_sentence", []) or []

        item = {
            "step": safe_get(tlog, "step"),
            "current_sentence_index": safe_get(tlog, "current_sentence_index"),
            "num_words_in_sentence": safe_get(tlog, "num_words_in_sentence", safe_get(ssum, "num_words_in_sentence")),
            "actual_reading_sentence_index": safe_get(tlog, "actual_reading_sentence_index"),
            "sampled_words_in_sentence": sampled_words,
        }
        out["text_reading_logs"].append(item)

    return out

def main():
    ap = argparse.ArgumentParser(description="Trim simulation results to reading content fields.")
    # ap.add_argument("input_json", type=Path, help="Path to all_simulation_results.json")
    ap.add_argument("-o", "--output_json", type=Path, default=Path("simulation_read_contents.json"),
                    help="Where to write the trimmed JSON (default: simulation_read_contents.json)")
    ap.add_argument("--indent", type=int, default=2, help="Indentation for output JSON")
    args = ap.parse_args()

    # sim_results_filepath = f"../../simulated_results/{args.input_json}/all_simulation_results.json"
    # sim_results_filepath = f"../../parameter_inference/simulation_data/rho_0.290__w_0.700__cov_1.30/all_simulation_results.json"    # TODO uncomment later, after sequentiality baseline check
    # NOTE: Updated on 0929. Use the best parameter from now on.

    # TODO comment later, doing the sequentiality baseline for text reader with gamma=0.2
    sim_results_filepath = f"/home/baiy4/reader-agent-zuco/step5/simulators/simulator_v20250604/simulated_results/20251011_0827_trials1_stims9_conds3/all_simulation_results.json"
    
    data = load_json(sim_results_filepath)

    if not isinstance(data, list):
        raise ValueError("Expected top-level list of episodes.")

    trimmed: List[Dict[str, Any]] = [transform_episode(ep) for ep in data]

    # args.output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json_filepath = Path(os.path.join('assets', 'comprehension_results', 'simulation', args.output_json))
    with open(output_json_filepath, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, ensure_ascii=False, indent=args.indent)

    print(f"[ok] Wrote {args.output_json} with {len(trimmed)} episode(s).")

if __name__ == "__main__":
    main()
