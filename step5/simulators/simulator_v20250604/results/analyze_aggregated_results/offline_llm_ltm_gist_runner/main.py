
import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List

from .ltm_gist_pipeline import run_trial_from_logs

def group_trials(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return records

def main():
    parser = argparse.ArgumentParser(description="Offline LTM gist runner (calls _LLMMemories directly)")
    parser.add_argument("--input_json", type=str, default="simulation_read_contents.json")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--config", type=str, default="/home/baiy4/reading-model/step5/config.yaml",
                        help="Path to the same config.yaml used by the simulator/_LLMMemories")
    parser.add_argument("--max_steps", type=int, default=None, help="Cap steps per episode (trial)")
    parser.add_argument("--max_episodes", type=int, default=None, help="Process only the first N episodes (after offset)")
    parser.add_argument("--episode_offset", type=int, default=0, help="Skip this many episodes before processing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_json, "r") as f:
        data = json.load(f)

    trials = group_trials(data)
    start = max(args.episode_offset, 0)
    end = start + args.max_episodes if args.max_episodes is not None else None
    trials = trials[start:end]

    results = []
    for t in trials:
        results.append(run_trial_from_logs(args.config, t, max_steps=args.max_steps))

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_json = os.path.join(args.output_dir, f"ltm_gists_{ts}.json")
    with open(out_json, "w") as f:
        json.dump({"created_utc": ts, "results": results}, f, indent=2)

    out_md = os.path.join(args.output_dir, f"ltm_gists_{ts}.md")
    with open(out_md, "w") as f:
        for r in results:
            f.write(f"### Episode {r['episode_index']} | Stimulus {r['stimulus_index']} | {r['time_condition']} ({r['total_time']}s)\n\n")
            f.write(r["outline"] + "\n\n")

    print(f"Saved: {out_json}\nSaved: {out_md}")

if __name__ == "__main__":
    main()
