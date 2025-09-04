
import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List
from .llm_agent import LLMAgent
from .ltm_gist_pipeline import run_trial

def group_trials(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The input file already looks grouped; just return the list.
       If you later need regrouping, modify here.
    """
    return records

def main():
    parser = argparse.ArgumentParser(description="Offline LLM LTM gist runner")
    parser.add_argument("--input_json", type=str, default="simulation_read_contents.json")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_steps", type=int, default=None, help="Cap steps per trial")
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--api_key", type=str, default="")
    # NEW: control how many episodes (trials) to process
    parser.add_argument("--max_episodes", type=int, default=None, help="Process only the first N episodes (after offset)")
    parser.add_argument("--episode_offset", type=int, default=0, help="Skip this many episodes before processing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_json, "r") as f:
        data = json.load(f)

    all_trials = group_trials(data)

    # Apply episode window selection
    start = max(args.episode_offset, 0)
    end = start + args.max_episodes if args.max_episodes is not None else None
    selected_trials = all_trials[start:end]

    print(f"Loaded {len(all_trials)} episodes. Processing episodes [{start}:{'end' if end is None else end}] -> {len(selected_trials)} episodes.")

    llm = LLMAgent(model_name=args.model_name, api_key=args.api_key)

    results = []
    for trial in selected_trials:
        gist = run_trial(llm, trial, max_steps=args.max_steps)
        results.append(gist)

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_json = os.path.join(args.output_dir, f"ltm_gists_{ts}.json")
    with open(out_json, "w") as f:
        json.dump({"created_utc": ts, "results": results}, f, indent=2)

    # Also save a readable md outline per trial
    out_md = os.path.join(args.output_dir, f"ltm_gists_{ts}.md")
    with open(out_md, "w") as f:
        for r in results:
            f.write(f"### Episode {r['episode_index']} | Stimulus {r['stimulus_index']} | {r['time_condition']} ({r['total_time']}s)\n\n")
            f.write(r["outline"] + "\n\n")

    print(f"Saved: {out_json}\nSaved: {out_md}")

if __name__ == "__main__":
    main()
