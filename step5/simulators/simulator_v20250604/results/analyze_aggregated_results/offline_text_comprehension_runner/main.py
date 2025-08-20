import json
import os
import argparse

from gpt_proposition_parser import LLMAgent
from ci_schema_pipeline import run_pipeline

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    # ap.add_argument("--json", default="/mnt/data/simulation_read_contents.json")
    ap.add_argument("--episodes", type=int, default=None, help="Process only the first N episodes")
    ap.add_argument("--start", type=int, default=0, help="Start offset for episodes")
    ap.add_argument("--wm_buffer", type=int, default=5)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--verbose", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = ap.parse_args()

    # init your agent (uses your Aalto gateway per your file)
    llm = LLMAgent(model_name="gpt-4o", api_key="")  # key handled in your file

    results = run_pipeline(
        json_path="../assets/comprehension_results/simulation/simulation_read_contents.json",
        llm_agent=llm,
        use_llm=True,
        wm_buffer=args.wm_buffer,
        limit_episodes=args.episodes,
        start=args.start,
        log_every=args.log_every,
        # llm_role="You are a reader with no prior knowledge about the reading content.",
        verbose=args.verbose,
        p_store=0.35,       # per-cycle consolidation prob for recall proxy
    )

    # write outputs to a compact JSON for analysis/plots
    output_filepatn = os.path.join('..', 'assets','comprehension_results','simulation','ci_gist_outputs.json')    
    with open(output_filepatn,"w",encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Saved -> ci_gist_outputs.json")