#!/usr/bin/env python3
"""
Convert simulated fixations to JSON aligned with image coordinates.

Output schema (list of trials):
[
  {
    "stimulus_index": int,
    "participant_index": "simulation",
    "time_constraint": int | null,
    "fixation_data": [
       {
         "fix_x": float,
         "fix_y": float,
         "norm_fix_x": float,
         "norm_fix_y": float,
         "fix_duration": float,         # milliseconds
         "word_index": int,
         "recognized_word": str | null, # first recognized word (if any)
         "recognized_words": [str, ...] # full list (if any)
       },
       ...
    ]
  },
  ...
]
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def letter_centers_from_metadata(word_meta: Dict[str, Any], letter_idxs: List[int]) -> Optional[list]:
    letters = word_meta.get("letters metadata", [])
    centers = []
    for li in letter_idxs or []:
        if 0 <= li < len(letters):
            lb = letters[li]["letter boxes"]
            cx = (lb["letter box left"] + lb["letter box right"]) / 2.0
            cy = (lb["letter box top"]  + lb["letter box bottom"]) / 2.0
            centers.append((cx, cy))
    return centers if centers else None

def word_center_from_bbox(word_meta: Dict[str, Any]) -> tuple:
    x1, y1, x2, y2 = word_meta["word_bbox"]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def choose_fixation_point(word_meta, letter_idxs):
    letters = word_meta.get("letters metadata", [])
    pts = []
    for li in (letter_idxs or []):
        if 0 <= li < len(letters):
            lb = letters[li]["letter boxes"]
            cx = (lb["letter box left"] + lb["letter box right"]) / 2.0
            cy = (lb["letter box top"]  + lb["letter box bottom"]) / 2.0
            pts.append((cx, cy))
    if not pts:
        x1, y1, x2, y2 = word_meta["word_bbox"]
        return ( (x1+x2)/2.0, (y1+y2)/2.0 )
    if len(pts) == 1:
        return pts[0]
    return ( sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts) )

def build_trials(sim: list, meta: dict) -> list:
    images = meta["images"]
    img_w, img_h = meta["config"]["img size"]

    trials = []

    for episode in sim:
        episode_index = episode.get("episode_index", 0)
        stimulus_index = episode.get("stimulus_index", 0)
        words_meta = images[stimulus_index]["words metadata"]

        trial = {
            "stimulus_index": stimulus_index,
            "participant_index": episode_index,
            "time_constraint": episode.get("time_condition", None),
            "fixation_data": []
        }

        for sent in episode.get("text_reading_logs", []):
            ssum = sent.get("sentence_reading_summary", {})
            global_fix_seq = ssum.get("global_actual_fixation_sequence_in_text", [])

            # steps that produced at least one recognized word
            fix_steps = [
                step for step in sent.get("sentence_reading_logs", [])
                if step.get("word_recognition_summary", {}).get("num_words_read_this_step", 0) > 0
            ]

            n = min(len(global_fix_seq), len(fix_steps))
            for i in range(n):
                step = fix_steps[i]
                wrs = step["word_recognition_summary"]

                gaze_ms = float(wrs.get("total_elapsed_time_in_s", 0.0)) * 1000.0
                gwi = int(global_fix_seq[i])

                # recognized words (list) and first recognized word (if any)
                rlist = wrs.get("recognized_words_list", []) or []
                rfirst = rlist[0] if isinstance(rlist, list) and len(rlist) > 0 else None

                # letters sampled for the first recognized word (key "0")
                sampled = wrs.get("sampled_letters_indexes_dict", {})
                letter_idxs = []
                if "0" in sampled:
                    letter_idxs = sampled["0"].get("letters_indexes", []) or []

                # if 0 <= gwi < len(words_meta):
                #     wmeta = words_meta[gwi]
                #     centers = letter_centers_from_metadata(wmeta, letter_idxs)
                #     if centers is None:
                #         fx, fy = word_center_from_bbox(wmeta)
                #     else:
                #         fx = sum(c[0] for c in centers) / len(centers)
                #         fy = sum(c[1] for c in centers) / len(centers)
                # else:
                #     # unmatched -> skip
                #     continue
                if 0 <= gwi < len(words_meta):
                    wmeta = words_meta[gwi]
                    fx, fy = choose_fixation_point(wmeta, letter_idxs)  # <- single letter, multi, or fallback
                else:
                    continue


                trial["fixation_data"].append({
                    "fix_x": fx,
                    "fix_y": fy,
                    "norm_fix_x": fx / img_w,
                    "norm_fix_y": fy / img_h,
                    "fix_duration": gaze_ms,
                    "word_index": gwi,
                    "recognized_word": rfirst,
                    "recognized_words": rlist
                })

        trials.append(trial)

    return trials

def main():
    ap = argparse.ArgumentParser(description="Export simulated fixations to JSON with image coordinates.")
    ap.add_argument("--simulation", "-s", type=Path, required=True, help="Folder name of the all_simulation_results.json")
    # ap.add_argument("--metadata", "-m", type=Path, required=True, help="Path to metadata.json")
    ap.add_argument("--out_json", "-o", type=Path, default=Path("simulation_scanpath.json"))
    args = ap.parse_args()

    # Get the simulation filepath
    sim_results_filepath = f"../../simulated_results/{args.simulation}/all_simulation_results.json"
    sim = load_json(sim_results_filepath)

    # Get the metadata filepath
    metadata_filepath = "../../../../data/gen_envs/08_15_09_07_10_images_W1920H1080WS16_LS40_MARGIN400/simulate/metadata.json"
    # meta = load_json(args.metadata)
    meta = load_json(metadata_filepath)

    trials = build_trials(sim, meta)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(trials, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(trials)} trials to {args.out_json}")

if __name__ == "__main__":
    main()


# TODO: change the simulation's episode index -- to trail index, then generate the participant_index as simulation-ID, and name the plots properly.