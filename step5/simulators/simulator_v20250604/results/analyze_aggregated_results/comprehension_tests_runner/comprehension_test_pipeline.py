
"""
comprehension_test_pipeline.py

Pipeline to evaluate comprehension (MCQ + free recall) from simulated LTM gists.

Inputs
------
- sim_ltm_gists.json : produced by the simulation. Keys like:
    "episode_0_stim_0_30s__LTM": { "p_store": ..., "items": {<proposition>: { ... } } }
- mcq_metadata.json   : MCQs grouped by stimulus index (string keys "0", "1", ...).
- gpt_comprehension_answer.py : provides LLMAgent with two methods:
      - get_mcq_answers(ltm_gist: list, question: str, options: dict) -> one of {'A','B','C','D','E'}
      - get_free_recall(ltm_gist: list) -> str

Outputs
-------
- JSON results with the same structure pattern as template.json
- (Optional) CSV summary with MCQ accuracy per trial and a free-recall length score.

Usage
-----
python comprehension_test_pipeline.py \
    --ltm /path/to/sim_ltm_gists.json \
    --mcq /path/to/mcq_metadata.json \
    --out_json /path/to/out_results.json \
    --out_csv /path/to/out_results.csv \
    --mode llm               # 'llm' calls Aalto OpenAI via LLMAgent
                             # 'heuristic' = offline keyword matching (no API calls, good for tests)
                             # 'gold' = use provided correct answers (upper bound, testing only)
    --max_props 40           # clamp propositions passed into LLM (prompt length control)
    --sort_by last_relevance # or 'total_strength' | 'visits'

Notes
-----
- The LLMAgent in gpt_comprehension_answer.py is hard-coded to use Aalto gateway.
  If you use --mode llm, ensure that file and its key are configured for your environment.
- The pipeline is robust to missing MCQs for a given stimulus (skips gracefully).
"""

import argparse
import json
import os
import sys
import csv
from dataclasses import dataclass, asdict
import re
from typing import Dict, List, Tuple, Any

# Allow imports from the same directory or explicit path (e.g., /mnt/data)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
if "/mnt/data" not in sys.path:
    sys.path.append("/mnt/data")

try:
    # This import will succeed when the file is present (user provided)
    from gpt_comprehension_answer import LLMAgent  # type: ignore
except Exception as e:
    LLMAgent = None  # We'll guard calls if not available


@dataclass
class TrialKey:
    episode_index: int
    stimulus_index: int
    time_condition: str  # '30s', '60s', '90s'


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_trial_key(key: str) -> TrialKey:
    """
    Expect keys like 'episode_0_stim_0_30s__LTM'.
    """
    m = re.match(r"episode_(\d+)_stim_(\d+)_(\d+)(s)__LTM", key)
    if not m:
        raise ValueError(f"Unrecognized trial key format: {key}")
    return TrialKey(episode_index=int(m.group(1)),
                    stimulus_index=int(m.group(2)),
                    time_condition=m.group(3) + m.group(4))


def extract_trials(ltm_json: Dict[str, Any]) -> List[Tuple[TrialKey, Dict[str, Any]]]:
    trials = []
    for k, v in ltm_json.items():
        if k.endswith("__LTM"):
            try:
                tk = parse_trial_key(k)
                trials.append((tk, v))
            except Exception as e:
                # Skip anything that doesn't match the expected pattern
                continue
    # sort by (episode, stim, time)
    trials.sort(key=lambda kv: (kv[0].episode_index, kv[0].stimulus_index, kv[0].time_condition))
    return trials


def rank_propositions(items: Dict[str, Dict[str, float]], sort_by: str, max_props: int) -> List[str]:
    # items: { proposition: { 'last_relevance':..., 'total_strength':..., 'visits':... } }
    def score(meta: Dict[str, float]) -> float:
        if sort_by == "last_relevance":
            return float(meta.get("last_relevance", 0.0))
        if sort_by == "total_strength":
            return float(meta.get("total_strength", 0.0))
        if sort_by == "visits":
            return float(meta.get("visits", 0.0))
        # default
        return float(meta.get("last_relevance", 0.0))

    ranked = sorted(items.items(), key=lambda kv: score(kv[1]), reverse=True)
    props = [p for p, _ in ranked[:max_props]]
    return props


def mcqs_for_stim(mcq_data: Dict[str, Any], stimulus_index: int) -> List[Tuple[str, Dict[str, Any]]]:
    # Returns list of (mcq_idx_str, mcq_obj) ordered by numerical index
    stim_key = str(stimulus_index)
    if stim_key not in mcq_data:
        return []
    items = mcq_data[stim_key]
    # sort by int index of string keys
    pairs = sorted(items.items(), key=lambda kv: int(kv[0]))
    return pairs


def heuristic_answer(ltm_props: List[str], question: str, options: Dict[str, str]) -> str:
    """
    Offline keyword match. Returns 'A'|'B'|'C'|'D' or 'E' if no evidence.
    """
    # Build a single searchable text blob from LTM props
    blob = " ".join([p.lower() for p in ltm_props])
    # Very light normalization
    blob = re.sub(r"[^a-z0-9\s]+", " ", blob)

    # Score each option by overlap with blob keywords
    scores = {}
    for k, text in options.items():
        t = text.lower()
        t = re.sub(r"[^a-z0-9\s]+", " ", t)
        # Simple token overlap
        toks = [tok for tok in t.split() if len(tok) > 2]
        score = sum(1 for tok in toks if tok in blob)
        scores[k] = score

    # If clear winner, return it; else E
    best = max(scores, key=lambda k: scores[k])
    # Ensure best is strictly greater than all others
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) == 0 or sorted_scores[0] == 0:
        return "E"
    if len(sorted_scores) > 1 and sorted_scores[0] == sorted_scores[1]:
        return "E"
    return best


def make_free_recall_from_props(ltm_props: List[str], max_sentences: int = 4) -> str:
    """
    Offline free recall: compress propositions into a brief paragraph.
    """
    # Clean and turn prop strings into short phrases
    phrases = []
    for p in ltm_props[: max_sentences * 3]:
        # e.g., "birth(gps tracking)" -> "gps tracking"
        inside = p.split("(", 1)[-1].rstrip(")")
        inside = inside.replace(")", "").replace("(", " ")
        inside = re.sub(r"\s+", " ", inside).strip()
        if inside:
            phrases.append(inside)

    # Group phrases into 2-4 sentences heuristically
    if not phrases:
        return ""
    chunks = []
    step = max(1, len(phrases) // max_sentences)
    for i in range(0, len(phrases), step):
        sent = ", ".join(phrases[i:i+step])
        sent = sent[0].upper() + sent[1:] if sent else sent
        if sent and not sent.endswith("."):
            sent += "."
        chunks.append(sent)
        if len(chunks) >= max_sentences:
            break
    return " ".join(chunks)


def run_pipeline(ltm_path: str,
                 mcq_path: str,
                 out_json: str,
                 out_csv: str = None,
                 mode: str = "llm",
                 sort_by: str = "last_relevance",
                 max_props: int = 40,
                 participant_id_default: int = 0,
                 verbose: int = 1) -> Dict[str, Any]:

    def vprint(level, *args, **kwargs):
        if verbose >= level:
            print(*args, **kwargs)

    vprint(1, f"[INIT] mode={mode} sort_by={sort_by} max_props={max_props}")
    vprint(2, f"[PATHS] LTM={ltm_path}\n[PATHS] MCQ={mcq_path}")

    ltm = load_json(ltm_path)
    mcq = load_json(mcq_path)
    trials = extract_trials(ltm)
    vprint(1, f"[LOAD] Extracted {len(trials)} trial(s) from LTM store.")

    agent = None
    if mode == "llm":
        if LLMAgent is None:
            raise RuntimeError("LLMAgent not importable; cannot run in 'llm' mode.")
        vprint(1, "[LLM] Initializing LLMAgent (gpt-4o)...")
        agent = LLMAgent(model_name="gpt-4o", api_key=None)  # key handled inside class
        vprint(1, "[LLM] LLMAgent ready.")

    results = []

    total_mcq = 0
    total_correct = 0

    for i, (tk, payload) in enumerate(trials, start=1):
        print(f"[TRIAL {i}/{len(trials)}] episode={tk.episode_index} stim={tk.stimulus_index} time={tk.time_condition}") if verbose>=1 else None
        items = payload.get("items", {})
        ltm_props = rank_propositions(items, sort_by=sort_by, max_props=max_props)
        print(f"  └─ selected {len(ltm_props)} proposition(s) for prompting") if verbose>=2 else None

        # Collect MCQ logs
        mcq_logs = []
        stim_mcqs = mcqs_for_stim(mcq, tk.stimulus_index)
        for j, (mcq_idx, obj) in enumerate(stim_mcqs, start=1):
            q = obj["question"]
            options = obj["options"]
            correct = obj["correct_answer"]

            if mode == "llm":
                ans = agent.get_mcq_answers(ltm_gist=ltm_props, question=q, options=options)
                ans = ans if ans in ["A","B","C","D","E"] else "E"
            elif mode == "gold":
                ans = correct
            else:  # heuristic
                ans = heuristic_answer(ltm_props, q, options)

            is_correct = (ans == correct)
            total_mcq += 1
            total_correct += int(is_correct)
            print(f"    [MCQ {j}/{len(stim_mcqs)}] idx={mcq_idx} ans={ans} correct={correct} ✓" if is_correct else f"    [MCQ {j}/{len(stim_mcqs)}] idx={mcq_idx} ans={ans} correct={correct} ✗") if verbose>=2 else None

            mcq_logs.append({
                "mcq_idx": str(mcq_idx),
                "answer": ans,
                "correct_answer": correct
            })

        # Free recall
        if mode == "llm":
            print("    [FR] Generating free recall via LLM...") if verbose>=2 else None
            free_recall = agent.get_free_recall(ltm_gist=ltm_props)
        else:
            print("    [FR] Generating heuristic free recall...") if verbose>=2 else None
            free_recall = make_free_recall_from_props(ltm_props)
        print(f"    [FR] length={len((free_recall or '').split())} words") if verbose>=2 else None

        # Build result entry (template-compatible)
        entry = {
            "episodic_info": {
                "episode_index": tk.episode_index,
                "participant_id": participant_id_default,
                "trial_condition": tk.time_condition.replace("s",""),  # "30", "60", "90"
                "stimulus": {
                    "stimulus_index": tk.stimulus_index,
                    # We don't have the full passage text here; keep fields minimal
                    "words_in_section": "",
                    "stimulus_width": None,
                    "stimulus_height": None
                },
                "task": {
                    "time_constraint": int(tk.time_condition.replace("s","")),
                    "task_type": "comprehension"
                },
                "mcq_logs": mcq_logs,
                "free_recall_answer": free_recall
            }
        }
        results.append(entry)

    # Save JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] JSON → {out_json}") if verbose>=1 else None

    # Optional CSV summary
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode_index", "stimulus_index", "trial_condition",
                "num_mcq", "num_correct", "mcq_accuracy",
                "free_recall_word_count"
            ])
            for row in results:
                info = row["episodic_info"]
                mcqs = info["mcq_logs"]
                num = len(mcqs)
                correct = sum(1 for m in mcqs if m["answer"] == m["correct_answer"])
                acc = (correct / num) if num > 0 else 0.0
                fr_wc = len((info.get("free_recall_answer") or "").split())
                writer.writerow([
                    info["episode_index"],
                    info["stimulus"]["stimulus_index"],
                    info["trial_condition"],
                    num, correct, round(acc, 3), fr_wc
                ])

    run_summary = {
        "num_trials": len(results),
        "out_json": out_json,
        "out_csv": out_csv
    }
    if total_mcq > 0:
        print(f"[DONE] Trials={len(results)} | MCQs={total_mcq} | Overall acc={total_correct/total_mcq:.3f}") if verbose>=1 else None
    else:
        print(f"[DONE] Trials={len(results)} | MCQs=0") if verbose>=1 else None
    return run_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["llm","heuristic","gold"], default="heuristic",
                    help="llm: call OpenAI via LLMAgent; heuristic: offline keyword; gold: use correct answers")
    ap.add_argument("--sort_by", choices=["last_relevance","total_strength","visits"], default="last_relevance")
    ap.add_argument("--max_props", type=int, default=40, help="Clamp number of LTM propositions per prompt")
    ap.add_argument("--participant_id", type=int, default=0, help="Participant ID to put in results (if unknown)")
    ap.add_argument("--verbose", type=int, default=1, help="0=silent, 1=high-level, 2=step-by-step")
    args = ap.parse_args()

    ltm_path = os.path.join('..', 'assets', 'comprehension_results', 'simulation', 'sim_ltm_gists.json')
    mcq_path = os.path.join('..', 'assets', 'comprehension_results', 'mcq_metadata.json')
    out_json = os.path.join('..', 'assets', 'comprehension_results', 'simulation', 'comprehension_test_results.json' )
    out_csv = os.path.join('..', 'assets', 'comprehension_results', 'simulation', 'comprehension_test_results.csv' )

    stats = run_pipeline(
        ltm_path=ltm_path,
        mcq_path=mcq_path,
        out_json=out_json,
        out_csv=out_csv,
        mode=args.mode,
        sort_by=args.sort_by,
        max_props=args.max_props,
        participant_id_default=args.participant_id,
        verbose=args.verbose
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
