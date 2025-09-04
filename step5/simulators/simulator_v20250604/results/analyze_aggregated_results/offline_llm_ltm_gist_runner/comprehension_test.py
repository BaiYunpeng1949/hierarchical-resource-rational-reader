
import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import numpy as np

from .LLMMemories import LLMWorkingMemory
from . import constants as const

from .ltm_loader import load_ltm_from_md, build_ltm_index, select_ltm, build_prompt_from_ltm


# ---- Free recall scoring (bag-of-words cosine) ----
def _free_recall_score(pred: str, ref: str) -> float:
    """Compute a very simple cosine similarity on token counts."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    cv = CountVectorizer(ngram_range=(1, 2), lowercase=True, stop_words="english")
    X = cv.fit_transform([pred or "", ref or ""])
    if X.shape[0] < 2:
        return 0.0
    sim = cosine_similarity(X[0], X[1])[0, 0]
    try:
        return float(np.clip(sim, 0.0, 1.0))
    except Exception:
        return 0.0

def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def _gather_source_texts_by_stimulus(sim_read_contents_path: Optional[str]) -> Dict[int, str]:
    """
    Build a mapping {stimulus_index: concatenated_source_text} from simulation_read_contents.json.
    We concatenate all sampled words in reading logs in order per stimulus.
    """
    if not sim_read_contents_path or not os.path.exists(sim_read_contents_path):
        return {}
    data = _load_json(sim_read_contents_path)
    by_idx: Dict[int, List[str]] = defaultdict(list)
    for ep in data:
        stim_idx = int(ep.get("stimulus_index", 0))
        logs = ep.get("text_reading_logs", [])
        for step in logs:
            words = step.get("sampled_words_in_sentence", [])
            words = [w for w in words if (isinstance(w, str) and w.strip())]
            if words:
                by_idx[stim_idx].append(" ".join(words))
    return {k: " ".join(v) for k, v in by_idx.items()}


def _load_optional_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def _safe_const(path_name: str, fallback: Optional[str] = None) -> Optional[str]:
    return getattr(const, path_name, fallback)

def _now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())

def run_comprehension(
    ltm_gists_json: str,
    output_dir: str,
    max_episodes: Optional[int] = None,
    mcq_metadata_path: Optional[str] = None,
    sim_read_contents_json: Optional[str] = None,
    human_metrics_json: Optional[str] = None,
    ltm_md_path: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Returns (results_json_path, metrics_json_path)
    """
    assert os.path.exists(ltm_gists_json), f"ltm_gists_json not found: {ltm_gists_json}"
    os.makedirs(output_dir, exist_ok=True)

    data = _load_json(ltm_gists_json)
    trials: List[Dict[str, Any]] = data.get("results", [])
    if max_episodes is not None:
        trials = trials[: int(max_episodes)]
    
    # Optional: load Markdown LTM once
    ltm_idx = None
    if ltm_md_path:
        md_path = Path(ltm_md_path)
        if md_path.exists():
            entries = load_ltm_from_md(md_path)
            ltm_idx = build_ltm_index(entries)

    # MCQ metadata
    if mcq_metadata_path is None:
        mcq_metadata_path = _safe_const("MCQ_METADATA_PATH", None)
    if not mcq_metadata_path or not os.path.exists(mcq_metadata_path):
        raise FileNotFoundError(
            f"MCQ metadata file not found. Looked for: {mcq_metadata_path!r}. "
            "Please set --mcq_metadata or define MCQ_METADATA_PATH in constants.py"
        )
    mcq_metadata: Dict[str, Any] = _load_json(mcq_metadata_path)

    # Optional: source texts for free recall scoring
    source_by_stim = _gather_source_texts_by_stimulus(sim_read_contents_json)

    wm = LLMWorkingMemory(config=_safe_const("CONFIG_PATH", "config.yaml"))  # will use your Aalto setup
    wm.reset()

    enriched: List[Dict[str, Any]] = []
    # Metrics collectors
    mcq_total = 0
    mcq_correct = 0
    mcq_by_time = defaultdict(lambda: {"n": 0, "correct": 0})
    frs_by_time = defaultdict(list)

    for t in trials:
        stim_idx = int(t.get("stimulus_index", 0))
        time_cond = t.get("time_condition", "NA")
        outline = t.get("outline") or t.get("gists") or ""
        # Prefer MD-based, coherent LTM; fallback to JSON outline/gists
        episode_idx = int(t.get("episode_index", 0))
        md_entry = select_ltm(ltm_idx, episode_idx, stim_idx, time_cond) if ltm_idx else None
        outline = (
            build_prompt_from_ltm(md_entry)
            if md_entry is not None
            else (t.get("outline") or t.get("gists") or "")
        )

        # ---- MCQs ----
        mcq_logs = []
        raw_mcqs = mcq_metadata.get(str(stim_idx), {})
        for mcq_idx, mcq in raw_mcqs.items():
            q = mcq.get("question", "")
            opts = mcq.get("options", {})
            pred = wm.retrieve_memory(
                question_type=const.QUESTION_TYPES["MCQ"],
                question=q,
                options=opts,
                ltm_gists=outline,
            )
            corr = mcq.get("correct_answer", "")
            is_ok = str(pred).strip().upper() == str(corr).strip().upper()
            mcq_logs.append(
                {
                    "mcq_idx": mcq_idx,
                    "question": q,
                    "options": opts,
                    "answer": pred,
                    "correct_answer": corr,
                    "is_correct": bool(is_ok),
                }
            )
            mcq_total += 1
            mcq_by_time[time_cond]["n"] += 1
            if is_ok:
                mcq_correct += 1
                mcq_by_time[time_cond]["correct"] += 1

        # ---- Free recall generation ----
        fr_text = wm.retrieve_memory(
            question_type=const.QUESTION_TYPES["FRS"], ltm_gists=outline
        )
        # Scoring target: prefer true source text; otherwise fall back to outline
        ref = source_by_stim.get(stim_idx, outline)
        fr_score = _free_recall_score(fr_text, ref)
        frs_by_time[time_cond].append(fr_score)

        enriched.append(
            {
                **t,
                "episodic_info": {
                    "mcq_logs": mcq_logs,
                    "free_recall_answer": fr_text,
                    "free_recall_score": fr_score,
                },
            }
        )

    # ---- Aggregate metrics ----
    metrics = {
        "mcq_accuracy_overall": (mcq_correct / mcq_total) if mcq_total else None,
        "mcq_accuracy_by_time": {
            k: (v["correct"] / v["n"]) if v["n"] else None for k, v in mcq_by_time.items()
        },
        "fr_mean_by_time": {k: float(np.mean(v)) if v else None for k, v in frs_by_time.items()},
        "fr_std_by_time": {k: float(np.std(v)) if v else None for k, v in frs_by_time.items()},
        "n_trials": len(trials),
    }

    # ---- Optional human comparison ----
    human = _load_optional_json(human_metrics_json) or _load_optional_json(_safe_const("HUMAN_METRICS_PATH", None))
    if human:
        metrics["human_comparison"] = {
            "mcq_accuracy_by_time": {
                k: {
                    "model": metrics["mcq_accuracy_by_time"].get(k),
                    "human": human.get("mcq_accuracy_by_time", {}).get(k),
                    "delta": (
                        (metrics["mcq_accuracy_by_time"].get(k) or 0.0)
                        - (human.get("mcq_accuracy_by_time", {}).get(k) or 0.0)
                    ),
                }
                for k in set(list(metrics["mcq_accuracy_by_time"].keys()) + list(human.get("mcq_accuracy_by_time", {}).keys()))
            },
            "fr_mean_by_time": {
                k: {
                    "model": metrics["fr_mean_by_time"].get(k),
                    "human": human.get("fr_mean_by_time", {}).get(k),
                    "delta": (
                        (metrics["fr_mean_by_time"].get(k) or 0.0)
                        - (human.get("fr_mean_by_time", {}).get(k) or 0.0)
                    ),
                }
                for k in set(list(metrics["fr_mean_by_time"].keys()) + list(human.get("fr_mean_by_time", {}).keys()))
            },
        }
    # ---- Save ----
    tag = _now_tag()
    results_path = os.path.join(output_dir, f"comprehension_results_{tag}.json")
    metrics_path = os.path.join(output_dir, f"comprehension_metrics_{tag}.json")
    with open(results_path, "w") as f:
        json.dump({"created_utc": tag, "results": enriched}, f, indent=2)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return results_path, metrics_path


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ltm_gists_json", required=True, help="Path to ltm_gists_*.json produced by the offline pipeline")
    ap.add_argument("--output_dir", required=True, help="Where to write comprehension results/metrics")
    ap.add_argument("--max_episodes", type=int, default=None, help="Limit how many trials to process")
    ap.add_argument("--mcq_metadata", default=None, help="Override path to MCQ metadata JSON")
    ap.add_argument("--input_json", default=None, help="Optional simulation_read_contents.json for FR scoring")
    ap.add_argument("--human_metrics", default=None, help="Optional JSON file with human benchmark metrics to compare against")
    ap.add_argument("--ltm_md_path", default=None, help="Optional path to Markdown LTM gists (preferred if provided)")
    args = ap.parse_args()

    results_path, metrics_path = run_comprehension(
        ltm_gists_json=args.ltm_gists_json,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
        mcq_metadata_path=args.mcq_metadata,
        sim_read_contents_json=args.input_json,
        human_metrics_json=args.human_metrics,
        ltm_md_path=args.ltm_md_path,
    )
    print(f"Saved detailed results to: {results_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    _main()
