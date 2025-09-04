
#!/usr/bin/env python3
"""
Compute human free-recall scores using the SAME method as comprehension_test.py.

- Cosine similarity over CountVectorizer(ngram_range=(1,2), lowercase=True, stop_words="english")
- Compare each human "recall" to the ORIGINAL passage from stimuli_texts.json
- Robust to unicode/strange punctuation
- Aggregates mean/std by 30s / 60s / 90s

IMPORTANT: Your CSV uses Stim IDs like "stimuli1" while stimuli_texts.json keys start at "0".
We handle this automatically by preferring (stim_idx-1) when looking up the passage.
You can override with --csv_stim_ids_are_one_based=[true|false].
"""
import argparse
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(s: Optional[str]) -> str:
    """Light cleanup for odd punctuation/Unicode while preserving words."""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    # Common punctuation harmonization
    replacements = {
        "，": ", ", "。": ". ", "；": "; ", "：": ": ",
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "–": "-", "—": "-",
        "\u00A0": " ",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # Strip control chars
    s = re.sub(r"[\u0000-\u001F\u007F]", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def free_recall_score_same_method(pred: str, ref: str) -> float:
    """Exactly the vectorizer/cosine setup used in comprehension_test.py."""
    cv = CountVectorizer(ngram_range=(1, 2), lowercase=True, stop_words="english")
    X = cv.fit_transform([pred or "", ref or ""])
    if X.shape[0] < 2:
        return 0.0
    sim = cosine_similarity(X[0], X[1])[0, 0]
    return float(np.clip(sim, 0.0, 1.0))


def load_stimuli(stimuli_json: Optional[str]) -> Dict[int, str]:
    if not stimuli_json:
        return {}
    p = Path(stimuli_json)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        # allow "0","1",... as well as "stimuli1" style keys
        m = re.search(r"(\d+)$", str(k))
        if m:
            out[int(m.group(1))] = v
    return out


def parse_stim_index_from_csv(stim_id: str) -> Optional[int]:
    """Extract trailing integer from strings like 'stimuli1' -> 1."""
    if not isinstance(stim_id, str):
        stim_id = str(stim_id)
    m = re.search(r"(\d+)$", stim_id.strip())
    return int(m.group(1)) if m else None


def parse_time_cond(raw: str) -> str:
    """From 'A.30s' -> '30s' (or original if no digits)."""
    if not isinstance(raw, str):
        raw = str(raw)
    m = re.search(r"(\d+)\s*s", raw)
    return f"{m.group(1)}s" if m else raw.strip() or "NA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human_csv", required=True, help="Path to comprehension_raw_data_p1_to_p32.csv")
    ap.add_argument("--stimuli_json", required=True, help="Path to stimuli_texts.json mapping to original passages")
    ap.add_argument("--out_dir", default="./out_human_scores", help="Directory to write outputs")
    ap.add_argument("--save_row_csv", action="store_true", help="Also save per-row scores to CSV")
    ap.add_argument("--csv_stim_ids_are_one_based", type=str, default="true",
                    help="If 'true' (default), a CSV Stim ID like 'stimuli1' maps to JSON key 0 (i.e., -1 offset). "
                         "Set to 'false' to disable the -1 offset.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load inputs
    df = pd.read_csv(args.human_csv, encoding="utf-8", engine="python")
    stimuli = load_stimuli(args.stimuli_json)

    assume_one_based = str(args.csv_stim_ids_are_one_based).lower() in ("1", "true", "yes", "y")

    rows = []
    for i, row in df.iterrows():
        recall_text = normalize_text(row.get("recall", ""))
        stim_id = row.get("Stim ID", "")
        raw_idx = parse_stim_index_from_csv(stim_id)
        time_cond = parse_time_cond(row.get("Trial Condition", ""))

        ref_text = None
        if raw_idx is not None:
            # Prefer zero-based mapping if asked (CSV "stimuli1" -> JSON key 0)
            candidate_keys = []
            if assume_one_based:
                candidate_keys.append(raw_idx - 1)
            candidate_keys.append(raw_idx)  # also try same index, just in case
            for k in candidate_keys:
                if k in stimuli:
                    ref_text = normalize_text(stimuli[k])
                    break

        if ref_text is None:
            # No reference found: skip scoring for this row
            fr = np.nan
        else:
            fr = free_recall_score_same_method(recall_text, ref_text)

        rows.append({
            "row": i,
            "participant": row.get("Participant ID"),
            "csv_stim": stim_id,
            "parsed_idx": raw_idx,
            "time_cond": time_cond,
            "fr_score": fr
        })

    out_df = pd.DataFrame(rows)

    # Aggregate by time condition (keep 30s/60s/90s)
    keep = out_df["time_cond"].isin(["30s", "60s", "90s"])
    by_time = out_df[keep].groupby("time_cond")["fr_score"]
    fr_mean_by_time = by_time.mean().to_dict()
    fr_std_by_time = by_time.std(ddof=0).to_dict()

    metrics = {
        "fr_mean_by_time": {k: float(fr_mean_by_time.get(k)) if k in fr_mean_by_time else None for k in ["30s", "60s", "90s"]},
        "fr_std_by_time": {k: float(fr_std_by_time.get(k)) if k in fr_std_by_time else None for k in ["30s", "60s", "90s"]},
        "n_rows": int(len(out_df)),
        "n_scored": int(out_df["fr_score"].notna().sum()),
        "missing_refs": int(out_df["fr_score"].isna().sum())
    }

    # Save outputs
    metrics_path = os.path.join(args.out_dir, "human_free_recall_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if args.save_row_csv:
        perrow_path = os.path.join(args.out_dir, "human_free_recall_scored_rows.csv")
        out_df.to_csv(perrow_path, index=False, encoding="utf-8")

    # Print concise summary
    print("=== Human Free Recall (same-method cosine) ===")
    print("Mean by time:", metrics["fr_mean_by_time"])
    print("Std  by time:", metrics["fr_std_by_time"])
    print("Scored rows:", metrics["n_scored"], "/", metrics["n_rows"], " (missing refs:", metrics["missing_refs"], ")")
    print("Saved metrics to:", metrics_path)
    if args.save_row_csv:
        print("Saved per-row scores to:", perrow_path)


if __name__ == "__main__":
    """
    NOTE: the running commands:
    python compute_human_free_recall_score.py --human_csv comprehension_raw_data_p1_to_p32.csv --stimuli_json ../stimuli_texts.json --out_dir _new_free_recall_scores --save_row_csv --csv_stim_ids_are_one_based true
    """


    main()


