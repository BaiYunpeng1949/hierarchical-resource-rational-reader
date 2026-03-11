#!/usr/bin/env python3
"""
Calibrate an appraisal threshold from human stimuli.

Input:
  - a text file like human_stimuli.txt containing lines with optional [Either]

What it does:
  1) parses the file
  2) expands [Either] into explicit sentence variants
  3) keeps S-coordination items as:
        ambiguous   = without "Either"
        unambiguous = with "Either"
  4) computes appraisal scores
  5) writes CSV + summary + suggested thresholds

Outputs:
  - stimuli_appraisal_scores.csv
  - stimuli_appraisal_summary.txt
"""

import argparse
import importlib.util
import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Parsing
# =========================================================

@dataclass
class StimulusRecord:
    section: str              # "S" or "NP"
    template: str             # original line
    sentence: str             # expanded sentence
    condition: str            # "ambiguous" / "unambiguous" / "either_np" / "noeither_np"
    has_either: int


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def expand_optional_either(line: str) -> List[Tuple[str, int]]:
    """
    Expand a sentence template with [Either] or [either].

    Returns list of:
      (expanded_sentence, has_either)
    """
    if "[Either]" in line:
        with_either = normalize_space(line.replace("[Either]", "Either"))
        without_either = normalize_space(line.replace("[Either]", ""))
        return [(with_either, 1), (without_either, 0)]

    if "[either]" in line:
        with_either = normalize_space(line.replace("[either]", "either"))
        without_either = normalize_space(line.replace("[either]", ""))
        return [(with_either, 1), (without_either, 0)]

    return [(normalize_space(line), int(re.search(r"\beither\b", line, flags=re.I) is not None))]


def parse_stimuli_txt(path: str) -> List[StimulusRecord]:
    """
    Parse the uploaded text file.
    """
    records: List[StimulusRecord] = []
    current_section: Optional[str] = None

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.lower().startswith("s-coordination sentences"):
                current_section = "S"
                continue

            if line.lower().startswith("np-coordination sentences"):
                current_section = "NP"
                continue

            if current_section not in {"S", "NP"}:
                continue

            expanded = expand_optional_either(line)
            for sent, has_either in expanded:
                if current_section == "S":
                    # this is the key mapping for calibration
                    condition = "unambiguous" if has_either else "ambiguous"
                else:
                    condition = "either_np" if has_either else "noeither_np"

                records.append(
                    StimulusRecord(
                        section=current_section,
                        template=line,
                        sentence=sent,
                        condition=condition,
                        has_either=has_either,
                    )
                )

    return records


# =========================================================
# Appraisal scoring
# =========================================================

class AppraisalScorer:
    """
    Two modes:
      - heuristic: immediate fallback
      - custom: import a user-provided scorer function
    """

    def __init__(
        self,
        mode: str = "heuristic",
        scorer_path: Optional[str] = None,
        scorer_function: str = "score_sentence_appraisal",
    ):
        self.mode = mode
        self.scorer_path = scorer_path
        self.scorer_function = scorer_function
        self.custom_fn: Optional[Callable[[str], float]] = None

        if self.mode == "custom":
            if scorer_path is None:
                raise ValueError("custom mode requires --scorer_path")
            self.custom_fn = self._load_custom_function(scorer_path, scorer_function)

    def _load_custom_function(self, path: str, fn_name: str) -> Callable[[str], float]:
        spec = importlib.util.spec_from_file_location("custom_scorer_mod", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load scorer module from: {path}")

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore

        if not hasattr(mod, fn_name):
            raise AttributeError(f"Function '{fn_name}' not found in {path}")

        fn = getattr(mod, fn_name)
        if not callable(fn):
            raise TypeError(f"Attribute '{fn_name}' in {path} is not callable")

        return fn

    def score(self, sentence: str) -> float:
        if self.mode == "custom":
            assert self.custom_fn is not None
            score = float(self.custom_fn(sentence))
            return float(np.clip(score, 0.0, 1.0))

        # -----------------------------
        # heuristic fallback
        # higher = easier / more appraisable
        # -----------------------------
        tokens = re.findall(r"\b[\w'-]+\b", sentence.lower())
        n = len(tokens)
        avg_len = np.mean([len(t) for t in tokens]) if tokens else 0.0

        # crude syntactic-load cues
        has_either = int(bool(re.search(r"\beither\b", sentence, flags=re.I)))
        has_or = int(bool(re.search(r"\bor\b", sentence, flags=re.I)))
        proper_names = sum(t[:1].isupper() for t in sentence.split())

        # a simple bounded heuristic:
        # shorter, more explicit, more signaled structure -> higher appraisal
        raw = (
            0.65
            + 0.10 * has_either
            + 0.03 * has_or
            - 0.015 * max(n - 10, 0)
            - 0.020 * max(avg_len - 4.5, 0)
            - 0.005 * max(proper_names - 2, 0)
        )

        return float(np.clip(raw, 0.0, 1.0))


# =========================================================
# Threshold estimation
# =========================================================

def summarize_thresholds(df: pd.DataFrame) -> dict:
    """
    Expects only S-coordination rows with conditions:
      ambiguous / unambiguous
    """
    amb = df.loc[df["condition"] == "ambiguous", "appraisal"].to_numpy()
    unamb = df.loc[df["condition"] == "unambiguous", "appraisal"].to_numpy()

    if len(amb) == 0 or len(unamb) == 0:
        raise ValueError("Need both ambiguous and unambiguous samples.")

    mean_thr = float((amb.mean() + unamb.mean()) / 2.0)
    median_thr = float((np.median(amb) + np.median(unamb)) / 2.0)

    # conservative non-overlap style thresholds
    amb_q75 = float(np.quantile(amb, 0.75))
    unamb_q25 = float(np.quantile(unamb, 0.25))

    # midpoint between extreme class means can also be useful
    amb_mean = float(amb.mean())
    unamb_mean = float(unamb.mean())

    # simple separation diagnostics
    summary = {
        "n_ambiguous": int(len(amb)),
        "n_unambiguous": int(len(unamb)),

        "ambiguous_mean": amb_mean,
        "unambiguous_mean": unamb_mean,
        "ambiguous_std": float(np.std(amb, ddof=1)) if len(amb) > 1 else 0.0,
        "unambiguous_std": float(np.std(unamb, ddof=1)) if len(unamb) > 1 else 0.0,

        "ambiguous_median": float(np.median(amb)),
        "unambiguous_median": float(np.median(unamb)),

        "ambiguous_min": float(amb.min()),
        "ambiguous_max": float(amb.max()),
        "unambiguous_min": float(unamb.min()),
        "unambiguous_max": float(unamb.max()),

        "recommended_threshold_mean_midpoint": mean_thr,
        "recommended_threshold_median_midpoint": median_thr,

        # conservative cluster rules
        "recommended_ambiguous_upper_bound_q75": amb_q75,
        "recommended_unambiguous_lower_bound_q25": unamb_q25,
        "gap_exists_between_q75_and_q25": bool(amb_q75 < unamb_q25),
    }

    return summary


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_txt",
        type=str,
        required=True,
        help="Path to human_stimuli.txt",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./stimuli_appraisal_calibration",
        help="Output directory",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="heuristic",
        choices=["heuristic", "custom"],
        help="Scoring mode",
    )
    parser.add_argument(
        "--scorer_path",
        type=str,
        default=None,
        help="Path to a python file containing score_sentence_appraisal(sentence)->float",
    )
    parser.add_argument(
        "--scorer_function",
        type=str,
        default="score_sentence_appraisal",
        help="Function name in --scorer_path",
    )
    parser.add_argument(
        "--include_np",
        action="store_true",
        help="Also score NP-coordination items in the CSV (not used for thresholding).",
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    records = parse_stimuli_txt(args.input_txt)
    scorer = AppraisalScorer(
        mode=args.mode,
        scorer_path=args.scorer_path,
        scorer_function=args.scorer_function,
    )

    rows = []
    for r in records:
        if (r.section == "NP") and (not args.include_np):
            continue

        appraisal = scorer.score(r.sentence)
        rows.append({
            "section": r.section,
            "condition": r.condition,
            "has_either": r.has_either,
            "sentence": r.sentence,
            "appraisal": appraisal,
        })

    df = pd.DataFrame(rows)

    # thresholding only on S-coordination ambiguous/unambiguous
    df_s = df[df["section"] == "S"].copy()
    df_s = df_s[df_s["condition"].isin(["ambiguous", "unambiguous"])].copy()

    summary = summarize_thresholds(df_s)

    # save per-sentence scores
    csv_path = os.path.join(args.out_dir, "stimuli_appraisal_scores.csv")
    df.to_csv(csv_path, index=False)

    # save summary
    txt_path = os.path.join(args.out_dir, "stimuli_appraisal_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Appraisal calibration summary\n")
        f.write(f"mode: {args.mode}\n\n")

        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

        f.write("\nSuggested usage\n")
        f.write(
            "- Default threshold: recommended_threshold_mean_midpoint\n"
            "- More robust alternative: recommended_threshold_median_midpoint\n"
            "- Conservative cluster approach:\n"
            "    ambiguous   if appraisal <= recommended_ambiguous_upper_bound_q75\n"
            "    unambiguous if appraisal >= recommended_unambiguous_lower_bound_q25\n"
            "    otherwise leave unclassified\n"
        )

    print(f"Saved sentence-level scores to: {csv_path}")
    print(f"Saved threshold summary to: {txt_path}")

    print("\nRecommended threshold candidates:")
    print(f"  mean midpoint   : {summary['recommended_threshold_mean_midpoint']:.4f}")
    print(f"  median midpoint : {summary['recommended_threshold_median_midpoint']:.4f}")
    print(f"  ambiguous q75   : {summary['recommended_ambiguous_upper_bound_q75']:.4f}")
    print(f"  unambiguous q25 : {summary['recommended_unambiguous_lower_bound_q25']:.4f}")
    print(f"  gap exists      : {summary['gap_exists_between_q75_and_q25']}")


if __name__ == "__main__":
    main()