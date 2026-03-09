import os
import re
import json
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =========================================================
# Configuration
# =========================================================

SCANPATH_JSON = "11_18_17_40_integrated_corrected_human_scanpath.json"
SENTENCE_METADATA_JSON = "metadata_sentence_indeces.json"

# Appraisal bins: same style as simulation, [0,1] in equal-width bins
DEFAULT_BINS = np.linspace(0.0, 1.0, 11)  # 10 bins


# =========================================================
# Loading helpers
# =========================================================

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_sentence_id(word_index: int, sentences: List[Dict]) -> int:
    """
    Map a word index to its sentence id using start_idx/end_idx.
    Returns -1 if not found.
    """
    for sent in sentences:
        if sent["start_idx"] <= word_index <= sent["end_idx"]:
            return sent["sentence_idx"]
    return -1

def get_sentence_info(word_index: int, sentences: List[Dict]) -> Tuple[int, Optional[Dict]]:
    """
    Map a word index to (sentence_id, sentence_dict).
    Returns (-1, None) if not found.
    """
    for sent in sentences:
        if sent["start_idx"] <= word_index <= sent["end_idx"]:
            return sent["sentence_idx"], sent
    return -1, None


# =========================================================
# Appraisal scoring
# =========================================================

class AppraisalScorer:
    """
    Computes an initial appraisal score in [0,1], where:
      higher = easier / better appraised / more coherent
      lower  = harder / worse appraised / more difficult to integrate

    Two modes are supported:
      - dummy: simple heuristic
      - openai: LLM-backed scoring (requires OPENAI_API_KEY and openai package)
    """

    def __init__(self, mode: str = "dummy", model_name: str = "gpt-4.1-mini"):
        self.mode = mode
        self.model_name = model_name

        if self.mode == "openai":
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai"
                ) from e
            self.client = OpenAI()

    def score_sentence(self, context: str, sentence: str) -> float:
        if self.mode == "dummy":
            return self._score_dummy(context, sentence)
        elif self.mode == "openai":
            return self._score_openai(context, sentence)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _score_dummy(self, context: str, sentence: str) -> float:
        """
        Cheap placeholder heuristic:
        shorter and lexically simpler sentences get higher appraisal.
        Replace later with a real model if needed.
        """
        words = re.findall(r"\b\w+\b", sentence.lower())
        if not words:
            return 0.5

        avg_word_len = np.mean([len(w) for w in words])
        sent_len = len(words)

        # Convert to rough difficulty, then invert to appraisal
        difficulty = 0.04 * sent_len + 0.08 * avg_word_len
        appraisal = 1.0 / (1.0 + difficulty)

        return float(np.clip(appraisal, 0.0, 1.0))

    def _score_openai(self, context: str, sentence: str) -> float:
        """
        Ask an LLM for a coherence / ease-of-integration appraisal in [0,1].
        Higher means easier / more coherent.
        """
        prompt = f"""
You are scoring how easy a sentence should be to understand and integrate
given the preceding context.

Return ONLY a single number between 0 and 1:
- 1.0 = very easy / highly coherent / highly expected / easy to integrate
- 0.0 = very difficult / incoherent / unexpected / hard to integrate

Preceding context:
{context if context.strip() else "[No prior context]"}

Current sentence:
{sentence}
""".strip()

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=0
        )

        text = response.output_text.strip()
        try:
            score = float(text)
        except ValueError:
            # fallback parse
            match = re.search(r"([01](?:\.\d+)?)", text)
            if not match:
                raise ValueError(f"Could not parse appraisal score from model output: {text}")
            score = float(match.group(1))

        return float(np.clip(score, 0.0, 1.0))


# def compute_sentence_reread_metrics(
#     fixation_data: List[Dict],
#     sentence_meta_for_stimulus: Dict,
#     reread_word_coverage_threshold: float = 0.3,
# ) -> Dict[int, Dict]:
#     """
#     For each visited sentence, compute rereading based on return-word coverage.

#     Definition:
#       - A sentence is considered "moved beyond" once the reader fixates any later sentence.
#       - After that point, any fixation returning to the earlier sentence counts as a return visit.
#       - reread_coverage =
#             (# unique words in the sentence fixated during return visits)
#             / (# total words in the sentence)

#       - reread = 1 if reread_coverage >= reread_word_coverage_threshold else 0

#     Returns:
#       dict:
#         sentence_id -> {
#             "reread": int,
#             "reread_coverage": float,
#             "n_words_in_sentence": int,
#             "n_unique_returned_words": int,
#             "visited": int,
#         }
#     """
#     sentences = sentence_meta_for_stimulus["sentences"]

#     # Sentence metadata lookup
#     sent_info = {}
#     for sent in sentences:
#         sent_id = sent["sentence_idx"]
#         word_indices = list(range(sent["start_idx"], sent["end_idx"] + 1))
#         sent_info[sent_id] = {
#             "word_indices": set(word_indices),
#             "n_words": len(word_indices),
#         }

#     # Track which sentences were ever visited
#     visited_sentences = set()

#     # Track whether we have already moved beyond each sentence
#     moved_beyond = {sent_id: False for sent_id in sent_info.keys()}

#     # Track unique returned words after moving beyond
#     returned_words = {sent_id: set() for sent_id in sent_info.keys()}

#     # Keep the maximum sentence index seen so far
#     max_seen_sentence = -1

#     # Process fixations in temporal order
#     for fix in fixation_data:
#         word_index = fix.get("word_index", -1)
#         if word_index is None or word_index == -1:
#             continue

#         current_sent_id, _ = get_sentence_info(word_index, sentences)
#         if current_sent_id == -1:
#             continue

#         visited_sentences.add(current_sent_id)

#         # Any earlier sentence is now "moved beyond"
#         if current_sent_id > max_seen_sentence:
#             for prev_sent_id in sent_info.keys():
#                 if prev_sent_id < current_sent_id:
#                     moved_beyond[prev_sent_id] = True
#             max_seen_sentence = current_sent_id

#         # If this sentence has already been moved beyond, then this fixation is part of a return visit
#         if moved_beyond.get(current_sent_id, False):
#             if word_index in sent_info[current_sent_id]["word_indices"]:
#                 returned_words[current_sent_id].add(word_index)

#     # Build output
#     results = {}
#     for sent_id in visited_sentences:
#         n_words = sent_info[sent_id]["n_words"]
#         n_returned = len(returned_words[sent_id])
#         coverage = (n_returned / n_words) if n_words > 0 else 0.0

#         results[sent_id] = {
#             "reread": int(coverage >= reread_word_coverage_threshold),
#             "reread_coverage": float(coverage),
#             "n_words_in_sentence": int(n_words),
#             "n_unique_returned_words": int(n_returned),
#             "visited": 1,
#         }

#     return results


def compute_sentence_reread_metrics(
    fixation_data: List[Dict],
    sentence_meta_for_stimulus: Dict,
    reread_word_coverage_threshold: float = 0.3,
    initial_word_coverage_threshold: float = 0.5,
) -> Dict[int, Dict]:
    """
    For each visited sentence, compute:
      - initial_coverage:
          proportion of unique words fixated before the reader first moved beyond the sentence
      - return_coverage:
          proportion of unique words fixated after the reader had moved beyond the sentence

    A sentence is eligible for rereading analysis only if:
      initial_coverage >= initial_word_coverage_threshold

    A sentence is classified as reread only if:
      return_coverage >= reread_word_coverage_threshold

    Returns:
      dict:
        sentence_id -> {
            "eligible": int,
            "reread": int,
            "initial_coverage": float,
            "return_coverage": float,
            "n_words_in_sentence": int,
            "n_unique_initial_words": int,
            "n_unique_returned_words": int,
            "visited": int,
        }
    """
    sentences = sentence_meta_for_stimulus["sentences"]

    # Sentence metadata lookup
    sent_info = {}
    for sent in sentences:
        sent_id = sent["sentence_idx"]
        word_indices = list(range(sent["start_idx"], sent["end_idx"] + 1))
        sent_info[sent_id] = {
            "word_indices": set(word_indices),
            "n_words": len(word_indices),
        }

    visited_sentences = set()
    moved_beyond = {sent_id: False for sent_id in sent_info.keys()}

    # unique words seen before moving beyond sentence
    initial_words = {sent_id: set() for sent_id in sent_info.keys()}

    # unique words seen after moving beyond sentence
    returned_words = {sent_id: set() for sent_id in sent_info.keys()}

    max_seen_sentence = -1

    # Process fixations in temporal order
    for fix in fixation_data:
        word_index = fix.get("word_index", -1)
        if word_index is None or word_index == -1:
            continue

        current_sent_id, _ = get_sentence_info(word_index, sentences)
        if current_sent_id == -1:
            continue

        visited_sentences.add(current_sent_id)

        # record as initial-pass coverage if sentence has not yet been moved beyond
        if not moved_beyond[current_sent_id]:
            if word_index in sent_info[current_sent_id]["word_indices"]:
                initial_words[current_sent_id].add(word_index)
        else:
            # otherwise it is a return fixation
            if word_index in sent_info[current_sent_id]["word_indices"]:
                returned_words[current_sent_id].add(word_index)

        # update moved_beyond after processing the current fixation
        if current_sent_id > max_seen_sentence:
            for prev_sent_id in sent_info.keys():
                if prev_sent_id < current_sent_id:
                    moved_beyond[prev_sent_id] = True
            max_seen_sentence = current_sent_id

    results = {}
    for sent_id in visited_sentences:
        n_words = sent_info[sent_id]["n_words"]

        n_initial = len(initial_words[sent_id])
        n_returned = len(returned_words[sent_id])

        initial_coverage = (n_initial / n_words) if n_words > 0 else 0.0
        return_coverage = (n_returned / n_words) if n_words > 0 else 0.0

        eligible = int(initial_coverage >= initial_word_coverage_threshold)
        reread = int(eligible and (return_coverage >= reread_word_coverage_threshold))

        results[sent_id] = {
            "eligible": eligible,
            "reread": reread,
            "initial_coverage": float(initial_coverage),
            "return_coverage": float(return_coverage),
            "n_words_in_sentence": int(n_words),
            "n_unique_initial_words": int(n_initial),
            "n_unique_returned_words": int(n_returned),
            "visited": 1,
        }

    return results


# def build_sentence_level_records(
#     scanpath_data: List[Dict],
#     sentence_metadata: List[Dict],
#     scorer: AppraisalScorer,
#     reread_word_coverage_threshold: float = 0.3,
# ) -> pd.DataFrame:
#     """
#     Create one row per participant x stimulus x sentence visited:
#       participant_id
#       stimulus_index
#       time_constraint
#       sentence_idx
#       sentence_text
#       initial_appraisal
#       reread
#       reread_coverage
#       n_words_in_sentence
#       n_unique_returned_words
#     """
#     meta_by_stimulus = {x["stimulus_id"]: x for x in sentence_metadata}
#     rows = []

#     for trial in scanpath_data:
#         stimulus_index = trial["stimulus_index"]
#         participant_id = trial.get("participant_index", None)
#         time_constraint = trial.get("time_constraint", None)
#         fixation_data = trial["fixation_data"]

#         if stimulus_index not in meta_by_stimulus:
#             continue

#         stim_meta = meta_by_stimulus[stimulus_index]
#         sentences = stim_meta["sentences"]

#         reread_metrics = compute_sentence_reread_metrics(
#             fixation_data=fixation_data,
#             sentence_meta_for_stimulus=stim_meta,
#             reread_word_coverage_threshold=reread_word_coverage_threshold,
#         )

#         if not reread_metrics:
#             continue

#         sentence_texts = {s["sentence_idx"]: s["sentence"] for s in sentences}

#         for sent_id in sorted(reread_metrics.keys()):
#             current_sentence = sentence_texts[sent_id]
#             prior_context = " ".join(
#                 sentence_texts[i] for i in sorted(sentence_texts.keys()) if i < sent_id
#             )

#             appraisal = scorer.score_sentence(prior_context, current_sentence)
#             metrics = reread_metrics[sent_id]

#             rows.append({
#                 "participant_id": participant_id,
#                 "stimulus_index": stimulus_index,
#                 "time_constraint": time_constraint,
#                 "sentence_idx": sent_id,
#                 "sentence_text": current_sentence,
#                 "initial_appraisal": appraisal,
#                 "reread": int(metrics["reread"]),
#                 "reread_coverage": float(metrics["reread_coverage"]),
#                 "n_words_in_sentence": int(metrics["n_words_in_sentence"]),
#                 "n_unique_returned_words": int(metrics["n_unique_returned_words"]),
#             })

#     return pd.DataFrame(rows)


def build_sentence_level_records(
    scanpath_data: List[Dict],
    sentence_metadata: List[Dict],
    scorer: AppraisalScorer,
    reread_word_coverage_threshold: float = 0.3,
    initial_word_coverage_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Create one row per participant x stimulus x sentence visited.
    Includes:
      - initial_appraisal
      - eligible_for_reread_analysis
      - initial_coverage
      - return_coverage
      - reread (thresholded binary)
    """
    meta_by_stimulus = {x["stimulus_id"]: x for x in sentence_metadata}
    rows = []

    for trial in scanpath_data:
        stimulus_index = trial["stimulus_index"]
        participant_id = trial.get("participant_index", None)
        time_constraint = trial.get("time_constraint", None)
        fixation_data = trial["fixation_data"]

        if stimulus_index not in meta_by_stimulus:
            continue

        stim_meta = meta_by_stimulus[stimulus_index]
        sentences = stim_meta["sentences"]

        reread_metrics = compute_sentence_reread_metrics(
            fixation_data=fixation_data,
            sentence_meta_for_stimulus=stim_meta,
            reread_word_coverage_threshold=reread_word_coverage_threshold,
            initial_word_coverage_threshold=initial_word_coverage_threshold,
        )

        if not reread_metrics:
            continue

        sentence_texts = {s["sentence_idx"]: s["sentence"] for s in sentences}

        for sent_id in sorted(reread_metrics.keys()):
            current_sentence = sentence_texts[sent_id]
            prior_context = " ".join(
                sentence_texts[i] for i in sorted(sentence_texts.keys()) if i < sent_id
            )

            appraisal = scorer.score_sentence(prior_context, current_sentence)
            metrics = reread_metrics[sent_id]

            rows.append({
                "participant_id": participant_id,
                "stimulus_index": stimulus_index,
                "time_constraint": time_constraint,
                "sentence_idx": sent_id,
                "sentence_text": current_sentence,
                "initial_appraisal": appraisal,
                "eligible_for_reread_analysis": int(metrics["eligible"]),
                "reread": int(metrics["reread"]),
                "initial_coverage": float(metrics["initial_coverage"]),
                "return_coverage": float(metrics["return_coverage"]),
                "n_words_in_sentence": int(metrics["n_words_in_sentence"]),
                "n_unique_initial_words": int(metrics["n_unique_initial_words"]),
                "n_unique_returned_words": int(metrics["n_unique_returned_words"]),
            })

    return pd.DataFrame(rows)


def normalize_time_constraint(x):
    """
    Normalize time-constraint labels into strings like '30s', '60s', '90s'.
    Handles values like 30, 30.0, '30', '30s', etc.
    """
    if pd.isna(x):
        return None

    s = str(x).strip().lower()
    s = s.replace("seconds", "s").replace("second", "s")

    if s in {"30", "30.0"}:
        return "30s"
    if s in {"60", "60.0"}:
        return "60s"
    if s in {"90", "90.0"}:
        return "90s"
    if s in {"30s", "60s", "90s"}:
        return s

    return s


def filter_by_time_conditions(df, selected_conditions=None):
    """
    Keep only rows whose time_constraint belongs to selected_conditions.
    selected_conditions example: ['30s'], ['30s', '60s'], ['60s', '90s']
    """
    df = df.copy()
    df["time_constraint_norm"] = df["time_constraint"].apply(normalize_time_constraint)

    if selected_conditions is None:
        return df

    selected_norm = [normalize_time_constraint(x) for x in selected_conditions]
    return df[df["time_constraint_norm"].isin(selected_norm)].copy()


# def bin_rereading_probability(
#     df: pd.DataFrame,
#     bins: np.ndarray = DEFAULT_BINS,
# ) -> pd.DataFrame:
#     """
#     Bin initial appraisal and compute:
#       P(reread | appraisal bin) = n_reread / n_sentences
#     """
#     if df.empty:
#         return pd.DataFrame(columns=[
#             "time_subset",
#             "initial_appraisal_bin",
#             "initial_appraisal_bin_left",
#             "initial_appraisal_bin_right",
#             "initial_appraisal_bin_center",
#             "n_sentences",
#             "n_reread",
#             "probability_of_rereading"
#         ])

#     df = df.copy()

#     eps = 1e-9
#     clipped = np.clip(df["initial_appraisal"].to_numpy(), 0.0, 1.0 - eps)
#     df["appraisal_bin_idx"] = np.digitize(clipped, bins[1:], right=False)

#     records = []
#     for bin_idx in range(len(bins) - 1):
#         left = float(bins[bin_idx])
#         right = float(bins[bin_idx + 1])

#         subset = df[df["appraisal_bin_idx"] == bin_idx]
#         n_sentences = int(len(subset))
#         n_reread = int(subset["reread"].sum())
#         prob = (n_reread / n_sentences) if n_sentences > 0 else np.nan

#         records.append({
#             "initial_appraisal_bin": f"[{left:.1f}, {right:.1f})",
#             "initial_appraisal_bin_left": left,
#             "initial_appraisal_bin_right": right,
#             "initial_appraisal_bin_center": (left + right) / 2.0,
#             "n_sentences": n_sentences,
#             "n_reread": n_reread,
#             "probability_of_rereading": prob,
#         })

#     return pd.DataFrame(records)


def bin_rereading_probability(
    df: pd.DataFrame,
    bins: np.ndarray = DEFAULT_BINS,
    eligible_only: bool = True,
) -> pd.DataFrame:
    """
    Bin initial appraisal and compute:
      P(reread | appraisal bin)

    If eligible_only=True, only include sentences with
      eligible_for_reread_analysis == 1
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "initial_appraisal_bin",
            "initial_appraisal_bin_left",
            "initial_appraisal_bin_right",
            "initial_appraisal_bin_center",
            "n_sentences",
            "n_reread",
            "probability_of_rereading",
            "mean_initial_coverage",
            "mean_return_coverage",
        ])

    df = df.copy()

    if eligible_only:
        df = df[df["eligible_for_reread_analysis"] == 1].copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "initial_appraisal_bin",
            "initial_appraisal_bin_left",
            "initial_appraisal_bin_right",
            "initial_appraisal_bin_center",
            "n_sentences",
            "n_reread",
            "probability_of_rereading",
            "mean_initial_coverage",
            "mean_return_coverage",
        ])

    eps = 1e-9
    clipped = np.clip(df["initial_appraisal"].to_numpy(), 0.0, 1.0 - eps)
    df["appraisal_bin_idx"] = np.digitize(clipped, bins[1:], right=False)

    records = []
    for bin_idx in range(len(bins) - 1):
        left = float(bins[bin_idx])
        right = float(bins[bin_idx + 1])

        subset = df[df["appraisal_bin_idx"] == bin_idx]
        n_sentences = int(len(subset))
        n_reread = int(subset["reread"].sum())
        prob = (n_reread / n_sentences) if n_sentences > 0 else np.nan

        mean_initial_coverage = float(subset["initial_coverage"].mean()) if n_sentences > 0 else np.nan
        mean_return_coverage = float(subset["return_coverage"].mean()) if n_sentences > 0 else np.nan

        records.append({
            "initial_appraisal_bin": f"[{left:.1f}, {right:.1f})",
            "initial_appraisal_bin_left": left,
            "initial_appraisal_bin_right": right,
            "initial_appraisal_bin_center": (left + right) / 2.0,
            "n_sentences": n_sentences,
            "n_reread": n_reread,
            "probability_of_rereading": prob,
            "mean_initial_coverage": mean_initial_coverage,
            "mean_return_coverage": mean_return_coverage,
        })

    return pd.DataFrame(records)


# def export_time_subset_binnings(raw_df, output_dir, bins=DEFAULT_BINS):
#     """
#     Export separate binned CSVs for:
#       - all
#       - 30s
#       - 60s
#       - 90s
#       - 30s_60s
#       - 60s_90s
#     """
#     subset_specs = {
#         "all": None,
#         "30s": ["30s"],
#         "60s": ["60s"],
#         "90s": ["90s"],
#         "30s_60s": ["30s", "60s"],
#         "60s_90s": ["60s", "90s"],
#     }

#     all_outputs = []

#     for subset_name, selected_conditions in subset_specs.items():
#         df_sub = filter_by_time_conditions(raw_df, selected_conditions)
#         binned = bin_rereading_probability(df_sub, bins=bins)
#         binned.insert(0, "time_subset", subset_name)

#         out_csv = os.path.join(
#             output_dir,
#             f"human_sentence_appraisal_reread_binned_{subset_name}.csv"
#         )
#         binned.to_csv(out_csv, index=False)
#         print(f"Saved: {out_csv}")

#         all_outputs.append(binned)

#     combined = pd.concat(all_outputs, ignore_index=True)
#     combined_csv = os.path.join(
#         output_dir,
#         "human_sentence_appraisal_reread_binned_all_subsets.csv"
#     )
#     combined.to_csv(combined_csv, index=False)
#     print(f"Saved combined: {combined_csv}")

def export_time_subset_binnings(
    raw_df,
    output_dir,
    bins=DEFAULT_BINS,
    reread_word_coverage_threshold: float = 0.3,
    initial_word_coverage_threshold: float = 0.5,
):
    """
    Export separate binned CSVs for:
      - all
      - 30s
      - 60s
      - 90s
      - 30s_60s
      - 60s_90s
    """
    subset_specs = {
        "all": None,
        "30s": ["30s"],
        "60s": ["60s"],
        "90s": ["90s"],
        "30s_60s": ["30s", "60s"],
        "60s_90s": ["60s", "90s"],
    }

    all_outputs = []

    for subset_name, selected_conditions in subset_specs.items():
        df_sub = filter_by_time_conditions(raw_df, selected_conditions)
        binned = bin_rereading_probability(df_sub, bins=bins, eligible_only=True)
        binned.insert(0, "time_subset", subset_name)

        out_csv = os.path.join(
            output_dir,
            f"human_sentence_appraisal_reread_binned_{subset_name}"
            f"_initcov_{int(initial_word_coverage_threshold * 100)}"
            f"_returncov_{int(reread_word_coverage_threshold * 100)}.csv"
        )
        binned.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

        all_outputs.append(binned)

    combined = pd.concat(all_outputs, ignore_index=True)
    combined_csv = os.path.join(
        output_dir,
        f"human_sentence_appraisal_reread_binned_all_subsets"
        f"_initcov_{int(initial_word_coverage_threshold * 100)}"
        f"_returncov_{int(reread_word_coverage_threshold * 100)}.csv"
    )
    combined.to_csv(combined_csv, index=False)
    print(f"Saved combined: {combined_csv}")


# def main(
#     scanpath_json_path: str,
#     sentence_metadata_path: str,
#     output_dir: str,
#     appraisal_mode: str = "dummy",
#     appraisal_model: str = "gpt-4.1-mini",
#     reread_word_coverage_threshold: float = 0.3,
# ):
#     os.makedirs(output_dir, exist_ok=True)

#     scanpath_data = load_json(scanpath_json_path)
#     sentence_metadata = load_json(sentence_metadata_path)

#     scorer = AppraisalScorer(mode=appraisal_mode, model_name=appraisal_model)

#     raw_df = build_sentence_level_records(
#         scanpath_data=scanpath_data,
#         sentence_metadata=sentence_metadata,
#         scorer=scorer,
#         reread_word_coverage_threshold=reread_word_coverage_threshold,
#     )

#     raw_csv = os.path.join(
#         output_dir,
#         f"human_sentence_appraisal_reread_raw_threshold_{int(reread_word_coverage_threshold * 100)}.csv"
#     )
#     raw_df.to_csv(raw_csv, index=False)

#     binned_df = bin_rereading_probability(raw_df, bins=DEFAULT_BINS)
#     binned_csv = os.path.join(
#         output_dir,
#         f"human_sentence_appraisal_reread_binned_threshold_{int(reread_word_coverage_threshold * 100)}.csv"
#     )
#     binned_df.to_csv(binned_csv, index=False)

#     export_time_subset_binnings(raw_df, output_dir)

#     print(f"Saved raw sentence-level data to: {raw_csv}")
#     print(f"Saved binned rereading-probability data to: {binned_csv}")


def main(
    scanpath_json_path: str,
    sentence_metadata_path: str,
    output_dir: str,
    appraisal_mode: str = "dummy",
    appraisal_model: str = "gpt-4.1-mini",
    reread_word_coverage_threshold: float = 0.3,
    initial_word_coverage_threshold: float = 0.5,
):
    os.makedirs(output_dir, exist_ok=True)

    scanpath_data = load_json(scanpath_json_path)
    sentence_metadata = load_json(sentence_metadata_path)

    scorer = AppraisalScorer(mode=appraisal_mode, model_name=appraisal_model)

    raw_df = build_sentence_level_records(
        scanpath_data=scanpath_data,
        sentence_metadata=sentence_metadata,
        scorer=scorer,
        reread_word_coverage_threshold=reread_word_coverage_threshold,
        initial_word_coverage_threshold=initial_word_coverage_threshold,
    )

    raw_csv = os.path.join(
        output_dir,
        f"human_sentence_appraisal_reread_raw"
        f"_initcov_{int(initial_word_coverage_threshold * 100)}"
        f"_returncov_{int(reread_word_coverage_threshold * 100)}.csv"
    )
    raw_df.to_csv(raw_csv, index=False)

    binned_df = bin_rereading_probability(raw_df, bins=DEFAULT_BINS, eligible_only=True)
    binned_csv = os.path.join(
        output_dir,
        f"human_sentence_appraisal_reread_binned"
        f"_initcov_{int(initial_word_coverage_threshold * 100)}"
        f"_returncov_{int(reread_word_coverage_threshold * 100)}.csv"
    )
    binned_df.to_csv(binned_csv, index=False)

    export_time_subset_binnings(
        raw_df,
        output_dir,
        bins=DEFAULT_BINS,
        reread_word_coverage_threshold=reread_word_coverage_threshold,
        initial_word_coverage_threshold=initial_word_coverage_threshold,
    )

    print(f"Saved raw sentence-level data to: {raw_csv}")
    print(f"Saved binned rereading-probability data to: {binned_csv}")


# if __name__ == "__main__":
#     # Example paths: update these to your local project structure
#     base_dir = "/home/baiy4/reader-agent-zuco"

#     scanpath_json_path = os.path.join(
#         base_dir,
#         "data/human_data/bai_read_under_time_pressure/corrected_data_by_fix8/11_all_corrected_scanpaths_across_stimuli",
#         SCANPATH_JSON,
#     )

#     sentence_metadata_path = os.path.join(
#         base_dir,
#         "data/human_data/bai_read_under_time_pressure/stimuli/10_27_15_58_100_images_W1920H1080WS16_LS40_MARGIN400/assets",
#         SENTENCE_METADATA_JSON,
#     )

#     output_dir = os.path.join(
#         base_dir,
#         "data/human_data/bai_read_under_time_pressure/calculated_effects_appraisal_reread"
#     )

#     main(
#         scanpath_json_path=scanpath_json_path,
#         sentence_metadata_path=sentence_metadata_path,
#         output_dir=output_dir,
#         appraisal_mode="dummy",      # change to "openai" when ready
#         appraisal_model="gpt-4.1-mini",
#         reread_word_coverage_threshold=0.3,
#     )


if __name__ == "__main__":
    base_dir = "/home/baiy4/reader-agent-zuco"

    scanpath_json_path = os.path.join(
        base_dir,
        "data/human_data/bai_read_under_time_pressure/corrected_data_by_fix8/11_all_corrected_scanpaths_across_stimuli",
        SCANPATH_JSON,
    )

    sentence_metadata_path = os.path.join(
        base_dir,
        "data/human_data/bai_read_under_time_pressure/stimuli/10_27_15_58_100_images_W1920H1080WS16_LS40_MARGIN400/assets",
        SENTENCE_METADATA_JSON,
    )

    output_dir = os.path.join(
        base_dir,
        "data/human_data/bai_read_under_time_pressure/calculated_effects_appraisal_reread"
    )

    main(
        scanpath_json_path=scanpath_json_path,
        sentence_metadata_path=sentence_metadata_path,
        output_dir=output_dir,
        appraisal_mode="dummy",
        appraisal_model="gpt-4.1-mini",
        reread_word_coverage_threshold=0.3,
        initial_word_coverage_threshold=0.5,
    )