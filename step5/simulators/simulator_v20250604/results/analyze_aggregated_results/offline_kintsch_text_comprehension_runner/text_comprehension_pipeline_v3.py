#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_v3b_stepwise_global_boost.py

V3b (stepwise, additive global-context boost + threshold):
- Strictly processes reading steps in order (no pooling, no dedup, no MMR).
- For each step's sentence S_t and its candidate facets {f_k}, compute:
    local_sim(f)  in [0,1]
    ctx_avg(f)    = weighted average of sim(f, S_i) over the most-recent K past sentences
    R(f)          = clip( local_sim(f) + ctx_boost * ctx_avg(f), 0, 1 )
  This guarantees R(f) >= local_sim(f) whenever ctx_avg > 0, i.e., global context *boosts* local evidence.
- Keep a facet iff R(f) >= tau_gist (same threshold for all conditions).
- Fully offline: SBERT if available (all-MiniLM-L6-v2), else TF-IDF (1–2 grams) with L2 cosine.

Why this design?
- Cognitively: readers integrate current and recent context; consistent context should increase belief in a proposition.
- Technically: additive boost prevents the mixture from diluting strong local evidence (unlike convex averages).

CLI:
  python build_v3b_stepwise_global_boost.py \
      --input /path/to/ltm_gists_v1.json \
      --output /path/to/ltm_gists_v3b.json \
      --tau_gist 0.4 \
      --context_window 3 \
      --half_life 3.0 \
      --ctx_boost 1.0
"""
import argparse, json, os, re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# -------- Embedding backend (SBERT if available; else TF-IDF) --------

class Embedder:
    """
    Wraps either SBERT ('all-MiniLM-L6-v2') or a TF-IDF (1-2grams) fallback.
    Ensures returned vectors are L2-normalized row-wise so dot==cosine.
    """
    def __init__(self):
        self.backend = None
        self.model = None
        self.vectorizer = None
        self.name = None
        # Try SBERT first
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.backend = "sbert"
            self.name = "all-MiniLM-L6-v2"
        except Exception:
            # Fallback to TF-IDF
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2))
            self.backend = "tfidf"
            self.name = "tfidf-1_2"

    def encode_sentence(self, sentence: str) -> np.ndarray:
        """Return L2-normalized vector of the sentence (shape: (d,))."""
        if self.backend == "sbert":
            vec = self.model.encode([sentence], convert_to_numpy=True, normalize_embeddings=True)[0]
            return vec  # already L2-normalized
        else:
            from sklearn.preprocessing import normalize
            X = self.vectorizer.fit_transform([sentence]).astype(np.float32)
            X = normalize(X, norm="l2", axis=1)
            return X.toarray()[0]

    def encode_pairset(self, sentence: str, facets: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (sent_vec, facet_mat) both L2-normalized row-wise.
        - sent_vec: shape (d,)
        - facet_mat: shape (N, d)
        """
        if self.backend == "sbert":
            texts = [sentence] + facets
            vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return vecs[0], vecs[1:]
        else:
            from sklearn.preprocessing import normalize
            corpus = [sentence] + facets
            X = self.vectorizer.fit_transform(corpus).astype(np.float32)
            X = normalize(X, norm="l2", axis=1)
            sent_vec = X[0].toarray()[0]
            facet_mat = X[1:].toarray()
            return sent_vec, facet_mat

    def sim_to_unit(self, cos: np.ndarray) -> np.ndarray:
        """
        Map cosine similarity to [0,1] only for SBERT; TF-IDF is already [0,1].
        cos: scalar or numpy array of cosine values (dot of L2-normalized vectors).
        """
        if self.backend == "sbert":
            return 0.5 * (cos + 1.0)
        else:
            # TF-IDF + L2 cosine already in [0,1]
            return cos

# -------- Utilities --------

def normalize_facet_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def recency_weights(num_past: int, half_life: float) -> np.ndarray:
    """
    Build unnormalized recency weights for indices [1..num_past] where index 1 is the *closest* past step.
    We use powers of 0.5 with a half-life measured in sentences.
    Returns an array w_raw of shape (num_past,) in *reverse* chronological order [t-1, t-2, ..., t-num_past].
    """
    if num_past <= 0:
        return np.zeros((0,), dtype=np.float32)
    # distances: 1,2,...,num_past
    d = np.arange(1, num_past + 1, dtype=np.float32)
    # r = 0.5 ** (d / half_life)
    r = np.power(0.5, d / max(half_life, 1e-6)).astype(np.float32)
    return r  # larger for recent (d small), smaller for older

# -------- Main transform --------

def transform_v1_to_v3b_stepwise(input_path: str,
                                 output_path: str,
                                 tau_gist: float = 0.4,
                                 context_window: int = 3,
                                 half_life: float = 3.0,
                                 ctx_boost: float = 1.0) -> None:
    """
    Transform v1 JSON into v3b additive-boost selection.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        v1 = json.load(f)

    embedder = Embedder()
    v3b = {
        "pipeline": "v3b_stepwise_global_context_boost_threshold",
        "source_pipeline": v1.get("pipeline"),
        "scoring_backend": embedder.backend,
        "scoring_model": embedder.name,
        "tau_gist": float(tau_gist),
        "context_window": int(context_window),
        "half_life": float(half_life),
        "ctx_boost": float(ctx_boost),
        "results": []
    }

    for ep in v1.get("results", []):
        items = list(ep.get("debug", {}).get("items", []))
        # Ensure reading sequence: sort by 'step' if present
        try:
            items.sort(key=lambda x: x.get("step", 0))
        except Exception:
            pass

        outline_lines = []
        per_step_scores = []
        prev_sent_vecs: List[np.ndarray] = []  # SBERT history
        prev_sent_texts: List[str] = []        # TF-IDF history

        for step_idx, it in enumerate(items, start=1):
            sent_text = (it.get("sentence_text") or "").strip()
            facets_raw = it.get("facets") or []
            facets = [normalize_facet_text(x) for x in facets_raw if x]

            if not facets or not sent_text:
                per_step_scores.append({
                    "step": it.get("step"),
                    "sent_id": it.get("sent_id"),
                    "sentence_text": sent_text,
                    "candidates": [],
                })
                # Append sentence to history if we have text
                if sent_text:
                    if embedder.backend == "sbert":
                        prev_sent_vecs.append(embedder.encode_sentence(sent_text))
                    else:
                        prev_sent_texts.append(sent_text)
                continue

            if embedder.backend == "sbert":
                # Encode current sentence and facets
                s_vec, facet_mat = embedder.encode_pairset(sent_text, facets)  # L2-normalized
                # Build matrix of most-recent K past sentence vectors
                if prev_sent_vecs:
                    selected = prev_sent_vecs[-context_window:] if context_window > 0 else prev_sent_vecs[:]
                    selected = selected[::-1]  # most recent first
                    W = len(selected)
                    w_raw = recency_weights(W, half_life)
                    w = w_raw / float(np.sum(w_raw)) if np.sum(w_raw) > 1e-12 else np.ones((W,), dtype=np.float32) / float(W)
                    P = np.stack(selected, axis=1)  # shape (d, W)
                    # Similarities to each past sentence: (N,d) @ (d,W) = (N,W)
                    sim_past = facet_mat @ P
                    sim_past = embedder.sim_to_unit(sim_past)
                    ctx_avg = (sim_past * w).sum(axis=1)  # (N,)
                else:
                    ctx_avg = np.zeros((facet_mat.shape[0],), dtype=np.float32)

                # Local similarity
                sim_local = embedder.sim_to_unit(facet_mat @ s_vec)  # (N,)

            else:
                # TF-IDF: build consistent space over [S_t] + facets + selected past texts
                from sklearn.preprocessing import normalize
                selected_texts = prev_sent_texts[-context_window:] if context_window > 0 else prev_sent_texts[:]
                selected_texts = selected_texts[::-1]  # most recent first
                corpus = [sent_text] + facets + selected_texts
                X = embedder.vectorizer.fit_transform(corpus).astype(np.float32)
                X = normalize(X, norm="l2", axis=1)
                s_vec = X[0].toarray()[0]
                f_mat = X[1:1+len(facets)].toarray()
                if selected_texts:
                    past_mat = X[1+len(facets):].toarray()  # (W,d)
                    W = past_mat.shape[0]
                    w_raw = recency_weights(W, half_life)
                    w = w_raw / float(np.sum(w_raw)) if np.sum(w_raw) > 1e-12 else np.ones((W,), dtype=np.float32) / float(W)
                    sim_past = f_mat @ past_mat.T  # (N,W), already [0,1]
                    ctx_avg = (sim_past * w).sum(axis=1)  # (N,)
                else:
                    ctx_avg = np.zeros((f_mat.shape[0],), dtype=np.float32)

                sim_local = f_mat @ s_vec  # (N,)

                facet_mat = f_mat  # for logging consistency

            # Additive boost: R = clip(local + ctx_boost * ctx_avg, 0, 1)
            R = sim_local + float(ctx_boost) * ctx_avg
            R = np.clip(R, 0.0, 1.0)

            # Threshold selection
            kept_idx = [i for i, r in enumerate(R) if r >= tau_gist]
            kept = [facets[i] for i in kept_idx]
            for facet in kept:
                outline_lines.append(f"- {facet}")

            # Log details per candidate facet for auditing
            cand_log = []
            for i in range(len(facets)):
                cand_log.append({
                    "facet": facets[i],
                    "local_sim": float(sim_local[i]),
                    "ctx_avg_sim": float(ctx_avg[i]),
                    "R": float(R[i]),
                    "selected": (i in kept_idx)
                })

            per_step_scores.append({
                "step": it.get("step"),
                "sent_id": it.get("sent_id"),
                "sentence_text": sent_text,
                "context_window_effective": int(min(len(prev_sent_vecs) if embedder.backend=='sbert' else len(prev_sent_texts), context_window)),
                "half_life": float(half_life),
                "ctx_boost": float(ctx_boost),
                "tau_gist": float(tau_gist),
                "candidates": cand_log
            })

            # Append this sentence to the history for *future* steps
            if embedder.backend == "sbert":
                prev_sent_vecs.append(s_vec)
            else:
                prev_sent_texts.append(sent_text)

        outline = "\n".join(outline_lines)

        v3b["results"].append({
            "episode_index": ep.get("episode_index"),
            "stimulus_index": ep.get("stimulus_index"),
            "time_condition": ep.get("time_condition"),
            "total_time": ep.get("total_time"),
            "outline": outline,
            "gists": outline,
            "debug": {
                "per_step_scores": per_step_scores
            }
        })

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(v3b, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser(description="V3b (stepwise): additive global-context boost with recency + threshold")
    ap.add_argument("--input", required=True, help="Path to ltm_gists_v1.json")
    ap.add_argument("--output", required=True, help="Path to write ltm_gists_v3b.json")
    ap.add_argument("--tau_gist", type=float, default=0.4, help="Relevance threshold for keeping a facet (in [0,1])")
    ap.add_argument("--context_window", type=int, default=3, help="How many recent sentences to include in context (K)")
    ap.add_argument("--half_life", type=float, default=3.0, help="Half-life (in sentences) for recency weighting")
    ap.add_argument("--ctx_boost", type=float, default=1.0, help="Additive weight on the context average similarity")
    args = ap.parse_args()

    transform_v1_to_v3b_stepwise(
        input_path=args.input,
        output_path=args.output,
        tau_gist=args.tau_gist,
        context_window=args.context_window,
        half_life=args.half_life,
        ctx_boost=args.ctx_boost
    )

if __name__ == "__main__":
    main()
