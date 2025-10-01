#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_v2a_stepwise.py

V2a (stepwise, strictly sequential, no dedup, no MMR):
- Reads v1 output: ltm_gists_v1.json (with debug.items[*] holding per-step sentence facets)
- Processes *each step in order*:
    * rank that step's facets by semantic relevance to that step's sentence
      (SBERT embeddings if available; otherwise TF-IDF cosine)
    * keep a FIXED top-K per step (same K across all time conditions)
    * append kept facets directly to the outline (no pooling, no dedup, no diversity)
- Emits ltm_gists_v2a_stepwise.json with the SAME `results[*].outline`/`gists` interface as v1.
"""
import argparse, json, os, re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

class Embedder:
    def __init__(self):
        self.backend = None
        self.model = None
        self.vectorizer = None
        self.name = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.backend = "sbert"
            self.name = "all-MiniLM-L6-v2"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize
            self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2))
            self.backend = "tfidf"
            self.name = "tfidf-1_2"

    def encode_pairset(self, sentence: str, facets: List[str]) -> Tuple[np.ndarray, np.ndarray]:
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

def normalize_facet_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def transform_v1_to_v2a_stepwise(input_path: str, output_path: str, k_per_step: int = 2) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        v1 = json.load(f)

    embedder = Embedder()
    v2 = {
        "pipeline": "v2a_stepwise_local_selector",
        "source_pipeline": v1.get("pipeline"),
        "scoring_backend": embedder.backend,
        "scoring_model": embedder.name,
        "k_per_step": int(k_per_step),
        "results": []
    }

    for ep in v1.get("results", []):
        items = list(ep.get("debug", {}).get("items", []))
        try:
            items.sort(key=lambda x: x.get("step", 0))
        except Exception:
            pass

        outline_lines = []
        per_step_scores = []

        for it in items:
            sent_text = (it.get("sentence_text") or "").strip()
            facets_raw = it.get("facets") or []
            facets = [normalize_facet_text(x) for x in facets_raw if x]

            if not facets or not sent_text:
                per_step_scores.append({
                    "step": it.get("step"),
                    "sent_id": it.get("sent_id"),
                    "sentence_text": sent_text,
                    "candidates": []
                })
                continue

            sent_vec, facet_mat = embedder.encode_pairset(sent_text, facets)
            base_scores = facet_mat @ sent_vec  # cosine

            N = len(facets)
            K = min(int(k_per_step), N)
            top_idx = np.argsort(-base_scores)[:K].tolist()
            kept = [facets[i] for i in top_idx]

            for facet in kept:
                outline_lines.append(f"- {facet}")

            per_step_scores.append({
                "step": it.get("step"),
                "sent_id": it.get("sent_id"),
                "sentence_text": sent_text,
                "candidates": [
                    {"facet": facets[i], "relevance": float(base_scores[i]), "selected": (i in top_idx)}
                    for i in range(N)
                ]
            })

        outline = "\n".join(outline_lines)

        v2["results"].append({
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
        json.dump(v2, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser(description="V2a (stepwise): keep top-K facets per step by local semantic relevance")
    ap.add_argument("--input", required=True, help="Path to ltm_gists_v1.json")
    ap.add_argument("--output", required=True, help="Path to write ltm_gists_v2a_stepwise.json")
    ap.add_argument("--k_per_step", type=int, default=2, help="Fixed number of facets to keep per step (all conditions)")
    args = ap.parse_args()
    transform_v1_to_v2a_stepwise(args.input, args.output, args.k_per_step)

if __name__ == "__main__":
    main()
