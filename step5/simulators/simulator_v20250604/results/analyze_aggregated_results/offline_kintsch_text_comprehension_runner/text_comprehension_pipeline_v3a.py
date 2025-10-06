#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, re, numpy as np
from typing import List, Dict

class Embedder:
    def __init__(self):
        self.backend = None
        self.model = None
        self.vectorizer = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.backend = "sbert"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2))
            self.backend = "tfidf"
    def encode(self, texts: List[str]):
        if self.backend == "sbert":
            return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        else:
            from sklearn.preprocessing import normalize
            X = self.vectorizer.fit_transform(texts).astype(np.float32)
            X = normalize(X, norm="l2", axis=1)
            return X.toarray()

def strip_bullet(line: str) -> str:
    return re.sub(r"^\s*[-•]\s*", "", (line or "").strip())

def parse_outline(outline: str):
    if not outline: return []
    return [strip_bullet(ln) for ln in outline.splitlines() if ln.strip()]

def rebuild_outline(facets):
    return "\n".join(f"- {f}" for f in facets if f and f.strip())

def normalize_for_embed(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def pairwise_drop_earlier(facets, tau_dup):
    n = len(facets)
    if n == 0:
        return {"kept_indices": [], "removed_indices": [], "pairs_triggered": []}
    emb = Embedder()
    texts = [normalize_for_embed(x) for x in facets]
    X = emb.encode(texts)
    S = X @ X.T
    drop = set()
    triggers = []
    for i in range(n-1):
        if i in drop: 
            continue
        sims_row = S[i, i+1:]
        if sims_row.size == 0:
            continue
        idxs = np.where(sims_row >= tau_dup)[0]
        if idxs.size > 0:
            j = int(idxs[0] + (i+1))
            drop.add(i)
            triggers.append((i, j, float(S[i, j])))
    kept = [idx for idx in range(n) if idx not in drop]
    removed = sorted(list(drop))
    return {"kept_indices": kept, "removed_indices": removed, "pairs_triggered": triggers}

def run(input_path, output_path, tau_dup):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {
        "pipeline": f"{data.get('pipeline','unknown')}_v3a_pairwise_dedupe",
        "source_pipeline": data.get("pipeline"),
        "tau_dup": float(tau_dup),
        "results": []
    }
    for ep in data.get("results", []):
        text = ep.get("outline") or ep.get("gists") or ""
        facets = parse_outline(text)
        info = pairwise_drop_earlier(facets, tau_dup)
        kept_facets = [facets[i] for i in info["kept_indices"]]
        dedup_outline = rebuild_outline(kept_facets)
        out["results"].append({
            "episode_index": ep.get("episode_index"),
            "stimulus_index": ep.get("stimulus_index"),
            "time_condition": ep.get("time_condition"),
            "total_time": ep.get("total_time"),
            "outline": dedup_outline,
            "gists": dedup_outline,
            "debug": {"dedupe_pairwise": info}
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Pairwise per-episode dedupe (drop earlier if later is similar)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tau_dup", type=float, default=0.87)
    args = ap.parse_args()
    run(args.input, args.output, args.tau_dup)

if __name__ == "__main__":
    main()
