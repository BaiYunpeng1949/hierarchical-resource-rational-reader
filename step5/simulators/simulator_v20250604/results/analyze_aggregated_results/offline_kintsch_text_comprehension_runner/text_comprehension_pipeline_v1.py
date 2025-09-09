#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
text_comprehension_pipeline.py (v1.2 — facets-only, comprehension_test compatible)

- Reads simulation_read_contents.json (episodes with text_reading_logs)
- Reconstructs sentence text from sampled tokens
- Uses LLM (or OFFLINE fallback) to extract SHORT facets per sentence
- Stacks facets in order into an outline string
- Writes a single JSON with top-level "results" and per-trial "outline"/"gists" strings
  that comprehension_test.py can read directly.

Run examples:
  # Offline fallback (no LLM), first 5 episodes
  python text_comprehension_pipeline.py \
    --input simulation_read_contents.json \
    --output ltm_gists_v1_compat.json \
    --max_facets 3 \
    --model offline \
    --max_episodes 5

  # Use your llm_agent gateway (model string is passed to llm_agent;
  # routing is decided by llm_agent.py itself)
  python text_comprehension_pipeline.py \
    --input simulation_read_contents.json \
    --output ltm_gists_v1_compat.json \
    --max_facets 3 \
    --model gpt-4o \
    --max_episodes 10 \
    --episode_offset 20 \
    --max_steps 120
"""
import argparse
import json
import os
import re
from typing import List, Dict, Any, Optional

# ----------------- minimal facetizer (fallback) -----------------

_STOP = set("""
a an and are as at be by for from has have he her hers him his i in is it its 
of on or our she that the their them they this to was we were will with you your yours not 
but if then than so such into over under between within without about above below after before
""".split())

def _word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-']+|\d+(?:\.\d+)?", text)

def _phrase_chunks(tokens: List[str]) -> List[str]:
    chunks, cur = [], []
    for t in tokens:
        low = t.lower()
        if low not in _STOP:
            cur.append(low)
        else:
            if cur:
                chunks.append(" ".join(cur))
                cur = []
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c.replace(" ", "")) > 2]

def _local_facets(text: str, k: int) -> List[str]:
    toks = _word_tokens(text)
    chunks = _phrase_chunks(toks)
    from collections import Counter
    wc = Counter()
    for ch in chunks:
        wc.update(ch.split())
    scored = []
    for ch in chunks:
        score = sum(wc[w] for w in ch.split()) + 0.2*(len(ch.split())-1)
        scored.append((ch, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    out, seen = [], set()
    for ch, _ in scored:
        if ch not in seen:
            seen.add(ch)
            out.append(ch)
            if len(out) >= k:
                break
    return out or (toks[:k] if toks else [])

# ----------------- sentence reconstruction -----------------

def reconstruct_sentence(sampled_words: List[str]) -> str:
    toks = [t for t in sampled_words if t and isinstance(t, str)]
    s = " ".join(toks)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s.strip()

# ----------------- LLM prompt -----------------

def build_prompt(sentence: str, max_facets: int) -> str:
    return (
        "Extract up to {k} SHORT facets from the sentence.\n"
        "Constraints: one facet per line; no numbering; <=12 tokens per facet; avoid function words.\n"
        "Aim for concrete, non-overlapping facts; no paraphrase of the whole sentence.\n\n"
        "Sentence: {s}\n\n"
        "Output: just the facets, one per line."
    ).format(k=max_facets, s=sentence)

DEFAULT_ROLE = (
    "You are an expert reader extracting small facets from sentences to build an ordered LTM gist.\n"
    "Only return concise facets; no commentary, no bullets, one per line."
)

def get_facets_with_llm(llm, sentence: str, max_facets: int) -> List[str]:
    try:
        prompt = build_prompt(sentence, max_facets)
        lines = llm.get_facet_summaries(role=DEFAULT_ROLE, prompt=prompt)
        facets = [ln.strip("-• ").strip() for ln in lines if str(ln).strip()]
        return facets[:max_facets] if max_facets > 0 else facets
    except Exception:
        return _local_facets(sentence, max_facets)

# ----------------- outline builder -----------------

def _build_outline_from_items(items: List[Dict[str, Any]]) -> str:
    lines = []
    for it in items:
        for f in (it.get("facets") or []):
            if not f:
                continue
            facet = re.sub(r"\s+", " ", str(f).strip())
            lines.append(f"- {facet}")
    return "\n".join(lines)

# ----------------- core processing -----------------

def process_sim_file(path: str,
                     max_facets: int,
                     model_name: str,
                     max_episodes: Optional[int],
                     episode_offset: int,
                     max_steps: Optional[int]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    # Episode slicing
    start = max(episode_offset, 0)
    end = start + max_episodes if (max_episodes is not None) else None
    episodes = episodes[start:end]

    # LLM wiring
    llm = None
    if model_name and model_name.lower() != "offline":
        try:
            from llm_agent import LLMAgent
            # NOTE: In your llm_agent.py, Aalto gateway decides routing; model_name is mostly informational.
            llm = LLMAgent(model_name=model_name, api_key="")
        except Exception:
            llm = None  # fallback

    results = []
    for ep in episodes:

        # Print log
        print(f"The episode index is: {ep['episode_index']}, the stimulus index is: {ep['stimulus_index']}, the time condition is: {ep['time_condition']}")

        debug_block = {"schemas": [], "items": []}  # keep raw facets per sentence
        logs = ep.get("text_reading_logs", [])
        if max_steps is not None and max_steps >= 0:
            logs = [lg for lg in logs if (lg.get("step") or 0) < max_steps]

        for step in logs:
            sent_id = step.get("actual_reading_sentence_index", step.get("current_sentence_index"))
            sampled = step.get("sampled_words_in_sentence", [])
            sentence_text = reconstruct_sentence(sampled)

            # Print log
            print(f"    The sent id is: {sent_id}")
            print(f"    The sampled text is: {sampled}")
            print(f"    The sentence text is: {sentence_text}")

            if not sentence_text:
                continue
            if llm is not None:
                facets = get_facets_with_llm(llm, sentence_text, max_facets)
            else:
                facets = _local_facets(sentence_text, max_facets)
            debug_block["items"].append({
                "step": step.get("step"),
                "sent_id": sent_id,
                "sentence_text": sentence_text,
                "facets": facets
            })

        outline = _build_outline_from_items(debug_block["items"])
        results.append({
            "episode_index": ep.get("episode_index"),
            "stimulus_index": ep.get("stimulus_index"),
            "time_condition": ep.get("time_condition"),
            "total_time": ep.get("total_time"),
            "outline": outline,
            "gists": outline,     # alias
            "debug": debug_block  # raw details for later analyses
        })

    return {
        "pipeline": "v1_facets_only",
        "model_role": DEFAULT_ROLE,
        "max_facets_per_sentence": max_facets,
        "results": results
    }

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="Minimal facets-only LTM gist builder (comprehension_test compatible)")
    ap.add_argument("--input", required=True, help="Path to simulation_read_contents.json")
    ap.add_argument("--output", required=True, help="Where to write the LTM gist JSON")
    ap.add_argument("--max_facets", type=int, default=3, help="Max facets per sentence")
    ap.add_argument("--model", default="offline",
                    help="Model name for llm_agent (use 'offline' for local facetizer). "
                         "Note: with Aalto gateway enabled in llm_agent.py, the routing ignores this name.")
    ap.add_argument("--max_episodes", type=int, default=None, help="Process only the first N episodes (after offset)")
    ap.add_argument("--episode_offset", type=int, default=0, help="Skip this many episodes before processing")
    ap.add_argument("--max_steps", type=int, default=None, help="Cap steps per episode (keep steps < max_steps)")
    args = ap.parse_args()

    out = process_sim_file(
        path=args.input,
        max_facets=args.max_facets,
        model_name=args.model,
        max_episodes=args.max_episodes,
        episode_offset=args.episode_offset,
        max_steps=args.max_steps
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote LTM gist to {args.output}")

if __name__ == "__main__":
    main()
