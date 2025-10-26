#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from collections import defaultdict
from statistics import mean
from typing import Dict, Tuple, List, Optional
import math
import re

EN_STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","when","while",
    "of","to","in","on","for","from","by","with","as","at","into","through",
    "is","am","are","was","were","be","been","being","do","does","did","doing",
    "this","that","these","those","it","its","him","her","his","hers","their","theirs","they","them",
    "i","you","we","he","she","my","your","our","ours","me","us","yours",
    "not","no","yes","too","very","just","only","than","so","such",
    "can","could","should","would","may","might","will","shall","must"
}

FR_STOPWORDS = {
    "le","la","les","un","une","des","et","ou","mais","si","alors","sinon",
    "de","du","des","à","au","aux","en","dans","par","pour","sur","avec","chez","sans","sous",
    "est","sont","étais","était","étaient","être","été","étant","ai","as","a","ont","avons","avez",
    "ce","cet","cette","ces","il","elle","ils","elles","on","nous","vous","tu","te","moi","toi",
    "mon","ma","mes","ton","ta","tes","son","sa","ses","leur","leurs",
    "ne","pas","plus","moins","très","trop","bien","mal"
}

def load_stopwords(lang: str, custom_file: Optional[Path]) -> set:
    if custom_file:
        words = set(w.strip().lower() for w in custom_file.read_text(encoding="utf-8").splitlines() if w.strip())
        return words
    if lang.lower().startswith("fr"):
        return FR_STOPWORDS
    return EN_STOPWORDS

def is_content_word(token: str, stopwords: set) -> bool:
    if token is None:
        return False
    t = token.strip().lower()
    if len(t) < 2:
        return False
    if not re.search(r"[a-zàâäçéèêëîïôöùûüÿœ]", t):
        return False
    return t not in stopwords

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def bbox_center(word: dict):
    bbox = word.get("word_bbox")
    if bbox and len(bbox) == 4:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
    return None

def pos_xy(word: dict):
    pos = word.get("position", {})
    x = pos.get("x", pos.get("x_init"))
    y = pos.get("y", pos.get("y_init"))
    if x is None or y is None: return None
    return (float(x), float(y))

def word_xy(word: dict):
    return bbox_center(word) or pos_xy(word)

def build_lookup(meta: dict):
    xy_lookup = {}
    txt_lookup = {}
    for img in meta.get("images", []):
        stim_idx = img.get("image index", img.get("image_index"))
        words = img.get("words metadata") or img.get("words_metadata") or []
        wmap_xy = {}
        wmap_txt = {}
        idxs = []
        for i, w in enumerate(words):
            gid = w.get("word_index_in_text", i)
            try:
                gid = int(gid)
            except Exception:
                continue
            xy = word_xy(w)
            if xy is not None:
                wmap_xy[gid] = (float(xy[0]), float(xy[1]))
            token = w.get("word") or w.get("text") or w.get("token")
            if isinstance(token, str):
                wmap_txt[gid] = token
            idxs.append(gid)
        if idxs and min(idxs) == 1 and 0 not in wmap_xy:
            wmap_xy = {k-1: v for k,v in wmap_xy.items()}
            wmap_txt = {k-1: v for k,v in wmap_txt.items()}
        xy_lookup[int(stim_idx)] = wmap_xy
        txt_lookup[int(stim_idx)] = wmap_txt
    return xy_lookup, txt_lookup

def get_episode_key(ep: dict):
    return (ep["episode_index"], ep["stimulus_index"], ep["time_condition"], float(ep["total_time"]))

def normalize_seq_zero_based(seq):
    nonneg = [i for i in seq if isinstance(i, int) and i >= 0]
    if nonneg and min(nonneg) == 1:
        return [i-1 if isinstance(i, int) and i >= 0 else i for i in seq]
    return seq

def compute_eye_metrics(seq, xy_map):
    sacc_count = 0
    ltr_dx_sum = 0.0
    amps = []
    prev_xy = None
    for idx in seq:
        if not isinstance(idx, int) or idx < 0:
            prev_xy = None
            continue
        xy = xy_map.get(idx)
        if xy is None:
            prev_xy = None
            continue
        if prev_xy is not None:
            dx = xy[0] - prev_xy[0]
            dy = xy[1] - prev_xy[1]
            amps.append(math.hypot(dx, dy))
            sacc_count += 1
            if dx > 0:
                ltr_dx_sum += dx
        prev_xy = xy
    amp_mean = mean(amps) if amps else 0.0
    return sacc_count, amp_mean, ltr_dx_sum

def percent_fixated_content_words(seq, txt_map, stopwords):
    content_indices = {i for i,tok in txt_map.items() if is_content_word(tok, stopwords)}
    if not content_indices:
        return 0.0
    fixated = set(i for i in seq if isinstance(i, int) and i in content_indices)
    return len(fixated) / len(content_indices)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--sim", type=Path, default=None)
    ap.add_argument("--seq", type=Path, default=None)
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("analyzed_fixation_metrics_core.json"))
    ap.add_argument("--lang", type=str, default="en")
    ap.add_argument("--stopwords", type=Path, default=None)
    args = ap.parse_args()

    sim_path  = args.sim  or (args.root / "all_simulation_results.json")
    seq_path  = args.seq  or (args.root / "processed_fixation_sequences.json")
    meta_path = args.meta or (args.root / "lightweight_metadata.json")

    sim  = load_json(sim_path)
    seq  = load_json(seq_path)
    meta = load_json(meta_path)

    xy_lookup, txt_lookup = build_lookup(meta)
    stops = load_stopwords(args.lang, args.stopwords)

    seq_by_key = { (e["episode_index"], e["stimulus_index"], e["time_condition"], float(e["total_time"])): e
                   for e in seq }

    rows = []
    for ep in sim:
        key = get_episode_key(ep)
        total_time = key[3]
        stim_idx = int(key[1])

        fix_durs_s = []
        for trl in ep.get("text_reading_logs", []):
            for step in trl.get("sentence_reading_logs", []):
                dur = step.get("word_recognition_summary", {}).get("individual_step_gaze_duration_in_s")
                if isinstance(dur, (int, float)):
                    fix_durs_s.append(float(dur))
        total_fix_time = sum(fix_durs_s)
        fix_mean_ms = (mean(fix_durs_s) * 1000.0) if fix_durs_s else 0.0
        pct_time_in_fix = (total_fix_time / total_time) if total_time > 0 else 0.0

        seq_ep = seq_by_key.get(key, {}).get("global_fixation_sequence", [])
        seq_ep = normalize_seq_zero_based(seq_ep)

        sacc_count, amp_mean_px, ltr_dx_sum_px = compute_eye_metrics(seq_ep, xy_lookup.get(stim_idx, {}))
        sacc_rate_hz = (sacc_count / total_time) if total_time > 0 else 0.0
        gaze_vel_px_s = (ltr_dx_sum_px / total_time) if total_time > 0 else 0.0

        pct_fix_content = percent_fixated_content_words(seq_ep, txt_lookup.get(stim_idx, {}), stops)

        rows.append({
            "episode_index": int(key[0]),
            "stimulus_index": stim_idx,
            "time_condition": key[2],
            "total_time_s": total_time,
            "fixation_duration_mean_ms": fix_mean_ms,
            "saccade_amplitude_mean_px": amp_mean_px,
            "saccade_rate_hz": sacc_rate_hz,
            "gaze_velocity_px_s": gaze_vel_px_s,
            "percent_fixated_content_words": pct_fix_content,
            "percent_time_in_fixation": pct_time_in_fix
        })

    agg = defaultdict(list)
    for r in rows:
        agg[r["time_condition"]].append(r)

    def stat(vals):
        if not vals:
            return (0.0, 0.0)
        m = mean(vals)
        sd = (mean([(v-m)**2 for v in vals]))**0.5
        return m, sd

    by_cond = {}
    for cond, rr in agg.items():
        by_cond[cond] = {
            "fixation_duration_mean_ms": stat([r["fixation_duration_mean_ms"] for r in rr])[0],
            "fixation_duration_std_ms":  stat([r["fixation_duration_mean_ms"] for r in rr])[1],
            "saccade_amplitude_mean_px": stat([r["saccade_amplitude_mean_px"] for r in rr])[0],
            "saccade_amplitude_std_px":  stat([r["saccade_amplitude_mean_px"] for r in rr])[1],
            "saccade_rate_hz_mean":      stat([r["saccade_rate_hz"] for r in rr])[0],
            "saccade_rate_hz_std":       stat([r["saccade_rate_hz"] for r in rr])[1],
            "gaze_velocity_px_s_mean":   stat([r["gaze_velocity_px_s"] for r in rr])[0],
            "gaze_velocity_px_s_std":    stat([r["gaze_velocity_px_s"] for r in rr])[1],
            "percent_fixated_content_words_mean": stat([r["percent_fixated_content_words"] for r in rr])[0],
            "percent_fixated_content_words_std":  stat([r["percent_fixated_content_words"] for r in rr])[1],
            "percent_time_in_fixation_mean": stat([r["percent_time_in_fixation"] for r in rr])[0],
            "percent_time_in_fixation_std":  stat([r["percent_time_in_fixation"] for r in rr])[1],
            "num_episodes": len(rr),
        }

    out = {
        "by_episode": rows,
        "by_condition": by_cond,
        "units": {
            "fixation_duration_mean_ms": "ms",
            "saccade_amplitude_mean_px": "px",
            "saccade_rate_hz_mean": "Hz",
            "gaze_velocity_px_s_mean": "px/s",
            "percent_fixated_content_words_mean": "proportion (0..1)",
            "percent_time_in_fixation_mean": "proportion (0..1)"
        },
        "notes": {
            "content_word_heuristic": "alphabetic tokens, len>=2, not in stopwords (lang configurable)",
            "positions": "word center (bbox center; fallback to position.x/y)",
            "indexing": "uses word_index_in_text if available; normalizes 1-based sequences to 0-based"
        }
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote:", args.out.resolve())

if __name__ == "__main__":

    """
    python build_french_corpus_effects_metrics.py   --root simulation_data_effects_replication/rho_0.290__w_0.700__cov_1.30   --lang en   --out analyzed_by_episode_fixation_metrics.json
    """

    main()