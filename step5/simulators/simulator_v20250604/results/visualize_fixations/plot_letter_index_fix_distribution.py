#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import math

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_sampled_letters(sim_json: List[Dict[str, Any]]):
    """
    Yields tuples (word_len, letter_idx) for every sampled letter index in the file.
    Walks: episode -> text_reading_logs[*] -> sentence_reading_logs[*] -> word_recognition_summary.sampled_letters_indexes_dict
    """
    for episode in sim_json:
        tr_logs = episode.get("text_reading_logs", [])
        for tlog in tr_logs:
            s_logs = tlog.get("sentence_reading_logs", [])
            for slog in s_logs:
                wrs = slog.get("word_recognition_summary", {}) or {}
                sampled = wrs.get("sampled_letters_indexes_dict", {}) or {}
                # multiple entries if multiple words were read on this step
                for k, item in sampled.items():
                    if not isinstance(item, dict):
                        continue
                    wlen = item.get("word_len")
                    letters = item.get("letters_indexes", []) or []
                    if wlen is None:
                        continue
                    try:
                        wlen = int(wlen)
                    except Exception:
                        continue
                    for idx in letters:
                        try:
                            li = int(idx)
                        except Exception:
                            continue
                        # keep only indices that are plausible for the word length
                        if 0 <= li < wlen and wlen > 0:
                            yield (wlen, li)

def collect_absolute_for_length(sim_json: List[Dict[str, Any]], word_length: int) -> np.ndarray:
    """
    Return counts array of size = word_length for absolute letter index distribution,
    considering only words whose length == word_length.
    """
    counts = np.zeros(word_length, dtype=np.int64)
    for wlen, li in iter_sampled_letters(sim_json):
        if wlen == word_length:
            counts[li] += 1
    return counts

def collect_normalized(sim_json: List[Dict[str, Any]], divisor: str = "Lminus1") -> np.ndarray:
    """
    Return an array of normalized positions in [0,1] for ALL word lengths.
    divisor: "L" => li / L,  "Lminus1" => li / max(L-1,1) so first=0, last=1.
    """
    values: List[float] = []
    for wlen, li in iter_sampled_letters(sim_json):
        if wlen <= 0:
            continue
        if divisor == "L":
            v = li / float(wlen)
        else:
            denom = float(max(wlen - 1, 1))
            v = li / denom
        # clamp numerically
        if v < 0: v = 0.0
        if v > 1: v = 1.0
        values.append(v)
    return np.array(values, dtype=np.float32)

def plot_absolute(counts: np.ndarray, word_length: int, out_path: Path, title: Optional[str] = None):
    x = np.arange(word_length)
    fig = plt.figure(figsize=(10, 4), dpi=120)
    ax = plt.gca()
    ax.bar(x, counts)  # default color; no explicit styling
    ax.set_xlabel("Letter index (0-based)")
    ax.set_ylabel("Count")
    if title is None:
        title = f"Absolute letter-index distribution (word length = {word_length})"
    ax.set_title(title)
    ax.set_xlim(-0.5, word_length - 0.5)
    ax.grid(True, axis="y", alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_normalized(values: np.ndarray, out_path: Path, bins: int = 20, title: Optional[str] = None):
    fig = plt.figure(figsize=(10, 4), dpi=120)
    ax = plt.gca()
    ax.hist(values, bins=bins, range=(0.0, 1.0))
    ax.set_xlabel("Normalized letter position (0 → start, 1 → end)")
    ax.set_ylabel("Count")
    if title is None:
        title = f"Normalized letter-index distribution"
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Plot distributions of sampled letter indexes from simulation logs.")
    ap.add_argument("simulation_folder", type=Path, help="Path to all_simulation_results.json's folder")
    ap.add_argument("--out_dir", type=Path, default=Path("letter_index_plots"), help="Output directory")
    ap.add_argument("--word_length", type=int, default=12, help="Word length to filter for absolute index distribution")
    ap.add_argument("--norm_bins", type=int, default=20, help="Number of bins for normalized histogram")
    ap.add_argument("--norm_divisor", choices=["Lminus1","L"], default="Lminus1",
                    help="Normalized position: li/(L-1) or li/L")
    args = ap.parse_args()

    sim_results_json_path = f"../../simulated_results/{args.simulation_folder}/all_simulation_results.json"

    data = load_json(sim_results_json_path)
    if not isinstance(data, list):
        raise ValueError("Expected the simulation JSON to be a list of episodes.")

    # Absolute distribution for a given word length
    counts = collect_absolute_for_length(data, args.word_length)
    abs_path = args.out_dir / f"abs_letter_index_len{args.word_length}.png"
    plot_absolute(counts, args.word_length, abs_path)

    # Normalized distribution across all word lengths
    values = collect_normalized(data, divisor=args.norm_divisor)
    norm_path = args.out_dir / f"norm_letter_index_{args.norm_divisor}_bins{args.norm_bins}.png"
    plot_normalized(values, norm_path, bins=args.norm_bins)

    print(f"[ok] Wrote {abs_path}")
    print(f"[ok] Wrote {norm_path}")

if __name__ == "__main__":
    main()
