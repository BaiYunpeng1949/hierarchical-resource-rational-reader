#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_mcq_acc_summary.py

Reads a CSV like 'processed_mcq_freerecall_scores_p1_to_p32.csv' and outputs
MCQ accuracy stats in the SAME STRUCTURE as your free-recall JSON, but with
'mcq_' prefixes:

{
  "mcq_mean_by_time": {"30s": 0.0, "60s": 0.0, "90s": 0.0},
  "mcq_std_by_time":  {"30s": 0.0, "60s": 0.0, "90s": 0.0},
  "n_rows": 0,
  "n_scored": 0,
  "missing_mcq": 0
}

Usage:
  python compute_mcq_acc_summary.py --input /path/to/processed_mcq_freerecall_scores_p1_to_p32.csv \
                                    --output /path/to/mcq_acc_summary.json
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def summarize_mcq_by_time(df: pd.DataFrame) -> dict:
    # Expected columns (based on the provided file):
    # - 'time_constraint' as 30/60/90
    # - 'MCQ Accuracy' as float in [0,1]
    if 'time_constraint' not in df.columns:
        raise ValueError("CSV missing 'time_constraint' column.")
    if 'MCQ Accuracy' not in df.columns:
        raise ValueError("CSV missing 'MCQ Accuracy' column.")

    # Prepare basic counts
    n_rows = len(df)
    n_scored = df['MCQ Accuracy'].notna().sum()
    missing_mcq = int(n_rows - n_scored)

    # Compute mean/std grouped by time
    # Normalize time labels to '30s', '60s', '90s'
    def fmt_time(v):
        try:
            iv = int(v)
            return f"{iv}s"
        except Exception:
            # If it's already like '30s', just pass through
            s = str(v).strip()
            return s if s.endswith('s') else s + 's'

    df = df.copy()
    df['time_label'] = df['time_constraint'].map(fmt_time)

    grouped = df.groupby('time_label')['MCQ Accuracy']
    mean_by_time = grouped.mean().to_dict()
    std_by_time = grouped.std(ddof=0).to_dict()  # population std to mirror many summary stats; change to ddof=1 if needed

    # Ensure keys exist for the expected conditions
    for k in ['30s', '60s', '90s']:
        mean_by_time.setdefault(k, float('nan'))
        std_by_time.setdefault(k, float('nan'))

    # Convert any nan to None for JSON cleanliness
    def clean_nan(d):
        out = {}
        for k, v in d.items():
            if pd.isna(v):
                out[k] = None
            else:
                out[k] = float(v)
        return out

    result = {
        "mcq_mean_by_time": clean_nan(mean_by_time),
        "mcq_std_by_time": clean_nan(std_by_time),
        "n_rows": int(n_rows),
        "n_scored": int(n_scored),
        "missing_mcq": int(missing_mcq),
    }
    return result


def main():
    ap = argparse.ArgumentParser(description="Summarize MCQ accuracy by time constraint into JSON")
    ap.add_argument("--input", required=True, help="Path to processed_mcq_freerecall_scores CSV")
    ap.add_argument("--output", required=False, help="Optional path to write JSON summary")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    summary = summarize_mcq_by_time(df)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Wrote MCQ summary to {out_path}")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
