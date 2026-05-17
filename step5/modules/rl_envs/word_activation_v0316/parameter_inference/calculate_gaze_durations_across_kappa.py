import json
import math
import re
from pathlib import Path

import pandas as pd


def parse_kappa_from_folder(folder_name: str) -> float:
    """
    Convert folder names like:
        kappa_1p0 -> 1.0
        kappa_2p4000000000000004 -> 2.4000000000000004
        kappa_3p5 -> 3.5
    """
    match = re.search(r"kappa_(.+)$", folder_name)
    if not match:
        raise ValueError(f"Cannot parse kappa from folder name: {folder_name}")

    value_str = match.group(1).replace("p", ".")
    return float(value_str)


def extract_episode_gaze_duration(episode: dict):
    """
    Extract one gaze_duration per episode.

    Priority:
    1. Use the gaze_duration from the fixation where done == True.
    2. If not found, use the last positive gaze_duration in the fixation list.
    3. If not found, return None.
    """
    fixations = episode.get("fixations", [])

    # Prefer the completed final recognition step
    for fixation in fixations:
        if fixation.get("done") is True:
            gd = fixation.get("gaze_duration", None)
            if gd is not None and gd > 0:
                return float(gd)

    # Fallback: use the last positive gaze_duration
    positive_gds = [
        float(fixation["gaze_duration"])
        for fixation in fixations
        if fixation.get("gaze_duration", 0) is not None
        and fixation.get("gaze_duration", 0) > 0
    ]

    if positive_gds:
        return positive_gds[-1]

    return None


def summarize_kappa_gaze_durations(
    simulation_root: str = "simulation_data",
    output_csv: str = "kappa_gaze_duration_summary.csv",
):
    simulation_root = Path(simulation_root)

    if not simulation_root.exists():
        raise FileNotFoundError(f"Simulation root not found: {simulation_root}")

    rows = []

    kappa_folders = sorted(
        [p for p in simulation_root.iterdir() if p.is_dir() and p.name.startswith("kappa_")],
        key=lambda p: parse_kappa_from_folder(p.name),
    )

    if not kappa_folders:
        raise RuntimeError(f"No kappa_* folders found under {simulation_root}")

    for folder in kappa_folders:
        kappa = parse_kappa_from_folder(folder.name)
        log_path = folder / "logs.json"

        if not log_path.exists():
            print(f"[Warning] Missing logs.json: {log_path}")
            continue

        with open(log_path, "r", encoding="utf-8") as f:
            logs = json.load(f)

        gaze_durations = []
        n_episodes_total = 0
        n_missing_gaze_duration = 0

        for episode in logs:
            n_episodes_total += 1
            gd = extract_episode_gaze_duration(episode)

            if gd is None:
                n_missing_gaze_duration += 1
            else:
                gaze_durations.append(gd)

        if len(gaze_durations) == 0:
            print(f"[Warning] No valid gaze durations found in {log_path}")
            continue

        series = pd.Series(gaze_durations, dtype="float64")

        n = int(series.count())
        mean = float(series.mean())
        std = float(series.std(ddof=1)) if n > 1 else 0.0
        sem = float(std / math.sqrt(n)) if n > 1 else 0.0

        rows.append(
            {
                "kappa": kappa,
                "folder": folder.name,
                "gaze_duration_mean": mean,
                "gaze_duration_std": std,
                "gaze_duration_sem": sem,
                "gaze_duration_min": float(series.min()),
                "gaze_duration_max": float(series.max()),
                "n_valid_episodes": n,
                "n_total_episodes": n_episodes_total,
                "n_missing_gaze_duration": n_missing_gaze_duration,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("kappa").reset_index(drop=True)

    summary_df.to_csv(output_csv, index=False)
    print(f"Saved summary to: {output_csv}")
    print(summary_df)

    return summary_df


if __name__ == "__main__":
    summarize_kappa_gaze_durations(
        simulation_root="simulation_data",
        output_csv="kappa_gaze_duration_summary.csv",
    )