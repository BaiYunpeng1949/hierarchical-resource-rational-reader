import json
import math
import re
from pathlib import Path

import pandas as pd


def parse_params_from_folder(folder_name: str) -> dict:
    """
    Parse folder names like:
        rho_0.300_w_0.800_cov_2.00
        rho_0.280__w_0.670__cov_1.00
        rho_0.300__w_1.000__cov_3.00

    Returns:
        {
            "rho": 0.300,
            "w": 0.800,
            "coverage_factor": 2.00
        }
    """
    pattern = (
        r"rho_(?P<rho>[-+]?\d*\.?\d+)"
        r"_+w_(?P<w>[-+]?\d*\.?\d+)"
        r"_+cov_(?P<cov>[-+]?\d*\.?\d+)"
    )

    match = re.search(pattern, folder_name)

    if match is None:
        raise ValueError(f"Cannot parse parameters from folder name: {folder_name}")

    return {
        "rho": float(match.group("rho")),
        "w": float(match.group("w")),
        "coverage_factor": float(match.group("cov")),
    }


def load_word_position_map(metadata_path: str, center_key: str = "word_center_px") -> dict:
    """
    Build mapping:
        position_map[stimulus_index][word_index] = (x_px, y_px)

    center_key can be:
        "word_center_px" or "text_center_px"

    I recommend "word_center_px" for saccade amplitude, because it maps
    the fixation sequence at the word level to the center of each word box.
    """
    metadata_path = Path(metadata_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    position_map = {}

    for stim in metadata["stimuli"]:
        stim_idx = int(stim["stimulus_index"])
        position_map[stim_idx] = {}

        for word_info in stim["words"]:
            word_idx = int(word_info["word_index"])

            if center_key not in word_info:
                raise KeyError(
                    f"Cannot find {center_key} for stimulus {stim_idx}, word {word_idx}. "
                    f"Available keys: {list(word_info.keys())}"
                )

            x, y = word_info[center_key]
            position_map[stim_idx][word_idx] = (float(x), float(y))

    return position_map


def compute_saccade_amplitudes_for_sequence(
    word_sequence: list,
    stimulus_index: int,
    position_map: dict,
    include_zero_amplitudes: bool = False,
) -> list:
    """
    Compute pixel-distance amplitudes between consecutive fixated words.

    If include_zero_amplitudes=False, repeated consecutive fixation on the
    same word is excluded. This is usually better for word-level saccade
    amplitude.
    """
    amplitudes = []

    if len(word_sequence) < 2:
        return amplitudes

    stim_positions = position_map.get(int(stimulus_index), None)
    if stim_positions is None:
        return amplitudes

    for prev_word_idx, curr_word_idx in zip(word_sequence[:-1], word_sequence[1:]):
        prev_word_idx = int(prev_word_idx)
        curr_word_idx = int(curr_word_idx)

        if not include_zero_amplitudes and prev_word_idx == curr_word_idx:
            continue

        if prev_word_idx not in stim_positions or curr_word_idx not in stim_positions:
            continue

        x1, y1 = stim_positions[prev_word_idx]
        x2, y2 = stim_positions[curr_word_idx]

        amp_px = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        amplitudes.append(amp_px)

    return amplitudes


def summarize_one_processed_file(
    processed_path: Path,
    position_map: dict,
    params: dict,
    include_zero_amplitudes: bool = False,
) -> list:
    """
    Returns episode-level rows.

    Each row is one episode/stimulus/time-condition sequence:
        rho, w, coverage_factor, time_condition, episode_index,
        stimulus_index, episode_mean_saccade_amplitude_px, ...
    """
    with open(processed_path, "r", encoding="utf-8") as f:
        sequences = json.load(f)

    rows = []

    for item in sequences:
        stimulus_index = int(item["stimulus_index"])
        episode_index = int(item.get("episode_index", -1))
        time_condition = str(item["time_condition"])
        total_time = item.get("total_time", None)
        word_sequence = item.get("global_fixation_sequence", [])

        amplitudes = compute_saccade_amplitudes_for_sequence(
            word_sequence=word_sequence,
            stimulus_index=stimulus_index,
            position_map=position_map,
            include_zero_amplitudes=include_zero_amplitudes,
        )

        if len(amplitudes) == 0:
            continue

        amp_series = pd.Series(amplitudes, dtype="float64")

        rows.append(
            {
                **params,
                "time_condition": time_condition,
                "total_time": total_time,
                "episode_index": episode_index,
                "stimulus_index": stimulus_index,
                "episode_mean_saccade_amplitude_px": float(amp_series.mean()),
                "episode_std_saccade_amplitude_px": float(amp_series.std(ddof=1)) if len(amp_series) > 1 else 0.0,
                "n_saccades_episode": int(len(amp_series)),
            }
        )

    return rows


def summarize_saccade_amplitude_grid(
    simulation_root: str = "simulation_data",
    metadata_path: str = "simulation_word_position_metadata.json",
    output_csv: str = "saccade_amplitude_summary.csv",
    episode_level_output_csv: str = "saccade_amplitude_episode_level.csv",
    center_key: str = "word_center_px",
    include_zero_amplitudes: bool = False,
):
    simulation_root = Path(simulation_root)

    if not simulation_root.exists():
        raise FileNotFoundError(f"simulation_root not found: {simulation_root}")

    position_map = load_word_position_map(
        metadata_path=metadata_path,
        center_key=center_key,
    )

    all_episode_rows = []

    def folder_sort_key(p: Path):
        params = parse_params_from_folder(p.name)
        return params["rho"], params["w"], params["coverage_factor"]


    param_folders = sorted(
        [
            p for p in simulation_root.iterdir()
            if p.is_dir() and p.name.startswith("rho_")
        ],
        key=folder_sort_key,
    )

    if len(param_folders) == 0:
        raise RuntimeError(f"No rho_* parameter folders found under {simulation_root}")

    for folder in param_folders:
        processed_path = folder / "processed_fixation_sequences.json"

        if not processed_path.exists():
            print(f"[Warning] Missing processed_fixation_sequences.json: {processed_path}")
            continue

        try:
            params = parse_params_from_folder(folder.name)
        except ValueError as e:
            print(f"[Warning] {e}")
            continue

        episode_rows = summarize_one_processed_file(
            processed_path=processed_path,
            position_map=position_map,
            params=params,
            include_zero_amplitudes=include_zero_amplitudes,
        )

        all_episode_rows.extend(episode_rows)

    if len(all_episode_rows) == 0:
        raise RuntimeError("No valid saccade amplitudes were computed.")

    episode_df = pd.DataFrame(all_episode_rows)

    # Save episode-level results too. This is useful for debugging and later mixed-effect/statistical analysis.
    episode_df = episode_df.sort_values(
        ["rho", "w", "coverage_factor", "time_condition", "episode_index", "stimulus_index"]
    ).reset_index(drop=True)

    episode_df.to_csv(episode_level_output_csv, index=False)

    # Main summary: mean over episode-level means.
    # This avoids overweighting episodes with longer fixation sequences.
    summary_rows = []

    group_cols = ["rho", "w", "coverage_factor", "time_condition", "total_time"]

    for group_values, group in episode_df.groupby(group_cols, dropna=False):
        rho, w, coverage_factor, time_condition, total_time = group_values

        episode_means = group["episode_mean_saccade_amplitude_px"].astype(float)
        n_episodes = int(len(episode_means))
        n_saccades_total = int(group["n_saccades_episode"].sum())

        mean_amp = float(episode_means.mean())
        std_amp = float(episode_means.std(ddof=1)) if n_episodes > 1 else 0.0
        sem_amp = float(std_amp / math.sqrt(n_episodes)) if n_episodes > 1 else 0.0

        summary_rows.append(
            {
                "rho": rho,
                "w": w,
                "coverage_factor": coverage_factor,
                "time_condition": time_condition,
                "total_time": total_time,
                "saccade_amplitude_mean_px": mean_amp,
                "saccade_amplitude_std_px": std_amp,
                "saccade_amplitude_sem_px": sem_amp,
                "n_episodes": n_episodes,
                "n_saccades_total": n_saccades_total,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["rho", "w", "coverage_factor", "total_time", "time_condition"]
    ).reset_index(drop=True)

    summary_df.to_csv(output_csv, index=False)

    print(f"Saved summary CSV to: {output_csv}")
    print(f"Saved episode-level CSV to: {episode_level_output_csv}")
    print(summary_df)

    return summary_df, episode_df


if __name__ == "__main__":
    summarize_saccade_amplitude_grid(
        simulation_root="simulation_data",
        metadata_path="simulation_word_position_metadata.json",
        output_csv="saccade_amplitude_summary.csv",
        episode_level_output_csv="saccade_amplitude_episode_level.csv",
        center_key="word_center_px",
        include_zero_amplitudes=False,
    )