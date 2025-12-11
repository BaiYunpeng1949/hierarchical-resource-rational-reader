#!/usr/bin/env python3
"""
Animate scanpaths (human-only and simulation-only) on stimulus images.

For each invocation, this script produces TWO videos:
  1) A video with only the human scanpath.
  2) A video with only the simulation scanpath.

Usage example:

python make_scanpath_video_separate.py \
    --human_json data/human_scanpaths.json \
    --sim_json data/sim_scanpaths.json \
    --stimulus_index 7 \
    --human_participant_index 23 \
    --sim_participant_index 0 \
    --time_condition 90 \
    --out_human videos/stim7_p23_90s_human.mp4 \
    --out_sim videos/stim7_p23_90s_sim.mp4 \
    --fps 5 \
    --human_y_offset_px -60 \
    --sim_y_offset_px 60
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from PIL import Image

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib import animation

# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

IMG_EXTS = [".png", ".jpg", ".jpeg", ".webp"]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_image(images_dir: Path, stimulus_index: int) -> Optional[Path]:
    # Try common extensions
    for ext in IMG_EXTS:
        p = images_dir / f"{stimulus_index}{ext}"
        if p.exists():
            return p
    # Try glob any file starting with the index +
    for p in images_dir.glob(f"{stimulus_index}.*"):
        if p.suffix.lower() in IMG_EXTS:
            return p
    return None


def normalize_time_condition(value) -> str:
    """
    Normalize a time condition / time_constraint value so that:
      - 30, "30", "30s"  -> "30s"
      - 90, "90", "90s"  -> "90s"
      - "NA", "na", None, "" -> "NA"
    This allows human JSON using 90 and sim JSON using "90s" to match.
    """
    if value is None:
        return "NA"

    if isinstance(value, (int, float)):
        if value <= 0:
            return "NA"
        return f"{int(value)}s"

    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("", "na", "none"):
            return "NA"
        # strip trailing 's' if any
        if s.endswith("s"):
            s = s[:-1]
        # if digits, treat as seconds
        if s.isdigit():
            return f"{int(s)}s"
        # fall back to uppercased raw
        return value.upper()

    # anything else – fall back to string
    return str(value)


def trial_time_constraint(trial: Dict[str, Any]) -> str:
    tc = trial.get("time_constraint")
    return normalize_time_condition(tc)


def extract_fixations(trial: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a clean list of fixations with x, y and word_index (may be None)."""
    fixes = []
    for row in trial.get("fixation_data", []):
        try:
            x = float(row.get("fix_x"))
            y = float(row.get("fix_y"))
            wi = row.get("word_index", None)
            wi = int(wi) if wi is not None else None
            fixes.append({"x": x, "y": y, "word_index": wi})
        except Exception:
            continue
    return fixes


def classify_saccades_by_rules(fixations: List[Dict[str, Any]]) -> Tuple[List[str], List[bool]]:
    """
    Saccades (i -> i+1):
      - 'regression' (green) if next_word < furthest_word_seen_so_far
      - 'skip' (blue) if forward jump > 1
      - 'forward' (red) otherwise (including refixations or adjacent forward)
    Also return per-fixation boolean 'is_regressive_fix' for destination dots.
    """
    n = len(fixations)
    labels: List[str] = ["forward"] * max(0, n - 1)
    is_reg_fix: List[bool] = [False] * n  # destination fixation flags

    furthest = -10**9
    for i in range(n):
        wi = fixations[i].get("word_index")
        if wi is not None and wi != -1:
            # classify the saccade into this fixation
            if i > 0:
                prev = fixations[i-1].get("word_index")
                label = "forward"
                if prev is not None and prev != -1:
                    # regression if current < furthest so far
                    if wi < furthest:
                        label = "regression"
                        is_reg_fix[i] = True
                    else:
                        # skip if forward jump > 1
                        if (wi - prev) > 1:
                            label = "skip"
                labels[i-1] = label
            # maintain furthest reached index
            if wi > furthest:
                furthest = wi
        else:
            if i > 0:
                labels[i-1] = "forward"
    return labels, is_reg_fix


def select_trial(
    trials: List[Dict[str, Any]],
    stimulus_index: int,
    participant_index: str,
    time_condition: str,
) -> Dict[str, Any]:
    """
    Select a single trial by:
      - stimulus_index
      - participant_index (matching common participant fields)
      - time_condition (normalized)
    """
    participant_index = str(participant_index)
    time_condition = normalize_time_condition(time_condition)

    candidate = None
    for tr in trials:
        if tr.get("stimulus_index") != stimulus_index:
            continue
        if trial_time_constraint(tr) != time_condition:
            continue

        # Match participant index in any of several common fields
        for key in (
            "participant_index",
            "participant_id",
            "participantID",
            "participantId",
            "subject_id",
            "subject",
            "user_id",
            "user",
            "id",
            "participant",
        ):
            v = tr.get(key)
            if v is None:
                continue
            if str(v) == participant_index:
                return tr

        # If no direct match, keep as last resort (if exactly one such trial)
        if candidate is None:
            candidate = tr

    if candidate is None:
        raise RuntimeError(
            f"No trial found for stimulus_index={stimulus_index}, "
            f"participant_index={participant_index}, time_condition={time_condition}"
        )
    return candidate


# ---------------------------------------------------------------------
# Animation helpers
# ---------------------------------------------------------------------

def prepare_fixation_series(trial: Dict[str, Any], y_offset_px: float):
    """Return (xs, ys, labels, is_reg_fix) for a trial, with y-offset applied."""
    fixations = extract_fixations(trial)
    if not fixations:
        return [], [], [], []

    labels, is_reg_fix = classify_saccades_by_rules(fixations)
    xs = [f["x"] for f in fixations]
    ys = [f["y"] + y_offset_px for f in fixations]
    return xs, ys, labels, is_reg_fix


def draw_scanpath_partial(
    ax,
    xs: List[float],
    ys: List[float],
    labels: List[str],
    is_reg_fix: List[bool],
    upto_fix_idx: int,
    alpha_lines: float = 0.3,
    alpha_dots: float = 0.3,
):
    """
    Draw scanpath up to fixation index `upto_fix_idx` (inclusive).
    Indices: 0..N-1.
    """
    n = len(xs)
    if n == 0:
        return
    upto_fix_idx = max(0, min(upto_fix_idx, n - 1))

    # Draw saccade segments
    for i in range(upto_fix_idx):
        if i >= len(labels):
            break
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]
        lab = labels[i]
        if lab == "regression":
            color = "green"
        elif lab == "skip":
            color = "blue"
        else:
            color = "red"
        ax.plot([x0, x1], [y0, y1], "-", linewidth=2, color=color,
                alpha=alpha_lines, zorder=2)

    # Draw dots
    # First fixation (if inside range)
    if upto_fix_idx >= 0:
        ax.scatter([xs[0]], [ys[0]], s=90, color="red",
                   alpha=alpha_dots, edgecolors="none", zorder=3)

    if upto_fix_idx >= 1:
        xs_dest = []
        ys_dest = []
        xs_reg = []
        ys_reg = []
        for i in range(1, upto_fix_idx + 1):
            if i >= len(is_reg_fix):
                break
            if is_reg_fix[i]:
                xs_reg.append(xs[i])
                ys_reg.append(ys[i])
            else:
                xs_dest.append(xs[i])
                ys_dest.append(ys[i])
        if xs_dest:
            ax.scatter(xs_dest, ys_dest, s=90, color="red",
                       alpha=alpha_dots, edgecolors="none", zorder=3)
        if xs_reg:
            ax.scatter(xs_reg, ys_reg, s=90, color="green",
                       alpha=alpha_dots, edgecolors="none", zorder=4)


def make_video_single(
    img_path: Path,
    series,
    out_path: Path,
    fps: int,
):
    """
    Render a video (.mp4) animating a single scanpath (human OR simulation).
    series: (xs, ys, labels, is_reg_fix)
    """
    xs, ys, labels, is_reg_fix = series
    n_fix = len(xs)

    if n_fix == 0:
        raise RuntimeError(f"No fixation data for {out_path}.")

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = plt.gca()

    def _setup_axis():
        ax.imshow(img, extent=[0, W, H, 0])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # image top at top
        ax.set_xticks([])
        ax.set_yticks([])

    _setup_axis()

    n_frames = n_fix  # one frame per fixation

    def init():
        ax.clear()
        _setup_axis()
        return []

    def update(frame_idx):
        ax.clear()
        _setup_axis()
        draw_scanpath_partial(
            ax,
            xs,
            ys,
            labels,
            is_reg_fix,
            upto_fix_idx=min(frame_idx, n_fix - 1),
            alpha_lines=0.3,
            alpha_dots=0.3,
        )
        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=1000.0 / fps,
        blit=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Writer = animation.writers["ffmpeg"]  # requires ffmpeg installed
    writer = Writer(fps=fps, metadata=dict(artist=""), bitrate=-1)

    print(f"[info] Saving video to {out_path} (frames={n_frames}, fps={fps})")
    anim.save(str(out_path), writer=writer)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Create TWO scanpath videos (human-only & sim-only) "
            "for a given stimulus, participants, and time condition."
        )
    )
    ap.add_argument("--human_json", type=Path, required=True,
                    help="Path to human scanpath JSON (list of trials).")
    ap.add_argument("--sim_json", type=Path, required=True,
                    help="Path to simulation scanpath JSON (list of trials).")
    ap.add_argument("--stimulus_index", type=int, required=True,
                    help="Stimulus index to select.")
    ap.add_argument("--human_participant_index", type=str, required=True,
                    help="Participant index/id for human trial (e.g., '23').")
    ap.add_argument("--sim_participant_index", type=str, required=True,
                    help="Participant index/id for simulation trial (e.g., '0').")
    ap.add_argument("--time_condition", type=str, required=True,
                    help="Time condition / constraint (e.g., 30, 30s, 90, 90s, NA).")
    ap.add_argument("--out_human", type=Path, required=True,
                    help="Output video path for HUMAN scanpath (e.g., human.mp4).")
    ap.add_argument("--out_sim", type=Path, required=True,
                    help="Output video path for SIMULATION scanpath (e.g., sim.mp4).")
    ap.add_argument("--fps", type=int, default=5,
                    help="Frames per second for both videos (default: 5).")
    ap.add_argument("--human_y_offset_px", type=float, default=-0.0,
                    help="Vertical pixel offset for human fixations (default: -60).")
    ap.add_argument("--sim_y_offset_px", type=float, default=0.0,
                    help="Vertical pixel offset for simulation fixations (default: 60).")
    ap.add_argument("--images_dir", type=Path, default=None,
                    help="Directory with stimulus images; "
                         "default matches plot_scanpaths.py assets path.")

    args = ap.parse_args()

    # Image directory: match your existing script by default
    if args.images_dir is None:
        img_dir = Path(
            os.path.join(
                "assets",
                "08_15_09_07_10_images_W1920H1080WS16_LS40_MARGIN400",
                "simulate",
            )
        )
    else:
        img_dir = args.images_dir

    # Load JSONs
    human_trials = load_json(args.human_json)
    sim_trials = load_json(args.sim_json)

    if not isinstance(human_trials, list):
        raise RuntimeError(f"{args.human_json} is not a list of trials.")
    if not isinstance(sim_trials, list):
        raise RuntimeError(f"{args.sim_json} is not a list of trials.")

    # Select trials
    human_trial = select_trial(
        human_trials,
        stimulus_index=args.stimulus_index,
        participant_index=args.human_participant_index,
        time_condition=args.time_condition,
    )
    sim_trial = select_trial(
        sim_trials,
        stimulus_index=args.stimulus_index,
        participant_index=args.sim_participant_index,
        time_condition=args.time_condition,
    )

    # Stimulus image
    img_path = find_image(img_dir, args.stimulus_index)
    if img_path is None:
        raise RuntimeError(
            f"Stimulus image not found for index={args.stimulus_index} in {img_dir}"
        )

    # Prepare fixation series
    human_series = prepare_fixation_series(human_trial, args.human_y_offset_px)
    sim_series = prepare_fixation_series(sim_trial, args.sim_y_offset_px)

    # Make videos
    make_video_single(
        img_path=img_path,
        series=human_series,
        out_path=args.out_human,
        fps=args.fps,
    )
    make_video_single(
        img_path=img_path,
        series=sim_series,
        out_path=args.out_sim,
        fps=args.fps,
    )

    print("[ok] Human and simulation videos written.")


if __name__ == "__main__":
    main()
