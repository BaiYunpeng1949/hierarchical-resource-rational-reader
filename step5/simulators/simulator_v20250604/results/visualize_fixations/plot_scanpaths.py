# import os
# import json
# import argparse
# from pathlib import Path
# from typing import List, Dict, Any, Optional, Tuple
# from PIL import Image
#
# # Use a non-interactive backend for headless environments
# import matplotlib
# matplotlib.use("Agg")
#
# matplotlib.rcParams["pdf.fonttype"] = 42
# matplotlib.rcParams["ps.fonttype"] = 42
# matplotlib.rcParams["font.family"] = "Arial"
# matplotlib.rcParams["svg.fonttype"] = "none"
#
# import matplotlib.pyplot as plt
#
# import numpy as np  # <<< NEW: for Gaussian noise
#
# IMG_EXTS = [".png", ".jpg", ".jpeg", ".webp"]
#
#
# def load_json(path: Path):
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)
#
#
# def find_image(images_dir: Path, stimulus_index: int) -> Optional[Path]:
#     # Try common extensions
#     for ext in IMG_EXTS:
#         p = images_dir / f"{stimulus_index}{ext}"
#         if p.exists():
#             return p
#     # Try glob any file starting with the index +
#     for p in images_dir.glob(f"{stimulus_index}.*"):
#         if p.suffix.lower() in IMG_EXTS:
#             return p
#     return None
#
#
# def trial_participant(trial: Dict[str, Any], default_participant: Optional[str]) -> str:
#     """
#     Decide whether this trial is from 'human', 'simulation', or 'unknown'.
#
#     Priority:
#       1) Explicit 'participant' field if present and non-empty.
#       2) Caller-provided default_participant.
#       3) Inspect common ID fields; if any string value starts with 'simulation',
#          treat as simulation. If numeric-only IDs and no strong signal, keep 'unknown'
#          (caller can still filter via --default_participant or filename inference).
#     """
#     p = trial.get("participant")
#     if p is not None and str(p).strip():
#         return str(p)
#
#     if default_participant:
#         return default_participant
#
#     # Inspect common id fields to infer type
#     for key in ("participant_index", "participant_id", "participantID",
#                 "participantId", "subject_id", "subject",
#                 "user_id", "user", "id"):
#         v = trial.get(key)
#         if v is None:
#             continue
#         s = str(v).strip().lower()
#         if not s:
#             continue
#         if s.startswith("simulation"):
#             return "simulation"
#         # If it's a clearly non-simulation string ID, we tentatively mark human.
#         # But if it's purely numeric (common in sims), avoid forcing 'human'.
#         if s not in ("human", "simulation", "unknown") and not s.isdigit():
#             return "human"
#
#     return "unknown"
#
#
# def trial_time_constraint(trial: Dict[str, Any]) -> str:
#     tc = trial.get("time_constraint")
#     if tc is None or (isinstance(tc, str) and not tc.strip()):
#         return "NA"
#     return str(tc)
#
#
# def _sanitize_id(text: str) -> str:
#     # keep alnum, dash, underscore
#     return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(text))
#
#
# def trial_participant_label(trial: Dict[str, Any], inferred: Optional[str]) -> str:
#     """
#     Return a participant label for filenames.
#
#     - For simulation trials with IDs like 'simulation-0', use that as-is (sanitized).
#     - For human trials with a concrete ID, use 'human-<id>'.
#     - Otherwise, fall back to the inferred base ('human'/'simulation'/'unknown').
#     """
#     base = trial_participant(trial, inferred)
#
#     def _sanitize_id(text: str) -> str:
#         return "".join(ch if (str(ch).isalnum() or ch in "-_") else "_" for ch in str(text))
#
#     for key in ("participant_index", "participant_id", "participantID",
#                 "participantId", "subject_id", "subject",
#                 "user_id", "user", "id", "participant"):
#         v = trial.get(key)
#         if v is None:
#             continue
#         s = str(v).strip()
#         if not s:
#             continue
#         low = s.lower()
#         if low.startswith("simulation"):
#             return _sanitize_id(s)
#         if low not in ("human", "simulation", "unknown"):
#             label_base = base if base.lower() not in ("unknown", "") else "human"
#             return f"{label_base}-{_sanitize_id(s)}"
#
#     return base
#
#
# def extract_fixations(trial: Dict[str, Any]) -> List[Dict[str, Any]]:
#     """Return a clean list of fixations with x, y and word_index (may be None)."""
#     fixes = []
#     for row in trial.get("fixation_data", []):
#         try:
#             x = float(row.get("fix_x"))
#             y = float(row.get("fix_y"))
#             wi = row.get("word_index", None)
#             wi = int(wi) if wi is not None else None
#             fixes.append({"x": x, "y": y, "word_index": wi})
#         except Exception:
#             continue
#     return fixes
#
#
# def choose_out_dir(base_out: Path, sim_out: Optional[Path],
#                    human_out: Optional[Path], participant: str) -> Path:
#     if participant.lower() == "simulation":
#         return Path(sim_out) if sim_out else (base_out / "simulation")
#     if participant.lower() == "human":
#         return Path(human_out) if human_out else (base_out / "human")
#     return base_out / "unknown"
#
#
# def classify_saccades_by_rules(fixations: List[Dict[str, Any]]) -> Tuple[List[str], List[bool]]:
#     """
#     Classify each saccade (between i -> i+1) as:
#       - 'regression' (green) if next_word < furthest_word_seen_so_far
#       - 'skip' (blue) if forward jump > 1
#       - 'forward' (red) otherwise (including refixations to same word or adjacent forward)
#     Also return a per-fixation boolean list 'is_regressive_fix' for coloring destination dots.
#     Note: saccades with missing/invalid word_index are labeled 'forward' as fallback.
#     """
#     n = len(fixations)
#     labels: List[str] = ["forward"] * max(0, n - 1)
#     is_reg_fix: List[bool] = [False] * n  # destination fixation flags
#
#     furthest = -10**9  # very small
#     for i in range(n):
#         wi = fixations[i].get("word_index")
#         if wi is not None and wi != -1:
#             # classify the saccade into this fixation
#             if i > 0:
#                 prev = fixations[i-1].get("word_index")
#                 label = "forward"
#                 if prev is not None and prev != -1:
#                     # regression if current < furthest so far
#                     if wi < furthest:
#                         label = "regression"
#                         is_reg_fix[i] = True
#                     else:
#                         # skip if forward jump > 1
#                         if (wi - prev) > 1:
#                             label = "skip"
#                 labels[i-1] = label
#             # maintain furthest reached index
#             if wi > furthest:
#                 furthest = wi
#         else:
#             if i > 0:
#                 labels[i-1] = "forward"
#     return labels, is_reg_fix
#
#
# def plot_trial_on_image(trial: Dict[str, Any],
#                         img_path: Path,
#                         out_path: Path,
#                         participant: str,
#                         label: str,
#                         dot_size: float,
#                         line_width: float,
#                         alpha_dots: float,
#                         alpha_lines: float,
#                         y_offset_px: float = 0.0,
#                         noise_sigma_px: float = 0.0,
#                         rng: Optional[np.random.Generator] = None):
#     """
#     Plot a single trial's scanpath on the stimulus image.
#
#     If noise_sigma_px > 0, add Gaussian noise N(0, sigma^2) in pixels to the
#     y-coordinate of each fixation once (before drawing), to visually restore
#     some vertical stochasticity.
#     """
#     img = Image.open(img_path).convert("RGB")
#     W, H = img.size
#
#     fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
#     ax = plt.gca()
#     ax.imshow(img, extent=[0, W, H, 0])  # top-left origin
#     ax.set_xlim(0, W)
#     ax.set_ylim(H, 0)  # y downward
#     ax.set_xticks([])
#     ax.set_yticks([])
#
#     fixes = extract_fixations(trial)
#     if len(fixes) >= 1:
#         xs = [f["x"] for f in fixes]
#         ys_base = [f["y"] for f in fixes]
#
#         # Add vertical noise if requested
#         if noise_sigma_px > 0.0:
#             if rng is None:
#                 rng = np.random.default_rng()
#             noise = rng.normal(loc=0.0, scale=noise_sigma_px, size=len(ys_base))
#             ys = [y + y_offset_px + dy for y, dy in zip(ys_base, noise)]
#         else:
#             ys = [y + y_offset_px for y in ys_base]
#
#         # Classify saccades + destination regression flags
#         labels, is_reg_fix = classify_saccades_by_rules(fixes)
#
#         # Draw saccades segment-by-segment for proper coloring
#         for i in range(len(fixes) - 1):
#             x0, y0 = xs[i], ys[i]
#             x1, y1 = xs[i+1], ys[i+1]
#             lab = labels[i]
#             if lab == "regression":
#                 color = "green"
#             elif lab == "skip":
#                 color = "blue"
#             else:
#                 color = "red"
#             ax.plot([x0, x1], [y0, y1], "-", linewidth=line_width,
#                     color=color, alpha=alpha_lines, zorder=2)
#
#         # Draw fixation dots: green if destination of a regressive saccade, else red
#         # First fixation is never destination; plot it as red by default.
#         ax.scatter([xs[0]], [ys[0]], s=dot_size, color="red",
#                    alpha=alpha_dots, edgecolors="none", zorder=3)
#         if len(fixes) > 1:
#             xs_dest: List[float] = []
#             ys_dest: List[float] = []
#             xs_reg: List[float] = []
#             ys_reg: List[float] = []
#             for i in range(1, len(fixes)):
#                 if is_reg_fix[i]:
#                     xs_reg.append(xs[i])
#                     ys_reg.append(ys[i])
#                 else:
#                     xs_dest.append(xs[i])
#                     ys_dest.append(ys[i])
#             if xs_dest:
#                 ax.scatter(xs_dest, ys_dest, s=dot_size, color="red",
#                            alpha=alpha_dots, edgecolors="none", zorder=3)
#             if xs_reg:
#                 ax.scatter(xs_reg, ys_reg, s=dot_size, color="green",
#                            alpha=alpha_dots, edgecolors="none", zorder=4)
#
#     # No title or in-figure annotations: trial info is only in the filename.
#
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     if out_path.exists():
#         out_path.unlink()  # replace automatically
#     fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
#     plt.close(fig)
#
#
# def main():
#     ap = argparse.ArgumentParser(
#         description="Plot scanpaths on stimulus images for each trial in JSON files."
#     )
#     ap.add_argument("--out_root", "-o", type=Path, default=Path("scanpath_plots"),
#                     help="Base output directory.")
#     ap.add_argument("--sim_out_dir", type=Path, default=None,
#                     help="Override output directory for simulation plots.")
#     ap.add_argument("--human_out_dir", type=Path, default=None,
#                     help="Override output directory for human plots.")
#     ap.add_argument("--human_y_offset_px", type=float, default=0.0,
#                     help="Vertical pixel offset added to human fixation y-coordinates (use negative to shift up).")
#     ap.add_argument("--sim_y_offset_px", type=float, default=0.0,
#                     help="Vertical pixel offset added to simulation fixation y-coordinates.")
#     ap.add_argument("--noise_sigma_px", type=float, default=0.0,
#                     help="Std dev of Gaussian noise (pixels) added to y per fixation (default: 0).")
#     ap.add_argument("--noise_seed", type=int, default=None,
#                     help="Random seed for y-noise (default: None, i.e., random).")
#     ap.add_argument("json_files", nargs="+", type=Path,
#                     help="One or more scanpath JSON files (each is a list of trials).")
#     ap.add_argument("--default_participant", type=str, default=None,
#                     help="Fallback participant label if missing in trials.")
#     ap.add_argument("--human_y_offset_after3_px", type=float, default=0.0,   # <<< NEW
#                     help="Additional vertical pixel offset for HUMAN trials when stimulus_index >= 3.")
#     args = ap.parse_args()
#
#     # Get the image dir
#     img_dir = Path(os.path.join(
#         "assets",
#         "08_15_09_07_10_images_W1920H1080WS16_LS40_MARGIN400",
#         "simulate",
#     ))
#
#     # Shared RNG for reproducible jitter across all plots (but still different per fixation)
#     rng = np.random.default_rng(args.noise_seed)
#
#     for jf in args.json_files:
#         trials = load_json(jf)
#         if not isinstance(trials, list):
#             print(f"[warn] {jf} did not contain a list; skipping.")
#             continue
#
#         # Determine a default participant label based on filename (optional)
#         inferred = args.default_participant
#         name = jf.stem.lower()
#         if inferred is None:
#             if "human" in name:
#                 inferred = "human"
#             elif "sim" in name or "simulation" in name:
#                 inferred = "simulation"
#             else:
#                 inferred = None
#
#         for idx, trial in enumerate(trials):
#             stim_idx = trial.get("stimulus_index")
#             img_path = find_image(img_dir, stim_idx) if stim_idx is not None else None
#             if img_path is None:
#                 print(f"[warn] Image not found for stimulus_index={stim_idx}; "
#                       f"skipping trial {idx} from {jf.name}.")
#                 continue
#
#             participant = trial_participant(trial, inferred)
#             label = trial_participant_label(trial, inferred)
#             tc = trial_time_constraint(trial)
#
#             out_dir = choose_out_dir(args.out_root, args.sim_out_dir,
#                                      args.human_out_dir, participant)
#             out_name = f"stim{stim_idx}_{label}_time{tc}.pdf"
#             out_path = out_dir / out_name
#
#             # Choose y-offset based on participant type
#             if participant.lower() == "human":
#                 y_offset_px = args.human_y_offset_px
#                 # Extra correction for later stimuli where human coordinates are shifted
#                 if stim_idx is not None and stim_idx >= 3:
#                     y_offset_px += args.human_y_offset_after3_px
#             elif participant.lower() == "simulation":
#                 y_offset_px = args.sim_y_offset_px
#             else:
#                 y_offset_px = 0.0
#
#             try:
#                 plot_trial_on_image(
#                     trial, img_path, out_path, participant, label,
#                     dot_size=90,
#                     line_width=2,
#                     alpha_dots=0.3,
#                     alpha_lines=0.3,
#                     y_offset_px=y_offset_px,
#                     noise_sigma_px=args.noise_sigma_px,
#                     rng=rng,
#                 )
#                 print(f"[ok] Wrote {out_path}")
#             except Exception as e:
#                 print(f"[error] Failed plotting stim={stim_idx}, "
#                       f"participant={participant}, time={tc}: {e}")
#
#
# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Generate publication-ready scanpath PDFs with:
  - editable stimulus text from publication metadata;
  - vector saccade lines and fixation markers;
  - the same 1920 x 1080 coordinate system used by the gaze data;
  - separate human/simulation output folders and optional y-offset corrections.

The output contains no rasterized stimulus screenshot. Text remains editable in
Adobe Illustrator, and scanpath lines/markers remain vector objects.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")

# Embed TrueType fonts in PDF rather than converting text to paths.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "Courier New"
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt


# ----------------------------- IO helpers -----------------------------

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_stimulus_metadata(
    metadata: Dict[str, Any],
    stimulus_index: int,
) -> Dict[str, Any]:
    """Find one stimulus record in publication_text_metadata.json."""
    idx = int(stimulus_index)

    for stimulus in metadata.get("stimuli", []):
        if int(stimulus.get("stimulus_index", -1)) == idx:
            return stimulus

    raise KeyError(
        f"Stimulus index {idx} was not found in the publication metadata."
    )


# ---------------------- Participant/file labeling ---------------------

def trial_participant(
    trial: Dict[str, Any],
    default_participant: Optional[str],
) -> str:
    """Infer whether a trial is human, simulation, or unknown."""
    participant = trial.get("participant")
    if participant is not None and str(participant).strip():
        return str(participant)

    if default_participant:
        return default_participant

    for key in (
        "participant_index", "participant_id", "participantID",
        "participantId", "subject_id", "subject", "user_id",
        "user", "id",
    ):
        value = trial.get(key)
        if value is None:
            continue

        text = str(value).strip().lower()
        if not text:
            continue

        if text.startswith("simulation"):
            return "simulation"

        # Clearly non-numeric, non-generic IDs are usually human IDs.
        if text not in ("human", "simulation", "unknown") and not text.isdigit():
            return "human"

    return "unknown"


def _sanitize_id(text: str) -> str:
    """Keep filename-safe alphanumeric, dash, and underscore characters."""
    return "".join(
        ch if (str(ch).isalnum() or ch in "-_") else "_"
        for ch in str(text)
    )


def trial_participant_label(
    trial: Dict[str, Any],
    inferred: Optional[str],
) -> str:
    """Return a participant label suitable for output filenames."""
    base = trial_participant(trial, inferred)

    for key in (
        "participant_index", "participant_id", "participantID",
        "participantId", "subject_id", "subject", "user_id",
        "user", "id", "participant",
    ):
        value = trial.get(key)
        if value is None:
            continue

        text = str(value).strip()
        if not text:
            continue

        lowered = text.lower()
        if lowered.startswith("simulation"):
            return _sanitize_id(text)

        if lowered not in ("human", "simulation", "unknown"):
            label_base = base if base.lower() not in ("unknown", "") else "human"
            return f"{label_base}-{_sanitize_id(text)}"

    return base


def trial_time_constraint(trial: Dict[str, Any]) -> str:
    tc = trial.get("time_constraint")
    if tc is None or (isinstance(tc, str) and not tc.strip()):
        return "NA"
    return str(tc)


def choose_out_dir(
    base_out: Path,
    sim_out: Optional[Path],
    human_out: Optional[Path],
    participant: str,
) -> Path:
    participant_lower = (participant or "").lower()

    if participant_lower == "simulation":
        return Path(sim_out) if sim_out else (base_out / "simulation")
    if participant_lower == "human":
        return Path(human_out) if human_out else (base_out / "human")
    return base_out / "unknown"


# ------------------------ Fixation processing -------------------------

def extract_fixations(trial: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return clean fixation records containing x, y, and word_index."""
    fixations: List[Dict[str, Any]] = []

    for row in trial.get("fixation_data", []):
        try:
            x = float(row.get("fix_x"))
            y = float(row.get("fix_y"))

            word_index = row.get("word_index", None)
            word_index = int(word_index) if word_index is not None else None

            fixations.append({
                "x": x,
                "y": y,
                "word_index": word_index,
            })
        except (TypeError, ValueError):
            continue

    return fixations


def classify_saccades_by_rules(
    fixations: List[Dict[str, Any]],
) -> Tuple[List[str], List[bool]]:
    """
    Classify each saccade between fixation i and i+1 as:
      - regression: destination word is behind the furthest word reached;
      - skip: forward jump is larger than one word;
      - forward: all other movements, including refixations.

    Returns:
      labels: one label per saccade;
      is_regressive_fix: one flag per fixation, marking regression destinations.
    """
    n_fixations = len(fixations)
    labels: List[str] = ["forward"] * max(0, n_fixations - 1)
    is_regressive_fix: List[bool] = [False] * n_fixations

    furthest_word_seen = -10**9

    for i, fixation in enumerate(fixations):
        current_word = fixation.get("word_index")

        if current_word is None or current_word == -1:
            if i > 0:
                labels[i - 1] = "forward"
            continue

        if i > 0:
            previous_word = fixations[i - 1].get("word_index")
            label = "forward"

            if previous_word is not None and previous_word != -1:
                if current_word < furthest_word_seen:
                    label = "regression"
                    is_regressive_fix[i] = True
                elif (current_word - previous_word) > 1:
                    label = "skip"

            labels[i - 1] = label

        if current_word > furthest_word_seen:
            furthest_word_seen = current_word

    return labels, is_regressive_fix


# ---------------------------- PDF drawing -----------------------------

def draw_editable_stimulus_text(
    ax,
    stimulus_metadata: Dict[str, Any],
    font_size: float,
    font_family: str,
    y_offset: float = 0.0,
):
    """
    Draw every stimulus word as an editable PDF text object.

    Word x/y coordinates must use the same coordinate system as the fixation
    data and the metadata width/height values.
    """
    for word in stimulus_metadata.get("words", []):
        text = str(word["text"])
        x = float(word["x"])
        y = float(word["y"]) + y_offset

        ax.text(
            x,
            y,
            text,
            fontsize=font_size,
            fontfamily=font_family,
            fontweight="normal",
            color="black",
            ha="left",
            va="top",
            zorder=1,
            clip_on=False,
        )


def plot_trial_on_editable_text(
    trial: Dict[str, Any],
    stimulus_metadata: Dict[str, Any],
    out_path: Path,
    font_size: float,
    font_family: str,
    text_y_offset_px: float,
    fixation_y_offset_px: float,
    noise_sigma_px: float,
    rng: np.random.Generator,
    dot_size: float,
    line_width: float,
    alpha_dots: float,
    alpha_lines: float,
):
    """Plot one trial as editable text plus vector scanpath objects."""
    width = int(stimulus_metadata["width"])
    height = int(stimulus_metadata["height"])

    # At 72 dpi, coordinate units correspond directly to PDF points.
    # This mirrors the working heatmap script and preserves exact page size.
    fig = plt.figure(
        figsize=(width / 72.0, height / 72.0),
        dpi=72,
        facecolor="white",
    )
    ax = fig.add_axes([0, 0, 1, 1])

    draw_editable_stimulus_text(
        ax=ax,
        stimulus_metadata=stimulus_metadata,
        font_size=font_size,
        font_family=font_family,
        y_offset=text_y_offset_px,
    )

    fixations = extract_fixations(trial)

    if fixations:
        xs = np.asarray([fixation["x"] for fixation in fixations], dtype=float)
        ys = np.asarray([fixation["y"] for fixation in fixations], dtype=float)

        ys = ys + fixation_y_offset_px

        if noise_sigma_px > 0.0:
            ys = ys + rng.normal(
                loc=0.0,
                scale=noise_sigma_px,
                size=len(ys),
            )

        saccade_labels, is_regressive_fix = classify_saccades_by_rules(fixations)

        # Saccades are individual vector line objects.
        for i, saccade_label in enumerate(saccade_labels):
            if saccade_label == "regression":
                color = "green"
            elif saccade_label == "skip":
                color = "blue"
            else:
                color = "red"

            ax.plot(
                [xs[i], xs[i + 1]],
                [ys[i], ys[i + 1]],
                linestyle="-",
                linewidth=line_width,
                color=color,
                alpha=alpha_lines,
                zorder=2,
                solid_capstyle="round",
                rasterized=False,
            )

        # Fixation markers are vector circles in the PDF.
        normal_indices = [
            i for i in range(len(fixations))
            if not is_regressive_fix[i]
        ]
        regression_indices = [
            i for i in range(len(fixations))
            if is_regressive_fix[i]
        ]

        if normal_indices:
            ax.scatter(
                xs[normal_indices],
                ys[normal_indices],
                s=dot_size,
                color="red",
                alpha=alpha_dots,
                edgecolors="none",
                zorder=3,
                rasterized=False,
            )

        if regression_indices:
            ax.scatter(
                xs[regression_indices],
                ys[regression_indices],
                s=dot_size,
                color="green",
                alpha=alpha_dots,
                edgecolors="none",
                zorder=4,
                rasterized=False,
            )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    fig.savefig(
        out_path,
        format="pdf",
        dpi=600,
        bbox_inches=None,
        pad_inches=0,
        facecolor="white",
    )
    plt.close(fig)


# ------------------------------- Main --------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot scanpaths as vector graphics over editable stimulus text."
        )
    )

    parser.add_argument(
        "json_files",
        nargs="+",
        type=Path,
        help="One or more scanpath JSON files, each containing a list of trials.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="JSON file containing stimulus words and their x/y coordinates.",
    )
    parser.add_argument(
        "--out_root", "-o",
        type=Path,
        default=Path("scanpath_plots"),
        help="Base output directory.",
    )
    parser.add_argument(
        "--sim_out_dir",
        type=Path,
        default=None,
        help="Optional separate output directory for simulation trials.",
    )
    parser.add_argument(
        "--human_out_dir",
        type=Path,
        default=None,
        help="Optional separate output directory for human trials.",
    )
    parser.add_argument(
        "--only",
        choices=["human", "simulation"],
        default=None,
        help="Only plot the selected participant type.",
    )
    parser.add_argument(
        "--exclude",
        choices=["human", "simulation"],
        default=None,
        help="Skip the selected participant type.",
    )
    parser.add_argument(
        "--default_participant",
        type=str,
        default=None,
        help="Fallback participant type if it is absent from trial records.",
    )

    # Coordinate adjustments retained from the original script.
    parser.add_argument(
        "--human_y_offset_px",
        type=float,
        default=0.0,
        help="Vertical offset applied to all human fixation y-coordinates.",
    )
    parser.add_argument(
        "--human_y_offset_after3_px",
        type=float,
        default=0.0,
        help=(
            "Additional human fixation y-offset when stimulus_index >= 3."
        ),
    )
    parser.add_argument(
        "--sim_y_offset_px",
        type=float,
        default=0.0,
        help="Vertical offset applied to simulation fixation y-coordinates.",
    )
    parser.add_argument(
        "--text_y_offset_px",
        type=float,
        default=0.0,
        help="Vertical offset applied to editable stimulus text.",
    )
    parser.add_argument(
        "--noise_sigma_px",
        type=float,
        default=0.0,
        help="SD of Gaussian noise added to fixation y-coordinates.",
    )
    parser.add_argument(
        "--noise_seed",
        type=int,
        default=None,
        help="Random seed for reproducible fixation y-noise.",
    )

    # Appearance controls; defaults match the original scanpath script.
    parser.add_argument(
        "--font_family",
        type=str,
        default="Courier New",
        help="Font used for editable stimulus text.",
    )
    parser.add_argument(
        "--font_size",
        type=float,
        default=None,
        help="Override metadata font size; otherwise metadata font_size is used.",
    )
    parser.add_argument(
        "--dot_size",
        type=float,
        default=90.0,
        help="Fixation marker area in points squared.",
    )
    parser.add_argument(
        "--line_width",
        type=float,
        default=2.0,
        help="Saccade line width in points.",
    )
    parser.add_argument(
        "--alpha_dots",
        type=float,
        default=0.3,
        help="Fixation marker opacity from 0 to 1.",
    )
    parser.add_argument(
        "--alpha_lines",
        type=float,
        default=0.3,
        help="Saccade line opacity from 0 to 1.",
    )

    args = parser.parse_args()

    metadata = load_json(args.metadata)
    metadata_font_size = float(metadata.get("font_size", 16))
    font_size = (
        float(args.font_size)
        if args.font_size is not None
        else metadata_font_size
    )

    rng = np.random.default_rng(args.noise_seed)

    for json_file in args.json_files:
        trials = load_json(json_file)

        if not isinstance(trials, list):
            print(f"[warn] {json_file} did not contain a list; skipping.")
            continue

        inferred = args.default_participant
        file_stem = json_file.stem.lower()

        if inferred is None:
            if "human" in file_stem:
                inferred = "human"
            elif "sim" in file_stem or "simulation" in file_stem:
                inferred = "simulation"

        for trial_index, trial in enumerate(trials):
            try:
                stimulus_index = trial.get("stimulus_index")
                if stimulus_index is None:
                    raise ValueError("Trial has no stimulus_index.")

                participant = trial_participant(trial, inferred)

                if args.only and participant.lower() != args.only:
                    continue
                if args.exclude and participant.lower() == args.exclude:
                    continue

                participant_label = trial_participant_label(trial, inferred)
                time_constraint = trial_time_constraint(trial)

                stimulus_metadata = get_stimulus_metadata(
                    metadata=metadata,
                    stimulus_index=int(stimulus_index),
                )

                if participant.lower() == "human":
                    fixation_y_offset = args.human_y_offset_px
                    if int(stimulus_index) >= 3:
                        fixation_y_offset += args.human_y_offset_after3_px
                elif participant.lower() == "simulation":
                    fixation_y_offset = args.sim_y_offset_px
                else:
                    fixation_y_offset = 0.0

                out_dir = choose_out_dir(
                    base_out=args.out_root,
                    sim_out=args.sim_out_dir,
                    human_out=args.human_out_dir,
                    participant=participant,
                )
                out_name = (
                    f"stim{stimulus_index}_{participant_label}_"
                    f"time{time_constraint}.pdf"
                )
                out_path = out_dir / out_name

                plot_trial_on_editable_text(
                    trial=trial,
                    stimulus_metadata=stimulus_metadata,
                    out_path=out_path,
                    font_size=font_size,
                    font_family=args.font_family,
                    text_y_offset_px=args.text_y_offset_px,
                    fixation_y_offset_px=fixation_y_offset,
                    noise_sigma_px=args.noise_sigma_px,
                    rng=rng,
                    dot_size=args.dot_size,
                    line_width=args.line_width,
                    alpha_dots=args.alpha_dots,
                    alpha_lines=args.alpha_lines,
                )

                print(f"[ok] Wrote {out_path}")

            except Exception as exc:
                print(
                    f"[error] Failed trial idx={trial_index} "
                    f"stim={trial.get('stimulus_index')}: {exc}"
                )


if __name__ == "__main__":
    main()