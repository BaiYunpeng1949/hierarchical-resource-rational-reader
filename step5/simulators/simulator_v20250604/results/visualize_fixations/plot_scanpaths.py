# #!/usr/bin/env python3
# import os
# import json
# import argparse
# from pathlib import Path
# from typing import List, Dict, Any, Optional, Tuple
# from PIL import Image

# # Use a non-interactive backend for headless environments
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# IMG_EXTS = [".png", ".jpg", ".jpeg", ".webp"]

# def load_json(path: Path):
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

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

# def trial_participant(trial: Dict[str, Any], default_participant: Optional[str]) -> str:
#     """
#     Decide whether this trial is from 'human', 'simulation', or 'unknown'.

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

#     if default_participant:
#         return default_participant

#     # Inspect common id fields to infer type
#     for key in ("participant_index","participant_id","participantID","participantId","subject_id","subject","user_id","user","id"):
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
#         if s not in ("human","simulation","unknown") and not s.isdigit():
#             return "human"

#     return "unknown"

# def trial_time_constraint(trial: Dict[str, Any]) -> str:
#     tc = trial.get("time_constraint")
#     if tc is None or (isinstance(tc, str) and not tc.strip()):
#         return "NA"
#     return str(tc)

# def _sanitize_id(text: str) -> str:
#     # keep alnum, dash, underscore
#     return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(text))

# def trial_participant_label(trial: Dict[str, Any], inferred: Optional[str]) -> str:
#     """
#     Return a participant label for filenames.

#     - For simulation trials with IDs like 'simulation-0', use that as-is (sanitized).
#     - For human trials with a concrete ID, use 'human-<id>'.
#     - Otherwise, fall back to the inferred base ('human'/'simulation'/'unknown').
#     """
#     base = trial_participant(trial, inferred)

#     def _sanitize_id(text: str) -> str:
#         return "".join(ch if (str(ch).isalnum() or ch in "-_") else "_" for ch in str(text))

#     for key in ("participant_index","participant_id","participantID","participantId","subject_id","subject","user_id","user","id","participant"):
#         v = trial.get(key)
#         if v is None:
#             continue
#         s = str(v).strip()
#         if not s:
#             continue
#         low = s.lower()
#         if low.startswith("simulation"):
#             return _sanitize_id(s)
#         if low not in ("human","simulation","unknown"):
#             label_base = base if base.lower() not in ("unknown","") else "human"
#             return f"{label_base}-{_sanitize_id(s)}"

#     return base

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

# def choose_out_dir(base_out: Path, sim_out: Optional[Path], human_out: Optional[Path], participant: str) -> Path:
#     if participant.lower() == "simulation":
#         return Path(sim_out) if sim_out else (base_out / "simulation")
#     if participant.lower() == "human":
#         return Path(human_out) if human_out else (base_out / "human")
#     return base_out / "unknown"

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

# def plot_trial_on_image(trial: Dict[str, Any],
#                         img_path: Path,
#                         out_path: Path,
#                         participant: str,
#                         label: str,
#                         dot_size: float,
#                         line_width: float,
#                         alpha_dots: float,
#                         alpha_lines: float,
#                         y_offset_px: float = 0.0):
#     img = Image.open(img_path).convert("RGB")
#     W, H = img.size

#     fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
#     ax = plt.gca()
#     ax.imshow(img, extent=[0, W, H, 0])  # top-left origin
#     ax.set_xlim(0, W)
#     ax.set_ylim(H, 0)  # y downward
#     ax.set_xticks([])
#     ax.set_yticks([])

#     fixes = extract_fixations(trial)
#     if len(fixes) >= 1:
#         xs = [f["x"] for f in fixes]
#         ys = [f["y"] + y_offset_px for f in fixes]  # apply vertical offset

#         # Classify saccades + destination regression flags
#         labels, is_reg_fix = classify_saccades_by_rules(fixes)

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
#             ax.plot([x0, x1], [y0, y1], "-", linewidth=line_width, color=color, alpha=alpha_lines, zorder=2)

#         # Draw fixation dots: green if destination of a regressive saccade, else red
#         # First fixation is never destination; plot it as red by default.
#         ax.scatter([xs[0]], [ys[0]], s=dot_size, color="red", alpha=alpha_dots, edgecolors="none", zorder=3)
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
#                 ax.scatter(xs_dest, ys_dest, s=dot_size, color="red", alpha=alpha_dots, edgecolors="none", zorder=3)
#             if xs_reg:
#                 ax.scatter(xs_reg, ys_reg, s=dot_size, color="green", alpha=alpha_dots, edgecolors="none", zorder=4)

#     # No title or in-figure annotations: trial info is only in the filename.

#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     if out_path.exists():
#         out_path.unlink()  # replace automatically
#     fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
#     plt.close(fig)

# def main():
#     ap = argparse.ArgumentParser(description="Plot scanpaths on stimulus images for each trial in JSON files.")
#     ap.add_argument("--out_root", "-o", type=Path, default=Path("scanpath_plots"), help="Base output directory.")
#     ap.add_argument("--sim_out_dir", type=Path, default=None, help="Override output directory for simulation plots.")
#     ap.add_argument("--human_out_dir", type=Path, default=None, help="Override output directory for human plots.")
#     ap.add_argument("--human_y_offset_px", type=float, default=0.0,
#                     help="Vertical pixel offset added to human fixation y-coordinates (use negative to shift up).")
#     ap.add_argument("--sim_y_offset_px", type=float, default=0.0,
#                     help="Vertical pixel offset added to simulation fixation y-coordinates.")
#     ap.add_argument("json_files", nargs="+", type=Path, help="One or more scanpath JSON files (each is a list of trials).")
#     ap.add_argument("--default_participant", type=str, default=None, help="Fallback participant label if missing in trials.")
#     args = ap.parse_args()
    
#     # Get the image dir
#     img_dir = Path(os.path.join("assets", "08_15_09_07_10_images_W1920H1080WS16_LS40_MARGIN400", "simulate"))

#     for jf in args.json_files:
#         trials = load_json(jf)
#         if not isinstance(trials, list):
#             print(f"[warn] {jf} did not contain a list; skipping.")
#             continue

#         # Determine a default participant label based on filename (optional)
#         inferred = args.default_participant
#         name = jf.stem.lower()
#         if inferred is None:
#             inferred = "human" if "human" in name else "simulation" if "sim" in name or "simulation" in name else None

#         for idx, trial in enumerate(trials):
#             stim_idx = trial.get("stimulus_index")
#             img_path = find_image(img_dir, stim_idx) if stim_idx is not None else None
#             if img_path is None:
#                 print(f"[warn] Image not found for stimulus_index={stim_idx}; skipping trial {idx} from {jf.name}.")
#                 continue

#             participant = trial_participant(trial, inferred)
#             label = trial_participant_label(trial, inferred)
#             tc = trial_time_constraint(trial)

#             out_dir = choose_out_dir(args.out_root, args.sim_out_dir, args.human_out_dir, participant)
#             out_name = f"stim{stim_idx}_{label}_time{tc}.pdf"
#             out_path = out_dir / out_name

#             # Choose y-offset based on participant type
#             if participant.lower() == "human":
#                 y_offset_px = args.human_y_offset_px
#             elif participant.lower() == "simulation":
#                 y_offset_px = args.sim_y_offset_px
#             else:
#                 y_offset_px = 0.0

#             try:
#                 plot_trial_on_image(
#                     trial, img_path, out_path, participant, label,
#                     dot_size=90,
#                     line_width=2,
#                     alpha_dots=0.3,
#                     alpha_lines=0.3,
#                     y_offset_px=y_offset_px,
#                 )
#                 print(f"[ok] Wrote {out_path}")
#             except Exception as e:
#                 print(f"[error] Failed plotting stim={stim_idx}, participant={participant}, time={tc}: {e}")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

# Use a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np  # <<< NEW: for Gaussian noise

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


def trial_participant(trial: Dict[str, Any], default_participant: Optional[str]) -> str:
    """
    Decide whether this trial is from 'human', 'simulation', or 'unknown'.

    Priority:
      1) Explicit 'participant' field if present and non-empty.
      2) Caller-provided default_participant.
      3) Inspect common ID fields; if any string value starts with 'simulation',
         treat as simulation. If numeric-only IDs and no strong signal, keep 'unknown'
         (caller can still filter via --default_participant or filename inference).
    """
    p = trial.get("participant")
    if p is not None and str(p).strip():
        return str(p)

    if default_participant:
        return default_participant

    # Inspect common id fields to infer type
    for key in ("participant_index", "participant_id", "participantID",
                "participantId", "subject_id", "subject",
                "user_id", "user", "id"):
        v = trial.get(key)
        if v is None:
            continue
        s = str(v).strip().lower()
        if not s:
            continue
        if s.startswith("simulation"):
            return "simulation"
        # If it's a clearly non-simulation string ID, we tentatively mark human.
        # But if it's purely numeric (common in sims), avoid forcing 'human'.
        if s not in ("human", "simulation", "unknown") and not s.isdigit():
            return "human"

    return "unknown"


def trial_time_constraint(trial: Dict[str, Any]) -> str:
    tc = trial.get("time_constraint")
    if tc is None or (isinstance(tc, str) and not tc.strip()):
        return "NA"
    return str(tc)


def _sanitize_id(text: str) -> str:
    # keep alnum, dash, underscore
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(text))


def trial_participant_label(trial: Dict[str, Any], inferred: Optional[str]) -> str:
    """
    Return a participant label for filenames.

    - For simulation trials with IDs like 'simulation-0', use that as-is (sanitized).
    - For human trials with a concrete ID, use 'human-<id>'.
    - Otherwise, fall back to the inferred base ('human'/'simulation'/'unknown').
    """
    base = trial_participant(trial, inferred)

    def _sanitize_id(text: str) -> str:
        return "".join(ch if (str(ch).isalnum() or ch in "-_") else "_" for ch in str(text))

    for key in ("participant_index", "participant_id", "participantID",
                "participantId", "subject_id", "subject",
                "user_id", "user", "id", "participant"):
        v = trial.get(key)
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("simulation"):
            return _sanitize_id(s)
        if low not in ("human", "simulation", "unknown"):
            label_base = base if base.lower() not in ("unknown", "") else "human"
            return f"{label_base}-{_sanitize_id(s)}"

    return base


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


def choose_out_dir(base_out: Path, sim_out: Optional[Path],
                   human_out: Optional[Path], participant: str) -> Path:
    if participant.lower() == "simulation":
        return Path(sim_out) if sim_out else (base_out / "simulation")
    if participant.lower() == "human":
        return Path(human_out) if human_out else (base_out / "human")
    return base_out / "unknown"


def classify_saccades_by_rules(fixations: List[Dict[str, Any]]) -> Tuple[List[str], List[bool]]:
    """
    Classify each saccade (between i -> i+1) as:
      - 'regression' (green) if next_word < furthest_word_seen_so_far
      - 'skip' (blue) if forward jump > 1
      - 'forward' (red) otherwise (including refixations to same word or adjacent forward)
    Also return a per-fixation boolean list 'is_regressive_fix' for coloring destination dots.
    Note: saccades with missing/invalid word_index are labeled 'forward' as fallback.
    """
    n = len(fixations)
    labels: List[str] = ["forward"] * max(0, n - 1)
    is_reg_fix: List[bool] = [False] * n  # destination fixation flags

    furthest = -10**9  # very small
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


def plot_trial_on_image(trial: Dict[str, Any],
                        img_path: Path,
                        out_path: Path,
                        participant: str,
                        label: str,
                        dot_size: float,
                        line_width: float,
                        alpha_dots: float,
                        alpha_lines: float,
                        y_offset_px: float = 0.0,
                        noise_sigma_px: float = 0.0,
                        rng: Optional[np.random.Generator] = None):
    """
    Plot a single trial's scanpath on the stimulus image.

    If noise_sigma_px > 0, add Gaussian noise N(0, sigma^2) in pixels to the
    y-coordinate of each fixation once (before drawing), to visually restore
    some vertical stochasticity.
    """
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = plt.gca()
    ax.imshow(img, extent=[0, W, H, 0])  # top-left origin
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # y downward
    ax.set_xticks([])
    ax.set_yticks([])

    fixes = extract_fixations(trial)
    if len(fixes) >= 1:
        xs = [f["x"] for f in fixes]
        ys_base = [f["y"] for f in fixes]

        # Add vertical noise if requested
        if noise_sigma_px > 0.0:
            if rng is None:
                rng = np.random.default_rng()
            noise = rng.normal(loc=0.0, scale=noise_sigma_px, size=len(ys_base))
            ys = [y + y_offset_px + dy for y, dy in zip(ys_base, noise)]
        else:
            ys = [y + y_offset_px for y in ys_base]

        # Classify saccades + destination regression flags
        labels, is_reg_fix = classify_saccades_by_rules(fixes)

        # Draw saccades segment-by-segment for proper coloring
        for i in range(len(fixes) - 1):
            x0, y0 = xs[i], ys[i]
            x1, y1 = xs[i+1], ys[i+1]
            lab = labels[i]
            if lab == "regression":
                color = "green"
            elif lab == "skip":
                color = "blue"
            else:
                color = "red"
            ax.plot([x0, x1], [y0, y1], "-", linewidth=line_width,
                    color=color, alpha=alpha_lines, zorder=2)

        # Draw fixation dots: green if destination of a regressive saccade, else red
        # First fixation is never destination; plot it as red by default.
        ax.scatter([xs[0]], [ys[0]], s=dot_size, color="red",
                   alpha=alpha_dots, edgecolors="none", zorder=3)
        if len(fixes) > 1:
            xs_dest: List[float] = []
            ys_dest: List[float] = []
            xs_reg: List[float] = []
            ys_reg: List[float] = []
            for i in range(1, len(fixes)):
                if is_reg_fix[i]:
                    xs_reg.append(xs[i])
                    ys_reg.append(ys[i])
                else:
                    xs_dest.append(xs[i])
                    ys_dest.append(ys[i])
            if xs_dest:
                ax.scatter(xs_dest, ys_dest, s=dot_size, color="red",
                           alpha=alpha_dots, edgecolors="none", zorder=3)
            if xs_reg:
                ax.scatter(xs_reg, ys_reg, s=dot_size, color="green",
                           alpha=alpha_dots, edgecolors="none", zorder=4)

    # No title or in-figure annotations: trial info is only in the filename.

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()  # replace automatically
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Plot scanpaths on stimulus images for each trial in JSON files."
    )
    ap.add_argument("--out_root", "-o", type=Path, default=Path("scanpath_plots"),
                    help="Base output directory.")
    ap.add_argument("--sim_out_dir", type=Path, default=None,
                    help="Override output directory for simulation plots.")
    ap.add_argument("--human_out_dir", type=Path, default=None,
                    help="Override output directory for human plots.")
    ap.add_argument("--human_y_offset_px", type=float, default=0.0,
                    help="Vertical pixel offset added to human fixation y-coordinates (use negative to shift up).")
    ap.add_argument("--sim_y_offset_px", type=float, default=0.0,
                    help="Vertical pixel offset added to simulation fixation y-coordinates.")
    ap.add_argument("--noise_sigma_px", type=float, default=0.0,
                    help="Std dev of Gaussian noise (pixels) added to y per fixation (default: 0).")
    ap.add_argument("--noise_seed", type=int, default=None,
                    help="Random seed for y-noise (default: None, i.e., random).")
    ap.add_argument("json_files", nargs="+", type=Path,
                    help="One or more scanpath JSON files (each is a list of trials).")
    ap.add_argument("--default_participant", type=str, default=None,
                    help="Fallback participant label if missing in trials.")
    ap.add_argument("--human_y_offset_after3_px", type=float, default=0.0,   # <<< NEW
                    help="Additional vertical pixel offset for HUMAN trials when stimulus_index >= 3.")
    args = ap.parse_args()

    # Get the image dir
    img_dir = Path(os.path.join(
        "assets",
        "08_15_09_07_10_images_W1920H1080WS16_LS40_MARGIN400",
        "simulate",
    ))

    # Shared RNG for reproducible jitter across all plots (but still different per fixation)
    rng = np.random.default_rng(args.noise_seed)

    for jf in args.json_files:
        trials = load_json(jf)
        if not isinstance(trials, list):
            print(f"[warn] {jf} did not contain a list; skipping.")
            continue

        # Determine a default participant label based on filename (optional)
        inferred = args.default_participant
        name = jf.stem.lower()
        if inferred is None:
            if "human" in name:
                inferred = "human"
            elif "sim" in name or "simulation" in name:
                inferred = "simulation"
            else:
                inferred = None

        for idx, trial in enumerate(trials):
            stim_idx = trial.get("stimulus_index")
            img_path = find_image(img_dir, stim_idx) if stim_idx is not None else None
            if img_path is None:
                print(f"[warn] Image not found for stimulus_index={stim_idx}; "
                      f"skipping trial {idx} from {jf.name}.")
                continue

            participant = trial_participant(trial, inferred)
            label = trial_participant_label(trial, inferred)
            tc = trial_time_constraint(trial)

            out_dir = choose_out_dir(args.out_root, args.sim_out_dir,
                                     args.human_out_dir, participant)
            out_name = f"stim{stim_idx}_{label}_time{tc}.pdf"
            out_path = out_dir / out_name

            # Choose y-offset based on participant type
            if participant.lower() == "human":
                y_offset_px = args.human_y_offset_px
                # Extra correction for later stimuli where human coordinates are shifted
                if stim_idx is not None and stim_idx >= 3:
                    y_offset_px += args.human_y_offset_after3_px
            elif participant.lower() == "simulation":
                y_offset_px = args.sim_y_offset_px
            else:
                y_offset_px = 0.0

            try:
                plot_trial_on_image(
                    trial, img_path, out_path, participant, label,
                    dot_size=90,
                    line_width=2,
                    alpha_dots=0.3,
                    alpha_lines=0.3,
                    y_offset_px=y_offset_px,
                    noise_sigma_px=args.noise_sigma_px,
                    rng=rng,
                )
                print(f"[ok] Wrote {out_path}")
            except Exception as e:
                print(f"[error] Failed plotting stim={stim_idx}, "
                      f"participant={participant}, time={tc}: {e}")


if __name__ == "__main__":
    main()
