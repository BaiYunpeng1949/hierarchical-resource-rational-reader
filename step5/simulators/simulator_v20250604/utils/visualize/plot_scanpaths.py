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

# # def trial_participant(trial: Dict[str, Any], default_participant: Optional[str]) -> str:
# #     p = trial.get("participant_index")
# #     if p is not None and str(p).strip():
# #         return str(p)
# #     return default_participant or "unknown"
# def trial_participant(trial: Dict[str, Any], default_participant: Optional[str]) -> str:
#     p = trial.get("participant")
#     if p is not None and str(p).strip():
#         return str(p)
#     if default_participant:
#         return default_participant
#     # Heuristic: if a participant index/id exists, assume human
#     if "participant_index" in trial or "participant_id" in trial or "subject_id" in trial:
#         return "human"
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
#     """Return a participant label for filenames.

#     For human trials, prefer a concrete participant ID if present and not just the string 'human'.
#     Keys we try (first non-empty wins): participant_index, participant_id, participantID, participantId, subject_id, subject, user_id, user, id, participant.
#     """
#     base = trial_participant(trial, inferred)
#     # search for a real id
#     for key in ("participant_index", "participant_id","participantID","participantId","subject_id","subject","user_id","user","id","participant"):
#         v = trial.get(key)
#         if v is not None:
#             s = str(v).strip()
#             if s and s.lower() not in ("human","simulation","unknown"):
#                 label_base = base if base.lower() not in ("unknown", "") else "human"
#                 return f"{label_base}-{_sanitize_id(s)}"
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
#                         dot_size: float,
#                         line_width: float,
#                         alpha_dots: float,
#                         alpha_lines: float):
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
#         ys = [f["y"] for f in fixes]

#         # Classify saccades + destination regression flags
#         labels, is_reg_fix = classify_saccades_by_rules(fixes)

#         # Draw saccades segment-by-segment for proper coloring
#         for i in range(len(fixes) - 1):
#             x0, y0 = fixes[i]["x"], fixes[i]["y"]
#             x1, y1 = fixes[i+1]["x"], fixes[i+1]["y"]
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

#     stim = trial.get("stimulus_index", "NA")
#     participant = trial_participant(trial, None)
#     label = trial_participant_label(trial, participant)

#     # TODO debug delete later
#     print(f"The label is: {label}, the participant is: {participant}")

#     tc = trial_time_constraint(trial)
#     ax.set_title(f"Stim {stim} | {label} | time={tc}", fontsize=10)

#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     if out_path.exists():
#         out_path.unlink()  # replace automatically
#     fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
#     plt.close(fig)

# def main():
#     ap = argparse.ArgumentParser(description="Plot scanpaths on stimulus images for each trial in JSON files.")
#     # ap.add_argument("--images_dir", "-i", type=Path, required=True, help="Directory containing stimulus images (indexed 0..N).")
#     ap.add_argument("--out_root", "-o", type=Path, default=Path("scanpath_plots"), help="Base output directory.")
#     ap.add_argument("--sim_out_dir", type=Path, default=None, help="Override output directory for simulation plots.")
#     ap.add_argument("--human_out_dir", type=Path, default=None, help="Override output directory for human plots.")
#     # ap.add_argument("--dot_size", type=float, default=48.0, help="Fixation dot size (default 48).")
#     # ap.add_argument("--line_width", type=float, default=2.0, help="Saccade line width (default 2.0).")
#     # ap.add_argument("--alpha_dots", type=float, default=0.6, help="Dot transparency (0..1, default 0.6).")
#     # ap.add_argument("--alpha_lines", type=float, default=0.6, help="Line transparency (0..1, default 0.6).")
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
#             tc = trial_time_constraint(trial)

#             out_dir = choose_out_dir(args.out_root, args.sim_out_dir, args.human_out_dir, participant)
#             out_name = f"stim{stim_idx}_{participant}_time{tc}.png"
#             out_path = out_dir / out_name

#             try:
#                 plot_trial_on_image(
#                     trial, img_path, out_path,
#                     dot_size=90,
#                     line_width=2,
#                     alpha_dots=0.3,
#                     alpha_lines=0.3,
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
    p = trial.get("participant")
    if p is not None and str(p).strip():
        return str(p)
    if default_participant:
        return default_participant
    # Heuristic: if a participant index/id exists, assume human
    if "participant_index" in trial or "participant_id" in trial or "subject_id" in trial:
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
    """Return a participant label for filenames.

    For human trials, prefer a concrete participant ID if present and not just the string 'human'.
    Keys we try (first non-empty wins): participant_index, participant_id, participantID, participantId, subject_id, subject, user_id, user, id, participant.
    """
    base = trial_participant(trial, inferred)
    # search for a real id
    for key in ("participant_index", "participant_id","participantID","participantId","subject_id","subject","user_id","user","id","participant"):
        v = trial.get(key)
        if v is not None:
            s = str(v).strip()
            if s and s.lower() not in ("human","simulation","unknown"):
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

def choose_out_dir(base_out: Path, sim_out: Optional[Path], human_out: Optional[Path], participant: str) -> Path:
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
                        dot_size: float,
                        line_width: float,
                        alpha_dots: float,
                        alpha_lines: float):
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
        ys = [f["y"] for f in fixes]

        # Classify saccades + destination regression flags
        labels, is_reg_fix = classify_saccades_by_rules(fixes)

        # Draw saccades segment-by-segment for proper coloring
        for i in range(len(fixes) - 1):
            x0, y0 = fixes[i]["x"], fixes[i]["y"]
            x1, y1 = fixes[i+1]["x"], fixes[i+1]["y"]
            lab = labels[i]
            if lab == "regression":
                color = "green"
            elif lab == "skip":
                color = "blue"
            else:
                color = "red"
            ax.plot([x0, x1], [y0, y1], "-", linewidth=line_width, color=color, alpha=alpha_lines, zorder=2)

        # Draw fixation dots: green if destination of a regressive saccade, else red
        # First fixation is never destination; plot it as red by default.
        ax.scatter([xs[0]], [ys[0]], s=dot_size, color="red", alpha=alpha_dots, edgecolors="none", zorder=3)
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
                ax.scatter(xs_dest, ys_dest, s=dot_size, color="red", alpha=alpha_dots, edgecolors="none", zorder=3)
            if xs_reg:
                ax.scatter(xs_reg, ys_reg, s=dot_size, color="green", alpha=alpha_dots, edgecolors="none", zorder=4)

    stim = trial.get("stimulus_index", "NA")
    participant = trial_participant(trial, None)
    label = trial_participant_label(trial, participant)


    tc = trial_time_constraint(trial)
    ax.set_title(f"Stim {stim} | {label} | time={tc}", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()  # replace automatically
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Plot scanpaths on stimulus images for each trial in JSON files.")
    # ap.add_argument("--images_dir", "-i", type=Path, required=True, help="Directory containing stimulus images (indexed 0..N).")
    ap.add_argument("--out_root", "-o", type=Path, default=Path("scanpath_plots"), help="Base output directory.")
    ap.add_argument("--sim_out_dir", type=Path, default=None, help="Override output directory for simulation plots.")
    ap.add_argument("--human_out_dir", type=Path, default=None, help="Override output directory for human plots.")
    # ap.add_argument("--dot_size", type=float, default=48.0, help="Fixation dot size (default 48).")
    # ap.add_argument("--line_width", type=float, default=2.0, help="Saccade line width (default 2.0).")
    # ap.add_argument("--alpha_dots", type=float, default=0.6, help="Dot transparency (0..1, default 0.6).")
    # ap.add_argument("--alpha_lines", type=float, default=0.6, help="Line transparency (0..1, default 0.6).")
    ap.add_argument("json_files", nargs="+", type=Path, help="One or more scanpath JSON files (each is a list of trials).")
    ap.add_argument("--default_participant", type=str, default=None, help="Fallback participant label if missing in trials.")
    args = ap.parse_args()
    
    # Get the image dir
    img_dir = Path(os.path.join("assets", "08_15_09_07_10_images_W1920H1080WS16_LS40_MARGIN400", "simulate"))

    for jf in args.json_files:
        trials = load_json(jf)
        if not isinstance(trials, list):
            print(f"[warn] {jf} did not contain a list; skipping.")
            continue

        # Determine a default participant label based on filename (optional)
        inferred = args.default_participant
        name = jf.stem.lower()
        if inferred is None:
            inferred = "human" if "human" in name else "simulation" if "sim" in name or "simulation" in name else None

        for idx, trial in enumerate(trials):
            stim_idx = trial.get("stimulus_index")
            img_path = find_image(img_dir, stim_idx) if stim_idx is not None else None
            if img_path is None:
                print(f"[warn] Image not found for stimulus_index={stim_idx}; skipping trial {idx} from {jf.name}.")
                continue

            participant = trial_participant(trial, inferred)
            label = trial_participant_label(trial, inferred)
            tc = trial_time_constraint(trial)

            out_dir = choose_out_dir(args.out_root, args.sim_out_dir, args.human_out_dir, participant)
            out_name = f"stim{stim_idx}_{label}_time{tc}.png"
            out_path = out_dir / out_name

            try:
                plot_trial_on_image(
                    trial, img_path, out_path,
                    dot_size=90,
                    line_width=2,
                    alpha_dots=0.3,
                    alpha_lines=0.3,
                )
                print(f"[ok] Wrote {out_path}")
            except Exception as e:
                print(f"[error] Failed plotting stim={stim_idx}, participant={participant}, time={tc}: {e}")

if __name__ == "__main__":
    main()
