#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_EXTS = [".png", ".jpg", ".jpeg", ".webp"]

# ----------------------- IO helpers -----------------------

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_image(images_dir: Path, stimulus_index: int, verbose: bool=False) -> Optional[Path]:
    """Try multiple filename patterns; fallback to glob + recursive search."""
    if stimulus_index is None:
        return None
    idx = int(stimulus_index)

    bases = [
        f"{idx}", f"{idx:02d}", f"{idx:03d}",
        f"image_{idx}", f"image_{idx:02d}",
        f"img_{idx}", f"img_{idx:02d}",
        f"stim{idx}", f"stim_{idx}",
    ]
    # direct candidates
    for b in bases:
        for ext in IMG_EXTS:
            p = images_dir / f"{b}{ext}"
            if p.exists():
                return p

    # shallow glob
    for b in bases:
        for ext in IMG_EXTS:
            for cand in images_dir.glob(f"{b}*{ext}"):
                if cand.is_file():
                    return cand

    # deep search
    for b in bases:
        for ext in IMG_EXTS:
            for cand in images_dir.rglob(f"{b}*{ext}"):
                if cand.is_file():
                    return cand

    if verbose:
        samples = list(images_dir.rglob("*.png"))[:5]
        samples += list(images_dir.rglob("*.jpg"))[:5]
        print(f"[warn] No image for stimulus_index={idx} under {images_dir}. Example files: {[str(s) for s in samples]}")
    return None

# ---------------- Participant labeling (for filenames) ----------------

def trial_participant(trial: Dict[str, Any], default_participant: Optional[str]) -> str:
    p = trial.get("participant")
    if p is not None and str(p).strip():
        return str(p)
    if default_participant:
        return default_participant

    # Inspect common id fields to infer type
    for key in ("participant_index","participant_id","participantID","participantId"):
        v = trial.get(key)
        if v is not None:
            s = str(v).strip().lower()
            if s.startswith("simulation"):
                return "simulation"
            return "human"
    for key in ("subject_id","subject","user_id","user","id"):
        if trial.get(key) is not None:
            return "human"
    return "unknown"

def _sanitize_id(text: str) -> str:
    return "".join(ch if (str(ch).isalnum() or ch in "-_") else "_" for ch in str(text))

def trial_participant_label(trial: Dict[str, Any], inferred: Optional[str]) -> str:
    """
    Return a participant label for filenames.
    - For simulation trials with participant_index like "simulation-0", use it as-is (sanitized).
    - For human trials with an ID (e.g., 23), output "human-23".
    - Otherwise, fall back to 'human'/'simulation'/inferred.
    """
    base = trial_participant(trial, inferred)
    for key in ("participant_index","participant_id","participantID","participantId","subject_id","subject","user_id","user","id","participant"):
        v = trial.get(key)
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("simulation"):
            return _sanitize_id(s)
        if low not in ("human","simulation","unknown"):
            label_base = base if base.lower() not in ("unknown","") else "human"
            return f"{label_base}-{_sanitize_id(s)}"
    return base

def trial_time_constraint(trial: Dict[str, Any]) -> str:
    tc = trial.get("time_constraint")
    if tc is None or (isinstance(tc, str) and not tc.strip()):
        return "NA"
    return str(tc)

def choose_out_dir(base_out: Path, sim_out: Optional[Path], human_out: Optional[Path], participant: str) -> Path:
    part = (participant or "").lower()
    if part == "simulation":
        return Path(sim_out) if sim_out else (base_out / "simulation")
    if part == "human":
        return Path(human_out) if human_out else (base_out / "human")
    return base_out / "unknown"

# ---------------- Fixations/Heat computation ----------------

def extract_fixations_for_heat(trial: Dict[str, Any]) -> List[Tuple[float, float, float]]:
    """
    Returns a list of (x, y, weight) where weight is fixation duration in milliseconds (default 1).
    """
    out: List[Tuple[float,float,float]] = []
    for row in trial.get("fixation_data", []):
        try:
            x = float(row.get("fix_x"))
            y = float(row.get("fix_y"))
        except Exception:
            continue
        dur = row.get("fix_duration", None)
        try:
            w = float(dur) if dur is not None else 1.0
        except Exception:
            w = 1.0
        out.append((x, y, w))
    return out

def add_gaussian(heat, cx: float, cy: float, sigma: float, weight: float):
    """
    Add a weighted Gaussian bump centered at (cx, cy) to 'heat' (shape HxW).
    Only updates a 3*sigma window for speed.
    """
    H, W = heat.shape
    rad = int(3 * sigma)
    x0 = max(0, int(np.floor(cx)) - rad)
    x1 = min(W, int(np.floor(cx)) + rad + 1)
    y0 = max(0, int(np.floor(cy)) - rad)
    y1 = min(H, int(np.floor(cy)) + rad + 1)
    if x0 >= x1 or y0 >= y1:
        return

    yy, xx = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    heat[y0:y1, x0:x1] += weight * g

def compute_heatmap(width: int, height: int, fixations: List[Tuple[float,float,float]], sigma_px: float):
    heat = np.zeros((height, width), dtype=np.float32)
    for (x, y, w) in fixations:
        add_gaussian(heat, x, y, sigma_px, w)
    return heat

def overlay_heatmap_on_image(img: Image.Image,
                             heat,
                             alpha_max: float = 0.65,
                             gamma: float = 0.6,
                             cmap_name: str = "RdBu_r",
                             vmax_ms: float = None):
    """
    Create an RGBA overlay from heat (HxW float) and alpha blend on the image.
    - alpha per pixel = alpha_max * (normalized_heat ** gamma)
    - colormap default 'RdBu_r' makes high dwell = blue (as in your draft).
    """
    W, H = img.size
    # Normalize heat 0..1
    h = heat.copy()
    h[h < 0] = 0
    if vmax_ms is not None and vmax_ms > 0:
        vmax = float(vmax_ms)
    else:
        vmax = float(h.max()) if float(h.max()) > 0 else 1.0
    # clip then scale
    h = np.clip(h, 0, vmax) / vmax

    # Colormap to RGBA (values 0..1)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(h)  # HxWx4, floats 0..1
    # Alpha based on intensity
    rgba[..., 3] = alpha_max * (h ** gamma)

    # Composite: draw background image, then overlay
    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    ax = plt.gca()
    ax.imshow(img, extent=[0, W, H, 0])
    ax.imshow(rgba, extent=[0, W, H, 0])
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_xticks([]); ax.set_yticks([])
    return fig, ax

# ---------------- Main plotting ----------------

def plot_heat_for_trial(trial: Dict[str, Any],
                        images_dir: Path,
                        out_path: Path,
                        sigma_px: float,
                        alpha_max: float,
                        gamma: float,
                        cmap_name: str,
                        vmax_ms: float = None,
                        verbose: bool=False):
    stim_idx = trial.get("stimulus_index")
    img_path = find_image(images_dir, stim_idx, verbose=verbose) if stim_idx is not None else None
    if img_path is None:
        raise FileNotFoundError(f"No image for stimulus_index={stim_idx} in {images_dir}")

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    fixes = extract_fixations_for_heat(trial)
    heat = compute_heatmap(W, H, fixes, sigma_px=sigma_px)

    fig, ax = overlay_heatmap_on_image(img, heat, alpha_max=alpha_max, gamma=gamma, cmap_name=cmap_name, vmax_ms=vmax_ms)

    participant = trial_participant(trial, None)
    label = trial_participant_label(trial, participant)
    tc = trial_time_constraint(trial)
    ax.set_title(f"Stim {stim_idx} | {label} | time={tc}", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Plot fixation-duration heatmaps over stimulus images.")
    ap.add_argument("--only", choices=["human","simulation"], default=None, help="Only plot trials of this participant type.")
    ap.add_argument("--exclude", choices=["human","simulation"], default=None, help="Skip trials of this participant type.")
    ap.add_argument("--norm_mode", choices=["per-image","fixed"], default="per-image", help="Normalization per image (independent) or fixed across images.")
    ap.add_argument("--vmax_ms", type=float, default=90000.0, help="If --norm_mode=fixed, treat this many milliseconds as the top of the color scale (default 90s).")
    # ap.add_argument("--images_dir", "-i", type=Path, required=True, help="Directory containing stimulus images.")
    ap.add_argument("--out_root", "-o", type=Path, default=Path("heatmap_plots"), help="Base output directory.")
    ap.add_argument("--sim_out_dir", type=Path, default=None, help="Override output directory for simulation plots.")
    ap.add_argument("--human_out_dir", type=Path, default=None, help="Override output directory for human plots.")
    ap.add_argument("--sigma_px", type=float, default=60.0, help="Gaussian sigma in pixels (spread of each fixation).")
    ap.add_argument("--alpha_max", type=float, default=0.65, help="Max overlay opacity.")
    ap.add_argument("--gamma", type=float, default=0.6, help="Alpha gamma correction for softer edges (0..1 strong).")
    ap.add_argument("--cmap", type=str, default="RdBu_r", help="Matplotlib colormap name; e.g., 'RdBu_r', 'turbo'.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging (image matching).")
    ap.add_argument("json_files", nargs="+", type=Path, help="One or more JSON files (each a list of trials).")
    ap.add_argument("--default_participant", type=str, default=None, help="Fallback participant label if missing in trials.")
    args = ap.parse_args()

    images_dir = Path(os.path.join("assets", "08_15_09_07_10_images_W1920H1080WS16_LS40_MARGIN400", "simulate"))

    for jf in args.json_files:
        trials = load_json(jf)
        if not isinstance(trials, list):
            print(f"[warn] {jf} is not a list of trials; skipping.")
            continue

        # infer default participant from file name if not provided
        inferred = args.default_participant
        name = jf.stem.lower()
        if inferred is None:
            if "human" in name:
                inferred = "human"
            elif "sim" in name or "simulation" in name:
                inferred = "simulation"

        for idx, trial in enumerate(trials):
            participant = trial_participant(trial, inferred)
            if args.only and participant != args.only:
                continue
            if args.exclude and participant == args.exclude:
                continue
            label = trial_participant_label(trial, inferred)
            tc = trial_time_constraint(trial)

            out_dir = choose_out_dir(args.out_root, args.sim_out_dir, args.human_out_dir, participant)
            stim_idx = trial.get("stimulus_index", "NA")
            out_name = f"stim{stim_idx}_{label}_time{tc}.png"
            out_path = out_dir / out_name

            try:
                plot_heat_for_trial(
                    trial=trial,
                    images_dir=images_dir,
                    out_path=out_path,
                    sigma_px=args.sigma_px,
                    alpha_max=args.alpha_max,
                    gamma=args.gamma,
                    cmap_name=args.cmap,
                    vmax_ms=(args.vmax_ms if args.norm_mode == "fixed" else None),
                    verbose=args.verbose,
                )
                print(f"[ok] Wrote {out_path}")
            except Exception as e:
                print(f"[error] Heatmap failed for trial idx={idx} stim={trial.get('stimulus_index')}: {e}")

if __name__ == "__main__":
    main()
