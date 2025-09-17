import os
import sys
import shutil
import json
import numpy as np
import pandas as pd

# ----------------------------
# Config (relative to *this* file)
# ----------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

HUMAN_DIR = os.path.join(HERE, "human_data")
SIM_ROOT = os.path.join(HERE, "simulation_data")

# Files we will compare (human vs sim)
FILES = {
    # metric_key: (human_csv, sim_csv_for_each_kappa, x_col, y_col)
    "length": (
        os.path.join(HUMAN_DIR, "gaze_duration_vs_word_length.csv"),
        "gaze_duration_vs_word_length.csv",
        "word_length",
        "average_gaze_duration",
    ),
    "logfreq": (
        os.path.join(HUMAN_DIR, "gaze_duration_vs_word_log_frequency.csv"),
        "gaze_duration_vs_word_log_frequency_binned.csv",  # use binned sim
        "log_frequency",
        "average_gaze_duration",
    ),
    "logitpred": (
        os.path.join(HUMAN_DIR, "gaze_duration_vs_word_logit_predictability.csv"),
        "gaze_duration_vs_word_logit_predictability_binned.csv",  # use binned sim
        "logit_predictability",
        "average_gaze_duration",
    ),
}

WEIGHTS = {"length": 1.0, "logfreq": 1.0, "logitpred": 1.0}

# ===== Curve-comparison knobs =====
# Resample both curves to the same number of x points inside their overlap
FIXED_POINTS_PER_METRIC = 20
# Use reduced chi-square if True; otherwise NRMSE
USE_REDUCED_CHI2 = True
# If SE/CI not available in CSVs, use this ms floor in the chi-square denominator
SE_FLOOR_MS = 5.0

# Where to dump the grid summary and the "best" sim files (so plot.py can use them)
GRID_OUT_CSV = os.path.join(SIM_ROOT, "grid_search_results.csv")
SIMULATED_RESULTS_FOR_PLOTS = os.path.join(HERE, "best_param_simulated_results")  # plot.py reads this


# ----------------------------
# Helpers
# ----------------------------
def load_xy_se(csv_path, x_col, y_col):
    """Load x, y and (optionally) per-bin SE from a CSV.
    Tries common column names: se/sem/std_err/stderr, or CI columns to derive SE.
    Falls back to None if unavailable.
    """
    df = pd.read_csv(csv_path)
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(
            f"Expected columns '{x_col}' and '{y_col}' in {csv_path}. "
            f"Got: {df.columns.tolist()}"
        )
    df = df[[x_col, y_col] + [c for c in df.columns if c not in (x_col, y_col)]].dropna(
        subset=[x_col, y_col]
    ).sort_values(x_col)

    # try to find SE
    se = None
    for cand in ["se", "sem", "std_err", "stderr", "se_ms"]:
        if cand in df.columns:
            se = df[cand].to_numpy(dtype=float)
            break
    if se is None:
        # try CI columns
        lo = next((c for c in ["ci_low", "ci95_low", "lower_ci", "ci_lower"] if c in df.columns), None)
        hi = next((c for c in ["ci_high", "ci95_high", "upper_ci", "ci_upper"] if c in df.columns), None)
        if lo and hi:
            se = (df[hi].to_numpy(dtype=float) - df[lo].to_numpy(dtype=float)) / (2 * 1.96)
        else:
            # try std + n
            std = next((c for c in ["std", "sd", "std_ms"] if c in df.columns), None)
            n = next((c for c in ["n", "count", "N", "Num"] if c in df.columns), None)
            if std is not None and n is not None:
                se = df[std].to_numpy(dtype=float) / np.sqrt(np.maximum(df[n].to_numpy(dtype=float), 1.0))

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    return x, y, (se.astype(float) if se is not None else None)


def resample_xy_se(x, y, se, lo, hi, K):
    xs = np.linspace(lo, hi, int(K))
    ys = np.interp(xs, x, y)
    if se is None:
        ses = None
    else:
        ses = np.interp(xs, x, se)
    return xs, ys, ses


def reduced_chi2(hx, hy, hse, sx, sy, sse, K=FIXED_POINTS_PER_METRIC):
    # overlapping support
    lo, hi = max(hx.min(), sx.min()), min(hx.max(), sx.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.inf
    hx, hy, hse = resample_xy_se(hx, hy, hse, lo, hi, K)
    sx, sy, sse = resample_xy_se(sx, sy, sse, lo, hi, K)

    if hse is None and sse is None:
        denom = (SE_FLOOR_MS ** 2)
        z2 = ((hy - sy) ** 2) / denom
    else:
        hvar = (hse if hse is not None else 0.0) ** 2
        svar = (sse if sse is not None else 0.0) ** 2
        denom = hvar + svar
        denom = np.where(~np.isfinite(denom) | (denom <= 0), SE_FLOOR_MS ** 2, denom)
        z2 = ((hy - sy) ** 2) / denom

    return float(np.mean(z2))  # reduced by K


def nrmse(hx, hy, sx, sy, K=FIXED_POINTS_PER_METRIC):
    lo, hi = max(hx.min(), sx.min()), min(hx.max(), sx.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.inf
    xs = np.linspace(lo, hi, int(K))
    hyi = np.interp(xs, hx, hy)
    syi = np.interp(xs, sx, sy)
    rng = np.ptp(hyi)
    rng = rng if rng > 0 else 1.0
    return float(np.sqrt(np.mean((hyi - syi) ** 2)) / rng)


def per_metric_loss(human_csv, sim_csv, xcol, ycol):
    hx, hy, hse = load_xy_se(human_csv, xcol, ycol)
    sx, sy, sse = load_xy_se(sim_csv, xcol, ycol)
    if USE_REDUCED_CHI2:
        return reduced_chi2(hx, hy, hse, sx, sy, sse)
    else:
        return nrmse(hx, hy, sx, sy)

def align_to_sim_bins_arrays(hx, hy, hse, sx, sy, sse, decimals=6):
    """
    Keep only human x-values that exist in SIM (match by rounded x).
    Returns matched arrays + count_removed.
    If fewer than 2 matched points remain, returns (None, ...), 0.
    """
    hr = np.round(hx, decimals)
    sr = np.round(sx, decimals)
    sim_bins = set(sr.tolist())

    hmask = np.array([x in sim_bins for x in hr])
    removed = int((~hmask).sum())

    hx2, hy2 = hx[hmask], hy[hmask]
    hse2 = (hse[hmask] if hse is not None else None)

    # Now keep sim rows only where sim bins are in the (filtered) human set, so order lines up
    common_bins = set(np.round(hx2, decimals).tolist())
    smask = np.array([x in common_bins for x in sr])
    sx2, sy2 = sx[smask], sy[smask]
    sse2 = (sse[smask] if sse is not None else None)

    if len(hx2) < 2 or len(sx2) < 2:
        return (None, None, None, None, None, None), removed

    # Sort both by x so indices align
    order_h = np.argsort(hx2); order_s = np.argsort(sx2)
    return (hx2[order_h], hy2[order_h], (hse2[order_h] if hse2 is not None else None),
            sx2[order_s], sy2[order_s], (sse2[order_s] if sse2 is not None else None)), removed



def parse_kappa(folder_name):
    # 'kappa_2p30' -> 2.30
    if not folder_name.startswith("kappa_"):
        return None
    s = folder_name[len("kappa_"):]
    try:
        return float(s.replace("p", "."))
    except Exception:
        return None


def find_kappa_folders(sim_root):
    out = []
    if not os.path.isdir(sim_root):
        return out
    for name in os.listdir(sim_root):
        k = parse_kappa(name)
        if k is not None and os.path.isdir(os.path.join(sim_root, name)):
            out.append((k, os.path.join(sim_root, name)))
    out.sort(key=lambda t: t[0])
    return out


def main():
    # sanity check human files exist
    for mkey, (h_path, _, xcol, ycol) in FILES.items():
        if not os.path.exists(h_path):
            raise FileNotFoundError(f"Human CSV missing: {h_path}")

    kappas = find_kappa_folders(SIM_ROOT)
    if not kappas:
        raise FileNotFoundError(f"No kappa_* folders found under {SIM_ROOT}")

    rows = []
    for kappa, kdir in kappas:
        per_metric = {}
        total = 0.0

        for mkey, (human_csv, sim_rel, xcol, ycol) in FILES.items():
            sim_csv = os.path.join(kdir, sim_rel)
            if not os.path.exists(sim_csv):
                # try un-binned fallback for freq/pred if needed
                if mkey in ("logfreq", "logitpred"):
                    sim_alt = sim_rel.replace("_binned", "")
                    sim_csv_alt = os.path.join(kdir, sim_alt)
                    if os.path.exists(sim_csv_alt):
                        sim_csv = sim_csv_alt
                    else:
                        per_metric[mkey] = np.inf
                        continue
                else:
                    per_metric[mkey] = np.inf
                    continue

            loss = per_metric_loss(human_csv, sim_csv, xcol, ycol)
            per_metric[mkey] = loss
            total += WEIGHTS[mkey] * loss

        row = {
            "kappa": kappa,
            "F_total": total,
            "F_length": per_metric.get("length", np.inf),
            "F_log_frequency": per_metric.get("logfreq", np.inf),
            "F_logit_predictability": per_metric.get("logitpred", np.inf),
            "kappa_folder": kdir,
        }
        rows.append(row)
        print(
            f"kappa={kappa:.3f}  -> F={total:.6f} | len={row['F_length']:.6f}, "
            f"logfreq={row['F_log_frequency']:.6f}, logitpred={row['F_logit_predictability']:.6f}"
        )

    res_df = pd.DataFrame(rows).sort_values(["F_total", "kappa"]).reset_index(drop=True)
    os.makedirs(SIM_ROOT, exist_ok=True)
    res_df.to_csv(GRID_OUT_CSV, index=False)
    print(f"\nGrid results written to: {GRID_OUT_CSV}")

    # best kappa
    best = res_df.iloc[0]
    best_kappa = best["kappa"]
    best_dir = best["kappa_folder"]
    print(f"\nBest kappa = {best_kappa:.3f} (F_total={best['F_total']:.6f})")
    print(f"Folder: {best_dir}")

    # prepare 'best_param_simulated_results' for plot.py (copy the 3 sim CSVs)
    os.makedirs(SIMULATED_RESULTS_FOR_PLOTS, exist_ok=True)
    copies = [
        ("gaze_duration_vs_word_length.csv", "gaze_duration_vs_word_length.csv"),
        ("gaze_duration_vs_word_log_frequency_binned.csv", "gaze_duration_vs_word_log_frequency_binned.csv"),
        ("gaze_duration_vs_word_logit_predictability_binned.csv", "gaze_duration_vs_word_logit_predictability_binned.csv"),
    ]
    for src_name, dst_name in copies:
        src = os.path.join(best_dir, src_name)
        if not os.path.exists(src) and "_binned" in src_name:
            # fallback to non-binned if binned missing
            src2 = os.path.join(best_dir, src_name.replace("_binned", ""))
            if os.path.exists(src2):
                src = src2
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(SIMULATED_RESULTS_FOR_PLOTS, dst_name))

    # optional: run plot.py programmatically if it's adjacent
    plot_py = os.path.join(HERE, "plot.py")
    if os.path.exists(plot_py):
        try:
            prev_cwd = os.getcwd()
            os.chdir(HERE)
            with open("plot.py", "rb") as f:
                code = compile(f.read(), "plot.py", "exec")
                exec(code, {"__name__": "__main__"})
            os.chdir(prev_cwd)
            print("Figures written via plot.py using best kappa.")
        except Exception as e:
            print(f"plot.py execution failed (skipping figure generation): {e}")
    else:
        print("plot.py not found next to this script; skipping automatic figure generation.")


if __name__ == "__main__":
    main()
