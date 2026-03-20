import os
import sys
import shutil
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
        "gaze_duration_vs_word_log_frequency_binned.csv",  # use binned sim if present
        "log_frequency",
        "average_gaze_duration",
    ),
    "logitpred": (
        os.path.join(HUMAN_DIR, "gaze_duration_vs_word_logit_predictability.csv"),
        "gaze_duration_vs_word_logit_predictability_binned.csv",  # use binned sim if present
        "logit_predictability",
        "average_gaze_duration",
    ),
}

WEIGHTS = {"length": 1.0, "logfreq": 1.0, "logitpred": 1.0}

# ===== Regression-line matching knobs =====
# Normalize within the overlap so losses are comparable across metrics
NORMALIZE = True
SLOPE_WEIGHT = 10.0   # prioritize trend
INTERCEPT_WEIGHT = 1.0
EPS = 1e-9

# Where to dump the grid summary and the "best" sim files (so plot.py can use them)
GRID_OUT_CSV = os.path.join(SIM_ROOT, "grid_search_results.csv")
SIMULATED_RESULTS_FOR_PLOTS = os.path.join(HERE, "best_param_simulated_results")  # plot.py reads this

# ----------------------------
# Helpers
# ----------------------------
def load_xy(csv_path, x_col, y_col):
    df = pd.read_csv(csv_path)
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(
            f"Expected columns '{x_col}' and '{y_col}' in {csv_path}. "
            f"Got: {df.columns.tolist()}"
        )
    df = df[[x_col, y_col]].dropna().sort_values(x_col)
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    return x, y

def line_loss_over_overlap(hx, hy, sx, sy):
    if hx.size == 0 or sx.size == 0:
        return np.inf
    lo, hi = max(hx.min(), sx.min()), min(hx.max(), sx.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.inf
    hmask = (hx >= lo) & (hx <= hi)
    smask = (sx >= lo) & (sx <= hi)
    xh, yh = hx[hmask], hy[hmask]
    xs, ys = sx[smask], sy[smask]
    if xh.size < 2 or xs.size < 2:
        return np.inf

    if NORMALIZE:
        # normalize x to [0,1] using human overlap
        x0, x1 = float(xh.min()), float(xh.max())
        xrng = max(x1 - x0, EPS)
        xh_n = (xh - x0) / xrng
        xs_n = (xs - x0) / xrng
        # normalize y to [0,1] using human overlap
        y0, y1 = float(yh.min()), float(yh.max())
        yrng = max(y1 - y0, EPS)
        yh_n = (yh - y0) / yrng
        ys_n = (ys - y0) / yrng
    else:
        xh_n, yh_n = xh, yh
        xs_n, ys_n = xs, ys

    # Fit lines exactly like plot.py (np.polyfit)
    sh, bh = np.polyfit(xh_n, yh_n, deg=1)  # slope, intercept
    ss, bs = np.polyfit(xs_n, ys_n, deg=1)

    slope_err2 = (ss - sh) ** 2
    int_err2   = (bs - bh) ** 2
    return float(SLOPE_WEIGHT * slope_err2 + INTERCEPT_WEIGHT * int_err2)

def per_metric_loss(human_csv, sim_csv, xcol, ycol):
    hx, hy = load_xy(human_csv, xcol, ycol)
    sx, sy = load_xy(sim_csv, xcol, ycol)
    return line_loss_over_overlap(hx, hy, sx, sy)

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
