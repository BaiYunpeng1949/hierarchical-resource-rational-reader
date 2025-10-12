import os, re, argparse
import numpy as np
import pandas as pd

HUMAN_DEFAULT = os.path.join("human_data", "all_words_regression_and_skip_probabilities.csv")
SIM_ROOT_DEFAULT = os.path.join("simulation_data")
SIM_REL_DEFAULT = "all_words_regression_and_skip_probabilities.csv"

# Four-panel figure definitions (skip + regression probabilities)
PANELS_DEF = [
    # 1) Skip vs Word Length (integer x)
    ("skip_vs_length", "length", "skip_probability", "Word Length", "Skip Probability", True),
    # 2) Skip vs Logit Predictability
    ("skip_vs_logit_predictability", "logit_predictability", "skip_probability", "Logit Predictability", "", False),
    # 3) Skip vs Log Frequency
    ("skip_vs_log_frequency", "log_frequency", "skip_probability", "Log Frequency", "", False),
    # 4) Regression vs Difficulty
    ("regression_vs_difficulty", "difficulty", "regression_probability", "Word Difficulty", "Regression Probability", False),
]

# For scoring during grid search (slope/intercept difference on overlapping ranges)
CURVES_FOR_SCORE = [
    ("skip_vs_length",               "length",              "skip_probability"),
    ("skip_vs_logit_predictability", "logit_predictability","skip_probability"),
    ("skip_vs_log_frequency",        "log_frequency",       "skip_probability"),
    ("regression_vs_difficulty",     "difficulty",          "regression_probability"),
]

WEIGHT_SLOPE = 1.0
WEIGHT_INTER = 1.0

def parse_w(folder_name: str):
    m = re.search(r"w_regression_cost_([0-9]+)p([0-9]+)", folder_name)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m2 = re.search(r"w_regression_cost_([0-9.]+)", folder_name)
    return float(m2.group(1)) if m2 else None

def _has_cols(df: pd.DataFrame, cols):
    return all(c in df.columns for c in cols)

def _xy_from(df: pd.DataFrame, xcol: str, ycol: str):
    if not _has_cols(df, [xcol, ycol]):
        return np.array([]), np.array([])
    sub = df[[xcol, ycol]].dropna()
    return sub[xcol].values.astype(float), sub[ycol].values.astype(float)

def _overlap(xh, xs):
    if xh.size == 0 or xs.size == 0:
        return None
    lo = max(np.min(xh), np.min(xs))
    hi = min(np.max(xh), np.max(xs))
    if lo < hi:
        return lo, hi
    return None

def _fit_line(x, y):
    if x.size < 2 or y.size < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)

def loss_slope_intercept(h_human: pd.DataFrame, h_sim: pd.DataFrame, xcol: str, ycol: str) -> float:
    xh, yh = _xy_from(h_human, xcol, ycol)
    xs, ys = _xy_from(h_sim,   xcol, ycol)
    ov = _overlap(xh, xs)
    if ov is None:
        return np.nan
    lo, hi = ov
    hmask = (xh >= lo) & (xh <= hi)
    smask = (xs >= lo) & (xs <= hi)
    xh2, yh2 = xh[hmask], yh[hmask]
    xs2, ys2 = xs[smask], ys[smask]
    sh, bh = _fit_line(xh2, yh2)
    ss, bs = _fit_line(xs2, ys2)
    if np.isnan(sh) or np.isnan(ss):
        return np.nan
    return WEIGHT_SLOPE * (ss - sh) ** 2 + WEIGHT_INTER * (bs - bh) ** 2

def score_folder(sim_csv: str, human_csv: str) -> dict:
    h = pd.read_csv(human_csv)
    s = pd.read_csv(sim_csv)
    scores = {}
    total_parts = []
    for key, xcol, ycol in CURVES_FOR_SCORE:
        val = loss_slope_intercept(h, s, xcol, ycol)
        scores[key] = val
        total_parts.append(val)
    total = float(np.nansum(total_parts)) if np.isfinite(np.nansum(total_parts)) else float("inf")
    scores["F_total"] = total
    return scores

def post_plot(human_csv: str, sim_root: str, best_folder: str, sim_relpath: str):
    """Create ONE 1x4 panel figure using plot.py's plot_in_a_row with binned dots + regression."""
    import importlib.util, sys
    here = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(here, "plot.py")
    if not os.path.exists(plot_path):
        print("plot.py not found; skipping figures.")
        return
    spec = importlib.util.spec_from_file_location("plot_module", plot_path)
    plot_module = importlib.util.module_from_spec(spec)
    sys.modules["plot_module"] = plot_module
    spec.loader.exec_module(plot_module)

    figures_dir = os.path.join(here, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    save_path = os.path.join(figures_dir, "probabilities_four_panel.png")

    panels = []
    for panel_name, x_col, y_col, x_label, y_label, x_integer in PANELS_DEF:
        panels.append({
            "panel_name": panel_name,
            "human_csv": human_csv,
            "sim_csv":   os.path.join(sim_root, best_folder, sim_relpath),
            "x_col": x_col,
            "y_col": y_col,
            "x_label": x_label,
            "y_label": y_label,
            "x_integer": x_integer,
        })

    plot_module.plot_in_a_row(panels, save_path)
    print(f"Saved four-panel figure to: {save_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", type=str, default=HUMAN_DEFAULT)
    ap.add_argument("--sim_root", type=str, default=SIM_ROOT_DEFAULT)
    ap.add_argument("--sim_relpath", type=str, default=SIM_REL_DEFAULT)
    ap.add_argument("--out_csv", type=str, default=None)
    args = ap.parse_args()

    human_csv = os.path.abspath(args.human)
    sim_root  = os.path.abspath(args.sim_root)
    sim_rel   = args.sim_relpath
    out_csv   = args.out_csv or os.path.join(sim_root, "grid_search_w_regression_cost_results.csv")

    rows = []
    for name in sorted(os.listdir(sim_root)):
        if not name.startswith("w_regression_cost_"):
            continue
        sim_csv = os.path.join(sim_root, name, sim_rel)
        if not os.path.exists(sim_csv):
            continue
        scores = score_folder(sim_csv, human_csv)
        w_val = parse_w(name)
        rows.append({"folder": name, "w_regression_cost": w_val, **scores})

    if not rows:
        raise RuntimeError("No simulation folders with the required CSV were found.")

    df = pd.DataFrame(rows).sort_values("F_total", ascending=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    best = df.iloc[0]
    print("Best w_regression_cost:", best["w_regression_cost"], "folder:", best["folder"])
    print("Per-metric:")
    for key, _, _ in CURVES_FOR_SCORE:
        v = best.get(key, np.nan)
        out = "NA" if pd.isna(v) else f"{float(v):.6f}"
        print(f"  {key}: {out}")
    print("F_total:", best["F_total"])

    # record best + draw figures
    note_path = os.path.join(os.path.dirname(__file__), "figures", "best_param.txt")
    os.makedirs(os.path.dirname(note_path), exist_ok=True)
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(f"Best w_regression_cost: {best['w_regression_cost']}\n")
        f.write(f"Best folder: {best['folder']}\n")
        for key, _, _ in CURVES_FOR_SCORE:
            v = best.get(key, np.nan)
            f.write(f"{key}: {v}\n")
        f.write(f"F_total: {best['F_total']}\n")

    post_plot(human_csv, sim_root, best["folder"], sim_rel)

if __name__ == "__main__":
    main()
