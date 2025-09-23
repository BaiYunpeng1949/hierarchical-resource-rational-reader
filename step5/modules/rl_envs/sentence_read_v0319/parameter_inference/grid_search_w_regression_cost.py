# grid_search_w_regression_cost.py (concise, plot-matching)
import os, re, argparse
import numpy as np
import pandas as pd

HUMAN_DEFAULT = os.path.join("human_data", "all_words_regression_and_skip_probabilities.csv")
SIM_ROOT_DEFAULT = os.path.join("simulation_data")

# Curves and the exact columns we’ll use (same as plot.py)
CURVES = [
    ("skip_vs_length",               "length",              "skip_probability"),
    ("skip_vs_logit_predictability", "logit_predictability","skip_probability"),
    ("skip_vs_log_frequency",        "log_frequency",       "skip_probability"),
    ("regression_vs_difficulty",     "difficulty",          "regression_probability"),
]

WEIGHT_SLOPE = 1.0   # you can tweak these if you want
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
    sub = df[[xcol, ycol]].copy()
    sub = sub.dropna()
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
    # exactly like plot.py: coeffs[0]=slope, coeffs[1]=intercept
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
    for key, xcol, ycol in CURVES:
        val = loss_slope_intercept(h, s, xcol, ycol)
        scores[key] = val
        total_parts.append(val)
    total = float(np.nansum(total_parts)) if np.isfinite(np.nansum(total_parts)) else float("inf")
    scores["F_total"] = total
    return scores

def post_plot(human_csv: str, sim_root: str, best_folder: str, sim_relpath: str):
    """Minimal: use plot.py exactly as-is to draw 4 figures + note."""
    try:
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

        human_df = pd.read_csv(human_csv)
        # sim_df = pd.read_csv(os.path.join(here, best_folder, sim_relpath))
        sim_df = pd.read_csv(os.path.join(sim_root, best_folder, sim_relpath))

        def col(df, names):
            for n in names:
                if n in df.columns: return df[n]
                for c in df.columns:
                    if c.lower() == n.lower(): return df[c]
            raise KeyError(str(names))

        # 1) Skip vs length
        plot_module.plot_comparison(
            x_human = col(human_df, ["length","word_length","len","L"]),
            y_human = col(human_df, ["skip_probability","P_skip","skip_prob","prob_skip","p_skip"]),
            x_sim   = col(sim_df,   ["length","word_length","len","L"]),
            y_sim   = col(sim_df,   ["skip_probability","P_skip","skip_prob","prob_skip","p_skip"]),
            x_name="Word Length", y_name="Skip Probability",
            output_dir=figures_dir, output_filename="skip_vs_length"
        )

        # 2) Skip vs logit predictability
        plot_module.plot_comparison(
            x_human = col(human_df, ["logit_predictability","p_logit","logit_pred","lp"]),
            y_human = col(human_df, ["skip_probability","P_skip","skip_prob","prob_skip","p_skip"]),
            x_sim   = col(sim_df,   ["logit_predictability","p_logit","logit_pred","lp"]),
            y_sim   = col(sim_df,   ["skip_probability","P_skip","skip_prob","prob_skip","p_skip"]),
            x_name="Logit Predictability", y_name="Skip Probability",
            output_dir=figures_dir, output_filename="skip_vs_logit_predictability"
        )

        # 3) Skip vs log frequency
        plot_module.plot_comparison(
            x_human = col(human_df, ["log_frequency","logfreq","log_freq","f_log"]),
            y_human = col(human_df, ["skip_probability","P_skip","skip_prob","prob_skip","p_skip"]),
            x_sim   = col(sim_df,   ["log_frequency","logfreq","log_freq","f_log"]),
            y_sim   = col(sim_df,   ["skip_probability","P_skip","skip_prob","prob_skip","p_skip"]),
            x_name="Log Frequency", y_name="Skip Probability",
            output_dir=figures_dir, output_filename="skip_vs_log_frequency"
        )

        # 4) Regression vs difficulty
        plot_module.plot_comparison(
            x_human = col(human_df, ["difficulty","diff","difficulty_bin","d"]),
            y_human = col(human_df, ["regression_probability","reg_prob","prob_reg","p_reg","regression"]),
            x_sim   = col(sim_df,   ["difficulty","diff","difficulty_bin","d"]),
            y_sim   = col(sim_df,   ["regression_probability","reg_prob","prob_reg","p_reg","regression"]),
            x_name="Word Difficulty", y_name="Regression Probability",
            output_dir=figures_dir, output_filename="regression_vs_difficulty"
        )
        print(f"Figures saved to: {figures_dir}")
    except Exception as e:
        print("Auto-plotting failed:", e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", type=str, default=HUMAN_DEFAULT)
    ap.add_argument("--sim_root", type=str, default=SIM_ROOT_DEFAULT)
    ap.add_argument("--sim_relpath", type=str, default="all_words_regression_and_skip_probabilities.csv")
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
    for key, _, _ in CURVES:
        v = best.get(key, np.nan)
        out = "NA" if pd.isna(v) else f"{float(v):.6f}"
        print(f"  {key}: {out}")
    print("F_total:", best["F_total"])

    # record best + draw figures
    figures_dir = os.path.join(os.path.dirname(human_csv), "..", "parameter_inference", "figures")
    try:
        os.makedirs(figures_dir, exist_ok=True)
    except Exception:
        pass
    note_path = os.path.join(os.path.dirname(__file__), "figures", "best_param.txt")
    os.makedirs(os.path.dirname(note_path), exist_ok=True)
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(f"Best w_regression_cost: {best['w_regression_cost']}\n")
        f.write(f"Best folder: {best['folder']}\n")
        for key, _, _ in CURVES:
            v = best.get(key, np.nan)
            f.write(f"{key}: {v}\n")
        f.write(f"F_total: {best['F_total']}\n")

    # post_plot(human_csv, best["folder"], sim_rel)
    post_plot(human_csv, sim_root, best["folder"], sim_rel)


if __name__ == "__main__":
    main()
