import os, re, math, argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

HUMAN_DEFAULT = os.path.join("human_data", "all_words_regression_and_skip_probabilities.csv")
SIM_ROOT_DEFAULT = os.path.join("simulation_data")

TARGETS = [
    ("skip", "length"),
    ("skip", "logit_predictability"),
    ("skip", "log_frequency"),
    ("regression", "difficulty"),
]

SIGMA0 = 1e-3  # stability floor

def _parse_w_from_folder(name: str) -> Optional[float]:
    m = re.search(r"w_regression_cost_([0-9]+)p([0-9]+)", name)
    if not m:
        m2 = re.search(r"w_regression_cost_([0-9\\.]+)", name)
        if m2:
            try:
                return float(m2.group(1))
            except Exception:
                return None
        return None
    try:
        return float(f"{m.group(1)}.{m.group(2)}")
    except Exception:
        return None

def _first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in low:
            return low[k]
    return None

def _any_cols(df: pd.DataFrame, prefixes: List[str], suffix: str) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for p in prefixes:
        name = f"{p}_{suffix}".lower()
        for lc, orig in low.items():
            if lc == name:
                return orig
    # try loose contains (e.g., "skip_prob_length")
    for p in prefixes:
        for c in df.columns:
            lc = c.lower()
            if p in lc and suffix in lc and ("se" not in lc or suffix=="se") and ("n" not in lc or suffix=="n"):
                return c
    return None

def _normalize_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flexible parser for our expected file:
    - One table that contains multiple 'condition' axes as separate columns:
      e.g., columns may include any subset of: length, logit_predictability, log_frequency, difficulty.
      In each row, only one of these columns is non-null/non-empty indicating which condition that row belongs to.
    - For each condition, we expect skip and/or regression probability columns, optionally with _se/_n.
    """
    df2 = df.copy()
    # Identify condition columns present
    cond_cols = {
        "length": _first_col(df2, ["length", "word_length", "l", "len", "len_bin"]),
        "logit_predictability": _first_col(df2, ["logit_predictability", "lp", "logit_pred", "p_logit"]),
        "log_frequency": _first_col(df2, ["log_frequency", "logfreq", "log_freq", "f_log", "freq_log"]),
        "difficulty": _first_col(df2, ["difficulty", "d", "diff", "difficulty_bin"]),
    }
    have_any_cond = any(v is not None for v in cond_cols.values())
    if not have_any_cond:
        # Try the previous "already long" pathway
        cols = {c.lower(): c for c in df2.columns}
        have = all(k in cols for k in ["metric","condition","x","mu"])
        if have:
            out = pd.DataFrame({
                "metric": df2[cols["metric"]].astype(str).str.lower(),
                "condition": df2[cols["condition"]].astype(str).str.lower(),
                "x": pd.to_numeric(df2[cols["x"]], errors="coerce"),
                "mu": pd.to_numeric(df2[cols["mu"]], errors="coerce"),
            })
            out["se"] = pd.to_numeric(df2[cols["se"]], errors="coerce") if "se" in cols else np.nan
            out["n"]  = pd.to_numeric(df2[cols["n"]],  errors="coerce") if "n"  in cols else np.nan
            return out
        raise ValueError("No recognizable condition columns (length/logit_predictability/log_frequency/difficulty).")

    # For each condition column that exists, subset rows where it's non-null and build long records
    records = []
    skip_prefixes = ["skip", "p_skip", "skip_prob", "prob_skip"]
    reg_prefixes  = ["regression", "reg", "p_reg", "reg_prob", "prob_reg", "regression_prob"]

    for cond, xcol in cond_cols.items():
        if xcol is None:
            continue
        sub = df2.copy()
        # keep rows where this condition has a value (others may be NaN)
        mask = sub[xcol].notna()
        # sometimes blanks/empty strings
        if sub[xcol].dtype == object:
            mask = mask & sub[xcol].astype(str).str.strip().ne("")
        sub = sub.loc[mask].copy()
        if sub.empty:
            continue

        # parse x
        sub["__x__"] = pd.to_numeric(sub[xcol], errors="coerce")

        # find y/se/n for skip
        y_skip = _any_cols(sub, skip_prefixes, "prob") or _any_cols(sub, skip_prefixes, "p") or _any_cols(sub, skip_prefixes, "mean") or _any_cols(sub, skip_prefixes, "mu")
        se_skip = _any_cols(sub, skip_prefixes, "se")
        n_skip  = _any_cols(sub, skip_prefixes, "n")

        # if y_skip is not None:
        #     part = pd.DataFrame({
        #         "metric": "skip",
        #         "condition": cond,
        #         "x": sub["__x__"],
        #         "mu": pd.to_numeric(sub[y_skip], errors="coerce"),
        #     })
        #     part["se"] = pd.to_numeric(sub[se_skip], errors="coerce") if se_skip and se_skip in sub.columns else np.nan
        #     part["n"]  = pd.to_numeric(sub[n_skip],  errors="coerce") if n_skip  and n_skip  in sub.columns else np.nan
        #     records.append(part)

        if y_skip is not None:
            sub_skip = sub[sub[y_skip].notna()].copy()
            if not sub_skip.empty:
                part = pd.DataFrame({
                    "metric": "skip",
                    "condition": cond,
                    "x": sub_skip["__x__"],
                    "mu": pd.to_numeric(sub_skip[y_skip], errors="coerce"),
                })
                part["se"] = pd.to_numeric(sub_skip[se_skip], errors="coerce") if se_skip and se_skip in sub_skip.columns else np.nan
                part["n"]  = pd.to_numeric(sub_skip[n_skip],  errors="coerce") if n_skip  and n_skip  in sub_skip.columns else np.nan
                records.append(part)

        # find y/se/n for regression
        y_reg = _any_cols(sub, reg_prefixes, "prob") or _any_cols(sub, reg_prefixes, "p") or _any_cols(sub, reg_prefixes, "mean") or _any_cols(sub, reg_prefixes, "mu")
        se_reg = _any_cols(sub, reg_prefixes, "se")
        n_reg  = _any_cols(sub, reg_prefixes, "n")

        # if y_reg is not None:
        #     part = pd.DataFrame({
        #         "metric": "regression",
        #         "condition": cond,
        #         "x": sub["__x__"],
        #         "mu": pd.to_numeric(sub[y_reg], errors="coerce"),
        #     })
        #     part["se"] = pd.to_numeric(sub[se_reg], errors="coerce") if se_reg and se_reg in sub.columns else np.nan
        #     part["n"]  = pd.to_numeric(sub[n_reg],  errors="coerce") if n_reg  and n_reg  in sub.columns else np.nan
        #     records.append(part)
        if y_reg is not None:
            sub_reg = sub[sub[y_reg].notna()].copy()
            if not sub_reg.empty:
                part = pd.DataFrame({
                    "metric": "regression",
                    "condition": cond,
                    "x": sub_reg["__x__"],
                    "mu": pd.to_numeric(sub_reg[y_reg], errors="coerce"),
                })
                part["se"] = pd.to_numeric(sub_reg[se_reg], errors="coerce") if se_reg and se_reg in sub_reg.columns else np.nan
                part["n"]  = pd.to_numeric(sub_reg[n_reg],  errors="coerce") if n_reg  and n_reg  in sub_reg.columns else np.nan
                records.append(part)

    if not records:
        raise ValueError("Found condition columns but could not locate skip/regression y columns.")

    return pd.concat(records, ignore_index=True)

def _load_long(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    long = _normalize_long(df)
    long["metric"] = long["metric"].str.lower().str.strip()
    long["condition"] = long["condition"].str.lower().str.strip()
    mask = False
    for m, c in TARGETS:
        mask = mask | ((long["metric"] == m) & (long["condition"] == c))
    long = long.loc[mask].copy()
    return long

def _reduced_chi2(h: pd.DataFrame, s: pd.DataFrame) -> float:
    h = h.dropna(subset=["x","mu"]).copy()
    s = s.dropna(subset=["x","mu"]).copy()
    if h.empty or s.empty:
        # return float("inf")
        return np.nan

    xs_h = np.array(sorted(h["x"].unique()))
    xs_s = np.array(sorted(s["x"].unique()))
    xs = np.intersect1d(xs_h, xs_s)
    if xs.size == 0:
        lo = max(min(xs_h), min(xs_s))
        hi = min(max(xs_h), max(xs_s))
        if not (lo < hi):
            # return float("inf")
            return np.nan
        xs = np.array(sorted(x for x in h["x"].unique() if (x >= lo and x <= hi)))
        if xs.size == 0:
            # return float("inf")
            return np.nan
        s_sorted = s.sort_values("x")
        s_mu = np.interp(xs, s_sorted["x"].values, s_sorted["mu"].values)
        if s_sorted["se"].notna().any():
            s_se = np.interp(xs, s_sorted["x"].values, s_sorted["se"].fillna(method="ffill").fillna(method="bfill").values)
        else:
            s_se = np.full_like(s_mu, np.nan, dtype=float)
        h_use = h[h["x"].isin(xs)].sort_values("x")
        s_use = pd.DataFrame({"x": xs, "mu": s_mu, "se": s_se})
    else:
        h_use = h[h["x"].isin(xs)].sort_values("x")
        s_use = s[s["x"].isin(xs)].sort_values("x")

    n = min(len(h_use), len(s_use))
    h_use = h_use.iloc[:n]
    s_use = s_use.iloc[:n]

    h_se = h_use["se"].fillna(0.0).values
    s_se = s_use["se"].fillna(0.0).values
    denom = (h_se**2 + s_se**2 + (SIGMA0**2))
    numer = (s_use["mu"].values - h_use["mu"].values)**2
    chi = np.mean(numer / denom)
    return float(chi)

def _bin_curve(df: pd.DataFrame, m: str, c: str, edges: np.ndarray) -> pd.DataFrame:
    """
    Bin a long-format df (one metric/condition) onto given bin edges.
    Returns columns: x (bin center), mu (mean in bin), se (binomial or sample SE), n (count).
    """
    sub = df[(df["metric"] == m) & (df["condition"] == c)].dropna(subset=["x","mu"]).copy()
    if sub.empty:
        return pd.DataFrame(columns=["x","mu","se","n"])

    # bin assignment
    bins = pd.cut(sub["x"], bins=edges, include_lowest=True, right=False)
    grp = sub.groupby(bins, observed=True)

    out = grp["mu"].agg(["mean","std","count"]).reset_index(drop=True)
    if out.empty:
        return pd.DataFrame(columns=["x","mu","se","n"])

    # bin centers
    # centers = 0.5 * (edges[:-1] + edges[-1:0:-1])  # WRONG DIRECTION
    centers = 0.5 * (edges[:-1] + edges[1:])

    out["x"] = centers[:len(out)]

    # SE: prefer binomial using count if mean in [0,1]; else sample SE
    p = out["mean"].values
    n = out["count"].values.astype(float)
    se_binom = np.sqrt(np.clip(p,0,1) * np.clip(1-p,0,1) / np.clip(n, 1, None))
    se_sample = (out["std"] / np.sqrt(np.clip(n, 1, None))).fillna(0.0).values
    # choose binomial where p is within [0,1] and finite; else sample
    se = np.where(np.isfinite(se_binom), se_binom, se_sample)

    return pd.DataFrame({
        "x": out["x"].values,
        "mu": out["mean"].values,
        "se": se,
        "n": out["count"].values.astype(int),
    })

def _chi2_on_binned(h: pd.DataFrame, s: pd.DataFrame) -> float:
    """
    h and s are already binned on the SAME edges (same 'x' centers).
    Returns reduced chi-square (mean over bins) with stability floor.
    """
    if h.empty or s.empty:
        return np.nan
    # align by exact x centers (they should match)
    xs = np.intersect1d(np.unique(h["x"].values), np.unique(s["x"].values))
    if xs.size == 0:
        return np.nan
    h_use = h[h["x"].isin(xs)].sort_values("x")
    s_use = s[s["x"].isin(xs)].sort_values("x")
    # protect lengths
    n = min(len(h_use), len(s_use))
    if n == 0:
        return np.nan
    h_use = h_use.iloc[:n]
    s_use = s_use.iloc[:n]

    denom = (h_use["se"].fillna(0.0).values**2 +
             s_use["se"].fillna(0.0).values**2 +
             (SIGMA0**2))
    numer = (s_use["mu"].values - h_use["mu"].values)**2
    return float(np.mean(numer / denom))

def score_folder(sim_csv: str, human_csv: str) -> dict:
    h_raw = _load_long(human_csv)
    s_raw = _load_long(sim_csv)

    per_metric = {}
    total_vals = []

    for m, c in TARGETS:
        # Get overlap range and shared edges (follow plot: ~12 bins is typical)
        h_mc = h_raw[(h_raw["metric"] == m) & (h_raw["condition"] == c)].dropna(subset=["x","mu"])
        s_mc = s_raw[(s_raw["metric"] == m) & (s_raw["condition"] == c)].dropna(subset=["x","mu"])
        if h_mc.empty or s_mc.empty:
            chi = np.nan
        else:
            lo = max(h_mc["x"].min(), s_mc["x"].min())
            hi = min(h_mc["x"].max(), s_mc["x"].max())
            if not (lo < hi):
                chi = np.nan
            else:
                n_bins = 12  # match your plot binning density
                edges = np.linspace(lo, hi, n_bins + 1)

                h_b = _bin_curve(h_raw, m, c, edges)
                s_b = _bin_curve(s_raw, m, c, edges)

                chi = _chi2_on_binned(h_b, s_b)

        per_metric[f"{m}_vs_{c}"] = chi
        total_vals.append(chi)

    total = float(np.nansum(total_vals)) if np.isfinite(np.nansum(total_vals)) else float("inf")
    per_metric["F_total"] = total
    return per_metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", type=str, default=HUMAN_DEFAULT)
    ap.add_argument("--sim_root", type=str, default=SIM_ROOT_DEFAULT)
    ap.add_argument("--sim_relpath", type=str, default="all_words_regression_and_skip_probabilities.csv",
                    help="CSV file inside each parameter folder")
    ap.add_argument("--out_csv", type=str, default=None)
    args = ap.parse_args()

    human_csv = os.path.abspath(args.human)
    sim_root = os.path.abspath(args.sim_root)
    sim_rel = args.sim_relpath
    out_csv = args.out_csv or os.path.join(sim_root, "grid_search_w_regression_cost_results.csv")

    if not os.path.exists(human_csv):
        raise FileNotFoundError(human_csv)
    if not os.path.isdir(sim_root):
        raise NotADirectoryError(sim_root)

    rows = []
    for name in sorted(os.listdir(sim_root)):
        if not name.startswith("w_regression_cost_"):
            continue
        folder = os.path.join(sim_root, name)
        if not os.path.isdir(folder):
            continue
        sim_csv = os.path.join(folder, sim_rel)
        if not os.path.exists(sim_csv):
            continue

        scores = score_folder(sim_csv, human_csv)
        w_val = _parse_w_from_folder(name)
        row = {"folder": name, "w_regression_cost": w_val}
        row.update(scores)
        rows.append(row)

    if not rows:
        raise RuntimeError("No simulation folders with the required CSV were found.")

    df = pd.DataFrame(rows)
    df = df.sort_values("F_total", ascending=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    best = df.iloc[0]
    print("Best w_regression_cost:", best["w_regression_cost"], "folder:", best["folder"])
    print("Per-metric:")
    for m, c in TARGETS:
        print(f"  {m}_vs_{c}: {best[f'{m}_vs_{c}']:.6f}")
    print("F_total:", best["F_total"])

    # Generate figures and record best param
    post_plot(human_csv, sim_root, sim_rel, df)

def post_plot(human_csv: str, sim_root: str, sim_relpath: str, df: pd.DataFrame):
    """Generate figures and record best param under ./figures using plot.py."""
    if df is None or df.empty:
        print("No results to plot; skipping figures.")
        return

    best = df.sort_values("F_total", ascending=True).iloc[0]

    try:
        import importlib.util, sys
        here = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(here, "plot.py")
        if not os.path.exists(plot_path):
            print(f"plot.py not found at {plot_path}; skipping figures.")
            return

        spec = importlib.util.spec_from_file_location("plot_module", plot_path)
        plot_module = importlib.util.module_from_spec(spec)
        sys.modules["plot_module"] = plot_module
        spec.loader.exec_module(plot_module)

        figures_dir = os.path.join(here, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        # Write summary
        note_path = os.path.join(figures_dir, "best_param.txt")
        with open(note_path, "w", encoding="utf-8") as f:
            f.write(f"Best w_regression_cost: {best['w_regression_cost']}\n")
            f.write(f"Best folder: {best['folder']}\n")
            f.write("Per-metric reduced chi-square (NaN = missing/non-overlap):\n")
            for col in df.columns:
                if col not in ("folder","w_regression_cost"):
                    f.write(f"  {col}: {best[col]}\n")
            f.write(f"F_total: {best['F_total']}\n")

        # Load CSVs
        human_df = pd.read_csv(human_csv)
        best_sim_csv = os.path.join(sim_root, best["folder"], sim_relpath)
        sim_df = pd.read_csv(best_sim_csv)

        def col(dframe, names):
            for n in names:
                if n in dframe.columns:
                    return dframe[n]
                for c in dframe.columns:
                    if c.lower() == n.lower():
                        return dframe[c]
            raise KeyError(str(names))

        # 1) Skip vs length
        try:
            plot_module.plot_comparison(
                x_human = col(human_df, ["length","word_length","len","L"]),
                y_human = col(human_df, ["skip_probability","P_skip","skip_prob","prob_skip","p_skip"]),
                x_sim   = col(sim_df,   ["length","word_length","len","L"]),
                y_sim   = col(sim_df,   ["skip_probability","P_skip","skip_prob","prob_skip","p_skip"]),
                x_name="Word Length", y_name="Skip Probability",
                output_dir=figures_dir, output_filename="skip_vs_length"
            )
        except Exception as e:
            print("Plot skip_vs_length failed:", e)

        # 2) Skip vs logit predictability
        try:
            plot_module.plot_comparison(
                x_human = col(human_df, ["logit_predictability","p_logit","logit_pred","lp"]),
                y_human = col(human_df, ["skip_probability","P_skip","skip_prob","prob_skip","p_skip"]),
                x_sim   = col(sim_df,   ["logit_predictability","p_logit","logit_pred","lp"]),
                y_sim   = col(sim_df,   ["skip_probability","P_skip","skip_prob","prob_skip","p_skip"]),
                x_name="Logit Predictability", y_name="Skip Probability",
                output_dir=figures_dir, output_filename="skip_vs_logit_predictability"
            )
        except Exception as e:
            print("Plot skip_vs_logit_predictability failed:", e)

        # 3) Skip vs log frequency
        try:
            plot_module.plot_comparison(
                x_human = col(human_df, ["log_frequency","logfreq","log_freq","f_log"]),
                y_human = col(human_df, ["skip_probability","P_skip","skip_prob","prob_skip","p_skip"]),
                x_sim   = col(sim_df,   ["log_frequency","logfreq","log_freq","f_log"]),
                y_sim   = col(sim_df,   ["skip_probability","P_skip","skip_prob","prob_skip","p_skip"]),
                x_name="Log Frequency", y_name="Skip Probability",
                output_dir=figures_dir, output_filename="skip_vs_log_frequency"
            )
        except Exception as e:
            print("Plot skip_vs_log_frequency failed:", e)

        # 4) Regression vs difficulty
        try:
            plot_module.plot_comparison(
                x_human = col(human_df, ["difficulty","diff","difficulty_bin","d"]),
                y_human = col(human_df, ["regression_probability","reg_prob","prob_reg","p_reg","regression"]),
                x_sim   = col(sim_df,   ["difficulty","diff","difficulty_bin","d"]),
                y_sim   = col(sim_df,   ["regression_probability","reg_prob","prob_reg","p_reg","regression"]),
                x_name="Word Difficulty", y_name="Regression Probability",
                output_dir=figures_dir, output_filename="regression_vs_difficulty"
            )
        except Exception as e:
            print("Plot regression_vs_difficulty failed:", e)

        print(f"Figures saved to: {figures_dir}")
        print(f"Best parameter recorded at: {note_path}")
    except Exception as e:
        print("Auto-plotting failed:", e)



if __name__ == "__main__":
    main()
