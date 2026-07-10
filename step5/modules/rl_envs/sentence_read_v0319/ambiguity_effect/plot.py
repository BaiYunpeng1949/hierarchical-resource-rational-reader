import os
import numpy as np
import pandas as pd

import matplotlib as mpl

# Keep text editable in PDF/EPS outputs for production artwork.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.use14corefonts"] = False
mpl.rcParams["text.usetex"] = False

# Use a production-friendly editable font.
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial"]

# Keep math text editable if any math labels are added later.
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Arial"
mpl.rcParams["mathtext.it"] = "Arial:italic"
mpl.rcParams["mathtext.bf"] = "Arial:bold"

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# =========================
# Universal configuration
# =========================
HUMAN_COLOR = "#1f77b4"   # blue for human
SIM_COLOR   = "#2ca02c"   # green for simulation
CI_ALPHA    = 0.5         # confidence band alpha

# Line/marker styles
LINE_WIDTH        = 2.0
REGRESSION_DASHED = True
REG_LINESTYLE     = "--" if REGRESSION_DASHED else "-"
SHOW_SCATTER      = True
SCATTER_SIZE      = 36
SCATTER_EDGEWIDTH = 1.0

# Font/size controls
FONT_SIZE_BASE = 12
TICK_SIZE      = 12
LEGEND_SIZE    = 12

# Tick granularity
MAX_X_TICKS = 6
MAX_Y_TICKS = 6

# ---- Binning controls ----
BIN_COUNT_CONT = 12
# --------------------------

# ---- Per-axes sizing controls ----
PANEL_AX_WIDTH_IN   = 3.0
PANEL_AX_HEIGHT_IN  = 3.0
# ----------------------------------

# Legend placement
LEGEND_LOC = "best"

# Output
DEFAULT_SAVE_DIR   = "figures"
REGRESSION_TXTNAME = "ambiguity_effect_regression_stats.txt"

# =========================
# Data paths
# =========================
HUMAN_CSV = "data/ambiguity_effect_human_binned.csv"
SIM_CSV   = "data/ambiguity_effect_simulation_binned.csv"


def _ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def _set_global_fonts():
    plt.rcParams.update({'font.size': FONT_SIZE_BASE})
    plt.rc('xtick', labelsize=TICK_SIZE)
    plt.rc('ytick', labelsize=TICK_SIZE)


def _linregress_basic(x, y):
    """Return (a, b, r2, sigma2, n) for y = a + b x"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan, n

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    Sxx = np.sum((x - x_mean) ** 2)
    Sxy = np.sum((x - x_mean) * (y - y_mean))

    b = Sxy / (Sxx if Sxx != 0 else 1e-12)
    a = y_mean - b * x_mean

    y_hat = a + b * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2) if np.any(y != y_mean) else 0.0
    r2 = 1.0 - ss_res / (ss_tot if ss_tot != 0 else 1e-12)

    dof = max(n - 2, 1)
    sigma2 = ss_res / dof
    return a, b, r2, sigma2, n


def _regress_and_ci(x, y, x_smooth=None):
    """
    Linear regression y = a + b x with 95% CI for mean prediction.
    Returns x_line, y_hat, y_low, y_high, (a,b,r2,n)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    a, b, r2, sigma2, n = _linregress_basic(x, y)

    if np.isnan(a):
        order = np.argsort(x)
        return x[order], y[order], None, None, (a, b, r2, len(x))

    if x_smooth is None:
        x_line = np.linspace(np.min(x), np.max(x), 200)
    else:
        x_line = np.asarray(x_smooth, dtype=float)

    y_hat_line = a + b * x_line

    x_mean = np.mean(x)
    Sxx = np.sum((x - x_mean) ** 2)
    tcrit = 1.96

    with np.errstate(divide='ignore', invalid='ignore'):
        se_mean = np.sqrt(
            sigma2 * (1.0 / len(x) + (x_line - x_mean) ** 2 / (Sxx if Sxx != 0 else 1e-12))
        )

    y_low = y_hat_line - tcrit * se_mean
    y_high = y_hat_line + tcrit * se_mean

    return x_line, y_hat_line, y_low, y_high, (a, b, r2, n)


def _style_axes(ax, force_integer_x=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.set_title("")

    if MAX_X_TICKS is not None:
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=MAX_X_TICKS, prune=None, integer=force_integer_x)
        )
    elif force_integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if MAX_Y_TICKS is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=MAX_Y_TICKS, prune=None))


def _bin_series(x, y, force_integer=False, n_bins=BIN_COUNT_CONT):
    """
    Return binned x (centers) and averaged y.
    If already using binned csv, this is effectively a passthrough-like regrouping.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if x.size == 0:
        return np.array([]), np.array([])

    if force_integer:
        xi = np.round(x).astype(int)
        order = np.argsort(xi)
        xi, y = xi[order], y[order]

        uniq = np.unique(xi)
        xb, yb = [], []
        for u in uniq:
            mask = (xi == u)
            xb.append(float(u))
            yb.append(float(np.mean(y[mask])))
        return np.array(xb, dtype=float), np.array(yb, dtype=float)

    lo, hi = np.min(x), np.max(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return np.array([]), np.array([])

    bins = np.linspace(lo, hi, n_bins + 1)
    xb, yb = [], []

    for i in range(len(bins) - 1):
        if i < len(bins) - 2:
            mask = (x >= bins[i]) & (x < bins[i + 1])
        else:
            mask = (x >= bins[i]) & (x <= bins[i + 1])

        if np.any(mask):
            xb.append((bins[i] + bins[i + 1]) / 2.0)
            yb.append(np.mean(y[mask]))

    return np.array(xb, dtype=float), np.array(yb, dtype=float)


def _plot_regression_binned(ax, x, y, label, color, force_integer_x=False):
    """
    Bin/average the series to dots, then fit regression and CI on the binned dots.
    Returns regression stats based on binned means.
    """
    xb, yb = _bin_series(x, y, force_integer=force_integer_x)

    # If the csv is already binned and the second-stage binning produced too few points,
    # fall back to original values.
    if xb.size < 2:
        xb = np.asarray(x, dtype=float)
        yb = np.asarray(y, dtype=float)
        m = np.isfinite(xb) & np.isfinite(yb)
        xb = xb[m]
        yb = yb[m]

    if xb.size < 2:
        return {"intercept": np.nan, "slope": np.nan, "r2": np.nan, "n": int(xb.size)}

    order = np.argsort(xb)
    xb = xb[order]
    yb = yb[order]

    x_line, y_hat, y_low, y_high, stats = _regress_and_ci(xb, yb)
    a, b, r2, n = stats[0], stats[1], stats[2], stats[3]

    if y_low is not None and y_high is not None:
        ax.fill_between(x_line, y_low, y_high, color=color, alpha=CI_ALPHA)

    ax.plot(x_line, y_hat, REG_LINESTYLE, linewidth=LINE_WIDTH, color=color, label=label)

    if SHOW_SCATTER:
        ax.scatter(
            xb, yb,
            s=SCATTER_SIZE,
            facecolor='none',
            edgecolor=color,
            linewidth=SCATTER_EDGEWIDTH
        )

    return {"intercept": a, "slope": b, "r2": r2, "n": n}


def _write_regression_stats(stats_rows, save_dir, filename):
    _ensure_dir(save_dir)
    out_path = os.path.join(save_dir, filename)

    lines = ["panel\tseries\tintercept\tslope\tr2\tn"]
    for row in stats_rows:
        lines.append(
            f"{row['panel']}\t{row['series']}\t"
            f"{row['intercept']:.6f}\t{row['slope']:.6f}\t"
            f"{row['r2']:.6f}\t{row['n']}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved regression stats to: {out_path}")


def save_panels_separately(panels, output_dir, base_name="ambiguity_regression"):
    """
    Save each panel as an individual PDF with identical axes size.
    """
    _ensure_dir(output_dir)
    _set_global_fonts()

    all_stats = []

    for idx, panel in enumerate(panels, start=1):
        fig, ax = plt.subplots(
            1, 1,
            figsize=(PANEL_AX_WIDTH_IN, PANEL_AX_HEIGHT_IN),
            constrained_layout=False,
        )

        human_df = pd.read_csv(panel["human_csv"])
        sim_df   = pd.read_csv(panel["sim_csv"])

        x_h = human_df[panel["x_col"]].values
        y_h = human_df[panel["y_col"]].values
        x_s = sim_df[panel["x_col"]].values
        y_s = sim_df[panel["y_col"]].values

        force_int = bool(panel.get("x_integer", False))

        human_stats = _plot_regression_binned(
            ax, x_h, y_h,
            label="Human",
            color=HUMAN_COLOR,
            force_integer_x=force_int,
        )
        sim_stats = _plot_regression_binned(
            ax, x_s, y_s,
            label="Simulation",
            color=SIM_COLOR,
            force_integer_x=force_int,
        )

        ax.set_xlabel(panel["x_label"])

        y_lab = panel.get("y_label", None)
        if y_lab is not None:
            ax.set_ylabel(y_lab)
        else:
            ax.set_ylabel(" ")

        _style_axes(ax, force_integer_x=force_int)

        if force_int:
            x_all = np.concatenate([x_h, x_s])
            x_all = x_all[np.isfinite(x_all)]
            if x_all.size:
                xmin, xmax = int(np.floor(x_all.min())), int(np.ceil(x_all.max()))
                ax.set_xlim(xmin, xmax)

        # ax.legend(loc=LEGEND_LOC, fontsize=LEGEND_SIZE, frameon=False)

        out_path = os.path.join(output_dir, f"{base_name}_{idx}.pdf")
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        print(f"Saved: {out_path}")

        all_stats.append({
            "panel": panel.get("panel_name", f"panel_{idx}"),
            "series": "human",
            **human_stats
        })
        all_stats.append({
            "panel": panel.get("panel_name", f"panel_{idx}"),
            "series": "simulation",
            **sim_stats
        })

    _write_regression_stats(all_stats, output_dir, REGRESSION_TXTNAME)


def main():
    panels = [
        {
            "panel_name": "ambiguity_vs_regression",
            "human_csv": HUMAN_CSV,
            "sim_csv": SIM_CSV,
            "x_col": "ambiguity_zscore",
            "y_col": "avg_regression_probability",
            "x_label": "Sentence ambiguity",
            "y_label": "",
            "x_integer": False,
        },
        {
            "panel_name": "ambiguity_vs_skip",
            "human_csv": HUMAN_CSV,
            "sim_csv": SIM_CSV,
            "x_col": "ambiguity_zscore",
            "y_col": "avg_skip_probability",
            "x_label": "Sentence ambiguity",
            "y_label": "",
            "x_integer": False,
        },
    ]

    save_panels_separately(
        panels=panels,
        output_dir=DEFAULT_SAVE_DIR,
        base_name="ambiguity_regression"
    )


if __name__ == "__main__":
    main()