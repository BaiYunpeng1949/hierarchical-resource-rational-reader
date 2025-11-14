import os
import numpy as np
import pandas as pd
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
REGRESSION_DASHED = True     # dashed regression line by default
REG_LINESTYLE     = "--" if REGRESSION_DASHED else "-"
SHOW_SCATTER      = True     # show *binned* dots (averaged)
SCATTER_SIZE      = 36       # matplotlib scatter size (points^2)
SCATTER_EDGEWIDTH = 1.0

# Font/size controls (adjust once here)
FONT_SIZE_BASE = 12
TICK_SIZE      = 12
LEGEND_SIZE    = 12

# Tick granularity (set to None to auto, or an int for max # major ticks)
MAX_X_TICKS = 6
MAX_Y_TICKS = 6

# ---- Binning controls ----
# For continuous x, use equal-width bins (per-series). For integer x (e.g., word length),
# aggregate by integer value instead of continuous bins.
BIN_COUNT_CONT = 12
# ----------------------------------

# ---- Per-axes sizing controls ----
# Keep the SAME axes size even if you have 2, 3, or 4 panels.
PANEL_AX_WIDTH_IN   = 3.0
PANEL_AX_HEIGHT_IN  = 3.0
SUBPLOT_WSPACE      = 0.25  # relative spacing between subplots
# ----------------------------------

# Legend placement (inside panel to avoid extra margins)
LEGEND_LOC      = "best"  # e.g., "best", "upper left", etc.

# Output defaults
DEFAULT_SAVE_DIR   = "figures"
REGRESSION_TXTNAME = "regression_stats.txt"

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
    Sxx = np.sum((x - x_mean)**2)
    Sxy = np.sum((x - x_mean)*(y - y_mean))
    b = Sxy / (Sxx if Sxx != 0 else 1e-12)
    a = y_mean - b * x_mean
    y_hat = a + b * x
    # R^2
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y_mean)**2) if np.any(y != y_mean) else 0.0
    r2 = 1.0 - ss_res / (ss_tot if ss_tot != 0 else 1e-12)
    # Residual variance
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

    # 95% CI for the mean prediction
    x_mean = np.mean(x)
    Sxx = np.sum((x - x_mean)**2)
    tcrit = 1.96
    with np.errstate(divide='ignore', invalid='ignore'):
        se_mean = np.sqrt(sigma2 * (1.0/len(x) + (x_line - x_mean)**2 / (Sxx if Sxx != 0 else 1e-12)))
    y_low = y_hat_line - tcrit * se_mean
    y_high = y_hat_line + tcrit * se_mean

    return x_line, y_hat_line, y_low, y_high, (a, b, r2, n)

def _style_axes(ax, force_integer_x=False):
    """Shared style: remove top/right spines, no grid/title, tick controls, integer x optional."""
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # No grid, no title
    ax.grid(False)
    ax.set_title("")

    # Tick granularity
    if MAX_X_TICKS is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=MAX_X_TICKS, prune=None, integer=force_integer_x))
    elif force_integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if MAX_Y_TICKS is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=MAX_Y_TICKS, prune=None))

def _bin_series(x, y, force_integer=False, n_bins=BIN_COUNT_CONT):
    """Return binned x (centers) and averaged y. If integer, group by int(x)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size == 0:
        return np.array([]), np.array([])

    if force_integer:
        # group by integer value of x
        xi = np.round(x).astype(int)
        # sort by integer x
        order = np.argsort(xi)
        xi, y = xi[order], y[order]
        # aggregate
        uniq = np.unique(xi)
        xb = []
        yb = []
        for u in uniq:
            mask = (xi == u)
            if np.any(mask):
                xb.append(float(u))
                yb.append(float(np.mean(y[mask])))
        return np.array(xb, dtype=float), np.array(yb, dtype=float)
    else:
        # equal-width bins across the observed range
        lo, hi = np.min(x), np.max(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return np.array([]), np.array([])
        bins = np.linspace(lo, hi, n_bins + 1)
        xb = []
        yb = []
        for i in range(len(bins) - 1):
            mask = (x >= bins[i]) & (x < bins[i + 1]) if i < len(bins) - 2 else (x >= bins[i]) & (x <= bins[i + 1])
            if np.any(mask):
                xb.append((bins[i] + bins[i + 1]) / 2.0)
                yb.append(np.mean(y[mask]))
        return np.array(xb, dtype=float), np.array(yb, dtype=float)

def _plot_regression_binned(ax, x, y, label, color, force_integer_x=False):
    """
    Bin/average the series to dots, then fit regression and CI on the binned dots.
    Returns a dict with regression stats (based on binned means).
    """
    xb, yb = _bin_series(x, y, force_integer=force_integer_x)
    if xb.size < 2:
        return {"intercept": np.nan, "slope": np.nan, "r2": np.nan, "n": int(xb.size)}

    order = np.argsort(xb)
    xb = xb[order]; yb = yb[order]

    # regression on binned dots
    x_line, y_hat, y_low, y_high, stats = _regress_and_ci(xb, yb)
    a, b, r2, n = stats[0], stats[1], stats[2], stats[3]

    # CI band
    if y_low is not None and y_high is not None:
        ax.fill_between(x_line, y_low, y_high, color=color, alpha=CI_ALPHA)

    # Regression line (dashed as requested)
    ax.plot(x_line, y_hat, REG_LINESTYLE, linewidth=LINE_WIDTH, color=color, label=label)

    # Binned dots
    if SHOW_SCATTER:
        ax.scatter(xb, yb, s=SCATTER_SIZE, facecolor='none', edgecolor=color, linewidth=SCATTER_EDGEWIDTH)

    return {"intercept": a, "slope": b, "r2": r2, "n": n}

def _write_regression_stats(stats_dict, save_dir, base_filename):
    """
    Write regression stats to a text file (tab-separated).
    Rows: series, intercept, slope, r2, n
    """
    _ensure_dir(save_dir)
    out_path = os.path.join(save_dir, f"{base_filename}_regression_stats.txt")
    lines = ["series\tintercept\tslope\tr2\tn"]
    for series in ("human", "simulation"):
        s = stats_dict[series]
        lines.append(f"{series}\t{s['intercept']:.6f}\t{s['slope']:.6f}\t{s['r2']:.6f}\t{s['n']}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path

def plot_comparison(x_human, y_human, x_sim, y_sim, x_name, y_name, output_dir, output_filename, x_integer=None):
    """
    Single-panel comparison plot from in-memory arrays with *binned dots*.
    - Human (blue) vs Simulation (green)
    - Dashed regression + 95% CI, binned dots
    - Legend inside axes, no grid/title, uniform sizes
    - Optionally force integer x ticks with x_integer=True.
      If x_integer is None, auto-detect when x label contains 'length' or values near integers.
    """
    _set_global_fonts()
    _ensure_dir(output_dir)

    # Heuristic for integer x-axis
    force_integer = False
    if x_integer is not None:
        force_integer = bool(x_integer)
    else:
        label_lower = (x_name or "").lower()
        if "length" in label_lower:
            force_integer = True
        else:
            # auto if >90% of values are near integers
            x_all = np.concatenate([np.asarray(x_human).ravel(), np.asarray(x_sim).ravel()])
            if x_all.size:
                near_int = np.mean(np.isclose(x_all, np.round(x_all), atol=1e-6)) >= 0.9
                force_integer = bool(near_int)

    fig, ax = plt.subplots(1, 1, figsize=(PANEL_AX_WIDTH_IN, PANEL_AX_HEIGHT_IN))

    human_stats = _plot_regression_binned(ax, x_human, y_human, label="Human",      color=HUMAN_COLOR, force_integer_x=force_integer)
    sim_stats   = _plot_regression_binned(ax, x_sim,   y_sim,   label="Simulation", color=SIM_COLOR,   force_integer_x=force_integer)

    ax.set_xlabel(x_name)
    if y_name is not None:
        ax.set_ylabel(y_name)

    _style_axes(ax, force_integer_x=force_integer)

    if force_integer:
        # For integer axis, set limits to observed integer range across both series
        x_all = np.concatenate([np.asarray(x_human).ravel(), np.asarray(x_sim).ravel()])
        x_all = x_all[np.isfinite(x_all)]
        if x_all.size:
            xmin, xmax = int(np.floor(x_all.min())), int(np.ceil(x_all.max()))
            ax.set_xlim(xmin, xmax)

    ax.legend(loc=LEGEND_LOC, fontsize=LEGEND_SIZE, frameon=False)

    out_path = os.path.join(output_dir, f"{output_filename}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    _write_regression_stats({"human": human_stats, "simulation": sim_stats}, output_dir, output_filename)

# CSV multi-panel utility
def plot_in_a_row(panels, save_path):
    """
    Generic N-panel plot with consistent per-axes size, reading CSVs internally.
    panels: list of dicts, each with keys:
        - human_csv, sim_csv, x_col, y_col, x_label, y_label
        - (optional) x_integer: bool (force integer x ticks)
    """
    assert len(panels) >= 2, "At least two panels are required."

    _ensure_dir(os.path.dirname(save_path))
    _set_global_fonts()

    n_panels = len(panels)
    fig_width  = PANEL_AX_WIDTH_IN * n_panels
    fig_height = PANEL_AX_HEIGHT_IN
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height), constrained_layout=False)
    if n_panels == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=SUBPLOT_WSPACE)

    def _read_csv(path): return pd.read_csv(path)

    all_stats = []
    for idx, (ax, panel) in enumerate(zip(axes, panels)):
        human_df = _read_csv(panel["human_csv"])
        sim_df   = _read_csv(panel["sim_csv"])

        x_h = human_df[panel["x_col"]].values
        y_h = human_df[panel["y_col"]].values
        x_s = sim_df[panel["x_col"]].values
        y_s = sim_df[panel["y_col"]].values

        force_int = bool(panel.get("x_integer", False))

        human_stats = _plot_regression_binned(ax, x_h, y_h, label="Human",      color=HUMAN_COLOR, force_integer_x=force_int)
        sim_stats   = _plot_regression_binned(ax, x_s, y_s, label="Simulation", color=SIM_COLOR,   force_integer_x=force_int)

        ax.set_xlabel(panel["x_label"])
        if panel.get("y_label") is not None:
            ax.set_ylabel(panel.get("y_label"))

        _style_axes(ax, force_integer_x=force_int)

        if force_int:
            x_all = np.concatenate([x_h, x_s])
            x_all = x_all[np.isfinite(x_all)]
            if x_all.size:
                xmin, xmax = int(np.floor(x_all.min())), int(np.ceil(x_all.max()))
                ax.set_xlim(xmin, xmax)

        if idx == 0:
            ax.legend(loc=LEGEND_LOC, fontsize=LEGEND_SIZE, frameon=False)

        all_stats.append({"human": human_stats, "simulation": sim_stats})

    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # Write a joint regression file per panel
    base = os.path.splitext(os.path.basename(save_path))[0]
    dirn = os.path.dirname(save_path) or "."
    lines = ["panel_index\tpanel_name\tseries\tintercept\tslope\tr2\tn"]
    for i, (stats, panel) in enumerate(zip(all_stats, panels), start=1):
        panel_name = panel.get("panel_name", f"panel_{i}")
        for series in ("human", "simulation"):
            s = stats[series]
            lines.append(f"{i}\t{panel_name}\t{series}\t{s['intercept']:.6f}\t{s['slope']:.6f}\t{s['r2']:.6f}\t{s['n']}")
    with open(os.path.join(dirn, f"{base}_{REGRESSION_TXTNAME}"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def save_panels_separately(panels, output_dir, base_name="panel"):
    """
    Save each panel in `panels` as an individual PDF with identical axes size.
    - Only the first panel shows its y-label; others use a blank placeholder
      to preserve left margin for alignment in Illustrator.
    - Legend is only shown on the first panel (as in plot_in_a_row).
    """
    _ensure_dir(output_dir)
    _set_global_fonts()

    for idx, panel in enumerate(panels, start=1):
        fig, ax = plt.subplots(
            1, 1,
            figsize=(PANEL_AX_WIDTH_IN, PANEL_AX_HEIGHT_IN),
            constrained_layout=False,
        )

        # Read CSVs (same as in plot_in_a_row)
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

        # Axis labels
        ax.set_xlabel(panel["x_label"])

        # First panel: real y-label (if provided)
        # Other panels: placeholder to keep margin identical
        y_lab = panel.get("y_label", None)
        if idx == 1 and y_lab is not None:
            ax.set_ylabel(y_lab)
        elif idx == 4 and y_lab is not None:
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

        out_path = os.path.join(output_dir, f"{base_name}_{idx}.pdf")
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        print(f"Saved: {out_path}")
