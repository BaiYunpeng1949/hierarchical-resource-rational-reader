import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# =========================
# Universal configuration
# =========================
HUMAN_COLOR = "#1f77b4"   # blue for human
SIM_COLOR   = "#2ca02c"   # green for simulation
CI_ALPHA    = 0.5         # confidence band alpha

# Line/marker styles
LINE_WIDTH        = 2.0
REGRESSION_DASHED = True     # dashed regression line per request
REG_LINESTYLE     = "--" if REGRESSION_DASHED else "-"
SHOW_SCATTER      = True     # show original dots
SCATTER_SIZE      = 36       # matplotlib scatter size (points^2)
SCATTER_EDGEWIDTH = 1.0

# Font/size controls (adjust once here)
FONT_SIZE_BASE = 14
TICK_SIZE      = 12
LEGEND_SIZE    = 14

# Tick granularity (set to None to auto, or an int for max # major ticks)
MAX_X_TICKS = 6
MAX_Y_TICKS = 6

# ---- Per-axes sizing controls ----
# Define the desired width/height for each subplot (inches). This lets you keep
# the SAME axes size even if you have 2, 3, or 4 panels.
PANEL_AX_WIDTH_IN   = 6.0
PANEL_AX_HEIGHT_IN  = 4.0
SUBPLOT_WSPACE      = 0.25  # relative spacing between subplots
# ----------------------------------

# Legend placement (inside the first panel to avoid extra margins)
LEGEND_LOC      = "best"  # e.g., "best", "upper left", etc.

# Output dirs
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
    Simple linear regression y = a + b x with 95% CI for mean prediction.
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

def _plot_regression(ax, df, x_col, y_col, label, color):
    """
    Plot regression line (dashed) + CI and raw scatter.
    Returns a dict with regression stats.
    """
    work = df[[x_col, y_col]].dropna().copy()
    work = work.sort_values(by=x_col)
    x = work[x_col].values
    y = work[y_col].values

    x_line, y_hat, y_low, y_high, stats = _regress_and_ci(x, y)
    a, b, r2, n = stats[0], stats[1], stats[2], stats[3]

    # CI band
    if y_low is not None and y_high is not None:
        ax.fill_between(x_line, y_low, y_high, color=color, alpha=CI_ALPHA)

    # Regression line (dashed as requested)
    ax.plot(x_line, y_hat, REG_LINESTYLE, linewidth=LINE_WIDTH, color=color, label=label)

    # Raw scatter
    if SHOW_SCATTER:
        ax.scatter(x, y, s=SCATTER_SIZE, facecolor='none', edgecolor=color, linewidth=SCATTER_EDGEWIDTH)

    return {"intercept": a, "slope": b, "r2": r2, "n": n}

def compare_one(ax, human_csv, sim_csv, x_col, y_col, x_label, y_label, want_legend=False, x_integer=False):
    """
    Draw one panel into 'ax' following the unified style.
    Returns dict with 'human' and 'simulation' regression stats.
    """
    human_df = pd.read_csv(human_csv)
    sim_df   = pd.read_csv(sim_csv)

    human_stats = _plot_regression(ax, human_df, x_col, y_col, label="Human",      color=HUMAN_COLOR)
    sim_stats   = _plot_regression(ax, sim_df,   x_col, y_col, label="Simulation", color=SIM_COLOR)

    ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    _style_axes(ax, force_integer_x=x_integer)

    if x_integer:
        # Ensure integer ticks span the observed domain neatly
        all_x = np.concatenate([human_df[x_col].dropna().values, sim_df[x_col].dropna().values])
        xmin, xmax = int(np.floor(all_x.min())), int(np.ceil(all_x.max()))
        ax.set_xlim(xmin, xmax)

    if want_legend:
        ax.legend(loc=LEGEND_LOC, fontsize=LEGEND_SIZE, frameon=False)

    return {"human": human_stats, "simulation": sim_stats}

def _write_regression_stats(stats_per_panel, save_dir, filename=REGRESSION_TXTNAME):
    """
    Write regression stats to a text file (tab-separated) for easy paper writing.
    Columns: panel_index, series, intercept, slope, r2, n
    """
    _ensure_dir(save_dir)
    out_path = os.path.join(save_dir, filename)
    lines = ["panel_index\tseries\tintercept\tslope\tr2\tn"]
    for i, panel_stats in enumerate(stats_per_panel, start=1):
        for series in ("human", "simulation"):
            s = panel_stats[series]
            lines.append(f"{i}\t{series}\t{s['intercept']:.6f}\t{s['slope']:.6f}\t{s['r2']:.6f}\t{s['n']}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path

def plot_in_a_row(panels, save_path):
    """
    Generic N-panel plot with consistent per-axes size.
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

    all_stats = []
    for idx, (ax, panel) in enumerate(zip(axes, panels)):
        stats = compare_one(
            ax=ax,
            human_csv=panel["human_csv"],
            sim_csv=panel["sim_csv"],
            x_col=panel["x_col"],
            y_col=panel["y_col"],
            x_label=panel["x_label"],
            y_label=panel.get("y_label", None),
            want_legend=(idx == 0),               # legend inside the first subplot
            x_integer=panel.get("x_integer", False),
        )
        all_stats.append(stats)

    # Save figure and stats
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    stats_path = _write_regression_stats(all_stats, os.path.dirname(save_path))
    print(f"Saved regression stats to: {stats_path}")

# Backward-compatible wrapper for 3 panels
def plot_three_in_a_row(panels, save_path):
    assert len(panels) == 3, "Exactly three panels are required."
    return plot_in_a_row(panels, save_path)

if __name__ == "__main__":
    # You can tweak these once here and re-run to affect all figures.
    human_data_dir = "human_data"
    sim_data_dir   = "best_param_simulated_results"
    save_dir       = DEFAULT_SAVE_DIR

    panels = [
        {
            "human_csv": os.path.join(human_data_dir, "gaze_duration_vs_word_length.csv"),
            "sim_csv":   os.path.join(sim_data_dir,   "gaze_duration_vs_word_length.csv"),
            "x_col": "word_length",
            "y_col": "average_gaze_duration",
            "x_label": "Word Length",
            "y_label": "Average Gaze Duration (ms)",
            "x_integer": True,  # <- force integer ticks for word length
        },
        {
            "human_csv": os.path.join(human_data_dir, "gaze_duration_vs_word_log_frequency.csv"),
            "sim_csv":   os.path.join(sim_data_dir,   "gaze_duration_vs_word_log_frequency_binned.csv"),
            "x_col": "log_frequency",
            "y_col": "average_gaze_duration",
            "x_label": "Log Frequency",
            "y_label": "",  # ylabel optional
        },
        {
            "human_csv": os.path.join(human_data_dir, "gaze_duration_vs_word_logit_predictability.csv"),
            "sim_csv":   os.path.join(sim_data_dir,   "gaze_duration_vs_word_logit_predictability_binned.csv"),
            "x_col": "logit_predictability",
            "y_col": "average_gaze_duration",
            "x_label": "Logit Predictability",
            "y_label": "",  # keep empty for consistency
        },
    ]

    _ensure_dir(save_dir)
    out_path = os.path.join(save_dir, "gaze_duration_three_panel.png")
    plot_in_a_row(panels, out_path)
    print(f"Saved figure to: {out_path}")
