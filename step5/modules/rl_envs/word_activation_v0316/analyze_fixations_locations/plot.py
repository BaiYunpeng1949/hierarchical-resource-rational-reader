import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# =========================
# Universal configuration
# =========================
HUMAN_COLOR = "#1f77b4"   # blue for human
SIM_COLOR   = "#2ca02c"   # green for simulation
CI_ALPHA    = 0.5

# Line/marker styles
LINE_WIDTH        = 2.0
REGRESSION_DASHED = True
REG_LINESTYLE     = "-" if REGRESSION_DASHED else "-"
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

# Per-axes sizing
PANEL_AX_WIDTH_IN   = 3.0
PANEL_AX_HEIGHT_IN  = 3.0
SUBPLOT_WSPACE      = 0.25

# Output dirs
DEFAULT_SAVE_DIR = "figures"

# =========================
# Data configuration
# =========================
DATA_DIR = "data"
HUMAN_DIR = os.path.join(DATA_DIR, "human")
SIM_DIR   = os.path.join(DATA_DIR, "simulation")

# Use single-only for both human and simulation fixation-location plots
FIXATION_HUMAN_CSV = os.path.join(HUMAN_DIR, "rayner_forward_fixations_single_only.csv")
FIXATION_SIM_CSV   = os.path.join(SIM_DIR,   "sim_forward_fixations_single_only.csv")

REGRESSION_HUMAN_CSV = os.path.join(HUMAN_DIR, "rayner_regressions_intraword_only.csv")
REGRESSION_SIM_CSV   = os.path.join(SIM_DIR,   "sim_intraword_regressions_only.csv")

FIXATION_MULTIPLE_HUMAN_CSV = os.path.join(HUMAN_DIR, "rayner_forward_fixations_multiple.csv")
FIXATION_MULTIPLE_SIM_CSV   = os.path.join(SIM_DIR,   "sim_forward_fixations_multiple.csv")

ACTION_SIM_CSV = os.path.join(SIM_DIR, "sim_first_fixation_actions.csv")

MCCONKIE_HUMAN_CSV = os.path.join(HUMAN_DIR, "mcconkie_processed.csv")
MCCONKIE_SIM_CSV   = os.path.join(SIM_DIR,   "sim_initial_fixations.csv")

FIXATION_MULTIPLE_SAVE_DIR = os.path.join(DEFAULT_SAVE_DIR, "forward_fixations_multiple")
FIXATION_SAVE_DIR   = os.path.join(DEFAULT_SAVE_DIR, "previewed_fixation_locations")
REGRESSION_SAVE_DIR = os.path.join(DEFAULT_SAVE_DIR, "intraword_regressions")
ACTION_SAVE_DIR = os.path.join(DEFAULT_SAVE_DIR, "first_fixation_actions")
MCCONKIE_SAVE_DIR = os.path.join(DEFAULT_SAVE_DIR, "mcconkie_initial_fixations")
MCCONKIE_WORD_LENGTHS_TO_PLOT = [3, 4, 5, 6, 7, 8]

WORD_LENGTHS_TO_PLOT = [5, 6, 7, 8, 9]


def _ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def _set_global_fonts():
    plt.rcParams.update({'font.size': FONT_SIZE_BASE})
    plt.rc('xtick', labelsize=TICK_SIZE)
    plt.rc('ytick', labelsize=TICK_SIZE)


def _style_axes(ax, force_integer_x=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.set_title("")

    if MAX_X_TICKS is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=MAX_X_TICKS, prune=None, integer=force_integer_x))
    elif force_integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if MAX_Y_TICKS is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=MAX_Y_TICKS, prune=None))


def _normalize_human_units_if_needed(df, y_col):
    """
    Human CSVs are sometimes stored in percent units (0-100),
    whereas simulation CSVs are already in probability units (0-1).
    If the column looks like percentages, convert to 0-1.
    """
    df = df.copy()
    if not df.empty and df[y_col].max() > 1.0:
        df[y_col] = df[y_col] / 100.0
    return df


def _trim_trailing_zeros(df, y_col):
    values = df[y_col].values
    nonzero_idx = [i for i, v in enumerate(values) if v > 1e-8]

    if not nonzero_idx:
        return df.iloc[:1]  # keep at least one point

    last_idx = max(nonzero_idx)
    return df.iloc[:last_idx + 1]


def _snap_to_zero(values, eps=1e-4):
    return [0.0 if abs(v) < eps else v for v in values]


def _load_and_filter(csv_path, word_length, y_col, is_human=False):
    df = pd.read_csv(csv_path)
    if "word_length" not in df.columns or "letter_number" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'word_length' and 'letter_number' columns.")
    if y_col not in df.columns:
        raise ValueError(f"{csv_path} must contain '{y_col}'.")

    df = df[df["word_length"] == word_length].copy()
    df = df.sort_values("letter_number").reset_index(drop=True)

    if is_human:
        df = _normalize_human_units_if_needed(df, y_col)

    return df


def _load_action_data(csv_path, word_length):
    df = pd.read_csv(csv_path)

    if "word_length" not in df.columns or "action" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'word_length' and 'action' columns.")

    df = df[df["word_length"] == word_length].copy()
    df = df.sort_values("action").reset_index(drop=True)

    return df


def _plot_series(ax, df, y_col, color, linestyle="-"):
    y_vals = _snap_to_zero(df[y_col].values)

    ax.plot(
        df["letter_number"].values,
        y_vals,
        linestyle=linestyle,
        linewidth=LINE_WIDTH,
        color=color,
    )
    if SHOW_SCATTER:
        ax.scatter(
            df["letter_number"].values,
            y_vals,
            s=SCATTER_SIZE,
            facecolor="none",
            edgecolor=color,
            linewidth=SCATTER_EDGEWIDTH,
        )


def _plot_word_length_comparison(
    human_csv,
    sim_csv,
    y_col,
    word_length,
    y_label,
    save_path,
    sim_linestyle="-"
):
    human_df = _load_and_filter(human_csv, word_length, y_col=y_col, is_human=True)
    sim_df   = _load_and_filter(sim_csv, word_length, y_col=y_col, is_human=False)

    if human_df.empty and sim_df.empty:
        print(f"[Skip] word_length={word_length} (both empty)")
        return

    _ensure_dir(os.path.dirname(save_path))
    _set_global_fonts()

    fig, ax = plt.subplots(
        1, 1,
        figsize=(PANEL_AX_WIDTH_IN, PANEL_AX_HEIGHT_IN),
        constrained_layout=False
    )

    # Plotting
    if not human_df.empty:
        _plot_series(ax, human_df, y_col=y_col, color=HUMAN_COLOR, linestyle="-")

    if not sim_df.empty:
        _plot_series(ax, sim_df, y_col=y_col, color=SIM_COLOR, linestyle=sim_linestyle)

    ax.set_xlabel("Letter position")
    ax.set_ylabel(y_label)

    _style_axes(ax, force_integer_x=True)

    x_candidates = []

    if not human_df.empty:
        x_candidates.append(int(human_df["letter_number"].max()))
    if not sim_df.empty:
        x_candidates.append(int(sim_df["letter_number"].max()))

    xmax = max(x_candidates)
    ax.set_xlim(0, xmax)
    ax.set_ylim(bottom=0)

    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_previewed_fixation_locations():
    for word_length in WORD_LENGTHS_TO_PLOT:
        out_path = os.path.join(
            FIXATION_SAVE_DIR,
            f"previewed_fixation_location_len{word_length}.pdf"
        )
        _plot_word_length_comparison(
            human_csv=FIXATION_HUMAN_CSV,
            sim_csv=FIXATION_SIM_CSV,
            y_col="proportion_of_fixation",
            word_length=word_length,
            y_label="Proportion of first fixation",
            save_path=out_path,
            sim_linestyle="-",
        )


def plot_intraword_regressions():
    for word_length in WORD_LENGTHS_TO_PLOT:
        out_path = os.path.join(
            REGRESSION_SAVE_DIR,
            f"intraword_regression_len{word_length}.pdf"
        )
        _plot_word_length_comparison(
            human_csv=REGRESSION_HUMAN_CSV,
            sim_csv=REGRESSION_SIM_CSV,
            y_col="probability_of_regression",
            word_length=word_length,
            y_label="Probability of regressions",
            save_path=out_path,
            sim_linestyle=REG_LINESTYLE,
        )


def _plot_action_distribution(
    sim_csv,
    word_length,
    save_path
):
    sim_df = _load_action_data(sim_csv, word_length)

    if sim_df.empty:
        print(f"[Skip] word_length={word_length} (empty)")
        return

    _ensure_dir(os.path.dirname(save_path))
    _set_global_fonts()

    fig, ax = plt.subplots(
        1, 1,
        figsize=(PANEL_AX_WIDTH_IN, PANEL_AX_HEIGHT_IN),
        constrained_layout=False
    )

    y_vals = _snap_to_zero(sim_df["proportion_of_action"].values)

    # line
    ax.plot(
        sim_df["action"].values,
        y_vals,
        linestyle="-",
        linewidth=LINE_WIDTH,
        color=SIM_COLOR,
    )

    # scatter
    if SHOW_SCATTER:
        ax.scatter(
            sim_df["action"].values,
            y_vals,
            s=SCATTER_SIZE,
            facecolor="none",
            edgecolor=SIM_COLOR,
            linewidth=SCATTER_EDGEWIDTH,
        )

    ax.set_xlabel("Action (0=beginning, 3=ending)")
    ax.set_ylabel("Proportion")

    _style_axes(ax, force_integer_x=True)

    ax.set_xlim(0, 4)
    ax.set_ylim(bottom=0)

    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"Saved: {save_path}")


def plot_first_fixation_actions():
    for word_length in WORD_LENGTHS_TO_PLOT:
        out_path = os.path.join(
            ACTION_SAVE_DIR,
            f"first_fixation_action_len{word_length}.pdf"
        )

        _plot_action_distribution(
            sim_csv=ACTION_SIM_CSV,
            word_length=word_length,
            save_path=out_path,
        )

def plot_forward_fixations_multiple():
    for word_length in WORD_LENGTHS_TO_PLOT:
        out_path = os.path.join(
            FIXATION_MULTIPLE_SAVE_DIR,
            f"forward_fixation_multiple_len{word_length}.pdf"
        )
        _plot_word_length_comparison(
            human_csv=FIXATION_MULTIPLE_HUMAN_CSV,
            sim_csv=FIXATION_MULTIPLE_SIM_CSV,
            y_col="proportion_of_fixation",
            word_length=word_length,
            y_label="Proportion of forward fixation",
            save_path=out_path,
            sim_linestyle="-",
        )

def plot_mcconkie_initial_fixations():
    for word_length in MCCONKIE_WORD_LENGTHS_TO_PLOT:
        out_path = os.path.join(
            MCCONKIE_SAVE_DIR,
            f"mcconkie_initial_fixation_len{word_length}.pdf"
        )
        _plot_word_length_comparison(
            human_csv=MCCONKIE_HUMAN_CSV,
            sim_csv=MCCONKIE_SIM_CSV,
            y_col="proportion_of_fixation",
            word_length=word_length,
            y_label="Proportion of initial fixation",
            save_path=out_path,
            sim_linestyle="-",
        )



if __name__ == "__main__":
    plot_previewed_fixation_locations()
    plot_forward_fixations_multiple()
    plot_intraword_regressions()
    plot_first_fixation_actions()
    plot_mcconkie_initial_fixations()