#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_eye_comp_and_baselines_from_aggregated_metrics_dot.py

Dot-distribution version of plot_eye_comp_and_baselines_from_aggregated_metrics.py.
Keeps the original figure layout, axes size, colours, font sizes, labels, and output format,
but replaces bars with point distributions plus mean ± SD markers.

Outputs:
    comparison_panel_baselines_clipped_dot.pdf
    comparison_panel_baselines_dot_inferential_stats.csv
    comparison_panel_baselines_dot_agreement_stats.csv
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from scipy import stats

# ================== House style ==================
HUMAN_COLOR = "#1f77b4"     # blue
SIM_COLOR   = "#2ca02c"     # green

BASELINE_FILL   = "#bdbdbd" # grey fill
BASELINE_EDGE   = SIM_COLOR # green edge
BASELINE_HATCHES = ["/", "\\", ".", "xx", "oo", "++", "--"]  # retained for compatibility
BASELINE_MARKERS = ["s", "^", "D", "v", "P", "X", "h"]

# For bounded metrics, draw a tiny mean marker when mean==0 so the point is visible
MIN_VISIBLE_BAR = 0.005   # retained name for compatibility with the original script

FONT_SIZE_BASE = 12
TICK_SIZE      = 12
LEGEND_SIZE    = 12

BAR_GROUP_WIDTH = 0.80
BAR_LINEWIDTH   = 1.2
BAR_CAPSIZE     = 3

# Per-axes size (inches)
AX_W_IN = 5.0
AX_H_IN = 3.0

# Absolute gaps (inches)
H_GAP_IN = 0.8   # horizontal gap between columns
V_GAP_IN = 0.8   # vertical gap between rows

# Legend target panel (we'll put legend in bottom-right blank slot)
LEGEND_LOC = "upper left"
LEGEND_BBOX = (0.0, 1.0)    # anchor inside the legend panel

# Bounded-metric handling
CLIP_BOUNDED_ERR = True
BOUNDED_METRICS = {"skip_rate", "regression_rate", "mcq_accuracy", "free_recall_score"}

# Dot/summary style, matched to the separate dot-plot template
POINT_SIZE = 20
POINT_ALPHA = 0.30
MEAN_MARKER_SIZE = 8
JITTER_WIDTH = 0.035
RNG_SEED = 7

# Files
AGG_MAIN_PATH = Path("assets/aggregated_panel_metrics.json")
AGG_BASE_PATH = Path("assets/aggregated_panel_metrics_baseline.json")
OUT_PDF       = Path("comparison_panel_baselines_clipped_dot.pdf")
OUT_STATS     = Path("comparison_panel_baselines_dot_inferential_stats.csv")
OUT_AGREE     = Path("comparison_panel_baselines_dot_agreement_stats.csv")

# ================== Helpers ==================
def _set_fonts():
    plt.rcParams.update({'font.size': FONT_SIZE_BASE})
    plt.rc('xtick', labelsize=TICK_SIZE)
    plt.rc('ytick', labelsize=TICK_SIZE)


def _style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.set_title("")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def _load_json_with_fallback(primary_path, fallback_name):
    candidates = [
        primary_path,
        Path(fallback_name),
        Path(__file__).with_name(fallback_name),
    ]
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Could not find {fallback_name}; tried: " + ", ".join(str(p) for p in candidates))


def bounded_yerr(means, stds, lo=0.0, hi=1.0):
    """Asymmetric yerr clipped to [lo, hi]."""
    m = np.array(means, dtype=float)
    s = np.array(stds, dtype=float)
    lower = np.minimum(s, m - lo)   # don't extend below lo
    upper = np.minimum(s, hi - m)   # don't extend above hi
    lower = np.clip(lower, 0, None)
    upper = np.clip(upper, 0, None)
    return [lower, upper]


def _series_positions(centers, n_series):
    """Keep the original bar-cluster geometry, but return point centers."""
    slot_w = BAR_GROUP_WIDTH / n_series
    return [np.array([c - BAR_GROUP_WIDTH/2 + (i + 0.5)*slot_w for c in centers], dtype=float)
            for i in range(n_series)]


def _jittered_x(x, n, width, rng):
    if n <= 1:
        return np.array([x], dtype=float)
    return x + rng.uniform(-width, width, size=n)


def _clean_values(block):
    vals = block.get("values", [])
    arr = np.array(vals, dtype=float)
    return arr[np.isfinite(arr)]


def _extract_metric(d, group, metric_key, conditions):
    return {
        "mean": [d[group][metric_key][c]["mean"] for c in conditions],
        "std":  [d[group][metric_key][c]["std"]  for c in conditions],
        "values": [_clean_values(d[group][metric_key][c]) for c in conditions],
        "value_unit": [d[group][metric_key][c].get("value_unit", "") for c in conditions],
        "values_source": [d[group][metric_key][c].get("values_source", "") for c in conditions],
    }


def _dot_cluster(ax, centers, series, colors, edges, markers,
                 ylab=None, xlabels=None, metric_key=None):
    n_series = len(series)
    xpos = _series_positions(centers, n_series)
    rng = np.random.default_rng(RNG_SEED)
    is_bounded = (metric_key in BOUNDED_METRICS)

    for i, y in enumerate(series):
        marker = markers[i]
        for j, cond_x in enumerate(xpos[i]):
            values = y.get("values", [])[j]
            if values.size:
                ax.scatter(
                    _jittered_x(cond_x, values.size, JITTER_WIDTH, rng),
                    values,
                    s=POINT_SIZE,
                    marker=marker,
                    facecolors=colors[i],
                    edgecolors=edges[i],
                    linewidths=0.5,
                    alpha=POINT_ALPHA,
                    zorder=2,
                )

        means = np.array(y["mean"], dtype=float)
        stds = np.array(y.get("std", [0] * len(centers)), dtype=float)

        if is_bounded and CLIP_BOUNDED_ERR:
            yerr = bounded_yerr(means, stds, lo=0.0, hi=1.0)
        else:
            yerr = stds

        means_plot = means.copy()
        if is_bounded and MIN_VISIBLE_BAR is not None:
            means_plot = np.where(means_plot == 0.0, MIN_VISIBLE_BAR, means_plot)

        ax.errorbar(
            xpos[i], means_plot, yerr=yerr,
            fmt=marker,
            markersize=MEAN_MARKER_SIZE,
            markerfacecolor=colors[i],
            markeredgecolor=edges[i],
            markeredgewidth=BAR_LINEWIDTH,
            ecolor=edges[i],
            elinewidth=BAR_LINEWIDTH,
            capsize=BAR_CAPSIZE,
            linestyle="none",
            zorder=4,
        )

    if ylab:
        ax.set_ylabel(ylab)

    _style_axes(ax)
    if is_bounded:
        ax.set_ylim(0.0, 1.0)

    if xlabels:
        ax.set_xticks(centers)
        ax.set_xticklabels(xlabels)
    else:
        ax.set_xticks([])

    ax.set_xlim(min(centers) - 0.55, max(centers) + 0.55)


def _prepare_handles_for_legend(baseline_labels, baseline_markers):
    h_human = Line2D(
        [0], [0], marker="o", linestyle="none",
        color=HUMAN_COLOR, markerfacecolor=HUMAN_COLOR,
        markeredgecolor=HUMAN_COLOR, markeredgewidth=BAR_LINEWIDTH,
        markersize=MEAN_MARKER_SIZE, label="Human"
    )
    h_sim = Line2D(
        [0], [0], marker="o", linestyle="none",
        color=SIM_COLOR, markerfacecolor=SIM_COLOR,
        markeredgecolor=SIM_COLOR, markeredgewidth=BAR_LINEWIDTH,
        markersize=MEAN_MARKER_SIZE, label="Simulation"
    )
    baseline_handles = []
    for i, lab in enumerate(baseline_labels):
        baseline_handles.append(
            Line2D(
                [0], [0], marker=baseline_markers[i % len(baseline_markers)], linestyle="none",
                color=BASELINE_EDGE, markerfacecolor=BASELINE_FILL,
                markeredgecolor=BASELINE_EDGE, markeredgewidth=BAR_LINEWIDTH,
                markersize=MEAN_MARKER_SIZE, label=lab
            )
        )
    return [h_human, h_sim] + baseline_handles


def _eta_squared(groups):
    clean = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    if not clean:
        return float("nan")
    all_values = np.concatenate(clean)
    grand_mean = np.mean(all_values)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in clean)
    ss_total = np.sum((all_values - grand_mean) ** 2)
    return float(ss_between / ss_total) if ss_total > 0 else float("nan")


def _cohens_d_independent(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1 = np.var(a, ddof=1)
    s2 = np.var(b, ddof=1)
    sp = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else float("nan")


def _safe_f_oneway(groups):
    try:
        if len(groups) >= 2 and all(len(g) > 1 for g in groups):
            return stats.f_oneway(*groups)
    except Exception:
        pass
    return None


def _write_statistics(data_main, data_base, metrics, variants, pretty, out_name=OUT_STATS):
    """
    Write descriptive statistics, one-way ANOVA across time conditions within each
    plotted source, and pairwise Welch t-tests between time conditions.

    All tests use the same point-wise values that are plotted.
    """
    conditions = data_main["conditions"]
    rows = []

    source_specs = [("human", "Human", data_main["human"]),
                    ("simulation", "Simulation", data_main["simulation"])]
    for v in variants:
        source_specs.append((v, pretty.get(v, v.replace("_", " ")), data_base["baselines"][v]))

    for metric_key, ylabel in metrics:
        for source_key, source_label, source_block in source_specs:
            groups = []
            value_units = []
            value_sources = []
            for cond in conditions:
                block = source_block[metric_key][cond]
                values = _clean_values(block)
                groups.append(values)
                value_units.append(block.get("value_unit", ""))
                value_sources.append(block.get("values_source", ""))

                rows.append({
                    "section": "descriptive",
                    "metric": metric_key,
                    "metric_label": ylabel,
                    "source": source_key,
                    "source_label": source_label,
                    "condition": cond,
                    "comparison": "",
                    "n": int(values.size),
                    "mean": float(np.mean(values)) if values.size else float("nan"),
                    "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "sem": float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0,
                    "test": "",
                    "df1": "",
                    "df2": "",
                    "statistic": "",
                    "p": "",
                    "effect_size": "",
                    "effect_size_name": "",
                    "value_unit": block.get("value_unit", ""),
                    "values_source": block.get("values_source", ""),
                })

            f_res = _safe_f_oneway(groups)
            if f_res is not None:
                df1 = len(groups) - 1
                df2 = sum(len(g) for g in groups) - len(groups)
                rows.append({
                    "section": "inferential",
                    "metric": metric_key,
                    "metric_label": ylabel,
                    "source": source_key,
                    "source_label": source_label,
                    "condition": "",
                    "comparison": "30s_vs_60s_vs_90s",
                    "n": int(sum(len(g) for g in groups)),
                    "mean": "",
                    "sd": "",
                    "sem": "",
                    "test": "one_way_anova",
                    "df1": int(df1),
                    "df2": int(df2),
                    "statistic": float(f_res.statistic),
                    "p": float(f_res.pvalue),
                    "effect_size": _eta_squared(groups),
                    "effect_size_name": "eta_squared",
                    "value_unit": ";".join(sorted(set(value_units))),
                    "values_source": ";".join(sorted(set(vs for vs in value_sources if vs))),
                })

            for i in range(len(conditions)):
                for j in range(i + 1, len(conditions)):
                    a = groups[i]
                    b = groups[j]
                    if len(a) > 1 and len(b) > 1:
                        try:
                            t_res = stats.ttest_ind(a, b, equal_var=False)
                            d = _cohens_d_independent(a, b)
                            rows.append({
                                "section": "inferential",
                                "metric": metric_key,
                                "metric_label": ylabel,
                                "source": source_key,
                                "source_label": source_label,
                                "condition": "",
                                "comparison": f"{conditions[i]}_vs_{conditions[j]}",
                                "n": int(len(a) + len(b)),
                                "mean": "",
                                "sd": "",
                                "sem": "",
                                "test": "welch_t_test",
                                "df1": "",
                                "df2": float(t_res.df) if hasattr(t_res, "df") else "",
                                "statistic": float(t_res.statistic),
                                "p": float(t_res.pvalue),
                                "effect_size": d,
                                "effect_size_name": "cohens_d",
                                "value_unit": ";".join(sorted(set(value_units))),
                                "values_source": ";".join(sorted(set(vs for vs in value_sources if vs))),
                            })
                        except Exception:
                            pass

    fieldnames = [
        "section", "metric", "metric_label", "source", "source_label", "condition",
        "comparison", "n", "mean", "sd", "sem", "test", "df1", "df2",
        "statistic", "p", "effect_size", "effect_size_name", "value_unit", "values_source",
    ]
    with open(out_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_agreement_stats(data_main, data_base, metrics, variants, pretty, out_name=OUT_AGREE):
    """Agreement of each simulated/baseline source with human means over all 15 metric-condition cells."""
    conditions = data_main["conditions"]
    rows = []

    compare_sources = [("simulation", "Simulation", data_main["simulation"])]
    for v in variants:
        compare_sources.append((v, pretty.get(v, v.replace("_", " ")), data_base["baselines"][v]))

    for source_key, source_label, source_block in compare_sources:
        human_means = []
        model_means = []
        for metric_key, ylabel in metrics:
            for cond in conditions:
                h = float(data_main["human"][metric_key][cond]["mean"])
                m = float(source_block[metric_key][cond]["mean"])
                human_means.append(h)
                model_means.append(m)
                rows.append({
                    "source": source_key,
                    "source_label": source_label,
                    "metric": metric_key,
                    "condition": cond,
                    "human_mean": h,
                    "comparison_mean": m,
                    "difference_human_minus_comparison": h - m,
                    "absolute_error": abs(h - m),
                    "squared_error": (h - m) ** 2,
                    "pearson_r": "",
                    "mae": "",
                    "rmse": "",
                })

        human_means = np.asarray(human_means, dtype=float)
        model_means = np.asarray(model_means, dtype=float)
        mae = float(np.mean(np.abs(human_means - model_means)))
        rmse = float(np.sqrt(np.mean((human_means - model_means) ** 2)))
        r = float(np.corrcoef(human_means, model_means)[0, 1]) if human_means.size > 1 else float("nan")
        rows.append({
            "source": source_key,
            "source_label": source_label,
            "metric": "OVERALL_15_MEANS",
            "condition": "",
            "human_mean": "",
            "comparison_mean": "",
            "difference_human_minus_comparison": "",
            "absolute_error": "",
            "squared_error": "",
            "pearson_r": r,
            "mae": mae,
            "rmse": rmse,
        })

    fieldnames = [
        "source", "source_label", "metric", "condition", "human_mean", "comparison_mean",
        "difference_human_minus_comparison", "absolute_error", "squared_error",
        "pearson_r", "mae", "rmse",
    ]
    with open(out_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ================== Main ==================
def main():
    data_main = _load_json_with_fallback(AGG_MAIN_PATH, "aggregated_panel_metrics.json")
    data_base = _load_json_with_fallback(AGG_BASE_PATH, "aggregated_panel_metrics_baseline.json")

    conditions = data_main["conditions"]
    pretty_conditions = [c.replace("s", " s") for c in conditions]
    centers = list(range(len(conditions)))

    row1 = [
        ("reading_speed", "Reading speed (WPM)"),
        ("skip_rate", "Skip rate"),
        ("regression_rate", "Regression rate"),
    ]
    row2 = [
        ("mcq_accuracy", "MCQ accuracy"),
        ("free_recall_score", "Free recall"),
    ]
    metrics = row1 + row2

    variants = data_base["meta"]["variants"]
    pretty = {
        "full_memory": "Sim with unlimited memory",
        "text_reader_gamma_0dot2": "Sim myopic text reader (\u03B3=0.2)",
        "text_reader_gamma_0dot6": "Sim myopic text reader (\u03B3=0.6)",
        "sentence_reader_gamma_0dot2": "Sim myopic sentence reader (\u03B3=0.2)",
        "sentence_reader_gamma_0dot6": "Sim myopic sentence reader (\u03B3=0.6)",
    }
    baseline_labels = [pretty.get(v, v.replace("_", " ")) for v in variants]

    # Figure with absolute spacer columns/rows
    fig_w = 3 * AX_W_IN + 2 * H_GAP_IN
    fig_h = 2 * AX_H_IN + 1 * V_GAP_IN
    fig = plt.figure(figsize=(fig_w, fig_h))

    widths  = [AX_W_IN, H_GAP_IN, AX_W_IN, H_GAP_IN, AX_W_IN]
    heights = [AX_H_IN, V_GAP_IN, AX_H_IN]
    gs = gridspec.GridSpec(
        nrows=3,
        ncols=5,
        figure=fig,
        width_ratios=widths,
        height_ratios=heights,
        wspace=0.0,
        hspace=0.0,
    )

    # Row 1 axes
    ax_speed = fig.add_subplot(gs[0, 0]); fig.add_subplot(gs[0, 1]).axis("off")
    ax_skip  = fig.add_subplot(gs[0, 2]); fig.add_subplot(gs[0, 3]).axis("off")
    ax_regr  = fig.add_subplot(gs[0, 4])
    # Row 2 axes
    ax_mcq   = fig.add_subplot(gs[2, 0]); fig.add_subplot(gs[2, 1]).axis("off")
    ax_free  = fig.add_subplot(gs[2, 2]); fig.add_subplot(gs[2, 3]).axis("off")
    ax_legend= fig.add_subplot(gs[2, 4]); ax_legend.set_axis_off(); ax_legend.set_facecolor("none")

    def series_for(metric_key):
        series = []
        colors = []
        edges  = []
        markers = []

        # Human
        series.append(_extract_metric(data_main, "human", metric_key, conditions))
        colors.append(HUMAN_COLOR); edges.append(HUMAN_COLOR); markers.append("o")

        # Main model
        series.append(_extract_metric(data_main, "simulation", metric_key, conditions))
        colors.append(SIM_COLOR); edges.append(SIM_COLOR); markers.append("o")

        # Baselines
        for i, v in enumerate(variants):
            s = _extract_metric({"baseline": data_base["baselines"][v]}, "baseline", metric_key, conditions)
            series.append(s)
            colors.append(BASELINE_FILL)
            edges.append(BASELINE_EDGE)
            markers.append(BASELINE_MARKERS[i % len(BASELINE_MARKERS)])

        return series, colors, edges, markers

    # Plot row 1
    for ax, (metric, ylab) in zip([ax_speed, ax_skip, ax_regr], row1):
        series, colors, edges, markers = series_for(metric)
        _dot_cluster(ax, centers, series, colors, edges, markers,
                     ylab=ylab, xlabels=pretty_conditions, metric_key=metric)

    # Plot row 2
    for ax, (metric, ylab) in zip([ax_mcq, ax_free], row2):
        series, colors, edges, markers = series_for(metric)
        _dot_cluster(ax, centers, series, colors, edges, markers,
                     ylab=ylab, xlabels=pretty_conditions, metric_key=metric)

    # Legend in bottom-right blank panel
    handles = _prepare_handles_for_legend(baseline_labels, BASELINE_MARKERS)
    labels  = [h.get_label() for h in handles]
    ax_legend.legend(
        handles, labels,
        loc=LEGEND_LOC, bbox_to_anchor=LEGEND_BBOX,
        frameon=True, facecolor="white", edgecolor="black",
        framealpha=0.9, fontsize=LEGEND_SIZE,
        borderpad=0.4, labelspacing=0.3,
    )

    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {OUT_PDF}")

    _write_statistics(data_main, data_base, metrics, variants, pretty, OUT_STATS)
    print(f"Saved: {OUT_STATS}")

    _write_agreement_stats(data_main, data_base, metrics, variants, pretty, OUT_AGREE)
    print(f"Saved: {OUT_AGREE}")


if __name__ == "__main__":
    _set_fonts()
    main()
