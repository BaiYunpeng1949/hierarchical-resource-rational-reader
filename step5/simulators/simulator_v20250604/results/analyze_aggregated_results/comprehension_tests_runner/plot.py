
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

HUMAN_COLOR = "blue"
SIM_COLOR = "green"
ERR_COLOR = "black"

def load_and_prepare(path, label):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "time_constraint": "time",
        "MCQ Accuracy": "mcq",
        "Free Recall Score": "fr",
    })
    # Ensure proper types
    df["time"] = df["time"].astype(str)
    df["mcq"] = pd.to_numeric(df["mcq"], errors="coerce")
    df["fr"] = pd.to_numeric(df["fr"], errors="coerce")
    g = df.groupby("time").agg(
        mcq_mean=("mcq","mean"),
        mcq_std=("mcq","std"),
        fr_mean=("fr","mean"),
        fr_std=("fr","std"),
        n=("mcq","count"),
    ).reset_index()
    g["label"] = label
    return g

def plot_grouped_bar(agg_df, value_col_mean, value_col_std, title, ylabel, outfile):
    times = ["30","60","90"]
    labels = ["Human", "Simulation"]
    x = np.arange(len(times))
    width = 0.35

    def get(series, t, lab):
        s = agg_df[(agg_df["time"]==t) & (agg_df["label"]==lab)][series]
        return float(s.iloc[0]) if len(s) else np.nan

    means_h = [get(value_col_mean, t, "Human") for t in times]
    means_s = [get(value_col_mean, t, "Simulation") for t in times]
    stds_h  = [get(value_col_std,  t, "Human") for t in times]
    stds_s  = [get(value_col_std,  t, "Simulation") for t in times]

    fig = plt.figure(figsize=(7,5))
    ax = plt.gca()

    bars_h = ax.bar(x - width/2, means_h, width, label="Human", color=HUMAN_COLOR, alpha=0.7)
    bars_s = ax.bar(x + width/2, means_s, width, label="Simulation", color=SIM_COLOR, alpha=0.7)

    ax.errorbar(x - width/2, means_h, yerr=stds_h, fmt='none', color=ERR_COLOR, capsize=5)
    ax.errorbar(x + width/2, means_s, yerr=stds_s, fmt='none', color=ERR_COLOR, capsize=5)

    # annotate
    for bars, stds in [(bars_h, stds_h), (bars_s, stds_s)]:
        for bar, s in zip(bars, stds):
            h = bar.get_height()
            if not np.isnan(h):
                txt = f"{h:.2f} ({s:.2f})" if not np.isnan(s) else f"{h:.2f}"
                ax.annotate(txt, xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom")

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}s" for t in times])
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)

def concat_side_by_side(img_left, img_right, out_path, padding=20, bg=(255,255,255,255)):
    L = Image.open(img_left).convert("RGBA")
    R = Image.open(img_right).convert("RGBA")
    H = max(L.height, R.height)
    W = L.width + R.width + padding
    canvas = Image.new("RGBA", (W, H), bg)
    canvas.paste(L, (0, 0))
    canvas.paste(R, (L.width + padding, 0))
    canvas.save(out_path)

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--sim_csv", default="/mnt/data/comprehension_test_results.csv")
    # ap.add_argument("--human_csv", default="/mnt/data/processed_mcq_freerecall_scores_p1_to_p32.csv")
    ap.add_argument("--out_dir", default="/mnt/data")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    human_csv= os.path.join('..', 'assets', 'comprehension_results', 'human', 'processed_mcq_freerecall_scores_p1_to_p32.csv')
    sim_csv = os.path.join('..', 'assets', 'comprehension_results', 'simulation', 'comprehension_test_results.csv')

    agg_sim = load_and_prepare(sim_csv, "Simulation")
    agg_hum = load_and_prepare(human_csv, "Human")
    agg = pd.concat([agg_hum, agg_sim], ignore_index=True)

    mcq_png = os.path.join(args.out_dir, "mcq_by_time.png")
    fr_png  = os.path.join(args.out_dir, "free_recall_by_time.png")
    plot_grouped_bar(agg, "mcq_mean", "mcq_std",
                     "MCQ Accuracy by Time (Human vs Simulation)",
                     "MCQ Accuracy", mcq_png)
    plot_grouped_bar(agg, "fr_mean", "fr_std",
                     "Free Recall Score by Time (Human vs Simulation)",
                     "Free Recall Score", fr_png)

    # Side-by-side combined image (no subplots)
    combined = os.path.join(args.out_dir, "mcq_fr_side_by_side.png")
    concat_side_by_side(mcq_png, fr_png, combined)

    # Also export the aggregation for reference
    agg_out = os.path.join(args.out_dir, "aggregated_mcq_fr_by_time.csv")
    agg.sort_values(["label","time"]).to_csv(agg_out, index=False)

    print("Saved:")
    print(mcq_png)
    print(fr_png)
    print(combined)
    print(agg_out)

if __name__ == "__main__":
    main()
