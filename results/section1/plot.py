import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def compare_gaze_duration(
    human_csv, 
    sim_csv, 
    x_col, 
    y_col, 
    x_label, 
    y_label, 
    title, 
    save_dir, 
    output_filename
):
    """
    Reads human and simulated data from CSV files, plots scatter points 
    with linear regression lines, and saves the figure.

    Parameters:
    -----------
    human_csv : str
        Path to the CSV file with human data.
    sim_csv : str
        Path to the CSV file with simulated data.
    x_col : str
        Name of the column in the CSV that will be used for the x-axis.
    y_col : str
        Name of the column in the CSV that will be used for the y-axis.
    x_label : str
        Label to use for the x-axis on the plot.
    y_label : str
        Label to use for the y-axis on the plot.
    title : str
        Plot title.
    save_dir : str
        Directory where the plot image will be saved.
    output_filename : str
        Name of the output image file (e.g., "comparison.png").
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    human_df = pd.read_csv(human_csv)
    sim_df = pd.read_csv(sim_csv)

    plt.figure(figsize=(8, 6))
    
    # Scatter plots
    plt.scatter(
        human_df[x_col], 
        human_df[y_col],
        color='blue', alpha=0.2, label="Human Data"
    )
    plt.scatter(
        sim_df[x_col], 
        sim_df[y_col],
        color='red', alpha=0.2, label="Simulated Data"
    )
    
    # Linear regression for human data
    human_slope, human_intercept, _, _, _ = linregress(human_df[x_col], human_df[y_col])
    plt.plot(
        human_df[x_col],
        human_slope * human_df[x_col] + human_intercept, 
        color='blue', linestyle='dashed',
        linewidth=3
    )

    # Linear regression for simulated data
    sim_slope, sim_intercept, _, _, _ = linregress(sim_df[x_col], sim_df[y_col])
    plt.plot(
        sim_df[x_col],
        sim_slope * sim_df[x_col] + sim_intercept,
        color='red', linestyle='dashed', 
        linewidth=3
    )

    # Labels and formatting
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Save plot
    save_path = os.path.join(save_dir, output_filename)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Comparison plot saved at {save_path}")


if __name__ == "__main__":
    
    human_data_dir = "human_data"
    sim_data_dir = "simulated_results"
    save_dir = "figures"

    compare_gaze_duration(
        human_csv=os.path.join(human_data_dir, "gaze_duration_vs_word_length.csv"),
        sim_csv=os.path.join(sim_data_dir, "gaze_duration_vs_word_length.csv"),
        x_col="word_length",
        y_col="average_gaze_duration",
        x_label="Word Length",
        y_label="Average Gaze Duration (ms)",
        title="Comparison: Gaze Duration vs. Word Length",
        save_dir=save_dir,
        output_filename="word_length_comparison.png"
    )

    compare_gaze_duration(
        human_csv=os.path.join(human_data_dir, "gaze_duration_vs_word_log_frequency.csv"),
        sim_csv=os.path.join(sim_data_dir, "gaze_duration_vs_word_log_frequency.csv"),
        x_col="log_frequency", 
        y_col="average_gaze_duration",
        x_label="Log Frequency",
        y_label="Average Gaze Duration (ms)",
        title="Comparison: Gaze Duration vs. Log Frequency",
        save_dir=save_dir,
        output_filename="log_frequency_comparison.png"
    )

    compare_gaze_duration(
        human_csv=os.path.join(human_data_dir, "gaze_duration_vs_word_logit_predictability.csv"),
        sim_csv=os.path.join(sim_data_dir, "gaze_duration_vs_word_logit_predictability.csv"),
        x_col="logit_predictability", 
        y_col="average_gaze_duration",
        x_label="Predictability",
        y_label="Average Gaze Duration (ms)",
        title="Comparison: Gaze Duration vs. Logit Predictability",
        save_dir=save_dir,
        output_filename="logit_predictability_comparison.png"
    )

    compare_gaze_duration(
        human_csv=os.path.join(human_data_dir, "gaze_duration_vs_word_log_frequency.csv"),
        sim_csv=os.path.join(sim_data_dir, "gaze_duration_vs_word_log_frequency_binned.csv"),
        x_col="log_frequency", 
        y_col="average_gaze_duration",
        x_label="Log Frequency",
        y_label="Average Gaze Duration (ms)",
        title="Comparison: Gaze Duration vs. Log Frequency",
        save_dir=save_dir,
        output_filename="log_frequency_binned_comparison.png"
    )

    compare_gaze_duration(
        human_csv=os.path.join(human_data_dir, "gaze_duration_vs_word_logit_predictability.csv"),
        sim_csv=os.path.join(sim_data_dir, "gaze_duration_vs_word_logit_predictability_binned.csv"),
        x_col="logit_predictability", 
        y_col="average_gaze_duration",
        x_label="Predictability",
        y_label="Average Gaze Duration (ms)",
        title="Comparison: Gaze Duration vs. Logit Predictability",
        save_dir=save_dir,
        output_filename="logit_predictability_binned_comparison.png"
    )

