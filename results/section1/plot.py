import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    output_filename,
    use_log_x=False,
    poly_degree=1,
    font_size=12,        # New parameter for base font size
    tick_size=10,        # New parameter for tick label size
    legend_size=10       # New parameter for legend font size
):
    """
    Reads human and simulated data from CSV files, plots scatter points 
    with polynomial (or simple linear) regression lines, and saves the figure.

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
        Label for the x-axis on the plot.
    y_label : str
        Label for the y-axis on the plot.
    title : str
        Title of the plot.
    save_dir : str
        Directory where the plot image will be saved.
    output_filename : str
        Name of the output image file (e.g., "comparison.png").
    use_log_x : bool
        If True, we apply log-transform to the x-values before fitting.
    poly_degree : int
        The degree of the polynomial to use in regression. 
        - 1 means simple linear.
        - 2, 3, etc. means polynomial fit.
    font_size : int
        Base font size for the plot.
    tick_size : int
        Tick label size for the plot.
    legend_size : int
        Legend font size for the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    human_df = pd.read_csv(human_csv)
    sim_df = pd.read_csv(sim_csv)

    # Set font sizes
    plt.rcParams.update({'font.size': font_size})
    plt.rc('xtick', labelsize=tick_size)
    plt.rc('ytick', labelsize=tick_size)

    # Plot 1: Regression lines
    plt.figure(figsize=(8, 6))
    
    # Scatter plots
    plt.scatter(human_df[x_col], human_df[y_col], color='blue', alpha=0.2, label="Human Data")
    plt.scatter(sim_df[x_col], sim_df[y_col], color='red', alpha=0.2, label="Simulated Data")

    # Fit and plot regression lines with statistics
    if use_log_x:
        human_df['transformed_x'] = np.log(human_df[x_col])
        sim_df['transformed_x'] = np.log(sim_df[x_col])
    else:
        human_df['transformed_x'] = human_df[x_col]
        sim_df['transformed_x'] = sim_df[x_col]

    x_min = min(human_df[x_col].min(), sim_df[x_col].min())
    x_max = max(human_df[x_col].max(), sim_df[x_col].max())
    x_line = np.linspace(x_min, x_max, 200)

    for df, color, label in [(human_df, 'blue', 'Human'), (sim_df, 'red', 'Simulation')]:
        # Calculate regression
        coeffs = np.polyfit(df['transformed_x'], df[y_col], deg=poly_degree)
        poly = np.poly1d(coeffs)
        
        # Calculate R-squared
        y_pred = poly(df['transformed_x'])
        r_squared = np.corrcoef(df[y_col], y_pred)[0,1]**2
        
        # Format equation string
        if poly_degree == 1:
            eq_str = f'y = {coeffs[1]:.2f} + {coeffs[0]:.2f}x'
        else:
            eq_str = f'y = {coeffs[-1]:.2f}'
            for i, coef in enumerate(coeffs[:-1][::-1]):
                eq_str += f' + {coef:.2f}x^{i+1}'
        
        # Plot with detailed legend
        if use_log_x:
            x_plot = x_line[x_line > 0]
            y_plot = poly(np.log(x_plot))
        else:
            x_plot = x_line
            y_plot = poly(x_plot)
            
        plt.plot(x_plot, y_plot, color=color, linestyle='dashed', 
                linewidth=3, label=f"{label} (R²={r_squared:.2f}):\n{eq_str}")

    plt.xlabel(x_label + (" (log scale for regression)" if use_log_x else ""), fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.title(title + " (Regression)", fontsize=font_size+2)
    plt.legend(fontsize=legend_size)
    plt.grid(True)
    
    base_name = os.path.splitext(output_filename)[0]
    regression_path = os.path.join(save_dir, f"{base_name}_regression.png")
    plt.savefig(regression_path)
    plt.close()

    # Plot 2: Connected points with confidence bands
    plt.figure(figsize=(8, 6))
    
    # Plot human data points connected
    plt.plot(human_df[x_col], human_df[y_col], 
            'o-', color='blue', label="Human Data", alpha=0.7)

    # Plot simulation data points with confidence band
    plt.plot(sim_df[x_col], sim_df[y_col], 
            'o-', color='red', label="Simulated Data", alpha=0.7)
    
    if all(col in sim_df.columns for col in ['ci_95_lower', 'ci_95_upper']):
        plt.fill_between(
            sim_df[x_col],
            sim_df['ci_95_lower'],
            sim_df['ci_95_upper'],
            color='red', alpha=0.1, label='95% CI'
        )

    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.title(title + " (Connected Points)", fontsize=font_size+2)
    plt.legend(fontsize=legend_size)
    plt.grid(True)
    
    connected_path = os.path.join(save_dir, f"{base_name}_connected.png")
    plt.savefig(connected_path)
    plt.close()
    
    print(f"Comparison plots saved at {regression_path} and {connected_path}")



if __name__ == "__main__":
    
    human_data_dir = "human_data"
    sim_data_dir = "simulated_results"
    save_dir = "figures"

    # Plot configurations
    poly_degree = 1
    font_size = 15
    tick_size = 15
    legend_size = 15

    compare_gaze_duration(
        human_csv=os.path.join(human_data_dir, "gaze_duration_vs_word_length.csv"),
        sim_csv=os.path.join(sim_data_dir, "gaze_duration_vs_word_length.csv"),
        x_col="word_length",
        y_col="average_gaze_duration",
        x_label="Word Length",
        y_label="Average Gaze Duration (ms)",
        title="Comparison: Gaze Duration vs. Word Length",
        save_dir=save_dir,
        output_filename="word_length_comparison.png",
        use_log_x=False,
        poly_degree=poly_degree,
        font_size=font_size,      # Larger base font size
        tick_size=tick_size,      # Slightly smaller tick labels
        legend_size=legend_size     # Legend font size
    )

    # compare_gaze_duration(
    #     human_csv=os.path.join(human_data_dir, "gaze_duration_vs_word_log_frequency.csv"),
    #     sim_csv=os.path.join(sim_data_dir, "gaze_duration_vs_word_log_frequency.csv"),
    #     x_col="log_frequency", 
    #     y_col="average_gaze_duration",
    #     x_label="Log Frequency",
    #     y_label="Average Gaze Duration (ms)",
    #     title="Comparison: Gaze Duration vs. Log Frequency",
    #     save_dir=save_dir,
    #     output_filename="log_frequency_comparison.png"
    # )

    # compare_gaze_duration(
    #     human_csv=os.path.join(human_data_dir, "gaze_duration_vs_word_logit_predictability.csv"),
    #     sim_csv=os.path.join(sim_data_dir, "gaze_duration_vs_word_logit_predictability.csv"),
    #     x_col="logit_predictability", 
    #     y_col="average_gaze_duration",
    #     x_label="Predictability",
    #     y_label="Average Gaze Duration (ms)",
    #     title="Comparison: Gaze Duration vs. Logit Predictability",
    #     save_dir=save_dir,
    #     output_filename="logit_predictability_comparison.png"
    # )

    compare_gaze_duration(
        human_csv=os.path.join(human_data_dir, "gaze_duration_vs_word_log_frequency.csv"),
        sim_csv=os.path.join(sim_data_dir, "gaze_duration_vs_word_log_frequency_binned.csv"),
        x_col="log_frequency", 
        y_col="average_gaze_duration",
        x_label="Log Frequency",
        y_label="Average Gaze Duration (ms)",
        title="Comparison: Gaze Duration vs. Log Frequency",
        save_dir=save_dir,
        output_filename="log_frequency_binned_comparison.png",
        use_log_x=False,
        poly_degree=poly_degree,
        font_size=font_size,      # Larger base font size
        tick_size=tick_size,      # Slightly smaller tick labels
        legend_size=legend_size     # Legend font size
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
        output_filename="logit_predictability_binned_comparison.png",
        use_log_x=False,
        poly_degree=poly_degree,
        font_size=font_size,      # Larger base font size
        tick_size=tick_size,      # Slightly smaller tick labels
        legend_size=legend_size     # Legend font size
    )

