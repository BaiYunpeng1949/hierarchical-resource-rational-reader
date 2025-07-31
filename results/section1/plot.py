import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import seaborn as sns
from matplotlib.lines import Line2D

# NOTE: should be the universal configuration
PLOT_WIDTH = 9
PLOT_HEIGHT = 6

def compare_gaze_duration(
    human_csv, 
    sim_csv, 
    x_col, 
    y_col, 
    x_label, 
    y_label=None, 
    title=None, 
    save_dir=None, 
    output_filename=None,
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

    # Plot 1: Regression lines with confidence bands
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    
    # Function to calculate regression stats and equation
    def get_regression_stats(df):
        if use_log_x:
            x = np.log(df[x_col])
        else:
            x = df[x_col]
        y = df[y_col]
        coeffs = np.polyfit(x, y, deg=poly_degree)
        poly = np.poly1d(coeffs)
        y_pred = poly(x)
        r_squared = np.corrcoef(y, y_pred)[0,1]**2
        
        if poly_degree == 1:
            eq_str = f'y = {coeffs[1]:.2f} + {coeffs[0]:.2f}x'
        else:
            eq_str = f'y = {coeffs[-1]:.2f}'
            for i, coef in enumerate(coeffs[:-1][::-1]):
                eq_str += f' + {coef:.2f}x^{i+1}'
        return r_squared, eq_str

    # Calculate regression stats for both datasets
    human_r2, human_eq = get_regression_stats(human_df)
    sim_r2, sim_eq = get_regression_stats(sim_df)
    
    # Plot regression with confidence bands for human data
    human_plot = sns.regplot(data=human_df, x=x_col, y=y_col,
                scatter=True, color='blue', 
                label=f'Human (R²={human_r2:.2f}):\n{human_eq}',
                scatter_kws={'alpha': 1.0},
                line_kws={'linestyle': 'dashed', 'linewidth': 2})

    # Plot regression with confidence bands for simulation data
    sim_plot = sns.regplot(data=sim_df, x=x_col, y=y_col,
                scatter=True, color='green', 
                label=f'Simulation (R²={sim_r2:.2f}):\n{sim_eq}',
                scatter_kws={'alpha': 1.0},
                line_kws={'linestyle': 'dashed', 'linewidth': 2})

    plt.xlabel(x_label + (" (log scale for regression)" if use_log_x else ""), fontsize=font_size)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=font_size)
    if title is not None:
        plt.title(title, fontsize=font_size+2)
    
    # Create custom legend handles with dashed lines
    legend_elements = [
        Line2D([0], [0], color='blue', linestyle='dashed', linewidth=2,
               label=f'Human (R²={human_r2:.2f}):\n{human_eq}'),
        Line2D([0], [0], color='green', linestyle='dashed', linewidth=2,
               label=f'Simulation (R²={sim_r2:.2f}):\n{sim_eq}')
    ]
    plt.legend(handles=legend_elements, fontsize=legend_size)
    plt.grid(True)
    
    base_name = os.path.splitext(output_filename)[0]
    regression_path = os.path.join(save_dir, f"{base_name}_regression.png")
    plt.savefig(regression_path)
    plt.close()

    # Plot 2: Connected points with confidence bands
    plt.figure(figsize=(8, 6))
    
    # Plot human data points connected
    plt.plot(human_df[x_col], human_df[y_col], 
            'o-', color='blue', label="Human Data", alpha=1.0)

    # Plot simulation data points with confidence band
    plt.plot(sim_df[x_col], sim_df[y_col], 
            'o-', color='green', label="Simulated Data", alpha=1.0)
    
    if all(col in sim_df.columns for col in ['ci_95_lower', 'ci_95_upper']):
        plt.fill_between(
            sim_df[x_col],
            sim_df['ci_95_lower'],
            sim_df['ci_95_upper'],
            color='green', alpha=1.0, label='95% CI'
        )

    plt.xlabel(x_label, fontsize=font_size)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=font_size)
    if title is not None:
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

    # Plot configurations -- NOTE: these should be the universal setting
    poly_degree = 1
    font_size = 20
    tick_size = 20
    legend_size = 20

    compare_gaze_duration(
        human_csv=os.path.join(human_data_dir, "gaze_duration_vs_word_length.csv"),
        sim_csv=os.path.join(sim_data_dir, "gaze_duration_vs_word_length.csv"),
        x_col="word_length",
        y_col="average_gaze_duration",
        x_label="Word Length",
        y_label="Average Gaze Duration (ms)",
        # title="Word Length's Effect on Gaze Duration",
        title = "",
        save_dir=save_dir,
        output_filename="word_length_comparison.png",
        use_log_x=False,
        poly_degree=poly_degree,
        font_size=font_size,      # Larger base font size
        tick_size=tick_size,      # Slightly smaller tick labels
        legend_size=legend_size     # Legend font size
    )

    compare_gaze_duration(
        human_csv=os.path.join(human_data_dir, "gaze_duration_vs_word_log_frequency.csv"),
        sim_csv=os.path.join(sim_data_dir, "gaze_duration_vs_word_log_frequency_binned.csv"),
        x_col="log_frequency", 
        y_col="average_gaze_duration",
        x_label="Log Frequency",
        # y_label="Average Gaze Duration (ms)",
        y_label="",
        # title="Frequency's Effect on Gaze Duration",
        title="",
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
        x_label="Logit Predictability",
        # y_label="Average Gaze Duration (ms)",
        y_label="",
        # title="Predictability's Effect on Gaze Duration",
        title="",
        save_dir=save_dir,
        output_filename="logit_predictability_binned_comparison.png",
        use_log_x=False,
        poly_degree=poly_degree,
        font_size=font_size,      # Larger base font size
        tick_size=tick_size,      # Slightly smaller tick labels
        legend_size=legend_size     # Legend font size
    )

