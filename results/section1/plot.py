import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# def compare_gaze_duration(
#     human_csv, 
#     sim_csv, 
#     x_col, 
#     y_col, 
#     x_label, 
#     y_label, 
#     title, 
#     save_dir, 
#     output_filename
# ):
#     """
#     Reads human and simulated data from CSV files, plots scatter points 
#     with linear regression lines, and saves the figure.

#     Parameters:
#     -----------
#     human_csv : str
#         Path to the CSV file with human data.
#     sim_csv : str
#         Path to the CSV file with simulated data.
#     x_col : str
#         Name of the column in the CSV that will be used for the x-axis.
#     y_col : str
#         Name of the column in the CSV that will be used for the y-axis.
#     x_label : str
#         Label to use for the x-axis on the plot.
#     y_label : str
#         Label to use for the y-axis on the plot.
#     title : str
#         Plot title.
#     save_dir : str
#         Directory where the plot image will be saved.
#     output_filename : str
#         Name of the output image file (e.g., "comparison.png").
#     """
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Load data
#     human_df = pd.read_csv(human_csv)
#     sim_df = pd.read_csv(sim_csv)

#     plt.figure(figsize=(8, 6))
    
#     # Scatter plots
#     plt.scatter(
#         human_df[x_col], 
#         human_df[y_col],
#         color='blue', alpha=0.2, label="Human Data"
#     )
#     plt.scatter(
#         sim_df[x_col], 
#         sim_df[y_col],
#         color='red', alpha=0.2, label="Simulated Data"
#     )
    
#     # Linear regression for human data
#     human_slope, human_intercept, _, _, _ = linregress(human_df[x_col], human_df[y_col])
#     plt.plot(
#         human_df[x_col],
#         human_slope * human_df[x_col] + human_intercept, 
#         color='blue', linestyle='dashed',
#         linewidth=3
#     )

#     # Linear regression for simulated data
#     sim_slope, sim_intercept, _, _, _ = linregress(sim_df[x_col], sim_df[y_col])
#     plt.plot(
#         sim_df[x_col],
#         sim_slope * sim_df[x_col] + sim_intercept,
#         color='red', linestyle='dashed', 
#         linewidth=3
#     )

#     # Labels and formatting
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
    
#     # Save plot
#     save_path = os.path.join(save_dir, output_filename)
#     plt.savefig(save_path)
#     plt.close()
    
#     print(f"Comparison plot saved at {save_path}")

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
    poly_degree=1
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
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    human_df = pd.read_csv(human_csv)
    sim_df   = pd.read_csv(sim_csv)

    # If using log scale for x, create log_x columns
    # (and filter out non-positive x)
    if use_log_x:
        human_df = human_df[human_df[x_col] > 0].copy()
        sim_df   = sim_df[sim_df[x_col] > 0].copy()
        
        human_df["transformed_x"] = np.log(human_df[x_col])
        sim_df["transformed_x"]   = np.log(sim_df[x_col])
    else:
        human_df["transformed_x"] = human_df[x_col]
        sim_df["transformed_x"]   = sim_df[x_col]

    plt.figure(figsize=(8, 6))
    
    # Scatter plots (use original x values for plotting)
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
    
    # Generate a smooth range of x values (for the final plotted line)
    x_min = min(human_df[x_col].min(), sim_df[x_col].min())
    x_max = max(human_df[x_col].max(), sim_df[x_col].max())
    x_line = np.linspace(x_min, x_max, 200)  # denser for a smoother curve

    # We'll fit polynomial of 'poly_degree' on "transformed_x".
    # Then, to get y_line, we compute p(poly_of_x_line).
    # For the transform, we either do log(x) or x, 
    # so we must generate the "transformed_x_line" accordingly:
    if use_log_x:
        # Avoid log(0) issues
        x_line_for_fit = np.log(x_line[x_line > 0])
        # If x_line includes values <= 0, we skip them
        x_line_for_plot = x_line[x_line > 0]
    else:
        x_line_for_fit = x_line
        x_line_for_plot = x_line
    
    # Fit polynomial for human data
    # polyfit(x_data, y_data, deg=poly_degree)
    human_coeffs = np.polyfit(human_df["transformed_x"], human_df[y_col], deg=poly_degree)
    human_poly   = np.poly1d(human_coeffs)
    # Evaluate on x_line_for_fit
    human_y_line = human_poly(x_line_for_fit)

    # Fit polynomial for sim data
    sim_coeffs = np.polyfit(sim_df["transformed_x"], sim_df[y_col], deg=poly_degree)
    sim_poly   = np.poly1d(sim_coeffs)
    sim_y_line = sim_poly(x_line_for_fit)

    # Plot the polynomial lines
    plt.plot(
        x_line_for_plot, human_y_line, 
        color='blue', linestyle='dashed', linewidth=3,
        label=f"Human Poly deg={poly_degree}"
    )
    plt.plot(
        x_line_for_plot, sim_y_line, 
        color='red', linestyle='dashed', linewidth=3,
        label=f"Sim Poly deg={poly_degree}"
    )

    # Labels and formatting
    x_label_final = x_label
    if use_log_x:
        x_label_final += " (log scale for regression)"
    plt.xlabel(x_label_final)
    plt.ylabel(y_label)
    plt.title(title)

    # We already added data labels, so let's remove duplicates in legend
    plt.legend()
    plt.grid(True)
    
    # Save plot
    save_path = os.path.join(save_dir, output_filename)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Comparison plot (poly_degree={poly_degree}) saved at {save_path}")



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
        output_filename="word_length_comparison.png",
        use_log_x=False,
        poly_degree=1
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

