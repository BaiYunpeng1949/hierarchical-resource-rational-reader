import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from matplotlib.lines import Line2D


# NOTE: A unified configuration
PLOT_WIDTH = 9 
PLOT_HEIGHT = 6 


def plot_comparison(x_human, y_human, x_sim, y_sim, x_name, y_name, output_dir, output_filename):
    """Plot regression lines with confidence bands for both human and simulated data, including binned regression lines"""
    try:
        # Check if all x values are identical for either dataset
        if len(np.unique(x_human)) <= 1 or len(np.unique(x_sim)) <= 1:
            print(f"Skipping plot for {y_name} vs {x_name} (all x values are identical)")
            return
            
        # Function to calculate regression stats and equation
        def get_regression_stats(x, y):
            coeffs = np.polyfit(x, y, deg=1)
            poly = np.poly1d(coeffs)
            y_pred = poly(x)
            r_squared = np.corrcoef(y, y_pred)[0,1]**2
            
            eq_str = f'y = {coeffs[1]:.2f} + {coeffs[0]:.2f}x'
            return r_squared, eq_str, coeffs

        # Function to bin data and compute means
        def bin_data(x, y, n_bins=12):
            # Create bins based on x range
            bins = np.linspace(min(x), max(x), n_bins + 1)
            bin_means_x = []
            bin_means_y = []
            
            for i in range(len(bins) - 1):
                mask = (x >= bins[i]) & (x < bins[i + 1])
                if np.any(mask):
                    bin_means_x.append((bins[i] + bins[i + 1]) / 2)
                    bin_means_y.append(np.mean(y[mask]))
            
            return np.array(bin_means_x), np.array(bin_means_y)

        # Calculate regression stats for both raw and binned data
        # Raw data stats
        human_r2, human_eq, human_coeffs = get_regression_stats(x_human, y_human)
        sim_r2, sim_eq, sim_coeffs = get_regression_stats(x_sim, y_sim)
        
        # Bin data and calculate stats
        human_bin_x, human_bin_y = bin_data(x_human, y_human)
        sim_bin_x, sim_bin_y = bin_data(x_sim, y_sim)
        human_bin_r2, human_bin_eq, human_bin_coeffs = get_regression_stats(human_bin_x, human_bin_y)
        sim_bin_r2, sim_bin_eq, sim_bin_coeffs = get_regression_stats(sim_bin_x, sim_bin_y)

        # Set the tick configurations
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        
        # Plot 3: Only binned data with confidence bands
        plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT), constrained_layout=True)
        
        # Plot binned data points with confidence bands
        sns.regplot(x=human_bin_x, y=human_bin_y, color='blue', 
                   scatter=True, scatter_kws={'s': 50}, 
                   label=f'Human (R²={human_bin_r2:.2f}):\n{human_bin_eq}')
        sns.regplot(x=sim_bin_x, y=sim_bin_y, color='green', 
                   scatter=True, scatter_kws={'s': 50},
                   label=f'Simulation (R²={sim_bin_r2:.2f}):\n{sim_bin_eq}')
        
        plt.xlabel(x_name, fontsize=20)
        plt.ylabel(y_name, fontsize=20)
        # plt.title(f'{y_name} vs {x_name} (Binned Only)', fontsize=14)
        # plt.title(f'{y_name} vs {x_name}', fontsize=20)
        
        # Create custom legend handles
        legend_elements = [
            Line2D([0], [0], color='blue', linestyle='-', linewidth=2,
                   label=f'Human (R²={human_bin_r2:.2f}):\n{human_bin_eq}'),
            Line2D([0], [0], color='green', linestyle='-', linewidth=2,
                   label=f'Simulation (R²={sim_bin_r2:.2f}):\n{sim_bin_eq}')
        ]
        plt.legend(handles=legend_elements, fontsize=20)
        plt.grid(True, alpha=0.3)
        
        # Save binned-only regression plot
        binned_only_path = os.path.join(output_dir, f"{output_filename}_binned_only_regression.png")
        plt.savefig(binned_only_path, 
                    dpi=300,                           # (or your preferred dpi)
                    bbox_inches='tight',               # <─ NEW
                    pad_inches=0.05
                    )
        plt.close()
        
        # print(f"Comparison plots saved at {regression_path}, {binned_path}, and {binned_only_path}")
        # print(f"Comparison plots saved at, {binned_path}, and {binned_only_path}")
        print(f"Comparison plot saved at {binned_only_path}")

        
    except Exception as e:
        print(f"Error plotting {y_name} vs {x_name}: {str(e)}")
        plt.close()

def main():
    # Set up directories
    base_dir = "/home/baiy4/reader-agent-zuco/results/section2"
    output_dir = os.path.join(base_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    human_data = pd.read_csv(os.path.join(base_dir, "processed_human_data/all_words_regression_and_skip_probabilities.csv"))
    sim_data = pd.read_csv(os.path.join(base_dir, "processed_simulated_results/all_words_regression_and_skip_probabilities.csv"))
    
    # Plot skip probability relationships
    # 1. Skip prob vs word length
    plot_comparison(
        human_data['length'], 
        human_data['skip_probability'],
        sim_data['length'],
        sim_data['skip_probability'],
        'Word Length',
        'Skip Probability',
        output_dir,
        'skip_probability_vs_length'
    )
    
    # 2. Skip prob vs log frequency
    plot_comparison(
        human_data['log_frequency'],
        human_data['skip_probability'],
        sim_data['log_frequency'],
        sim_data['skip_probability'],
        'Log Word Frequency',
        # 'Skip Probability',
        '',
        output_dir,
        'skip_probability_vs_log_frequency'
    )
    
    # 3. Skip prob vs logit predictability
    plot_comparison(
        human_data['logit_predictability'],
        human_data['skip_probability'],
        sim_data['logit_predictability'],
        sim_data['skip_probability'],
        'Logit Predictability',
        # 'Skip Probability',
        '',
        output_dir,
        'skip_probability_vs_logit_predictability'
    )
    
    # Plot regression probability relationships
    # 1. Regression prob vs difficulty
    plot_comparison(
        human_data['difficulty'],
        human_data['regression_probability'],
        sim_data['difficulty'],
        sim_data['regression_probability'],
        'Word Difficulty',
        'Regression Probability',
        output_dir,
        'regression_probability_vs_difficulty'
    )
    
    # # 2. Regression prob vs skip prob
    # plot_comparison(
    #     human_data['skip_probability'],
    #     human_data['regression_probability'],
    #     sim_data['skip_probability'],
    #     sim_data['regression_probability'],
    #     'Skip Probability',
    #     'Regression Probability',
    #     output_dir,
    #     'regression_probability_vs_skip_probability'
    # )
    
    print(f"All plots have been saved to {output_dir}")

if __name__ == "__main__":
    main()
