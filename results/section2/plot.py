import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from matplotlib.lines import Line2D

def plot_comparison(x_human, y_human, x_sim, y_sim, x_name, y_name, output_dir, output_filename):
    """Plot regression lines and connected points with confidence bands for both human and simulated data"""
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
            return r_squared, eq_str

        # Calculate regression stats for both datasets
        human_r2, human_eq = get_regression_stats(x_human, y_human)
        sim_r2, sim_eq = get_regression_stats(x_sim, y_sim)
        
        # Plot 1: Regression lines with confidence bands
        plt.figure(figsize=(8, 6))
        
        # Plot human data
        human_plot = sns.regplot(x=x_human, y=y_human,
                    scatter=True, color='blue', 
                    label=f'Human (R²={human_r2:.2f}):\n{human_eq}',
                    scatter_kws={'alpha': 0.1},
                    line_kws={'linestyle': 'dashed', 'linewidth': 2})

        # Plot simulated data
        sim_plot = sns.regplot(x=x_sim, y=y_sim,
                    scatter=True, color='red', 
                    label=f'Simulation (R²={sim_r2:.2f}):\n{sim_eq}',
                    scatter_kws={'alpha': 0.1},
                    line_kws={'linestyle': 'dashed', 'linewidth': 2})

        plt.xlabel(x_name, fontsize=12)
        plt.ylabel(y_name, fontsize=12)
        plt.title(f'{y_name} vs {x_name}', fontsize=14)
        
        # Create custom legend handles with dashed lines
        legend_elements = [
            Line2D([0], [0], color='blue', linestyle='dashed', linewidth=2,
                   label=f'Human (R²={human_r2:.2f}):\n{human_eq}'),
            Line2D([0], [0], color='red', linestyle='dashed', linewidth=2,
                   label=f'Simulation (R²={sim_r2:.2f}):\n{sim_eq}')
        ]
        plt.legend(handles=legend_elements, fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save regression plot
        regression_path = os.path.join(output_dir, f"{output_filename}_regression.png")
        plt.savefig(regression_path)
        plt.close()

        # Plot 2: Connected points
        plt.figure(figsize=(8, 6))
        
        # Sort data for connected lines
        human_sorted = sorted(zip(x_human, y_human))
        sim_sorted = sorted(zip(x_sim, y_sim))
        
        # Plot human data points connected
        plt.plot([x for x, _ in human_sorted], [y for _, y in human_sorted], 
                'o-', color='blue', label="Human Data", alpha=0.4)

        # Plot simulation data points connected
        plt.plot([x for x, _ in sim_sorted], [y for _, y in sim_sorted], 
                'o-', color='red', label="Simulated Data", alpha=0.4)

        plt.xlabel(x_name, fontsize=12)
        plt.ylabel(y_name, fontsize=12)
        plt.title(f'{y_name} vs {x_name} (Connected Points)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save connected points plot
        connected_path = os.path.join(output_dir, f"{output_filename}_connected.png")
        plt.savefig(connected_path)
        plt.close()
        
        print(f"Comparison plots saved at {regression_path} and {connected_path}")
        
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
        'Skip Probability',
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
        'Skip Probability',
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
    
    # 2. Regression prob vs skip prob
    plot_comparison(
        human_data['skip_probability'],
        human_data['regression_probability'],
        sim_data['skip_probability'],
        sim_data['regression_probability'],
        'Skip Probability',
        'Regression Probability',
        output_dir,
        'regression_probability_vs_skip_probability'
    )
    
    print(f"All plots have been saved to {output_dir}")

if __name__ == "__main__":
    main()
