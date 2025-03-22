import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def plot_regression_with_confidence(x_human, y_human, x_sim, y_sim, x_name, y_name, output_dir):
    """Plot regression lines with confidence bands for both human and simulated data"""
    try:
        # Check if all x values are identical for either dataset
        if len(np.unique(x_human)) <= 1 or len(np.unique(x_sim)) <= 1:
            print(f"Skipping plot for {y_name} vs {x_name} (all x values are identical)")
            return
            
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot human data
        slope_h, intercept_h, r_value_h, p_value_h, std_err_h = stats.linregress(x_human, y_human)
        x_line_h = np.linspace(min(x_human), max(x_human), 100)
        y_line_h = slope_h * x_line_h + intercept_h
        
        # Calculate confidence interval for human data
        confidence = 0.95
        n_h = len(x_human)
        mean_x_h = np.mean(x_human)
        std_x_h = np.std(x_human)
        std_err_pred_h = std_err_h * np.sqrt(1/n_h + (x_line_h - mean_x_h)**2 / (n_h * std_x_h**2))
        
        # Plot human confidence band
        plt.fill_between(x_line_h, 
                        y_line_h - stats.t.ppf((1 + confidence) / 2, n_h-2) * std_err_pred_h,
                        y_line_h + stats.t.ppf((1 + confidence) / 2, n_h-2) * std_err_pred_h,
                        alpha=0.2, color='blue')
        
        # Plot simulated data
        slope_s, intercept_s, r_value_s, p_value_s, std_err_s = stats.linregress(x_sim, y_sim)
        x_line_s = np.linspace(min(x_sim), max(x_sim), 100)
        y_line_s = slope_s * x_line_s + intercept_s
        
        # Calculate confidence interval for simulated data
        n_s = len(x_sim)
        mean_x_s = np.mean(x_sim)
        std_x_s = np.std(x_sim)
        std_err_pred_s = std_err_s * np.sqrt(1/n_s + (x_line_s - mean_x_s)**2 / (n_s * std_x_s**2))
        
        # Plot simulated confidence band
        plt.fill_between(x_line_s, 
                        y_line_s - stats.t.ppf((1 + confidence) / 2, n_s-2) * std_err_pred_s,
                        y_line_s + stats.t.ppf((1 + confidence) / 2, n_s-2) * std_err_pred_s,
                        alpha=0.2, color='red')
        
        # Plot regression lines
        plt.plot(x_line_h, y_line_h, 'b-', label=f'Human (R² = {r_value_h**2:.3f})')
        plt.plot(x_line_s, y_line_s, 'r-', label=f'Simulated (R² = {r_value_s**2:.3f})')
        
        # Customize plot
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title(f'{y_name} vs {x_name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'comparison_{y_name.lower()}_vs_{x_name.lower()}.png'))
        plt.close()
        
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
    plot_regression_with_confidence(
        human_data['length'], 
        human_data['skip_probability'],
        sim_data['length'],
        sim_data['skip_probability'],
        'Word Length',
        'Skip Probability',
        output_dir
    )
    
    # 2. Skip prob vs log frequency
    plot_regression_with_confidence(
        human_data['log_frequency'],
        human_data['skip_probability'],
        sim_data['log_frequency'],
        sim_data['skip_probability'],
        'Log Word Frequency',
        'Skip Probability',
        output_dir
    )
    
    # Plot regression probability relationships
    # 1. Regression prob vs difficulty
    plot_regression_with_confidence(
        human_data['difficulty'],
        human_data['regression_probability'],
        sim_data['difficulty'],
        sim_data['regression_probability'],
        'Word Difficulty',
        'Regression Probability',
        output_dir
    )
    
    # 2. Regression prob vs skip prob
    plot_regression_with_confidence(
        human_data['skip_probability'],
        human_data['regression_probability'],
        sim_data['skip_probability'],
        sim_data['regression_probability'],
        'Skip Probability',
        'Regression Probability',
        output_dir
    )
    
    print(f"Plots have been saved to {output_dir}")

if __name__ == "__main__":
    main()
