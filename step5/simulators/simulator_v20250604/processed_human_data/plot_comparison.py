import json
import matplotlib.pyplot as plt
import numpy as np

def load_data(human_metrics_path, sim_metrics_path):
    with open(human_metrics_path, 'r') as f:
        human_data = json.load(f)
    with open(sim_metrics_path, 'r') as f:
        sim_data = json.load(f)
    return human_data, sim_data

def plot_metrics_comparison(human_data, sim_data, output_path):
    # Set up the figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Time conditions
    time_conditions = ['30s', '60s', '90s']
    x = np.arange(len(time_conditions))
    width = 0.35
    
    # Plot reading speed
    human_speeds = [human_data[cond]['reading_speed_mean'] for cond in time_conditions]
    sim_speeds = [sim_data[cond]['reading_speed_mean'] for cond in time_conditions]
    
    ax1.bar(x - width/2, human_speeds, width, label='Human', color='blue', alpha=0.7)
    ax1.bar(x + width/2, sim_speeds, width, label='Simulation', color='green', alpha=0.7)
    ax1.set_ylabel('Reading Speed (wpm)')
    ax1.set_title('Reading Speed Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(time_conditions)
    ax1.legend()
    
    # Plot skip rate
    human_skips = [human_data[cond]['skip_rate_mean'] for cond in time_conditions]
    sim_skips = [sim_data[cond]['skip_rate_mean'] for cond in time_conditions]
    
    ax2.bar(x - width/2, human_skips, width, label='Human', color='blue', alpha=0.7)
    ax2.bar(x + width/2, sim_skips, width, label='Simulation', color='green', alpha=0.7)
    ax2.set_ylabel('Skip Rate')
    ax2.set_title('Skip Rate Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(time_conditions)
    ax2.legend()
    
    # Plot regression rate
    human_regress = [human_data[cond]['regression_rate_mean'] for cond in time_conditions]
    sim_regress = [sim_data[cond]['regression_rate_mean'] for cond in time_conditions]
    
    ax3.bar(x - width/2, human_regress, width, label='Human', color='blue', alpha=0.7)
    ax3.bar(x + width/2, sim_regress, width, label='Simulation', color='green', alpha=0.7)
    ax3.set_ylabel('Regression Rate')
    ax3.set_title('Regression Rate Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(time_conditions)
    ax3.legend()
    
    # Add error bars
    for ax, human_std, sim_std in zip(
        [ax1, ax2, ax3],
        [[human_data[cond]['reading_speed_std'] for cond in time_conditions],
         [human_data[cond]['skip_rate_std'] for cond in time_conditions],
         [human_data[cond]['regression_rate_std'] for cond in time_conditions]],
        [[sim_data[cond]['reading_speed_std'] for cond in time_conditions],
         [sim_data[cond]['skip_rate_std'] for cond in time_conditions],
         [sim_data[cond]['regression_rate_std'] for cond in time_conditions]]
    ):
        ax.errorbar(x - width/2, human_std, yerr=human_std, fmt='none', color='black', capsize=5)
        ax.errorbar(x + width/2, sim_std, yerr=sim_std, fmt='none', color='black', capsize=5)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    # File paths
    human_metrics_path = "analyzed_human_metrics.json"
    sim_metrics_path = "../simulated_results/20250614_2133_trials1_stims9_conds3/analyzed_fixation_metrics.json"
    output_path = "metrics_comparison.png"
    
    # Load data and create plots
    human_data, sim_data = load_data(human_metrics_path, sim_metrics_path)
    plot_metrics_comparison(human_data, sim_data, output_path) 