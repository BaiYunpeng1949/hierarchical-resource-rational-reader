import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison

def load_simulation_data(file_path):
    """Load simulation data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_metrics(data):
    """Calculate metrics for each time condition using pre-calculated values from episode logs."""
    metrics = {
        '30s': {'regression_rate': [], 'skip_rate': [], 'reading_speed': []},
        '60s': {'regression_rate': [], 'skip_rate': [], 'reading_speed': []},
        '90s': {'regression_rate': [], 'skip_rate': [], 'reading_speed': []}
    }
    
    for episode in data:
        time_condition = episode['time_condition']
        sentence_len = episode['sentence_len']
        elapsed_time = episode['elapsed_time']
        
        # Get pre-calculated metrics from episode log
        regression_rate = episode['sentence_wise_regression_rate'] * 100  # Convert to percentage
        skip_rate = episode['sentence_wise_skip_rate'] * 100  # Convert to percentage
        
        # Calculate reading speed: number of words / total time spent
        reading_speed = (sentence_len / elapsed_time) * 60 if elapsed_time > 0 else 0  # Convert to words per minute
        
        # Store metrics
        metrics[time_condition]['regression_rate'].append(regression_rate)
        metrics[time_condition]['skip_rate'].append(skip_rate)
        metrics[time_condition]['reading_speed'].append(reading_speed)
    
    return metrics

def perform_statistical_analysis(metrics):
    """Perform statistical analysis on the metrics."""
    stats_results = {}
    
    for metric_name in ['regression_rate', 'skip_rate', 'reading_speed']:
        # Prepare data for ANOVA
        groups = []
        group_labels = []
        
        for time_condition in ['30s', '60s', '90s']:
            values = metrics[time_condition][metric_name]
            groups.append(values)
            group_labels.extend([time_condition] * len(values))
        
        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Store ANOVA results
        stats_results[f'{metric_name}_anova'] = {
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        }
        
        # Perform post-hoc Tukey HSD test if ANOVA is significant
        if p_value < 0.05:
            all_values = []
            for group in groups:
                all_values.extend(group)
            
            mc = MultiComparison(all_values, group_labels)
            tukey_result = mc.tukeyhsd()
            
            stats_results[f'{metric_name}_tukey'] = {
                'groups': tukey_result.groupsunique.tolist(),
                'meandiffs': [float(x) for x in tukey_result.meandiffs],
                'p_values': [float(x) for x in tukey_result.pvalues],
                'significant': [bool(x) for x in (tukey_result.pvalues < 0.05)]
            }
        
        # Calculate means and standard deviations
        stats_results[f'{metric_name}_values'] = {
            'time_constraints': ['30s', '60s', '90s'],
            'values': [np.mean(group) for group in groups],
            'stds': [np.std(group) for group in groups]
        }
    
    return stats_results

def plot_metrics(metrics, stats_results, output_dir):
    """Create plots for the metrics with statistical significance indicators."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn')
    
    # Define metrics to plot
    metrics_to_plot = {
        'regression_rate': 'Regression Rate (%)',
        'skip_rate': 'Word Skip Rate (%)',
        'reading_speed': 'Reading Speed (wpm)'
    }
    
    for metric_key, ylabel in metrics_to_plot.items():
        plt.figure(figsize=(12, 8), dpi=100)
        
        # Get values from statistical analysis results
        values_data = stats_results[f'{metric_key}_values']
        time_constraints = values_data['time_constraints']
        metric_values = values_data['values']
        metric_stds = values_data['stds']
        
        # Create bar plot with error bars
        bars = plt.bar(range(len(metric_values)), metric_values, alpha=0.7, width=0.6,
                      yerr=metric_stds, capsize=5)
        
        # Add value annotations on top of each bar
        for i, (value, std) in enumerate(zip(metric_values, metric_stds)):
            plt.text(i, value + std + 0.02 * (max(metric_values) - min(metric_values)),
                    f'{value:.1f} ± {std:.1f}',
                    ha='center', va='bottom', fontsize=10)
        
        # Customize x-axis
        plt.xticks(range(len(time_constraints)), time_constraints, fontsize=12)
        plt.xlabel('Time Constraint', fontsize=12, labelpad=10)
        
        # Customize y-axis
        plt.ylabel(ylabel, fontsize=12, labelpad=10)
        plt.yticks(fontsize=10)
        
        # Add title
        plt.title(f'{ylabel} by Time Constraint', fontsize=14, pad=20)
        
        # Add statistical significance indicators if available
        if f'{metric_key}_anova' in stats_results and f'{metric_key}_tukey' in stats_results:
            anova_results = stats_results[f'{metric_key}_anova']
            tukey_results = stats_results[f'{metric_key}_tukey']
            
            if anova_results['significant']:
                # Calculate y-axis limits for significance bars
                y_max = max([v + s for v, s in zip(metric_values, metric_stds)])
                y_min = min([v - s for v, s in zip(metric_values, metric_stds)])
                y_range = y_max - y_min
                
                # Add significance bars
                bar_height = y_max + 0.15 * y_range
                
                # Add significance bars for each significant comparison
                for i, (g1, g2, p_val, mean_diff) in enumerate(zip(
                    tukey_results['groups'][:-1],
                    tukey_results['groups'][1:],
                    tukey_results['p_values'],
                    tukey_results['meandiffs']
                )):
                    if p_val < 0.05:
                        # Draw significance bar
                        plt.plot([i, i+1], [bar_height, bar_height], 'k-', lw=1.5)
                        
                        # Add p-value and effect size annotation
                        if p_val < 1e-10:
                            p_stars = '***'
                            p_text = 'p < 1e-10'
                        else:
                            p_stars = '*' * min(3, 1 + int(-np.log10(p_val)))
                            p_text = f'p = {p_val:.3e}'
                        
                        plt.text((i + i+1)/2, bar_height + 0.02 * y_range,
                                f'{p_stars}\n{p_text}\nΔ = {mean_diff:.1f}',
                                ha='center', va='bottom', fontsize=9)
                        
                        # Add a second bar if needed for multiple comparisons
                        if i < len(tukey_results['groups']) - 2:
                            bar_height += 0.1 * y_range
                
                # Adjust y-axis limits
                plt.ylim(y_min, y_max + 0.4 * y_range)
        
        # Add ANOVA results to plot
        if f'{metric_key}_anova' in stats_results:
            anova_results = stats_results[f'{metric_key}_anova']
            stats_text = f'ANOVA Results:\n'
            stats_text += f'F = {anova_results["f_statistic"]:.2f}\n'
            
            if anova_results['p_value'] < 1e-10:
                stats_text += 'p < 1e-10\n'
            else:
                stats_text += f'p = {anova_results["p_value"]:.3e}\n'
            
            # Add statistics text in a box
            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5),
                    fontsize=10)
        
        # Add grid for better readability
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'{metric_key}_with_stats.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Define paths
    base_dir = "/home/baiy4/reader-agent-zuco"
    folder_name = "0609_sentence_reading_under_time_pressure_v0604_01__rl_model_100000000_steps__1000"
    input_file = os.path.join(base_dir, f"step5/simulators/simulator_v20250604/sub_models/sentence_read_v0604/simulated_results/{folder_name}/simulated_episode_logs.json")
    output_dir = os.path.join(base_dir, f"step5/simulators/simulator_v20250604/sub_models/sentence_read_v0604/simulated_results/{folder_name}/analysis")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process data
    data = load_simulation_data(input_file)
    metrics = calculate_metrics(data)
    
    # Perform statistical analysis
    stats_results = perform_statistical_analysis(metrics)
    
    # Save metrics and statistical results
    results = {
        'metrics': metrics,
        'statistical_analysis': stats_results
    }
    with open(os.path.join(output_dir, 'metrics_and_stats.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create plots
    plot_metrics(metrics, stats_results, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
