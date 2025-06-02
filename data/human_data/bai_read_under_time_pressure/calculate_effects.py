import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison

INTEGRATED_CORRECTED_SCANPATH_FILE_NAME = "11_18_17_40_integrated_corrected_human_scanpath.json"

def load_bbox_metadata(metadata_dir: str) -> Dict:
    """
    Load the bounding box metadata from the JSON file.
    
    Args:
        metadata_dir: Directory containing the metadata files
        
    Returns:
        Dictionary containing the metadata
    """
    metadata_path = os.path.join(metadata_dir, "simulate", "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def load_corrected_scanpath_data(data_dir: str) -> Dict:
    """
    Load the corrected scanpath data from a JSON file.
    
    Args:
        data_dir: Directory containing the corrected scanpath data
        
    Returns:
        Dictionary containing the scanpath data
    """
    # Look for the integrated corrected human scanpath file
    data_path = os.path.join(data_dir, INTEGRATED_CORRECTED_SCANPATH_FILE_NAME)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find scanpath data at {data_path}")
        
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def process_scanpath_data(scanpath_data: Dict, metadata: Dict) -> pd.DataFrame:
    """
    Process the scanpath data and convert it to a pandas DataFrame.
    
    Args:
        scanpath_data: Dictionary containing the scanpath data
        metadata: Dictionary containing the stimulus metadata
        
    Returns:
        DataFrame containing the processed scanpath data
    """
    processed_data = []
    
    for trial in scanpath_data:
        stimulus_index = trial['stimulus_index']
        time_constraint = trial['time_constraint']
        participant_id = trial.get('participant_index', None)
        
        # Get fixation data
        fixations = trial['fixation_data']
        
        # Process each fixation
        for fixation in fixations:
            processed_fixation = {
                'stimulus_index': stimulus_index,
                'time_constraint': time_constraint,
                'participant_id': participant_id,
                'fix_x': fixation.get('fix_x'),
                'fix_y': fixation.get('fix_y'),
                'fix_duration': fixation.get('fix_duration'),
                'word_index': fixation.get('word_index', -1),
                'norm_fix_x': fixation.get('norm_fix_x'),
                'norm_fix_y': fixation.get('norm_fix_y')
            }
            processed_data.append(processed_fixation)
    
    return pd.DataFrame(processed_data)

def calculate_saccade_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate saccade lengths between consecutive fixations.
    
    Args:
        df: DataFrame containing the processed scanpath data
        
    Returns:
        DataFrame with added saccade length column
    """
    # Sort by participant, stimulus, and fixation order
    df = df.sort_values(['participant_id', 'stimulus_index', 'fix_x'])
    
    # Calculate saccade lengths
    df['prev_fix_x'] = df.groupby(['participant_id', 'stimulus_index'])['fix_x'].shift(1)
    df['prev_fix_y'] = df.groupby(['participant_id', 'stimulus_index'])['fix_y'].shift(1)
    
    df['saccade_length'] = np.sqrt(
        (df['fix_x'] - df['prev_fix_x'])**2 + 
        (df['fix_y'] - df['prev_fix_y'])**2
    )
    
    return df

def calculate_regression_rate(df: pd.DataFrame) -> Dict:
    """
    Calculate regression rate (backward saccades) by time constraint.
    
    Args:
        df: DataFrame containing the processed scanpath data
        
    Returns:
        Dictionary containing regression rates by time constraint
    """
    regression_rates = {}
    
    for time_constraint in df['time_constraint'].unique():
        time_data = df[df['time_constraint'] == time_constraint].copy()
        
        # Sort by participant, stimulus, and fixation order
        time_data = time_data.sort_values(['participant_id', 'stimulus_index', 'fix_x'])
        
        # Calculate word index differences
        time_data['next_word_index'] = time_data.groupby(['participant_id', 'stimulus_index'])['word_index'].shift(-1)
        time_data['is_regression'] = time_data['next_word_index'] < time_data['word_index']
        
        # Calculate regression rate
        regression_rate = time_data['is_regression'].mean() * 100
        regression_rates[f'regression_rate_{time_constraint}'] = regression_rate
    
    return regression_rates

def calculate_skip_rate(df: pd.DataFrame) -> Dict:
    """
    Calculate word skip rate by time constraint.
    
    Args:
        df: DataFrame containing the processed scanpath data
        
    Returns:
        Dictionary containing skip rates by time constraint
    """
    skip_rates = {}
    
    for time_constraint in df['time_constraint'].unique():
        time_data = df[df['time_constraint'] == time_constraint].copy()
        
        # Sort by participant, stimulus, and fixation order
        time_data = time_data.sort_values(['participant_id', 'stimulus_index', 'fix_x'])
        
        # Calculate word index differences
        time_data['next_word_index'] = time_data.groupby(['participant_id', 'stimulus_index'])['word_index'].shift(-1)
        time_data['word_diff'] = time_data['next_word_index'] - time_data['word_index']
        
        # Calculate skip rate (words skipped > 1)
        skip_rate = (time_data['word_diff'] > 1).mean() * 100
        skip_rates[f'skip_rate_{time_constraint}'] = skip_rate
    
    return skip_rates

def calculate_gaze_duration(df: pd.DataFrame) -> Dict:
    """
    Calculate gaze duration (total fixation time on a word) by time constraint.
    
    Args:
        df: DataFrame containing the processed scanpath data
        
    Returns:
        Dictionary containing gaze durations by time constraint
    """
    gaze_durations = {}
    
    for time_constraint in df['time_constraint'].unique():
        time_data = df[df['time_constraint'] == time_constraint].copy()
        
        # Calculate gaze duration per word
        gaze_duration = time_data.groupby(['participant_id', 'stimulus_index', 'word_index'])['fix_duration'].sum()
        
        # Calculate mean gaze duration
        mean_gaze = gaze_duration.mean()
        gaze_durations[f'mean_gaze_duration_{time_constraint}'] = mean_gaze
    
    return gaze_durations

def calculate_reading_speed(df: pd.DataFrame) -> Dict:
    """
    Calculate reading speed (words per minute) by time constraint.
    
    Args:
        df: DataFrame containing the processed scanpath data
        
    Returns:
        Dictionary containing reading speeds by time constraint
    """
    reading_speeds = {}
    
    for time_constraint in df['time_constraint'].unique():
        time_data = df[df['time_constraint'] == time_constraint].copy()
        
        # Calculate total words read and total time per trial
        trial_stats = time_data.groupby(['participant_id', 'stimulus_index']).agg({
            'word_index': lambda x: len(set(x[x != -1])),  # Unique words read
            'fix_duration': 'sum'  # Total reading time in ms
        })
        
        # Calculate reading speed (words per minute)
        reading_speed = (trial_stats['word_index'] / (trial_stats['fix_duration'] / 60000)).mean()
        reading_speeds[f'reading_speed_{time_constraint}'] = reading_speed
    
    return reading_speeds

def calculate_all_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate all metrics from the processed scanpath data.
    
    Args:
        df: DataFrame containing the processed scanpath data
        
    Returns:
        Dictionary containing all calculated metrics
    """
    # Calculate saccade lengths
    df = calculate_saccade_lengths(df)
    
    # Calculate all metrics
    metrics = {}
    
    # Basic metrics
    for time_constraint in df['time_constraint'].unique():
        time_data = df[df['time_constraint'] == time_constraint]
        
        # Average fixation duration
        metrics[f'avg_fix_duration_{time_constraint}'] = time_data['fix_duration'].mean()
        
        # Number of fixations per trial
        fixations_per_trial = time_data.groupby(['stimulus_index', 'participant_id']).size()
        metrics[f'avg_fixations_per_trial_{time_constraint}'] = fixations_per_trial.mean()
        
        # Average saccade length
        metrics[f'avg_saccade_length_{time_constraint}'] = time_data['saccade_length'].mean()
    
    # Add regression rates
    metrics.update(calculate_regression_rate(df))
    
    # Add skip rates
    metrics.update(calculate_skip_rate(df))
    
    # Add gaze durations
    metrics.update(calculate_gaze_duration(df))
    
    # Add reading speeds
    metrics.update(calculate_reading_speed(df))
    
    return metrics

def perform_statistical_analysis(df: pd.DataFrame) -> Dict:
    """
    Perform statistical analysis (ANOVA and post-hoc tests) on the metrics.
    
    Args:
        df: DataFrame containing the processed scanpath data
        
    Returns:
        Dictionary containing statistical test results
    """
    # Calculate saccade lengths if not already present
    if 'saccade_length' not in df.columns:
        df = calculate_saccade_lengths(df)
    
    # Prepare data for statistical analysis
    stats_results = {}
    
    # Metrics to analyze
    metrics_to_analyze = {
        'fix_duration': 'Fixation Duration (ms)',
        'saccade_length': 'Saccade Length (pixels)',
        'num_fixations': 'Number of Fixations',
        'skip_rate': 'Word Skip Rate (%)',
        'regression_rate': 'Regression Rate (%)'
    }
    
    # Calculate number of fixations per trial
    fixations_per_trial = df.groupby(['time_constraint', 'stimulus_index', 'participant_id']).size().reset_index(name='num_fixations')
    
    # Calculate skip rates and regression rates per trial
    skip_rates = []
    regression_rates = []
    for time_constraint in df['time_constraint'].unique():
        time_data = df[df['time_constraint'] == time_constraint].copy()
        time_data = time_data.sort_values(['participant_id', 'stimulus_index', 'fix_x'])
        time_data['next_word_index'] = time_data.groupby(['participant_id', 'stimulus_index'])['word_index'].shift(-1)
        
        # Calculate skip rate
        time_data['word_diff'] = time_data['next_word_index'] - time_data['word_index']
        skip_rate = time_data.groupby(['participant_id', 'stimulus_index'])['word_diff'].apply(lambda x: (x > 1).mean() * 100)
        skip_rates.extend([{'time_constraint': time_constraint, 'skip_rate': rate} for rate in skip_rate])
        
        # Calculate regression rate
        time_data['is_regression'] = time_data['next_word_index'] < time_data['word_index']
        regression_rate = time_data.groupby(['participant_id', 'stimulus_index'])['is_regression'].mean() * 100
        regression_rates.extend([{'time_constraint': time_constraint, 'regression_rate': rate} for rate in regression_rate])
    
    skip_rates_df = pd.DataFrame(skip_rates)
    regression_rates_df = pd.DataFrame(regression_rates)
    
    # Perform ANOVA for each metric
    for metric, metric_name in metrics_to_analyze.items():
        if metric == 'num_fixations':
            data = fixations_per_trial
        elif metric == 'skip_rate':
            data = skip_rates_df
        elif metric == 'regression_rate':
            data = regression_rates_df
        else:
            data = df
        
        # Perform one-way ANOVA
        groups = [group for _, group in data.groupby('time_constraint')[metric]]
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Store ANOVA results
        stats_results[f'{metric}_anova'] = {
            'f_statistic': float(f_stat),  # Convert numpy float to Python float
            'p_value': float(p_value),     # Convert numpy float to Python float
            'significant': bool(p_value < 0.05)  # Convert numpy bool to Python bool
        }
        
        # Perform post-hoc Tukey HSD test if ANOVA is significant
        if p_value < 0.05:
            mc = MultiComparison(data[metric], data['time_constraint'])
            tukey_result = mc.tukeyhsd()
            
            # Store Tukey HSD results
            stats_results[f'{metric}_tukey'] = {
                'groups': tukey_result.groupsunique.tolist(),
                'meandiffs': [float(x) for x in tukey_result.meandiffs],  # Convert numpy floats to Python floats
                'p_values': [float(x) for x in tukey_result.pvalues],     # Convert numpy floats to Python floats
                'significant': [bool(x) for x in (tukey_result.pvalues < 0.05)]  # Convert numpy bools to Python bools
            }
    
    # Calculate effect sizes (eta-squared) for significant results
    for metric, metric_name in metrics_to_analyze.items():
        if stats_results[f'{metric}_anova']['significant']:
            if metric == 'num_fixations':
                data = fixations_per_trial
            elif metric == 'skip_rate':
                data = skip_rates_df
            elif metric == 'regression_rate':
                data = regression_rates_df
            else:
                data = df
            
            # Calculate eta-squared
            groups = [group for _, group in data.groupby('time_constraint')[metric]]
            grand_mean = np.mean(data[metric])
            ss_total = np.sum((data[metric] - grand_mean) ** 2)
            ss_between = np.sum([len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups])
            eta_squared = ss_between / ss_total
            
            stats_results[f'{metric}_anova']['eta_squared'] = float(eta_squared)  # Convert numpy float to Python float
    
    return stats_results

def plot_metrics_with_stats(df: pd.DataFrame, stats_results: Dict, output_dir: str):
    """
    Create bar plots of the scanpath metrics with statistical significance indicators and standard deviation.
    
    Args:
        df: DataFrame containing the processed scanpath data
        stats_results: Dictionary containing statistical test results
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate saccade lengths if not already present
    if 'saccade_length' not in df.columns:
        df = calculate_saccade_lengths(df)
    
    # Calculate number of fixations per trial
    fixations_per_trial = df.groupby(['time_constraint', 'stimulus_index', 'participant_id']).size().reset_index(name='num_fixations')
    
    # Calculate skip rates and regression rates per trial
    skip_rates = []
    regression_rates = []
    for time_constraint in df['time_constraint'].unique():
        time_data = df[df['time_constraint'] == time_constraint].copy()
        time_data = time_data.sort_values(['participant_id', 'stimulus_index', 'fix_x'])
        time_data['next_word_index'] = time_data.groupby(['participant_id', 'stimulus_index'])['word_index'].shift(-1)
        
        # Calculate skip rate
        time_data['word_diff'] = time_data['next_word_index'] - time_data['word_index']
        skip_rate = time_data.groupby(['participant_id', 'stimulus_index'])['word_diff'].apply(lambda x: (x > 1).mean() * 100)
        skip_rates.extend([{'time_constraint': time_constraint, 'skip_rate': rate} for rate in skip_rate])
        
        # Calculate regression rate
        time_data['is_regression'] = time_data['next_word_index'] < time_data['word_index']
        regression_rate = time_data.groupby(['participant_id', 'stimulus_index'])['is_regression'].mean() * 100
        regression_rates.extend([{'time_constraint': time_constraint, 'regression_rate': rate} for rate in regression_rate])
    
    skip_rates_df = pd.DataFrame(skip_rates)
    regression_rates_df = pd.DataFrame(regression_rates)
    
    # Plot metrics with statistical significance
    metrics_to_plot = {
        'fix_duration': 'Fixation Duration (ms)',
        'saccade_length': 'Saccade Length (pixels)',
        'num_fixations': 'Number of Fixations',
        'skip_rate': 'Word Skip Rate (%)',
        'regression_rate': 'Regression Rate (%)'
    }
    
    # Set style
    plt.style.use('seaborn')
    
    for metric, ylabel in metrics_to_plot.items():
        # Create figure with specific size and DPI
        plt.figure(figsize=(10, 8), dpi=100)
        
        if metric == 'num_fixations':
            data = fixations_per_trial
        elif metric == 'skip_rate':
            data = skip_rates_df
        elif metric == 'regression_rate':
            data = regression_rates_df
        else:
            data = df
        
        # Create bar plot with error bars
        means = data.groupby('time_constraint')[metric].mean()
        stds = data.groupby('time_constraint')[metric].std()
        
        # Create bar plot
        bars = plt.bar(range(len(means)), means, yerr=stds, capsize=10, alpha=0.7, width=0.6)
        
        # Customize x-axis
        plt.xticks(range(len(means)), means.index, fontsize=12)
        plt.xlabel('Time Constraint (seconds)', fontsize=12, labelpad=10)
        
        # Customize y-axis
        plt.ylabel(ylabel, fontsize=12, labelpad=10)
        plt.yticks(fontsize=10)
        
        # Add title
        plt.title(f'{ylabel} by Time Constraint', fontsize=14, pad=20)
        
        # Add statistical significance indicators
        if stats_results[f'{metric}_anova']['significant']:
            tukey_results = stats_results[f'{metric}_tukey']
            groups = tukey_results['groups']
            p_values = tukey_results['p_values']
            
            # Calculate y-axis limits for significance bars
            y_max = means.max() + stds[means.idxmax()]
            y_min = means.min() - stds[means.idxmin()]
            y_range = y_max - y_min
            
            # Add significance bars with proper spacing
            for i, (g1, g2) in enumerate(zip(groups[:-1], groups[1:])):
                if p_values[i] < 0.05:
                    # Draw significance bar
                    bar_height = y_max + 0.05 * y_range
                    plt.plot([i, i+1], [bar_height, bar_height], 'k-', lw=1.5)
                    plt.text((i + i+1)/2, bar_height + 0.02 * y_range, '*', 
                            ha='center', va='bottom', fontsize=14)
            
            # Adjust y-axis limits to accommodate significance bars
            plt.ylim(y_min, y_max + 0.15 * y_range)
        
        # Add ANOVA results to plot
        anova_results = stats_results[f'{metric}_anova']
        stats_text = f'ANOVA: F={anova_results["f_statistic"]:.2f}, p={anova_results["p_value"]:.3f}'
        if anova_results['significant']:
            stats_text += f'\nη²={anova_results["eta_squared"]:.3f}'
        
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
        
        # Save figure with high quality
        plt.savefig(os.path.join(output_dir, f'{metric}_with_stats.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Define paths
    base_dir = "/home/baiy4/reader-agent-zuco"
    data_dir = os.path.join(base_dir, "data/human_data/bai_read_under_time_pressure/corrected_data_by_fix8/11_all_corrected_scanpaths_across_stimuli")
    metadata_dir = os.path.join(base_dir, "data/human_data/bai_read_under_time_pressure/stimuli/10_27_15_58_100_images_W1920H1080WS16_LS40_MARGIN400")
    output_dir = os.path.join(base_dir, "data/human_data/bai_read_under_time_pressure/calculated_effects")
    
    # Load data
    metadata = load_bbox_metadata(metadata_dir)
    scanpath_data = load_corrected_scanpath_data(data_dir)
    
    # Process data
    df = process_scanpath_data(scanpath_data, metadata)
    
    # Calculate metrics
    metrics = calculate_all_metrics(df)
    
    # Perform statistical analysis
    stats_results = perform_statistical_analysis(df)
    
    # Save metrics and statistical results to file
    results = {
        'metrics': metrics,
        'statistical_analysis': stats_results
    }
    with open(os.path.join(output_dir, 'metrics_and_stats.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create plots with statistical significance
    plot_metrics_with_stats(df, stats_results, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
