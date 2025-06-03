import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison
import sys
from calculate_metrics_for_user_study import AggregatedFixationMetrics

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

def load_sentence_metadata(metadata_dir: str) -> Dict:
    """
    Load the sentence metadata from the JSON file.
    
    Args:
        metadata_dir: Directory containing the metadata files
        
    Returns:
        Dictionary containing the sentence metadata
    """
    metadata_path = os.path.join(metadata_dir, "assets", "metadata_sentence_indeces.json")
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
        
        # Calculate regression rate by reading progress
        last_read_word_index = time_data.groupby(['participant_id', 'stimulus_index'])['word_index'].max()
        total_words_to_last_read = last_read_word_index + 1  # Assuming word indices start from 0
        
        # Calculate regression rate by fixations
        regression_rate_by_fixations = time_data['is_regression'].mean() * 100
        
        # Calculate regression rate by reading progress
        regression_rate_by_progress = (time_data['is_regression'].sum() / total_words_to_last_read.mean()) * 100
        
        regression_rates[f'regression_rate_by_fixations_{time_constraint}'] = regression_rate_by_fixations
        regression_rates[f'regression_rate_by_progress_{time_constraint}'] = regression_rate_by_progress
    
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
        
        # Calculate skip rate by reading progress
        last_read_word_index = time_data.groupby(['participant_id', 'stimulus_index'])['word_index'].max()
        total_words_to_last_read = last_read_word_index + 1  # Assuming word indices start from 0
        
        # Calculate skip rate by saccades
        skip_rate_by_saccades = (time_data['word_diff'] > 1).mean() * 100
        
        # Calculate skip rate by reading progress
        skip_rate_by_progress = (time_data['word_diff'].apply(lambda x: x > 1).sum() / total_words_to_last_read.mean()) * 100
        
        skip_rates[f'skip_rate_by_saccades_{time_constraint}'] = skip_rate_by_saccades
        skip_rates[f'skip_rate_by_progress_{time_constraint}'] = skip_rate_by_progress
    
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
        
        # Calculate reading speed using time constraint
        reading_speed_by_constraint = (trial_stats['word_index'] / time_constraint * 60).mean()
        
        # Calculate reading speed using total fixation duration
        reading_speed_by_fixations = (trial_stats['word_index'] / (trial_stats['fix_duration'] / 60000)).mean()
        
        reading_speeds[f'reading_speed_by_constraint_{time_constraint}'] = reading_speed_by_constraint
        reading_speeds[f'reading_speed_by_fixations_{time_constraint}'] = reading_speed_by_fixations
    
    return reading_speeds

def get_sentence_id(word_index: int, sentences: List[Dict]) -> int:
    """
    Get the sentence ID for a given word index.
    
    Args:
        word_index: The word index to look up
        sentences: List of sentence dictionaries with start_idx and end_idx
        
    Returns:
        The sentence ID (0-based index) or -1 if not found
    """
    for i, sentence in enumerate(sentences):
        if sentence['start_idx'] <= word_index <= sentence['end_idx']:
            return i
    return -1

def calculate_sentence_regression_metrics(df: pd.DataFrame, sentence_metadata: List[Dict]) -> Dict:
    """
    Calculate sentence-level regression metrics.
    
    Args:
        df: DataFrame containing the processed scanpath data
        sentence_metadata: List of stimulus metadata with sentence information
        
    Returns:
        Dictionary containing sentence regression metrics by time constraint
    """
    regression_metrics = {}
    
    for time_constraint in df['time_constraint'].unique():
        time_data = df[df['time_constraint'] == time_constraint].copy()
        
        # Initialize metrics
        total_fixations = 0
        regressed_fixations = 0
        total_sentences_read = 0
        regressed_sentences = set()
        revisited_sentences = {}  # Track how many times each sentence is revisited
        
        # Process each trial
        for (stimulus_id, participant_id), trial_data in time_data.groupby(['stimulus_index', 'participant_id']):
            # Get sentence metadata for this stimulus
            stimulus_meta = next((s for s in sentence_metadata if s['stimulus_id'] == stimulus_id), None)
            if not stimulus_meta:
                print(f"Warning: No metadata found for stimulus {stimulus_id}")
                continue
                
            # Sort fixations by time
            trial_data = trial_data.sort_values('fix_x')
            
            # Track reading progress
            current_sentence_id = -1
            max_sentence_id = -1
            sentence_fixations = {}  # Track fixations per sentence
            sentence_visits = {}  # Track number of visits per sentence
            
            # Process each fixation
            for _, fixation in trial_data.iterrows():
                if fixation['word_index'] == -1:  # Skip invalid word indices
                    continue
                    
                total_fixations += 1
                
                # Get sentence ID for this fixation
                sentence_id = get_sentence_id(fixation['word_index'], stimulus_meta['sentences'])
                if sentence_id == -1:
                    print(f"Warning: No sentence found for word index {fixation['word_index']} in stimulus {stimulus_id}")
                    continue
                
                # Update sentence fixations
                if sentence_id not in sentence_fixations:
                    sentence_fixations[sentence_id] = 0
                sentence_fixations[sentence_id] += 1
                
                # Update sentence visits
                if sentence_id not in sentence_visits:
                    sentence_visits[sentence_id] = 0
                sentence_visits[sentence_id] += 1
                
                # Update reading progress
                if sentence_id > max_sentence_id:
                    max_sentence_id = sentence_id
                    current_sentence_id = sentence_id
                
                # Check for regression
                if sentence_id < current_sentence_id:
                    regressed_fixations += 1
                    regressed_sentences.add(sentence_id)
            
            # Count revisited sentences (sentences visited more than once)
            for sentence_id, visits in sentence_visits.items():
                if visits > 1:
                    if sentence_id not in revisited_sentences:
                        revisited_sentences[sentence_id] = 0
                    revisited_sentences[sentence_id] += 1
            
            # Count total sentences read
            total_sentences_read += len(sentence_fixations)
        
        # Calculate metrics
        if total_fixations > 0:
            fixation_regression_rate = (regressed_fixations / total_fixations * 100)
        else:
            fixation_regression_rate = 0.0
            
        if total_sentences_read > 0:
            sentence_regression_rate = (len(regressed_sentences) / total_sentences_read * 100)
        else:
            sentence_regression_rate = 0.0
            
        num_trials = len(time_data.groupby(['stimulus_index', 'participant_id']))
        if num_trials > 0:
            avg_revisited_sentences = len(revisited_sentences) / num_trials
        else:
            avg_revisited_sentences = 0.0
        
        # Store metrics with explicit float conversion
        regression_metrics[f'fixation_regression_rate_{time_constraint}'] = float(fixation_regression_rate)
        regression_metrics[f'sentence_regression_rate_{time_constraint}'] = float(sentence_regression_rate)
        regression_metrics[f'avg_revisited_sentences_{time_constraint}'] = float(avg_revisited_sentences)
    
    return regression_metrics

def calculate_all_metrics(df: pd.DataFrame, sentence_metadata: List[Dict]) -> Dict:
    """
    Calculate all metrics from the processed scanpath data using AggregatedFixationMetrics.
    
    Args:
        df: DataFrame containing the processed scanpath data
        sentence_metadata: List of stimulus metadata with sentence information
        
    Returns:
        Dictionary containing all calculated metrics
    """
    metrics = {}
    
    for time_constraint in df['time_constraint'].unique():
        time_data = df[df['time_constraint'] == time_constraint].copy()
        
        # Process each trial
        for (stimulus_id, participant_id), trial_data in time_data.groupby(['stimulus_index', 'participant_id']):
            # Convert trial data to list of fixation dictionaries
            fixation_data = trial_data.to_dict('records')
            
            # Create AggregatedFixationMetrics instance
            metrics_calculator = AggregatedFixationMetrics(
                fixation_data=fixation_data,
                time_constraint=time_constraint
            )
            
            # Calculate metrics
            trial_metrics = metrics_calculator.compute_all_metrics()
            
            # Store metrics with time constraint
            for metric_name, value in trial_metrics.items():
                key = f'{metric_name}_{time_constraint}'
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
    
    # Calculate means for each metric
    final_metrics = {}
    for key, values in metrics.items():
        final_metrics[key] = float(np.mean(values))
    
    return final_metrics

def perform_statistical_analysis(df: pd.DataFrame, metrics: Dict) -> Dict:
    """
    Perform statistical analysis (ANOVA and post-hoc tests) on the metrics.
    
    Args:
        df: DataFrame containing the processed scanpath data
        metrics: Dictionary containing the calculated metrics
        
    Returns:
        Dictionary containing statistical test results
    """
    # Prepare data for statistical analysis
    stats_results = {}
    
    # Calculate metrics for each trial
    trial_metrics = {}
    
    # Group data by time constraint and trial
    for time_constraint in df['time_constraint'].unique():
        time_data = df[df['time_constraint'] == time_constraint].copy()
        
        # Process each trial
        for (stimulus_id, participant_id), trial_data in time_data.groupby(['stimulus_index', 'participant_id']):
            # Convert trial data to list of fixation dictionaries
            fixation_data = trial_data.to_dict('records')
            
            # Create AggregatedFixationMetrics instance
            metrics_calculator = AggregatedFixationMetrics(
                fixation_data=fixation_data,
                time_constraint=time_constraint
            )
            
            # Calculate metrics
            trial_metrics[f'{time_constraint}_{stimulus_id}_{participant_id}'] = metrics_calculator.compute_all_metrics()
    
    # Metrics to analyze
    metrics_to_analyze = {
        'Number of Fixations': 'Number of Fixations',
        'Average Saccade Length (px)': 'Average Saccade Length (pixels)',
        'Word Skip Percentage by Saccades V2 With Word Index Correction': 'Word Skip Rate (%)',
        'Revisit Percentage by Saccades V2 With Word Index Correction': 'Word Regression Rate (%)',
        'Average Reading Speed (wpm)': 'Reading Speed (wpm)'
    }
    
    # Perform ANOVA for each metric
    for metric_key, metric_name in metrics_to_analyze.items():
        # Extract values for each time constraint
        time_constraints = []
        metric_values = []
        metric_stds = []
        
        # Group trial metrics by time constraint
        time_groups = {}
        for key, value in trial_metrics.items():
            time_constraint = int(key.split('_')[0])
            if time_constraint not in time_groups:
                time_groups[time_constraint] = []
            if metric_key in value:
                time_groups[time_constraint].append(value[metric_key])
        
        # Calculate means and standard deviations
        for time_constraint in sorted(time_groups.keys()):
            values = time_groups[time_constraint]
            if values:
                time_constraints.append(time_constraint)
                metric_values.append(np.mean(values))
                metric_stds.append(np.std(values))
        
        if not metric_values:
            print(f"Warning: No data found for metric {metric_key}")
            continue
        
        # Perform one-way ANOVA
        groups = [time_groups[tc] for tc in sorted(time_groups.keys())]
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Store ANOVA results
            stats_results[f'{metric_key}_anova'] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
            
            # Perform post-hoc Tukey HSD test if ANOVA is significant
            if p_value < 0.05:
                # Prepare data for Tukey test
                values = []
                groups_labels = []
                for tc in sorted(time_groups.keys()):
                    values.extend(time_groups[tc])
                    groups_labels.extend([tc] * len(time_groups[tc]))
                
                mc = MultiComparison(values, groups_labels)
                tukey_result = mc.tukeyhsd()
                
                # Store Tukey HSD results
                stats_results[f'{metric_key}_tukey'] = {
                    'groups': tukey_result.groupsunique.tolist(),
                    'meandiffs': [float(x) for x in tukey_result.meandiffs],
                    'p_values': [float(x) for x in tukey_result.pvalues],
                    'significant': [bool(x) for x in (tukey_result.pvalues < 0.05)]
                }
            
            # Calculate eta-squared
            all_values = []
            for group in groups:
                all_values.extend(group)
            grand_mean = np.mean(all_values)
            ss_total = np.sum([(v - grand_mean) ** 2 for v in all_values])
            ss_between = np.sum([len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups])
            eta_squared = ss_between / ss_total if ss_total != 0 else 0
            
            stats_results[f'{metric_key}_anova']['eta_squared'] = float(eta_squared)
        else:
            print(f"Warning: Not enough data for ANOVA on metric {metric_key}")
            stats_results[f'{metric_key}_anova'] = {
                'f_statistic': None,
                'p_value': None,
                'significant': False,
                'eta_squared': None
            }
        
        # Store the values for plotting
        stats_results[f'{metric_key}_values'] = {
            'time_constraints': time_constraints,
            'values': metric_values,
            'stds': metric_stds
        }
    
    return stats_results

def plot_metrics_with_stats(df: pd.DataFrame, results: Dict, output_dir: str):
    """
    Create bar plots of the scanpath metrics with statistical significance indicators and standard deviation.
    
    Args:
        df: DataFrame containing the processed scanpath data
        results: Dictionary containing metrics and statistical analysis results
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get metrics from results
    metrics = results['metrics']
    stats_results = results['statistical_analysis']
    
    # Define metrics to plot with their display names
    metrics_to_plot = {
        'Number of Fixations': 'Number of Fixations',
        'Average Saccade Length (px)': 'Average Saccade Length (pixels)',
        'Word Skip Percentage by Saccades V2 With Word Index Correction': 'Word Skip Rate (%)',
        'Revisit Percentage by Saccades V2 With Word Index Correction': 'Word Regression Rate (%)',
        'Average Reading Speed (wpm)': 'Reading Speed (wpm)'
    }
    
    # Set style
    plt.style.use('seaborn')
    
    for metric_key, ylabel in metrics_to_plot.items():
        # Create figure with specific size and DPI
        plt.figure(figsize=(12, 8), dpi=100)
        
        # Get values from statistical analysis results
        if f'{metric_key}_values' not in stats_results:
            print(f"Warning: No data found for metric {metric_key}")
            plt.close()
            continue
            
        values_data = stats_results[f'{metric_key}_values']
        time_constraints = values_data['time_constraints']
        metric_values = values_data['values']
        metric_stds = values_data['stds']
        
        # Create bar plot with error bars
        bars = plt.bar(range(len(metric_values)), metric_values, alpha=0.7, width=0.6,
                      yerr=metric_stds, capsize=5)
        
        # Add value annotations on top of each bar
        for i, (value, std) in enumerate(zip(metric_values, metric_stds)):
            if np.isfinite(value):
                plt.text(i, value + std + 0.02 * (max(metric_values) - min(metric_values)),
                        f'{value:.1f} ± {std:.1f}',
                        ha='center', va='bottom', fontsize=10)
        
        # Customize x-axis
        plt.xticks(range(len(time_constraints)), time_constraints, fontsize=12)
        plt.xlabel('Time Constraint (seconds)', fontsize=12, labelpad=10)
        
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
                
                # Add significance bars with detailed annotations
                bar_height = y_max + 0.15 * y_range
                if np.isfinite(bar_height):
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
                            # Handle very small p-values
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
                if np.isfinite(y_min) and np.isfinite(y_max):
                    plt.ylim(y_min, y_max + 0.4 * y_range)
        
        # Add ANOVA results to plot if available
        if f'{metric_key}_anova' in stats_results:
            anova_results = stats_results[f'{metric_key}_anova']
            stats_text = f'ANOVA Results:\n'
            stats_text += f'F({len(time_constraints)-1}, {len(time_constraints)*len(metric_values)-len(time_constraints)}) = {anova_results["f_statistic"]:.2f}\n'
            
            # Handle very small p-values
            if anova_results['p_value'] < 1e-10:
                stats_text += 'p < 1e-10\n'
            else:
                stats_text += f'p = {anova_results["p_value"]:.3e}\n'
            
            if anova_results['significant']:
                stats_text += f'η² = {anova_results["eta_squared"]:.3f}'
            
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
        plt.savefig(os.path.join(output_dir, f'{metric_key.replace(" ", "_")}_with_stats.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def calculate_sentence_level_revisit(df: pd.DataFrame, sentence_metadata: List[Dict]) -> Dict:
    """
    Calculate sentence-level revisit metrics.
    A sentence is considered regressed if it falls behind the latest sentence index that has fixations.
    
    Args:
        df: DataFrame containing the processed scanpath data
        sentence_metadata: List of stimulus metadata with sentence information
        
    Returns:
        Dictionary containing sentence-level revisit metrics by time constraint
    """
    revisit_metrics = {}
    
    for time_constraint in df['time_constraint'].unique():
        time_data = df[df['time_constraint'] == time_constraint].copy()
        
        # Initialize metrics
        total_trials = 0
        total_regressed_sentences = 0
        total_sentences_read = 0
        
        # Process each trial
        for (stimulus_id, participant_id), trial_data in time_data.groupby(['stimulus_index', 'participant_id']):
            # Get sentence metadata for this stimulus
            stimulus_meta = next((s for s in sentence_metadata if s['stimulus_id'] == stimulus_id), None)
            if not stimulus_meta:
                print(f"Warning: No metadata found for stimulus {stimulus_id}")
                continue
            
            # Sort fixations by time
            trial_data = trial_data.sort_values('fix_x')
            
            # Track reading progress
            max_sentence_idx = -1
            regressed_sentences = set()
            sentences_read = set()
            
            # Process each fixation
            for _, fixation in trial_data.iterrows():
                if fixation['word_index'] == -1:  # Skip invalid word indices
                    continue
                
                # Get sentence ID for this fixation
                sentence_id = get_sentence_id(fixation['word_index'], stimulus_meta['sentences'])
                if sentence_id == -1:
                    continue
                
                # Update sentences read
                sentences_read.add(sentence_id)
                
                # Update max sentence index
                if sentence_id > max_sentence_idx:
                    max_sentence_idx = sentence_id
                # If current sentence is behind max sentence, it's a regression
                elif sentence_id < max_sentence_idx:
                    regressed_sentences.add(sentence_id)
            
            # Update metrics
            total_trials += 1
            total_regressed_sentences += len(regressed_sentences)
            total_sentences_read += len(sentences_read)
        
        # Calculate metrics
        if total_trials > 0:
            avg_regressed_sentences = total_regressed_sentences / total_trials
            avg_sentences_read = total_sentences_read / total_trials
            regression_rate = (total_regressed_sentences / total_sentences_read * 100) if total_sentences_read > 0 else 0
        else:
            avg_regressed_sentences = 0
            avg_sentences_read = 0
            regression_rate = 0
        
        # Store metrics
        revisit_metrics[f'sentence_level_revisit_{time_constraint}'] = {
            'avg_regressed_sentences': float(avg_regressed_sentences),
            'avg_sentences_read': float(avg_sentences_read),
            'regression_rate': float(regression_rate)
        }
    
    return revisit_metrics

def plot_sentence_revisit_metrics(metrics: Dict, output_dir: str):
    """
    Create simple bar plots for sentence-level revisit metrics.
    
    Args:
        metrics: Dictionary containing the calculated metrics
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract sentence-level revisit metrics
    time_constraints = []
    avg_regressed = []
    regression_rates = []
    
    for time_constraint in [30, 60, 90]:
        key = f'sentence_level_revisit_{time_constraint}'
        if key in metrics:
            time_constraints.append(time_constraint)
            avg_regressed.append(metrics[key]['avg_regressed_sentences'])
            regression_rates.append(metrics[key]['regression_rate'])
    
    # Set style
    plt.style.use('seaborn')
    
    # Plot 1: Average Number of Regressed Sentences
    plt.figure(figsize=(10, 6))
    bars = plt.bar(time_constraints, avg_regressed, width=20, alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.xlabel('Time Constraint (seconds)')
    plt.ylabel('Average Number of Regressed Sentences')
    plt.title('Average Number of Regressed Sentences by Time Constraint')
    plt.xticks(time_constraints)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentence_regression_count.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Regression Rate
    plt.figure(figsize=(10, 6))
    bars = plt.bar(time_constraints, regression_rates, width=20, alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.xlabel('Time Constraint (seconds)')
    plt.ylabel('Regression Rate (%)')
    plt.title('Sentence Regression Rate by Time Constraint')
    plt.xticks(time_constraints)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentence_regression_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Define paths
    base_dir = "/home/baiy4/reader-agent-zuco"
    data_dir = os.path.join(base_dir, "data/human_data/bai_read_under_time_pressure/corrected_data_by_fix8/11_all_corrected_scanpaths_across_stimuli")
    metadata_dir = os.path.join(base_dir, "data/human_data/bai_read_under_time_pressure/stimuli/10_27_15_58_100_images_W1920H1080WS16_LS40_MARGIN400")
    output_dir = os.path.join(base_dir, "data/human_data/bai_read_under_time_pressure/calculated_effects")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    metadata = load_bbox_metadata(metadata_dir)
    sentence_metadata = load_sentence_metadata(metadata_dir)
    scanpath_data = load_corrected_scanpath_data(data_dir)
    
    # Process data
    df = process_scanpath_data(scanpath_data, metadata)
    
    # Calculate metrics
    metrics = calculate_all_metrics(df, sentence_metadata)
    
    # Calculate sentence-level revisit metrics
    revisit_metrics = calculate_sentence_level_revisit(df, sentence_metadata)
    
    # Add revisit metrics to the main metrics dictionary
    metrics.update(revisit_metrics)
    
    # Save final_metrics to a separate JSON file
    final_metrics_file = os.path.join(output_dir, 'final_metrics.json')
    with open(final_metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Final metrics saved to {final_metrics_file}")
    
    # Perform statistical analysis
    stats_results = perform_statistical_analysis(df, metrics)
    
    # Save metrics and statistical results to file
    results = {
        'metrics': metrics,
        'statistical_analysis': stats_results
    }
    with open(os.path.join(output_dir, 'metrics_and_stats.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create plots with statistical significance
    plot_metrics_with_stats(df, results, output_dir)
    
    # Create sentence-level revisit plots
    plot_sentence_revisit_metrics(metrics, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
