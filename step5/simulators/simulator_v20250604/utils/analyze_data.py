import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple

def calculate_reading_metrics_for_trial(episode_log: Dict) -> Dict:
    """
    Calculate reading metrics for a single trial/episode.
    """
    # Initialize metrics
    furthest_word_index = -1  # Track the furthest word index reached
    total_fixations = 0
    total_skips = 0
    total_regressions = 0
    total_regressed_fixations = 0
    
    # Process each text reading log within the episode
    for individual_episode_log in episode_log['text_reading_logs']:
        sentence_index = individual_episode_log['current_sentence_index']
        sentence_summary = individual_episode_log['sentence_reading_summary']
        
        # Skip if this is a regressed sentence
        if individual_episode_log['action_information']['read_or_regress_action'] == 'regress':
            continue
            
        # Update total fixations
        total_fixations += sentence_summary['num_steps_or_fixations']
        
        # Calculate skips (only for first-pass reading)
        total_skips += len(sentence_summary['skipped_words_indexes'])
        
        # Calculate regressions and regressed fixations
        regressed_words = set(sentence_summary['regressed_words_indexes'])
        for reading_log in individual_episode_log['sentence_reading_logs']:
            if reading_log['action'] == 'regress':
                total_regressions += 1
                if reading_log['current_word_index'] not in regressed_words:
                    total_regressed_fixations += 1
        
        # Update reading progress - track the furthest word index reached
        current_sentence_words = sentence_summary['num_words_in_sentence']
        current_word_index = sentence_summary.get('current_word_index', current_sentence_words - 1)
        furthest_word_index = max(furthest_word_index, current_word_index)
    
    # Calculate metrics
    time_condition_str_key = episode_log['time_condition']
    time_condition_value = episode_log['total_time']
    
    # Calculate reading speed based on furthest word index reached
    reading_speed = ((furthest_word_index + 1) / time_condition_value) * 60  # words per minute
    skip_rate = total_skips / total_fixations if total_fixations > 0 else 0
    regression_rate = total_regressed_fixations / total_fixations if total_fixations > 0 else 0
    
    return {
        'time_condition': time_condition_str_key,
        'time_condition_value': time_condition_value,
        'reading_speed': reading_speed,
        'skip_rate': skip_rate,
        'regression_rate': regression_rate,
        'furthest_word_index': furthest_word_index,
        'total_fixations': total_fixations,
        'total_skips': total_skips,
        'total_regressions': total_regressions,
        'total_regressed_fixations': total_regressed_fixations
    }

def calculate_statistics(metrics_list: List[Dict]) -> Dict:
    """
    Calculate mean and standard deviation for each metric across trials.
    """
    # Initialize dictionaries to store arrays of values
    metric_arrays = {
        'reading_speed': [],
        'skip_rate': [],
        'regression_rate': [],
        'furthest_word_index': [],
        'total_fixations': [],
        'total_skips': [],
        'total_regressions': [],
        'total_regressed_fixations': []
    }
    
    # Collect values for each metric
    for trial_metrics in metrics_list:
        for metric_name in metric_arrays.keys():
            metric_arrays[metric_name].append(trial_metrics[metric_name])
    
    # Calculate statistics
    statistics = {}
    for metric_name, values in metric_arrays.items():
        statistics[f'{metric_name}_mean'] = np.mean(values)
        statistics[f'{metric_name}_std'] = np.std(values)
    
    # Add time condition
    statistics['time_condition'] = metrics_list[0]['time_condition']
    
    return statistics

def analyze_simulation_results(input_file: str, output_file: str):
    """
    Analyze simulation results and output metrics to a JSON file.
    """
    # Read input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Group results by time condition
    time_condition_results = defaultdict(list)
    for episode_log in data:
        time_condition = episode_log['time_condition']
        time_condition_results[time_condition].append(episode_log)
    
    # Calculate metrics for each time condition
    metrics_by_time = {}
    for time_condition, episode_logs in time_condition_results.items():
        # Calculate metrics for each trial
        trial_metrics = [calculate_reading_metrics_for_trial(episode_log) for episode_log in episode_logs]
        # Calculate statistics across trials
        metrics_by_time[time_condition] = calculate_statistics(trial_metrics)
    
    # Write results to output file
    with open(output_file, 'w') as f:
        json.dump(metrics_by_time, f, indent=4)

def process_simulation_results_to_fixation_sequence(input_file: str, output_file: str):
    """
    Process simulation results to extract concatenated global fixation sequences.
    Output format: [{episode_index, stimulus_index, time_condition, total_time, global_fixation_sequence}, ...]
    """
    # Read input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Process each episode
    processed_results = []
    for episode in data:
        # Initialize the processed episode data
        processed_episode = {
            'episode_index': episode['episode_index'],
            'stimulus_index': episode['stimulus_index'],
            'time_condition': episode['time_condition'],
            'total_time': episode['total_time'],
            'global_fixation_sequence': []
        }
        
        # Concatenate global fixation sequences from all sentences
        for text_reading_log in episode['text_reading_logs']:
            if 'sentence_reading_summary' in text_reading_log:
                global_fixations = text_reading_log['sentence_reading_summary'].get('global_actual_fixation_sequence_in_text', [])
                processed_episode['global_fixation_sequence'].extend(global_fixations)
        
        processed_results.append(processed_episode)
    
    # Write results to output file
    with open(output_file, 'w') as f:
        json.dump(processed_results, f, indent=4)

def analyze_fixation_sequences(input_file: str, output_file: str):
    """
    Analyze reading metrics from processed fixation sequences.
    Metrics:
    1. Reading speed: furthest index reached + 1 / time
    2. First-pass skip rate: number of skips (non-consecutive fixations) in first pass / furthest index + 1
    3. Regression rate: number of regressions (fixations behind current furthest) / furthest index + 1
    """
    # Read input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Process each episode
    processed_metrics = []
    for episode in data:
        fixation_sequence = episode['global_fixation_sequence']
        total_time = episode['total_time']
        
        # Find furthest index reached
        furthest_index = max(fixation_sequence) if fixation_sequence else -1
        
        # Calculate reading speed
        reading_speed = ((furthest_index + 1) / total_time) * 60  # words per minute
        
        # Calculate first-pass skip rate
        first_pass_fixations = []
        current_max = -1
        skips = 0
        
        for i, fix in enumerate(fixation_sequence):
            if fix > current_max:
                current_max = fix
                first_pass_fixations.append(fix)
                # Check if this fixation is a skip (non-consecutive)
                if i > 0 and fix - first_pass_fixations[-2] > 1:
                    skips += 1
        
        skip_rate = skips / (furthest_index + 1) if furthest_index >= 0 else 0
        
        # Calculate regression rate
        regressions = 0
        current_max = -1
        
        for fix in fixation_sequence:
            if fix < current_max:
                regressions += 1
            current_max = max(current_max, fix)
        
        regression_rate = regressions / (furthest_index + 1) if furthest_index >= 0 else 0
        
        # Store metrics
        episode_metrics = {
            'episode_index': episode['episode_index'],
            'stimulus_index': episode['stimulus_index'],
            'time_condition': episode['time_condition'],
            'total_time': total_time,
            'furthest_index': furthest_index,
            'reading_speed': reading_speed,
            'skip_rate': skip_rate,
            'regression_rate': regression_rate,
            'num_skips': skips,
            'num_regressions': regressions
        }
        
        processed_metrics.append(episode_metrics)
    
    # Group by time condition and calculate statistics
    time_condition_metrics = defaultdict(list)
    for metrics in processed_metrics:
        time_condition_metrics[metrics['time_condition']].append(metrics)
    
    # Calculate statistics for each time condition
    final_results = {}
    for time_condition, metrics_list in time_condition_metrics.items():
        # Calculate means and standard deviations
        reading_speeds = [m['reading_speed'] for m in metrics_list]
        skip_rates = [m['skip_rate'] for m in metrics_list]
        regression_rates = [m['regression_rate'] for m in metrics_list]
        
        final_results[time_condition] = {
            'reading_speed_mean': np.mean(reading_speeds),
            'reading_speed_std': np.std(reading_speeds),
            'skip_rate_mean': np.mean(skip_rates),
            'skip_rate_std': np.std(skip_rates),
            'regression_rate_mean': np.mean(regression_rates),
            'regression_rate_std': np.std(regression_rates),
            'num_episodes': len(metrics_list)
        }
    
    # Write results to output file
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=4)

def plot_metrics_comparison(human_data_path: str, sim_data: Dict, output_path: str):
    """
    Create comparison plots between human and simulation metrics.
    """
    # Load human data
    with open(human_data_path, 'r') as f:
        human_data = json.load(f)
    
    # Set up the figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Time conditions
    time_conditions = ['30s', '60s', '90s']
    x = np.arange(len(time_conditions))
    width = 0.35
    
    # Plot reading speed
    human_speeds = [human_data[cond]['reading_speed_mean'] for cond in time_conditions]
    sim_speeds = [sim_data[cond]['reading_speed_mean'] for cond in time_conditions]
    human_speeds_std = [human_data[cond]['reading_speed_std'] for cond in time_conditions]
    sim_speeds_std = [sim_data[cond]['reading_speed_std'] for cond in time_conditions]
    
    bars1 = ax1.bar(x - width/2, human_speeds, width, label='Human', color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, sim_speeds, width, label='Simulation', color='green', alpha=0.7)
    ax1.errorbar(x - width/2, human_speeds, yerr=human_speeds_std, fmt='none', color='black', capsize=5)
    ax1.errorbar(x + width/2, sim_speeds, yerr=sim_speeds_std, fmt='none', color='black', capsize=5)
    
    # Add value annotations
    for bar, std in zip(bars1, human_speeds_std):
        height = bar.get_height()
        ax1.annotate(f'{height:.1f} ({std:.1f})',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    for bar, std in zip(bars2, sim_speeds_std):
        height = bar.get_height()
        ax1.annotate(f'{height:.1f} ({std:.1f})',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax1.set_ylabel('Reading Speed (wpm)')
    ax1.set_title('Reading Speed Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(time_conditions)
    ax1.legend()
    
    # Plot skip rate
    human_skips = [human_data[cond]['skip_rate_mean'] for cond in time_conditions]
    sim_skips = [sim_data[cond]['skip_rate_mean'] for cond in time_conditions]
    human_skips_std = [human_data[cond]['skip_rate_std'] for cond in time_conditions]
    sim_skips_std = [sim_data[cond]['skip_rate_std'] for cond in time_conditions]
    
    bars1 = ax2.bar(x - width/2, human_skips, width, label='Human', color='blue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, sim_skips, width, label='Simulation', color='green', alpha=0.7)
    ax2.errorbar(x - width/2, human_skips, yerr=human_skips_std, fmt='none', color='black', capsize=5)
    ax2.errorbar(x + width/2, sim_skips, yerr=sim_skips_std, fmt='none', color='black', capsize=5)
    
    # Add value annotations
    for bar, std in zip(bars1, human_skips_std):
        height = bar.get_height()
        ax2.annotate(f'{height:.2f} ({std:.2f})',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    for bar, std in zip(bars2, sim_skips_std):
        height = bar.get_height()
        ax2.annotate(f'{height:.2f} ({std:.2f})',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax2.set_ylabel('Skip Rate')
    ax2.set_title('Skip Rate Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(time_conditions)
    ax2.legend()
    
    # Plot regression rate
    human_regress = [human_data[cond]['regression_rate_mean'] for cond in time_conditions]
    sim_regress = [sim_data[cond]['regression_rate_mean'] for cond in time_conditions]
    human_regress_std = [human_data[cond]['regression_rate_std'] for cond in time_conditions]
    sim_regress_std = [sim_data[cond]['regression_rate_std'] for cond in time_conditions]
    
    bars1 = ax3.bar(x - width/2, human_regress, width, label='Human', color='blue', alpha=0.7)
    bars2 = ax3.bar(x + width/2, sim_regress, width, label='Simulation', color='green', alpha=0.7)
    ax3.errorbar(x - width/2, human_regress, yerr=human_regress_std, fmt='none', color='black', capsize=5)
    ax3.errorbar(x + width/2, sim_regress, yerr=sim_regress_std, fmt='none', color='black', capsize=5)
    
    # Add value annotations
    for bar, std in zip(bars1, human_regress_std):
        height = bar.get_height()
        ax3.annotate(f'{height:.2f} ({std:.2f})',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    for bar, std in zip(bars2, sim_regress_std):
        height = bar.get_height()
        ax3.annotate(f'{height:.2f} ({std:.2f})',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax3.set_ylabel('Regression Rate')
    ax3.set_title('Regression Rate Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(time_conditions)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Get the current directory
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # File name
    file_name = "20250614_2133_trials1_stims9_conds3"

    input_file = os.path.join(current_dir, "simulated_results", file_name, "all_simulation_results.json")
    output_file = os.path.join(current_dir, "simulated_results", file_name, "processed_reading_metrics.json")
    fixation_sequence_file = os.path.join(current_dir, "simulated_results", file_name, "processed_fixation_sequences.json")
    fixation_metrics_file = os.path.join(current_dir, "simulated_results", file_name, "analyzed_fixation_metrics.json")
    human_metrics_file = os.path.join(current_dir, "processed_human_data", "analyzed_human_metrics.json")
    comparison_plot_file = os.path.join(current_dir, "simulated_results", file_name, "metrics_comparison.png")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process both metrics and fixation sequences
    # analyze_simulation_results(input_file, output_file)
    process_simulation_results_to_fixation_sequence(input_file, fixation_sequence_file)
    analyze_fixation_sequences(fixation_sequence_file, fixation_metrics_file)
    
    # Load simulation results and create comparison plots
    with open(fixation_metrics_file, 'r') as f:
        sim_data = json.load(f)
    plot_metrics_comparison(human_metrics_file, sim_data, comparison_plot_file)
