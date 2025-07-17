import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple


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
    1. Reading speed: unique words read / time (wpm)
    2. Skip rate: number of skips (word_diff > 1) / total fixations
    3. Regression rate: number of regressions (next_word < current_word) / total fixations
    """
    # Read input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Process each episode
    processed_metrics = []
    for episode in data:
        fixation_sequence = episode['global_fixation_sequence']
        total_time = episode['total_time']
        
        # Calculate reading speed (wpm)
        # Count unique words read (excluding -1)
        unique_words_read = len(set([w for w in fixation_sequence if w != -1]))
        reading_speed = (unique_words_read / total_time) * 60
        
        # Calculate skip rate (using word_skip_percentage_by_saccades method)
        total_num_skip_saccades = 0
        for i in range(len(fixation_sequence) - 1):
            current_word_idx = fixation_sequence[i]
            next_word_idx = fixation_sequence[i + 1]
            if current_word_idx == -1 or next_word_idx == -1:
                continue  # Skip if word index is -1 (not mapped)
            skipped = next_word_idx - current_word_idx - 1
            if skipped > 0:
                total_num_skip_saccades += 1
        
        total_saccades = len(fixation_sequence) - 1
        skip_rate = (total_num_skip_saccades / total_saccades) if total_saccades > 0 else 0
        
        # Calculate regression rate (using revisit_percentage_by_fixations method)
        last_read_word_index = -1
        num_revisit_words = 0
        
        for word_idx in fixation_sequence:
            if word_idx == -1:
                continue  # Skip if word index is -1 (not mapped)
            if word_idx < last_read_word_index:
                num_revisit_words += 1
            else:
                last_read_word_index = word_idx
        
        total_fixations = len([idx for idx in fixation_sequence if idx != -1])
        regression_rate = (num_revisit_words / total_fixations) if total_fixations > 0 else 0
        
        # Store metrics
        episode_metrics = {
            'episode_index': episode['episode_index'],
            'stimulus_index': episode['stimulus_index'],
            'time_condition': episode['time_condition'],
            'total_time': total_time,
            'unique_words': unique_words_read,
            'reading_speed': reading_speed,
            'skip_rate': skip_rate,
            'regression_rate': regression_rate,
            'num_skips': total_num_skip_saccades,
            'num_regressions': num_revisit_words
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

def plot_metrics_comparison(human_data_path: str, sim_data: List[Dict], output_path: str):
    """
    Create comparison plots between human and simulation metrics.
    """
    # Load human data
    with open(human_data_path, 'r') as f:
        human_data = json.load(f)
    
    # Group simulation data by time condition
    sim_data_by_condition = defaultdict(list)
    for episode in sim_data:
        sim_data_by_condition[episode['time_condition']].append(episode)
    
    # Calculate means and stds for simulation data
    sim_metrics = {}
    for condition, episodes in sim_data_by_condition.items():
        reading_speeds = [ep['reading_speed'] for ep in episodes]
        skip_rates = [ep['skip_rate'] for ep in episodes]
        regression_rates = [ep['regression_rate'] for ep in episodes]
        
        sim_metrics[condition] = {
            'reading_speed_mean': np.mean(reading_speeds),
            'reading_speed_std': np.std(reading_speeds),
            'skip_rate_mean': np.mean(skip_rates),
            'skip_rate_std': np.std(skip_rates),
            'regression_rate_mean': np.mean(regression_rates),
            'regression_rate_std': np.std(regression_rates)
        }
    
    # Set up the figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Time conditions
    time_conditions = ['30s', '60s', '90s']
    x = np.arange(len(time_conditions))
    width = 0.35
    
    # Plot reading speed
    human_speeds = [human_data[cond]['reading_speed_mean'] for cond in time_conditions]
    sim_speeds = [sim_metrics[cond]['reading_speed_mean'] for cond in time_conditions]
    human_speeds_std = [human_data[cond]['reading_speed_std'] for cond in time_conditions]
    sim_speeds_std = [sim_metrics[cond]['reading_speed_std'] for cond in time_conditions]
    
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
    sim_skips = [sim_metrics[cond]['skip_rate_mean'] for cond in time_conditions]
    human_skips_std = [human_data[cond]['skip_rate_std'] for cond in time_conditions]
    sim_skips_std = [sim_metrics[cond]['skip_rate_std'] for cond in time_conditions]
    
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
    sim_regress = [sim_metrics[cond]['regression_rate_mean'] for cond in time_conditions]
    human_regress_std = [human_data[cond]['regression_rate_std'] for cond in time_conditions]
    sim_regress_std = [sim_metrics[cond]['regression_rate_std'] for cond in time_conditions]
    
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

def process_fixation_sequences_to_metrics(input_file: str, output_file: str):
    """
    Process fixation sequences to calculate metrics for each episode.
    Output format: [{episode_index, stimulus_index, time_condition, total_time, reading_speed, skip_rate, regression_rate}, ...]
    """
    # Read input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Process each episode
    processed_metrics = []
    for episode in data:
        fixation_sequence = episode['global_fixation_sequence']
        total_time = episode['total_time']
        
        # Calculate reading speed (wpm)
        # unique_words_read = len(set([w for w in fixation_sequence if w != -1]))
        words_read = len(fixation_sequence)
        reading_speed = (words_read / total_time) * 60
        
        # Calculate skip rate (using word_skip_percentage_by_saccades method)
        total_num_skip_saccades = 0
        for i in range(len(fixation_sequence) - 1):
            current_word_idx = fixation_sequence[i]
            next_word_idx = fixation_sequence[i + 1]
            if current_word_idx == -1 or next_word_idx == -1:
                continue  # Skip if word index is -1 (not mapped)
            skipped = next_word_idx - current_word_idx - 1
            if skipped > 0:
                total_num_skip_saccades += 1
        
        total_saccades = len(fixation_sequence) - 1
        skip_rate = (total_num_skip_saccades / total_saccades) if total_saccades > 0 else 0
        
        # Calculate regression rate (using revisit_percentage_by_fixations method)
        last_read_word_index = -1
        num_revisit_words = 0
        
        for word_idx in fixation_sequence:
            if word_idx == -1:
                continue  # Skip if word index is -1 (not mapped)
            if word_idx < last_read_word_index:
                num_revisit_words += 1
            else:
                last_read_word_index = word_idx
        
        total_fixations = len([idx for idx in fixation_sequence if idx != -1])
        regression_rate = (num_revisit_words / total_fixations) if total_fixations > 0 else 0
        
        # Store metrics
        episode_metrics = {
            'episode_index': episode['episode_index'],
            'stimulus_index': episode['stimulus_index'],
            'time_condition': episode['time_condition'],
            'total_time': total_time,
            'reading_speed': reading_speed,
            'skip_rate': skip_rate,
            'regression_rate': regression_rate,
            'num_skips': total_num_skip_saccades,
            'num_regressions': num_revisit_words,
            # 'unique_words': unique_words_read
        }
        
        processed_metrics.append(episode_metrics)
    
    # Write results to output file
    with open(output_file, 'w') as f:
        json.dump(processed_metrics, f, indent=4)

if __name__ == "__main__":
    # Get the current directory
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # File name
    # file_name = "20250619_2202_trials1_stims9_conds3"
    file_name = "20250717_2124_trials1_stims9_conds3"

    input_file = os.path.join(current_dir, "simulated_results", file_name, "all_simulation_results.json")
    output_file = os.path.join(current_dir, "simulated_results", file_name, "processed_reading_metrics.json")
    fixation_sequence_file = os.path.join(current_dir, "simulated_results", file_name, "processed_fixation_sequences.json")
    fixation_metrics_file = os.path.join(current_dir, "simulated_results", file_name, "analyzed_fixation_metrics.json")
    human_metrics_file = os.path.join(current_dir, "processed_human_data", "analyzed_human_metrics.json")
    comparison_plot_file = os.path.join(current_dir, "simulated_results", file_name, "metrics_comparison.png")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process both metrics and fixation sequences
    process_simulation_results_to_fixation_sequence(input_file, fixation_sequence_file)
    process_fixation_sequences_to_metrics(fixation_sequence_file, fixation_metrics_file)
    
    # Load simulation results and create comparison plots
    with open(fixation_metrics_file, 'r') as f:
        sim_data = json.load(f)
    plot_metrics_comparison(human_metrics_file, sim_data, comparison_plot_file)
