import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def plot_regression_analysis(json_file_path):
    """
    Create box plots showing the distribution of regression vs progression timing across all episodes.
    
    Args:
        json_file_path (str): Path to the JSON file containing simulation results
    """
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Collect all regression events across episodes
    all_regression_events = []
    
    for episode in data:
        episode_id = episode['episode_id']
        step_logs = episode['step_wise_log']
        max_steps = len(step_logs) - 1  # Exclude the final terminate step
        
        # Track the current sentence index to detect regression
        current_sentence_index = -1
        
        for step_idx, step in enumerate(step_logs):
            if step['terminate']:
                continue
                
            actual_reading_sentence_index = step['actual_reading_sentence_index']
            
            # Determine if this is a regression (going back to a previous sentence)
            # or progression (reading next sentence)
            if actual_reading_sentence_index is not None:
                if actual_reading_sentence_index < current_sentence_index:
                    # Regression: going back to a previous sentence
                    normalized_step = step_idx / max_steps
                    all_regression_events.append({
                        'episode': episode_id,
                        'normalized_step': normalized_step,
                        'action': 'regress',
                        'from_sentence': current_sentence_index,
                        'to_sentence': actual_reading_sentence_index
                    })
                elif actual_reading_sentence_index > current_sentence_index:
                    # Progression: reading next sentence
                    normalized_step = step_idx / max_steps
                    all_regression_events.append({
                        'episode': episode_id,
                        'normalized_step': normalized_step,
                        'action': 'progress',
                        'from_sentence': current_sentence_index,
                        'to_sentence': actual_reading_sentence_index
                    })
                
                current_sentence_index = actual_reading_sentence_index
    
    # Separate regression and progression events
    regression_steps = [event['normalized_step'] for event in all_regression_events if event['action'] == 'regress']
    progression_steps = [event['normalized_step'] for event in all_regression_events if event['action'] == 'progress']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create box plot data
    box_data = [progression_steps, regression_steps]
    box_labels = ['Read Next Sentence', 'Regress to Previous']
    box_colors = ['lightblue', 'lightcoral']
    
    # Create box plots
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                   boxprops=dict(facecolor='white', alpha=0.8),
                   medianprops=dict(color='black', linewidth=2),
                   whiskerprops=dict(color='black', linewidth=1.5),
                   capprops=dict(color='black', linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor='red', markersize=4))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize the plot
    ax.set_ylabel('Normalized Reading Steps (0 = start, 1 = end of longest episode)', fontsize=12)
    ax.set_title('Distribution of Reading Actions: When Do Regressions vs Progressions Occur?', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics as text
    total_events = len(all_regression_events)
    regression_count = len(regression_steps)
    progression_count = len(progression_steps)
    
    stats_text = f'Total Events: {total_events}\nRegressions: {regression_count} ({regression_count/total_events*100:.1f}%)\nProgressions: {progression_count} ({progression_count/total_events*100:.1f}%)'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add detailed statistics for each box
    if progression_steps:
        prog_stats = f'Progress (n={len(progression_steps)}):\nMean: {np.mean(progression_steps):.3f}\nMedian: {np.median(progression_steps):.3f}\nQ1: {np.percentile(progression_steps, 25):.3f}\nQ3: {np.percentile(progression_steps, 75):.3f}'
        ax.text(0.02, 0.85, prog_stats, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    
    if regression_steps:
        reg_stats = f'Regression (n={len(regression_steps)}):\nMean: {np.mean(regression_steps):.3f}\nMedian: {np.median(regression_steps):.3f}\nQ1: {np.percentile(regression_steps, 25):.3f}\nQ3: {np.percentile(regression_steps, 75):.3f}'
        ax.text(0.02, 0.65, reg_stats, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('regression_analysis_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Print summary statistics
    print(f"Analysis Summary:")
    print(f"Total reading events: {total_events}")
    print(f"Regressions: {regression_count} ({regression_count/total_events*100:.1f}%)")
    print(f"Progressions: {progression_count} ({progression_count/total_events*100:.1f}%)")
    
    if regression_steps:
        print(f"\nRegression timing statistics:")
        print(f"  Mean: {np.mean(regression_steps):.3f}")
        print(f"  Median: {np.median(regression_steps):.3f}")
        print(f"  Q1 (25th percentile): {np.percentile(regression_steps, 25):.3f}")
        print(f"  Q3 (75th percentile): {np.percentile(regression_steps, 75):.3f}")
        print(f"  Min: {min(regression_steps):.3f}")
        print(f"  Max: {max(regression_steps):.3f}")
    
    if progression_steps:
        print(f"\nProgression timing statistics:")
        print(f"  Mean: {np.mean(progression_steps):.3f}")
        print(f"  Median: {np.median(progression_steps):.3f}")
        print(f"  Q1 (25th percentile): {np.percentile(progression_steps, 25):.3f}")
        print(f"  Q3 (75th percentile): {np.percentile(progression_steps, 75):.3f}")
        print(f"  Min: {min(progression_steps):.3f}")
        print(f"  Max: {max(progression_steps):.3f}")
    
    return all_regression_events

def plot_appraisal_vs_regression(json_file_path):
    """
    Create a plot showing the relationship between sentence appraisal scores and regression frequency.
    
    Args:
        json_file_path (str): Path to the JSON file containing simulation results
    """
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Collect regression events and their corresponding appraisal scores
    regression_appraisals = []
    
    for episode in data:
        episode_id = episode['episode_id']
        step_logs = episode['step_wise_log']
        init_appraisals = episode['init_sentence_appraisal_scores_distribution']
        
        # Track the current sentence index to detect regression
        current_sentence_index = -1
        
        for step_idx, step in enumerate(step_logs):
            if step['terminate']:
                continue
                
            actual_reading_sentence_index = step['actual_reading_sentence_index']
            
            # Check if this is a regression
            if actual_reading_sentence_index is not None and actual_reading_sentence_index < current_sentence_index:
                # This is a regression - get the appraisal score of the target sentence
                if actual_reading_sentence_index < len(init_appraisals):
                    appraisal_score = init_appraisals[actual_reading_sentence_index]
                    if appraisal_score != -1:  # Only include valid appraisal scores
                        regression_appraisals.append(appraisal_score)
            
            if actual_reading_sentence_index is not None:
                current_sentence_index = actual_reading_sentence_index
    
    if not regression_appraisals:
        print("No regression events found!")
        return
    
    # Create histogram bins for appraisal scores
    appraisal_bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
    bin_centers = (appraisal_bins[:-1] + appraisal_bins[1:]) / 2
    
    # Count regressions in each appraisal bin
    regression_counts, _ = np.histogram(regression_appraisals, bins=appraisal_bins)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Histogram of regression counts vs appraisal scores
    bars = ax1.bar(bin_centers, regression_counts, width=0.08, alpha=0.7, color='red', edgecolor='black')
    ax1.set_xlabel('Sentence Appraisal Score', fontsize=12)
    ax1.set_ylabel('Number of Regressions', fontsize=12)
    ax1.set_title('Regressions vs Sentence Appraisal Scores', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, regression_counts):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Scatter plot with trend line
    # Group by appraisal scores and count regressions
    appraisal_counts = defaultdict(int)
    for appraisal in regression_appraisals:
        appraisal_counts[round(appraisal, 1)] += 1
    
    appraisals = list(appraisal_counts.keys())
    counts = list(appraisal_counts.values())
    
    ax2.scatter(appraisals, counts, s=100, alpha=0.7, color='red', edgecolor='black')
    ax2.set_xlabel('Sentence Appraisal Score', fontsize=12)
    ax2.set_ylabel('Number of Regressions', fontsize=12)
    ax2.set_title('Regressions vs Appraisal Scores (Scatter)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add trend line if we have enough data points
    if len(appraisals) > 1:
        z = np.polyfit(appraisals, counts, 1)
        p = np.poly1d(z)
        ax2.plot(appraisals, p(appraisals), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        correlation = np.corrcoef(appraisals, counts)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add statistics
    total_regressions = len(regression_appraisals)
    mean_appraisal = np.mean(regression_appraisals)
    median_appraisal = np.median(regression_appraisals)
    
    stats_text = f'Total Regressions: {total_regressions}\nMean Appraisal: {mean_appraisal:.3f}\nMedian Appraisal: {median_appraisal:.3f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('appraisal_vs_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Print analysis
    print(f"\nAppraisal vs Regression Analysis:")
    print(f"Total regressions analyzed: {total_regressions}")
    print(f"Mean appraisal score of regressed sentences: {mean_appraisal:.3f}")
    print(f"Median appraisal score of regressed sentences: {median_appraisal:.3f}")
    
    # Check if lower appraisals are more likely to receive regressions
    low_appraisal_regressions = sum(1 for a in regression_appraisals if a < 0.5)
    high_appraisal_regressions = sum(1 for a in regression_appraisals if a >= 0.5)
    
    print(f"Regressions to low appraisal sentences (<0.5): {low_appraisal_regressions} ({low_appraisal_regressions/total_regressions*100:.1f}%)")
    print(f"Regressions to high appraisal sentences (≥0.5): {high_appraisal_regressions} ({high_appraisal_regressions/total_regressions*100:.1f}%)")
    
    return regression_appraisals

def plot_regression_comprehension_simple(json_file_path):
    """
    Create a simple plot showing comprehension scores after regressions vs number of regressions.
    
    Args:
        json_file_path (str): Path to the JSON file containing simulation results
    """
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Collect comprehension scores after regressions
    comprehension_after_regressions = []
    
    for episode in data:
        episode_id = episode['episode_id']
        step_logs = episode['step_wise_log']
        
        # Track the current sentence index to detect regression
        current_sentence_index = -1
        
        for step_idx, step in enumerate(step_logs):
            if step['terminate']:
                continue
                
            actual_reading_sentence_index = step['actual_reading_sentence_index']
            
            # Check if this is a regression
            if actual_reading_sentence_index is not None and actual_reading_sentence_index < current_sentence_index:
                # This is a regression - get the comprehension score after this step
                if step_idx < len(step_logs) - 1 and not step_logs[step_idx + 1]['terminate']:
                    comprehension_after = step_logs[step_idx + 1]['on_going_comprehension_log_scalar']
                    comprehension_after_regressions.append(comprehension_after)
            
            if actual_reading_sentence_index is not None:
                current_sentence_index = actual_reading_sentence_index
    
    if not comprehension_after_regressions:
        print("No regression events found!")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # X-axis: number of regressions (1, 2, 3, ...)
    x_values = list(range(1, len(comprehension_after_regressions) + 1))
    
    # Y-axis: comprehension scores after regressions
    y_values = comprehension_after_regressions
    
    # Create scatter plot
    ax.scatter(x_values, y_values, s=80, alpha=0.7, color='red', edgecolor='black')
    
    # Add trend line
    if len(x_values) > 1:
        z = np.polyfit(x_values, y_values, 1)
        p = np.poly1d(z)
        ax.plot(x_values, p(x_values), "r--", alpha=0.8, linewidth=2, label=f'Trend line')
        
        # Calculate correlation
        correlation = np.corrcoef(x_values, y_values)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Customize the plot
    ax.set_xlabel('Number of Regressions', fontsize=12)
    ax.set_ylabel('Comprehension Score After Regression', fontsize=12)
    ax.set_title('Comprehension Scores After Regressions', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_comprehension = np.mean(y_values)
    median_comprehension = np.median(y_values)
    
    stats_text = f'Total Regressions: {len(comprehension_after_regressions)}\nMean Comprehension: {mean_comprehension:.3f}\nMedian Comprehension: {median_comprehension:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('regression_comprehension_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Print analysis
    print(f"\nRegression Comprehension Analysis:")
    print(f"Total regressions analyzed: {len(comprehension_after_regressions)}")
    print(f"Mean comprehension after regressions: {mean_comprehension:.3f}")
    print(f"Median comprehension after regressions: {median_comprehension:.3f}")
    print(f"Min comprehension: {min(y_values):.3f}")
    print(f"Max comprehension: {max(y_values):.3f}")
    
    return comprehension_after_regressions

def plot_regression_timing_vs_completion(json_file_path):
    """
    Plots when regressions occur relative to the completion of the first pass of reading.
    A scatter plot is generated where each point is a regression. The y-axis shows the episode ID
    and the x-axis shows the normalized time of the regression. A vertical line indicates
    the average time at which the first full read-through of all sentences is completed.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    regression_timings = []
    finish_timings = []
    
    regression_timings_per_episode = defaultdict(list)

    for episode in data:
        episode_id = episode['episode_id']
        step_logs = episode['step_wise_log']
        
        max_steps = len(step_logs) - 1
        if max_steps <= 0:
            continue

        first_finish_step = -1
        for i, step in enumerate(step_logs):
            if step['is_sentence_all_finished']:
                first_finish_step = i
                break
        
        if first_finish_step != -1:
            finish_timings.append(first_finish_step / max_steps)

        current_sentence_index = -1
        for step_idx, step in enumerate(step_logs):
            if step['terminate']:
                continue
                
            actual_reading_sentence_index = step.get('actual_reading_sentence_index')
            
            if actual_reading_sentence_index is not None:
                if step.get('is_regress', False) and actual_reading_sentence_index < current_sentence_index:
                    normalized_time = step_idx / max_steps
                    regression_timings.append(normalized_time)
                    regression_timings_per_episode[episode_id].append(normalized_time)

                current_sentence_index = actual_reading_sentence_index
    
    if not regression_timings:
        print("No regression events found to plot.")
        return

    avg_finish_time = np.mean(finish_timings) if finish_timings else -1

    plt.figure(figsize=(15, 8))
    
    y_vals = []
    x_vals = []
    for episode_id, timings in sorted(regression_timings_per_episode.items()):
        y_vals.extend([episode_id] * len(timings))
        x_vals.extend(timings)

    plt.scatter(x_vals, y_vals, alpha=0.6, label='Regression Event', s=50)

    if avg_finish_time != -1:
        plt.axvline(x=avg_finish_time, color='r', linestyle='--', linewidth=2, 
                    label=f'Avg. First Full Read Completion (t={avg_finish_time:.2f})')
        
    plt.xlabel("Normalized Time Step in Episode")
    plt.ylabel("Episode ID")
    plt.title("Timing of Regression Events During Reading Episodes")
    if regression_timings_per_episode:
        plt.yticks(sorted(list(regression_timings_per_episode.keys())))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('regression_timing_vs_completion.png', dpi=300)
    plt.show()
    plt.close()

    print("Regression timing plot generated: 'regression_timing_vs_completion.png'")
    if avg_finish_time != -1:
        before_count = sum(1 for t in regression_timings if t < avg_finish_time)
        after_count = sum(1 for t in regression_timings if t >= avg_finish_time)
        print(f"Average finish time for first pass: {avg_finish_time:.3f}")
        if regression_timings:
            print(f"Regressions before first pass completion: {before_count} ({before_count/len(regression_timings)*100:.1f}%)")
            print(f"Regressions after first pass completion: {after_count} ({after_count/len(regression_timings)*100:.1f}%)")

def plot_regression_count_vs_appraisal(json_file_path):
    """
    Generates a scatter plot of regression counts versus sentence appraisal scores.
    Each point represents a unique sentence appraisal score, with the y-axis showing
    how many times sentences with that score were regressed to.
    This version uses the dynamic appraisal score of the target sentence at the moment of regression.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    regression_target_appraisals = []
    for episode in data:
        step_logs = episode['step_wise_log']
        
        for step_idx, step in enumerate(step_logs):
            if step_idx == 0:
                continue

            if step.get('is_regress'):
                target_idx = step.get('actual_reading_sentence_index')
                if target_idx is None:
                    continue

                # Get appraisal distribution from the previous step
                prev_step = step_logs[step_idx - 1]
                appraisal_dist_before = prev_step.get('already_read_sentences_appraisal_scores_distribution')

                if appraisal_dist_before and target_idx < len(appraisal_dist_before):
                    appraisal_score = appraisal_dist_before[target_idx]
                    if appraisal_score is not None and appraisal_score != -1:
                        regression_target_appraisals.append(appraisal_score)

    if not regression_target_appraisals:
        print("No regression events with valid dynamic appraisals found.")
        return

    appraisal_counts = defaultdict(int)
    for appraisal in regression_target_appraisals:
        # Round to group similar appraisal scores
        appraisal_counts[round(appraisal, 2)] += 1
    
    appraisals = list(appraisal_counts.keys())
    counts = list(appraisal_counts.values())
    
    plt.figure(figsize=(12, 8))
    plt.scatter(appraisals, counts, s=100, alpha=0.7, color='blue', edgecolor='black')
    
    plt.xlabel("Target Sentence Appraisal at Time of Regression", fontsize=12)
    plt.ylabel('Number of Regressions', fontsize=12)
    plt.title('Regressions Count vs. Dynamic Appraisal of Target Sentence', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
    plt.tight_layout()
    plt.savefig('regression_count_vs_dynamic_appraisal.png', dpi=300)
    plt.show()
    plt.close()

    print("Regression count vs dynamic appraisal plot generated: 'regression_count_vs_dynamic_appraisal.png'")

def plot_triggering_appraisal_distribution(json_file_path):
    """
    Plots the distribution of sentence appraisal scores that trigger a regression.
    It collects 'sentence_appraisal_before_regress' for all regression events
    and plots a histogram to show the distribution of these scores.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    triggering_appraisals = []
    for episode in data:
        for step in episode['step_wise_log']:
            if step.get('is_regress'):
                appraisal = step.get('sentence_appraisal_before_regress')
                if appraisal is not None and appraisal != -1:
                    triggering_appraisals.append(appraisal)

    if not triggering_appraisals:
        print("No valid regression-triggering appraisal scores found.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(triggering_appraisals, bins=20, kde=True, color='purple')
    
    plt.xlabel("Sentence Appraisal Score Before Regression")
    plt.ylabel("Frequency")
    plt.title("Distribution of Appraisal Scores Triggering a Regression")
    
    mean_val = np.mean(triggering_appraisals)
    median_val = np.median(triggering_appraisals)
    
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='-', label=f'Median: {median_val:.2f}')
    
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('regression_triggering_appraisal_distribution.png', dpi=300)
    plt.show()
    plt.close()

    print("Plot generated: 'regression_triggering_appraisal_distribution.png'")
    print(f"Found {len(triggering_appraisals)} regression-triggering events with valid appraisal scores.")
    print(f"Mean appraisal score: {mean_val:.3f}")
    print(f"Median appraisal score: {median_val:.3f}")

if __name__ == "__main__":
    # Example usage
    # json_file_path = "/home/baiy4/reader-agent-zuco/step5/modules/rl_envs/text_comprehension_v0516/temp_sim_data/0706_text_comprehension_v0516_no_time_decay_00_rl_model_continual_270000000_steps/1000ep/raw_sim_results.json"
    json_file_path = "/home/baiy4/reader-agent-zuco/step5/modules/rl_envs/text_comprehension_v0516/temp_sim_data/0707_text_comprehension_v0516_no_time_decay_softmin_reward_function_01_rl_model_80000000_steps/1000ep/raw_sim_results.json"

    # Generate all three plots
    # events = plot_regression_analysis(json_file_path)
    # regression_appraisals = plot_appraisal_vs_regression(json_file_path)
    # comprehension_scores = plot_regression_comprehension_simple(json_file_path)

    # New plots
    plot_regression_timing_vs_completion(json_file_path)
    plot_regression_count_vs_appraisal(json_file_path)
    plot_triggering_appraisal_distribution(json_file_path)


    # TODO: I can plot:
    # 1. regress probability vs. appraisal score (then a regression line)
    # 2. TODO: depict how regressions are happening after the first passes (link this to the whole text reading time, use it as a bar, a threshold, then position the regressions)
