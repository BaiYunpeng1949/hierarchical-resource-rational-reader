import os
import json
import numpy as np
import matplotlib.pyplot as plt

def save_json_file(logs_across_episodes, log_dir):
    """
    Save the logs to a JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Save to JSON file
    json_path = os.path.join(log_dir, 'reading_behavior_logs.json')
    with open(json_path, 'w') as f:
        json.dump(logs_across_episodes, f, indent=4)
    print(f"Logs saved to {json_path}")
    return json_path

def plot_reading_behavior_summary(log_dir, skipping_rates, regression_rates, sentence_lengths):
    """
    Create summary plots for reading behavior analysis
    """
    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Plot skipping rates distribution
    plt.figure(figsize=(10, 6))
    plt.hist(skipping_rates, bins=20, alpha=0.7)
    plt.xlabel('Skipping Rate (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Word Skipping Rates')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'skipping_rates_distribution.png'))
    plt.close()

    # Plot regression rates distribution
    plt.figure(figsize=(10, 6))
    plt.hist(regression_rates, bins=20, alpha=0.7)
    plt.xlabel('Regression Rate (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Regression Rates')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'regression_rates_distribution.png'))
    plt.close()

    # Plot relationship between sentence length and reading behaviors
    plt.figure(figsize=(10, 6))
    plt.scatter(sentence_lengths, skipping_rates, alpha=0.5, label='Skipping Rate')
    plt.scatter(sentence_lengths, regression_rates, alpha=0.5, label='Regression Rate')
    
    # Add trend lines
    z1 = np.polyfit(sentence_lengths, skipping_rates, 1)
    z2 = np.polyfit(sentence_lengths, regression_rates, 1)
    p1 = np.poly1d(z1)
    p2 = np.poly1d(z2)
    plt.plot(sentence_lengths, p1(sentence_lengths), "--", alpha=0.8)
    plt.plot(sentence_lengths, p2(sentence_lengths), "--", alpha=0.8)
    
    plt.xlabel('Sentence Length')
    plt.ylabel('Rate (%)')
    plt.title('Reading Behaviors vs Sentence Length')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'reading_behaviors_vs_length.png'))
    plt.close()

def plot_skipping_vs_predictability(log_dir, word_predictabilities, skipping_decisions):
    """
    Create an enhanced visualization of the relationship between word predictability and skipping probability.
    
    Args:
        log_dir: Directory to save the plot
        word_predictabilities: List of word predictability values
        skipping_decisions: List of binary skipping decisions (1 for skipped, 0 for not skipped)
    """
    plt.figure(figsize=(10, 6))
    
    # Convert inputs to numpy arrays
    word_predictabilities = np.array(word_predictabilities)
    skipping_decisions = np.array(skipping_decisions)
    
    # Create bins for predictability values
    num_bins = 10
    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Initialize arrays to store probabilities and counts
    skip_probs = []
    skip_probs_std = []
    sample_sizes = []
    
    # Calculate skipping probability for each bin
    for i in range(len(bins) - 1):
        mask = (word_predictabilities >= bins[i]) & (word_predictabilities < bins[i + 1])
        decisions_in_bin = skipping_decisions[mask]
        
        if len(decisions_in_bin) > 0:
            prob = np.mean(decisions_in_bin)
            std = np.std(decisions_in_bin) / np.sqrt(len(decisions_in_bin))  # Standard error
            skip_probs.append(prob)
            skip_probs_std.append(std)
            sample_sizes.append(len(decisions_in_bin))
        else:
            skip_probs.append(0)
            skip_probs_std.append(0)
            sample_sizes.append(0)
    
    # Create bar plot
    bars = plt.bar(bin_centers, skip_probs, width=(bins[1]-bins[0])*0.8, 
                  yerr=skip_probs_std, capsize=5, alpha=0.6, color='skyblue',
                  label='Skipping probability')
    
    # Add trend line using polynomial fit
    valid_indices = np.array(sample_sizes) > 0
    if np.sum(valid_indices) > 1:  # Need at least 2 points for fitting
        z = np.polyfit(bin_centers[valid_indices], np.array(skip_probs)[valid_indices], 1)
        p = np.poly1d(z)
        plt.plot(bin_centers, p(bin_centers), 'r--', 
                label=f'Trend line (slope: {z[0]:.3f})', linewidth=2)
    
    # Calculate correlation coefficient
    valid_mask = np.array(sample_sizes) > 0
    if np.sum(valid_mask) > 1:
        correlation = np.corrcoef(bin_centers[valid_mask], np.array(skip_probs)[valid_mask])[0,1]
    else:
        correlation = 0
    
    # Add sample size annotations on top of bars
    for i, (bar, size) in enumerate(zip(bars, sample_sizes)):
        if size > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'n={size}', ha='center', va='bottom')
    
    plt.xlabel('Word Predictability')
    plt.ylabel('Probability of Skipping')
    plt.title('Relationship between Word Predictability and Skipping Probability\n' + 
              f'Correlation coefficient: {correlation:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(log_dir, 'skipping_vs_predictability.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print statistical summary
    print("\nStatistical Summary of Predictability vs Skipping:")
    print(f"Overall correlation coefficient: {correlation:.3f}")
    print(f"Total number of words analyzed: {sum(sample_sizes)}")
    print("\nPredictability bins and skipping probabilities:")
    for i in range(len(bins)-1):
        if sample_sizes[i] > 0:
            print(f"Bin [{bins[i]:.2f}-{bins[i+1]:.2f}]: {skip_probs[i]:.3f} ± {skip_probs_std[i]:.3f} (n={sample_sizes[i]})")

def plot_regression_vs_appraisals(log_dir, regressed_words, word_appraisals):
    """
    Create visualization showing relationship between word appraisals and regression decisions.
    
    Args:
        log_dir: Directory to save the plot
        regressed_words: List of word indices that were regressed to
        word_appraisals: List of appraisal values for each word
    """
    plt.figure(figsize=(10, 6))
    
    # Convert inputs to numpy arrays if they aren't already
    word_appraisals = np.array(word_appraisals)
    
    # Create bins for appraisal values
    num_bins = 10
    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Initialize arrays to store regression probabilities and counts
    regression_probs = []
    regression_probs_std = []
    sample_sizes = []
    
    # Calculate regression probability for each bin
    for i in range(len(bins) - 1):
        mask = (word_appraisals >= bins[i]) & (word_appraisals < bins[i + 1])
        words_in_bin = np.where(mask)[0]
        if len(words_in_bin) > 0:
            regression_prob = np.mean([1 if idx in regressed_words else 0 for idx in words_in_bin])
            std = np.std([1 if idx in regressed_words else 0 for idx in words_in_bin]) / np.sqrt(len(words_in_bin))
            regression_probs.append(regression_prob)
            regression_probs_std.append(std)
            sample_sizes.append(len(words_in_bin))
        else:
            regression_probs.append(0)
            regression_probs_std.append(0)
            sample_sizes.append(0)
    
    # Create bar plot
    bars = plt.bar(bin_centers, regression_probs, width=(bins[1]-bins[0])*0.8,
                  yerr=regression_probs_std, capsize=5, alpha=0.6, color='salmon',
                  label='Regression probability')
    
    # Add trend line
    valid_indices = np.array(sample_sizes) > 0
    if np.sum(valid_indices) > 1:
        z = np.polyfit(bin_centers[valid_indices], np.array(regression_probs)[valid_indices], 1)
        p = np.poly1d(z)
        plt.plot(bin_centers, p(bin_centers), 'r--',
                label=f'Trend line (slope: {z[0]:.3f})', linewidth=2)
    
    # Calculate correlation coefficient
    valid_mask = np.array(sample_sizes) > 0
    if np.sum(valid_mask) > 1:
        correlation = np.corrcoef(bin_centers[valid_mask], np.array(regression_probs)[valid_mask])[0,1]
    else:
        correlation = 0
    
    # Add sample size annotations
    for i, (bar, size) in enumerate(zip(bars, sample_sizes)):
        if size > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'n={size}', ha='center', va='bottom')
    
    plt.xlabel('Word Appraisal Value')
    plt.ylabel('Probability of Regression')
    plt.title('Relationship between Word Appraisal and Regression Probability\n' +
              f'Correlation coefficient: {correlation:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(log_dir, 'regression_vs_appraisals.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print statistical summary
    print("\nStatistical Summary of Appraisals vs Regression:")
    print(f"Overall correlation coefficient: {correlation:.3f}")
    print(f"Total number of words analyzed: {sum(sample_sizes)}")
    print("\nAppraisal bins and regression probabilities:")
    for i in range(len(bins)-1):
        if sample_sizes[i] > 0:
            print(f"Bin [{bins[i]:.2f}-{bins[i+1]:.2f}]: {regression_probs[i]:.3f} ± {regression_probs_std[i]:.3f} (n={sample_sizes[i]})")

def plot_reading_behavior_metrics(log_dir, logs_across_episodes):
    """
    Create comprehensive visualization of reading behavior metrics.
    
    Args:
        log_dir: Directory to save the plot
        logs_across_episodes: List of episode logs containing reading behavior data
    """
    # Extract metrics
    skipping_rates = [log['skipping_rate'] for log in logs_across_episodes]
    regression_rates = [log['regression_rate'] for log in logs_across_episodes]
    sentence_lengths = [log['sentence_length'] for log in logs_across_episodes]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Skipping vs Regression Rate Scatter
    plt.subplot(2, 2, 1)
    plt.scatter(skipping_rates, regression_rates, alpha=0.5)
    plt.xlabel('Skipping Rate (%)')
    plt.ylabel('Regression Rate (%)')
    plt.title('Skipping vs Regression Rates')
    
    # Add trend line
    z = np.polyfit(skipping_rates, regression_rates, 1)
    p = np.poly1d(z)
    plt.plot(skipping_rates, p(skipping_rates), "r--", alpha=0.8)
    correlation = np.corrcoef(skipping_rates, regression_rates)[0,1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top')
    
    # 2. Metrics Distribution
    plt.subplot(2, 2, 2)
    plt.hist(skipping_rates, bins=20, alpha=0.5, label='Skipping', color='blue')
    plt.hist(regression_rates, bins=20, alpha=0.5, label='Regression', color='red')
    plt.xlabel('Rate (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reading Behaviors')
    plt.legend()
    
    # 3. Metrics vs Sentence Length
    plt.subplot(2, 2, 3)
    plt.scatter(sentence_lengths, skipping_rates, alpha=0.5, label='Skipping', color='blue')
    plt.scatter(sentence_lengths, regression_rates, alpha=0.5, label='Regression', color='red')
    plt.xlabel('Sentence Length')
    plt.ylabel('Rate (%)')
    plt.title('Reading Behaviors vs Sentence Length')
    plt.legend()
    
    # 4. Summary Statistics
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats_text = (
        f"Summary Statistics:\n\n"
        f"Skipping Rate:\n"
        f"  Mean: {np.mean(skipping_rates):.2f}%\n"
        f"  Std: {np.std(skipping_rates):.2f}%\n"
        f"  Median: {np.median(skipping_rates):.2f}%\n\n"
        f"Regression Rate:\n"
        f"  Mean: {np.mean(regression_rates):.2f}%\n"
        f"  Std: {np.std(regression_rates):.2f}%\n"
        f"  Median: {np.median(regression_rates):.2f}%\n\n"
        f"Sentence Length:\n"
        f"  Mean: {np.mean(sentence_lengths):.1f}\n"
        f"  Std: {np.std(sentence_lengths):.1f}\n"
        f"  Range: [{min(sentence_lengths)}, {max(sentence_lengths)}]"
    )
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'reading_behavior_metrics.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print overall statistics
    print("\nOverall Reading Behavior Statistics:")
    print(f"Average Skipping Rate: {np.mean(skipping_rates):.2f}% (±{np.std(skipping_rates):.2f}%)")
    print(f"Average Regression Rate: {np.mean(regression_rates):.2f}% (±{np.std(regression_rates):.2f}%)")
    print(f"Average Sentence Length: {np.mean(sentence_lengths):.1f} (±{np.std(sentence_lengths):.1f})")
    print(f"Correlation between Skipping and Regression: {correlation:.3f}")