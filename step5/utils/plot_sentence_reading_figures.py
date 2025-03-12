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