import os
import matplotlib.pyplot as plt
import json
import numpy as np
from collections import defaultdict
from scipy.stats import linregress

def compute_average_fixations(xs, ys):
    """
    Groups data by unique x-values and computes the mean of y-values per group.
    Returns (x_sorted, y_means) for plotting.
    """
    aggregator = defaultdict(list)
    for x_val, y_val in zip(xs, ys):
        aggregator[x_val].append(y_val)
    
    x_unique_sorted = sorted(aggregator.keys())
    y_means = [np.mean(aggregator[xv]) for xv in x_unique_sorted]
    return x_unique_sorted, y_means

def analyze_priors_effect(json_data, save_file_dir):
    """
    Generates fixation analysis plots for word frequency and predictability
    across different word lengths. Saves results in separate folders.
    Includes linear regression, averaged line plots, and universal plots.
    """
    data = json.loads(json_data)
    os.makedirs(save_file_dir, exist_ok=True)
    
    # Collect data by word length
    word_lengths = set()
    fixations_by_length = defaultdict(lambda: {"freq": [], "fixations": [], "pred": []})
    universal_freq = []
    universal_pred = []
    universal_fixations = []
    universal_word_lengths = []
    
    for episode in data:
        w_len = episode["word_len"]
        w_freq = episode["word_frequency"]
        w_pred = episode.get("Word predictability", 0.0)
        num_fixations = sum(1 for f in episode["fixations"] if not f["done"])
        
        word_lengths.add(w_len)
        fixations_by_length[w_len]["freq"].append(w_freq)
        fixations_by_length[w_len]["fixations"].append(num_fixations)
        fixations_by_length[w_len]["pred"].append(w_pred)
        
        universal_freq.append(w_freq)
        universal_pred.append(w_pred)
        universal_fixations.append(num_fixations)
        universal_word_lengths.append(w_len)
    
    # Create separate directories
    freq_dir = os.path.join(save_file_dir, "word_frequency")
    pred_dir = os.path.join(save_file_dir, "word_predictability")
    universal_dir = os.path.join(save_file_dir, "universal_plots")
    os.makedirs(freq_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(universal_dir, exist_ok=True)

    # Generate plots for each word length
    for w_len in sorted(word_lengths):
        freq_x = fixations_by_length[w_len]["freq"]
        fix_y = fixations_by_length[w_len]["fixations"]
        pred_x = fixations_by_length[w_len]["pred"]
        
        # Scatter plot: Fixations vs. Word Frequency
        plt.figure(figsize=(8, 6))
        plt.scatter(freq_x, fix_y, alpha=0.7)
        plt.xlabel("Word Frequency")
        plt.ylabel("Number of Fixations")
        plt.title(f"Fixations vs. Word Frequency | Word Length = {w_len}")
        plt.grid(True)
        plt.savefig(os.path.join(freq_dir, f"fixations_vs_word_freq_len_{w_len}.png"))
        plt.close()
        
        # Linear regression plot
        slope, intercept, _, _, _ = linregress(freq_x, fix_y)
        plt.figure(figsize=(8, 6))
        plt.scatter(freq_x, fix_y, alpha=0.7, label="Data")
        plt.plot(freq_x, np.array(freq_x) * slope + intercept, color='red', label="Linear Fit")
        plt.xlabel("Word Frequency")
        plt.ylabel("Number of Fixations")
        plt.title(f"Linear Regression: Fixations vs. Word Frequency | Word Length = {w_len}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(freq_dir, f"linear_fixations_vs_word_freq_len_{w_len}.png"))
        plt.close()
        
        # Line chart connecting averages
        x_sorted, y_means = compute_average_fixations(freq_x, fix_y)
        plt.figure(figsize=(8, 6))
        plt.plot(x_sorted, y_means, marker='o', linestyle='-', color='blue')
        plt.xlabel("Word Frequency")
        plt.ylabel("Average Number of Fixations")
        plt.title(f"Average Fixations vs. Word Frequency | Word Length = {w_len}")
        plt.grid(True)
        plt.savefig(os.path.join(freq_dir, f"avg_fixations_vs_word_freq_len_{w_len}.png"))
        plt.close()
        
        # Scatter plot: Fixations vs. Word Predictability
        plt.figure(figsize=(8, 6))
        plt.scatter(pred_x, fix_y, alpha=0.7)
        plt.xlabel("Word Predictability")
        plt.ylabel("Number of Fixations")
        plt.title(f"Fixations vs. Word Predictability | Word Length = {w_len}")
        plt.grid(True)
        plt.savefig(os.path.join(pred_dir, f"fixations_vs_word_pred_len_{w_len}.png"))
        plt.close()
        
        # Linear regression plot
        slope, intercept, _, _, _ = linregress(pred_x, fix_y)
        plt.figure(figsize=(8, 6))
        plt.scatter(pred_x, fix_y, alpha=0.7, label="Data")
        plt.plot(pred_x, np.array(pred_x) * slope + intercept, color='red', label="Linear Fit")
        plt.xlabel("Word Predictability")
        plt.ylabel("Number of Fixations")
        plt.title(f"Linear Regression: Fixations vs. Word Predictability | Word Length = {w_len}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(pred_dir, f"linear_fixations_vs_word_pred_len_{w_len}.png"))
        plt.close()
        
        # Line chart connecting averages
        x_sorted, y_means = compute_average_fixations(pred_x, fix_y)
        plt.figure(figsize=(8, 6))
        plt.plot(x_sorted, y_means, marker='o', linestyle='-', color='green')
        plt.xlabel("Word Predictability")
        plt.ylabel("Average Number of Fixations")
        plt.title(f"Average Fixations vs. Word Predictability | Word Length = {w_len}")
        plt.grid(True)
        plt.savefig(os.path.join(pred_dir, f"avg_fixations_vs_word_pred_len_{w_len}.png"))
        plt.close()
    
    # Generate universal plots
    def generate_universal_plots(x_data, y_data, xlabel, filename, color):
        plt.figure(figsize=(8, 6))
        plt.scatter(x_data, y_data, alpha=0.7)
        plt.xlabel(xlabel)
        plt.ylabel("Number of Fixations")
        plt.title(f"Fixations vs. {xlabel} (Universal)")
        plt.grid(True)
        plt.savefig(os.path.join(universal_dir, f"fixations_vs_{filename}.png"))
        plt.close()
        
        slope, intercept, _, _, _ = linregress(x_data, y_data)
        plt.figure(figsize=(8, 6))
        plt.scatter(x_data, y_data, alpha=0.7, label="Data")
        plt.plot(x_data, np.array(x_data) * slope + intercept, color='red', label="Linear Fit")
        plt.xlabel(xlabel)
        plt.ylabel("Number of Fixations")
        plt.title(f"Linear Regression: Fixations vs. {xlabel} (Universal)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(universal_dir, f"linear_fixations_vs_{filename}.png"))
        plt.close()
        
        x_sorted, y_means = compute_average_fixations(x_data, y_data)
        plt.figure(figsize=(8, 6))
        plt.plot(x_sorted, y_means, marker='o', linestyle='-', color=color)
        plt.xlabel(xlabel)
        plt.ylabel("Average Number of Fixations")
        plt.title(f"Average Fixations vs. {xlabel} (Universal)")
        plt.grid(True)
        plt.savefig(os.path.join(universal_dir, f"avg_fixations_vs_{filename}.png"))
        plt.close()
    
    # generate_universal_plots(universal_word_lengths, universal_fixations, "Word Length", "word_length", "blue")
    generate_universal_plots(universal_freq, universal_fixations, "Word Frequency", "word_frequency", "green")
    generate_universal_plots(universal_pred, universal_fixations, "Word Predictability", "word_predictability", "purple")
    
    print(f"Plots saved successfully in {save_file_dir}")


def analyze_word_length_effect(json_data, save_file_dir):
    """
    Generates fixation analysis plots for word prior probability.
    Splits data into 10 bins based on word prior probability (0.0-1.0 range) and 
    creates scatter plots, linear regression plots, and averaged line plots.
    Additionally, generates universal plots without bin separation.
    """
    data = json.loads(json_data)
    os.makedirs(save_file_dir, exist_ok=True)
    
    # Collect data by word prior probability bins
    bins = np.linspace(0, 1, 11)  # 10 bins: [0-0.1), [0.1-0.2), ..., [0.9-1.0]
    fixations_by_prior = defaultdict(lambda: {"word_lengths": [], "fixations": []})
    universal_word_lengths = []
    universal_fixations = []
    
    for episode in data:
        w_prior = episode["word_prior_prob"]
        w_len = episode["word_len"]
        num_fixations = sum(1 for f in episode["fixations"] if not f["done"])
        
        universal_word_lengths.append(w_len)
        universal_fixations.append(num_fixations)
        
        # Assign to the appropriate bin
        for i in range(10):
            if bins[i] <= w_prior < bins[i + 1]:
                fixations_by_prior[i]["word_lengths"].append(w_len)
                fixations_by_prior[i]["fixations"].append(num_fixations)
                break
    
    # Create separate directory for prior probability
    prior_dir = os.path.join(save_file_dir, "word_prior_probability")
    universal_dir = os.path.join(save_file_dir, "universal_plots")
    os.makedirs(prior_dir, exist_ok=True)
    os.makedirs(universal_dir, exist_ok=True)
    
    # Generate plots for each prior bin
    for i in range(10):
        word_x = fixations_by_prior[i]["word_lengths"]
        fix_y = fixations_by_prior[i]["fixations"]
        
        if not word_x:  # Skip empty bins
            continue
        
        # Scatter plot: Fixations vs. Word Length
        plt.figure(figsize=(8, 6))
        plt.scatter(word_x, fix_y, alpha=0.7)
        plt.xlabel("Word Length")
        plt.ylabel("Number of Fixations")
        plt.title(f"Fixations vs. Word Length | Prior Bin {bins[i]:.1f}-{bins[i+1]:.1f}")
        plt.grid(True)
        plt.savefig(os.path.join(prior_dir, f"fixations_vs_word_length_bin_{i}.png"))
        plt.close()
        
        # Linear regression plot
        slope, intercept, _, _, _ = linregress(word_x, fix_y)
        plt.figure(figsize=(8, 6))
        plt.scatter(word_x, fix_y, alpha=0.7, label="Data")
        plt.plot(word_x, np.array(word_x) * slope + intercept, color='red', label="Linear Fit")
        plt.xlabel("Word Length")
        plt.ylabel("Number of Fixations")
        plt.title(f"Linear Regression: Fixations vs. Word Length | Prior Bin {bins[i]:.1f}-{bins[i+1]:.1f}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(prior_dir, f"linear_fixations_vs_word_length_bin_{i}.png"))
        plt.close()
        
        # Line chart connecting averages
        x_sorted, y_means = compute_average_fixations(word_x, fix_y)
        plt.figure(figsize=(8, 6))
        plt.plot(x_sorted, y_means, marker='o', linestyle='-', color='blue')
        plt.xlabel("Word Length")
        plt.ylabel("Average Number of Fixations")
        plt.title(f"Average Fixations vs. Word Length | Prior Bin {bins[i]:.1f}-{bins[i+1]:.1f}")
        plt.grid(True)
        plt.savefig(os.path.join(prior_dir, f"avg_fixations_vs_word_length_bin_{i}.png"))
        plt.close()
    
    # Generate universal plots without bin separation
    plt.figure(figsize=(8, 6))
    plt.scatter(universal_word_lengths, universal_fixations, alpha=0.7)
    plt.xlabel("Word Length")
    plt.ylabel("Number of Fixations")
    plt.title("Fixations vs. Word Length (Universal)")
    plt.grid(True)
    plt.savefig(os.path.join(universal_dir, "fixations_vs_word_length.png"))
    plt.close()
    
    slope, intercept, _, _, _ = linregress(universal_word_lengths, universal_fixations)
    plt.figure(figsize=(8, 6))
    plt.scatter(universal_word_lengths, universal_fixations, alpha=0.7, label="Data")
    plt.plot(universal_word_lengths, np.array(universal_word_lengths) * slope + intercept, color='red', label="Linear Fit")
    plt.xlabel("Word Length")
    plt.ylabel("Number of Fixations")
    plt.title("Linear Regression: Fixations vs. Word Length (Universal)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(universal_dir, "linear_fixations_vs_word_length.png"))
    plt.close()
    
    x_sorted, y_means = compute_average_fixations(universal_word_lengths, universal_fixations)
    plt.figure(figsize=(8, 6))
    plt.plot(x_sorted, y_means, marker='o', linestyle='-', color='green')
    plt.xlabel("Word Length")
    plt.ylabel("Average Number of Fixations")
    plt.title("Average Fixations vs. Word Length (Universal)")
    plt.grid(True)
    plt.savefig(os.path.join(universal_dir, "avg_fixations_vs_word_length.png"))
    plt.close()
    
    print(f"Plots saved successfully in {save_file_dir}")
