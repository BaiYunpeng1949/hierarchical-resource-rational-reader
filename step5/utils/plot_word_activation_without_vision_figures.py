import os
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import linregress

from modules.rl_envs.word_activation_v0218 import Constants

def clamp(value, low, high):
    """Helper to keep 'value' between 'low' and 'high'."""
    return max(low, min(value, high))

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

def pseudo_freq_to_raw(p, min_freq=1, max_freq=1000000):
    """
    Converts pseudo-frequency values back to raw frequency values.
    """
    return min_freq + (max_freq - min_freq) * p

def bin_and_aggregate_frequency(df, bin_dict, freq_col="raw_data", logfreq_col="log_frequency", gaze_col="average_gaze_duration"):
    """
    Bins rows by their raw frequency per million (from 'freq_col'),
    then computes the mean log_frequency & mean gaze_duration
    for each bin.

    Returns a DataFrame with columns: 
        ['class', 'log_frequency', 'gaze_duration', 'count']
    """
    rows = []
    for class_label, (low, high) in bin_dict.items():
        # Select rows whose freq_per_million is in [low, high)
        mask = (df[freq_col] >= low) & (df[freq_col] < high)
        subset = df[mask]
        
        if len(subset) > 0:
            mean_log_freq = subset[logfreq_col].mean()
            mean_gaze = subset[gaze_col].mean()
            n = len(subset)
        else:
            # If no data falls in this bin, store NaN
            mean_log_freq = np.nan
            mean_gaze = np.nan
            n = 0
        
        rows.append({
            "class": class_label,
            "log_frequency": mean_log_freq,
            "average_gaze_duration": mean_gaze,
            "count": n
        })
    
    return pd.DataFrame(rows)

def bin_and_aggregate_predictability(df, bin_dict, logit_col="logit_predictability", gaze_col="average_gaze_duration"):
    """
    Bins rows by their logit_predictability (from 'logit_col'),
    then computes the mean logit_predictability & mean gaze_duration
    for each bin.

    Returns a DataFrame with columns:
        ['class', 'logit_predictability', 'gaze_duration', 'count']
    """
    rows = []
    for class_label, (low, high) in bin_dict.items():
        mask = (df[logit_col] >= low) & (df[logit_col] < high)
        subset = df[mask]
        
        if len(subset) > 0:
            mean_logit = subset[logit_col].mean()
            mean_gaze = subset[gaze_col].mean()
            n = len(subset)
        else:
            mean_logit = np.nan
            mean_gaze = np.nan
            n = 0
        
        rows.append({
            "class": class_label,
            "logit_predictability": mean_logit,
            "average_gaze_duration": mean_gaze,
            "count": n
        })
    
    return pd.DataFrame(rows)

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
        w_pred = episode.get("word_predictability", 0.0)
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
    
    print(f"Word' Prior Effect Plots saved successfully in {save_file_dir}")

def analyze_priors_effect_on_gaze_duration(
        json_data, save_file_dir, csv_log_freq_file_path, csv_logit_pred_file_path, csv_binned_log_freq_file_path, csv_binned_logit_pred_file_path,                                
    ):
    """
    Generates gaze duration analysis plots for word frequency and predictability.
    """
    data = json.loads(json_data)
    os.makedirs(save_file_dir, exist_ok=True)
    
    word_frequencies = []
    word_predictabilities = []
    word_predictabilities_clamped = []
    gaze_durations = []
    
    for episode in data:
        w_freq = episode["word_prior_prob"]             # All get the default prior's values, refine in the data processing parts later
        w_pred = episode.get("word_prior_prob", 0.0)    # For freq and pred, we could use different non-linear gaze duration models, or even rl models to generate data 
        last_fixation = next(f for f in reversed(episode["fixations"]) if f["done"])
        gaze_duration = last_fixation["gaze_duration"]
        
        # Clamp the word predictability values
        w_pred_clamped = clamp(w_pred, Constants.PREDICTABILITY_MIN, Constants.PREDICTABILITY_MAX)

        word_frequencies.append(w_freq)
        word_predictabilities.append(w_pred)
        word_predictabilities_clamped.append(w_pred_clamped)
        gaze_durations.append(gaze_duration)
    
    # Convert pseudo freq -> raw freq -> log10
    # NOTE: reference: Length, frequency, and predictability effects of words on eye movements in reading
    raw_freqs = [pseudo_freq_to_raw(f) for f in word_frequencies]
    log_freqs = [np.log10(f) for f in raw_freqs]

    # Convert pseudo pred -> clamp -> logit
    clamped_preds = [clamp(p, Constants.PREDICTABILITY_MIN, Constants.PREDICTABILITY_MAX) for p in word_predictabilities]

    # TODO debug delete later
    print(f"The clamp range is: {Constants.PREDICTABILITY_MIN} - {Constants.PREDICTABILITY_MAX}")

    logit_preds = [0.5 * np.log(p / (1.0 - p)) for p in clamped_preds]

    # # Save to csv
    # df_freq = pd.DataFrame({"frequency": word_frequencies, "average_gaze_duration": gaze_durations})
    # df_pred = pd.DataFrame({"predictability": word_predictabilities, "average_gaze_duration": gaze_durations})

    # Create a log-frequency DataFrame
    df_log_freq = pd.DataFrame({
        "raw_data": raw_freqs,
        "log_frequency": log_freqs,
        "average_gaze_duration": gaze_durations
    })
    df_log_freq.to_csv(csv_log_freq_file_path, index=False)

    # Create a logit-predictability DataFrame
    df_logit_pred = pd.DataFrame({
        "logit_predictability": logit_preds,
        "average_gaze_duration": gaze_durations
    })
    df_logit_pred.to_csv(csv_logit_pred_file_path, index=False)

    # Get the binned and aggregated version
    df_log_freq_binned = bin_and_aggregate_frequency(df=df_log_freq, bin_dict=Constants.LOG_FREQ_BINS)
    df_logit_pred_binned = bin_and_aggregate_predictability(df_logit_pred, Constants.LOGIT_PRED_BINS)

    # Save to csv
    df_log_freq_binned.to_csv(csv_binned_log_freq_file_path, index=False)
    df_logit_pred_binned.to_csv(csv_binned_logit_pred_file_path, index=False)

    # df_freq.to_csv(csv_freq_file_path, index=False)
    # df_pred.to_csv(csv_pred_file_path, index=False)

    # Scatter plot - Word Frequency
    plt.figure(figsize=(8, 6))
    plt.scatter(word_frequencies, gaze_durations, alpha=0.7)
    plt.xlabel("Word Frequency")
    plt.ylabel("Gaze Duration (ms)")
    plt.title("Gaze Duration vs. Word Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "gaze_duration_vs_word_frequency.png"))
    plt.close()
    
    # Linear regression - Word Frequency
    slope, intercept, _, _, _ = linregress(word_frequencies, gaze_durations)
    plt.figure(figsize=(8, 6))
    plt.scatter(word_frequencies, gaze_durations, alpha=0.7, label="Data")
    plt.plot(word_frequencies, np.array(word_frequencies) * slope + intercept, color='red', label="Linear Fit")
    plt.xlabel("Word Frequency")
    plt.ylabel("Gaze Duration (ms)")
    plt.title("Linear Regression: Gaze Duration vs. Word Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "linear_gaze_duration_vs_word_frequency.png"))
    plt.close()
    
    # Mean value line chart - Word Frequency
    x_sorted, y_means = compute_average_gaze(word_frequencies, gaze_durations)
    plt.figure(figsize=(8, 6))
    plt.plot(x_sorted, y_means, marker='o', linestyle='-', color='green')
    plt.xlabel("Word Frequency")
    plt.ylabel("Average Gaze Duration (ms)")
    plt.title("Average Gaze Duration vs. Word Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "avg_gaze_duration_vs_word_frequency.png"))
    plt.close()
    
    # Scatter plot - Word Predictability
    plt.figure(figsize=(8, 6))
    plt.scatter(word_predictabilities, gaze_durations, alpha=0.7)
    plt.xlabel("Word Predictability")
    plt.ylabel("Gaze Duration (ms)")
    plt.title("Gaze Duration vs. Word Predictability")
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "gaze_duration_vs_word_predictability.png"))
    plt.close()
    
    # Linear regression - Word Predictability
    slope, intercept, _, _, _ = linregress(word_predictabilities, gaze_durations)
    plt.figure(figsize=(8, 6))
    plt.scatter(word_predictabilities, gaze_durations, alpha=0.7, label="Data")
    plt.plot(word_predictabilities, np.array(word_predictabilities) * slope + intercept, color='red', label="Linear Fit")
    plt.xlabel("Word Predictability")
    plt.ylabel("Gaze Duration (ms)")
    plt.title("Linear Regression: Gaze Duration vs. Word Predictability")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "linear_gaze_duration_vs_word_predictability.png"))
    plt.close()
    
    # Mean value line chart - Word Predictability
    x_sorted, y_means = compute_average_gaze(word_predictabilities, gaze_durations)
    plt.figure(figsize=(8, 6))
    plt.plot(x_sorted, y_means, marker='o', linestyle='-', color='purple')
    plt.xlabel("Word Predictability")
    plt.ylabel("Average Gaze Duration (ms)")
    plt.title("Average Gaze Duration vs. Word Predictability")
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "avg_gaze_duration_vs_word_predictability.png"))
    plt.close()
    
    print(f"Prior Effects Analysis Plots saved successfully in {save_file_dir}")

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
    
    print(f"Word Length's Effect Analyze Plots saved successfully in {save_file_dir}")

def compute_average_gaze(word_lengths, gaze_durations):
    """Computes average gaze duration per word length."""
    unique_lengths = sorted(set(word_lengths))
    avg_gazes = [np.mean([g for w, g in zip(word_lengths, gaze_durations) if w == ul]) for ul in unique_lengths]
    return unique_lengths, avg_gazes

def analyze_word_length_gaze_duration(json_data, save_file_dir, csv_file_path):
    """
    Generates gaze duration analysis plots for word length.
    """
    data = json.loads(json_data)
    os.makedirs(save_file_dir, exist_ok=True)
    
    word_lengths = []
    gaze_durations = []
    
    for episode in data:
        w_len = episode["word_len"]
        last_fixation = next(f for f in reversed(episode["fixations"]) if f["done"])
        gaze_duration = last_fixation["gaze_duration"]
        
        word_lengths.append(w_len)
        gaze_durations.append(gaze_duration)
    
    # Compute mean values
    x_sorted, y_means = compute_average_gaze(word_lengths, gaze_durations)
    
    # Save to CSV
    df = pd.DataFrame({"word_length": x_sorted, "average_gaze_duration": y_means})
    df.to_csv(csv_file_path, index=False)
    
    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(word_lengths, gaze_durations, alpha=0.7)
    plt.xlabel("Word Length")
    plt.ylabel("Gaze Duration (ms)")
    plt.title("Gaze Duration vs. Word Length")
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "gaze_duration_vs_word_length.png"))
    plt.close()
    
    # Linear regression plot
    slope, intercept, _, _, _ = linregress(word_lengths, gaze_durations)
    plt.figure(figsize=(8, 6))
    plt.scatter(word_lengths, gaze_durations, alpha=0.7, label="Data")
    plt.plot(word_lengths, np.array(word_lengths) * slope + intercept, color='red', label="Linear Fit")
    plt.xlabel("Word Length")
    plt.ylabel("Gaze Duration (ms)")
    plt.title("Linear Regression: Gaze Duration vs. Word Length")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "linear_gaze_duration_vs_word_length.png"))
    plt.close()
    
    # Mean value line chart
    x_sorted, y_means = compute_average_gaze(word_lengths, gaze_durations)
    plt.figure(figsize=(8, 6))
    plt.plot(x_sorted, y_means, marker='o', linestyle='-', color='green')
    plt.xlabel("Word Length")
    plt.ylabel("Average Gaze Duration (ms)")
    plt.title("Average Gaze Duration vs. Word Length")
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "avg_gaze_duration_vs_word_length.png"))
    plt.close()
    
    print(f"Gaze Duration Analysis Plots saved successfully in {save_file_dir}")


def analyze_prior_vs_word_length(json_data, save_file_dir):
    """
    Generates fixation analysis plots for word prior probability vs. word length.
    Includes scatter plots, linear regression plots, and averaged line plots.
    """
    data = json.loads(json_data)
    os.makedirs(save_file_dir, exist_ok=True)
    
    prior_values = []
    word_lengths = []
    
    for episode in data:
        w_prior = episode["word_prior_prob"]
        w_len = episode["word_len"]
        
        prior_values.append(w_prior)
        word_lengths.append(w_len)
    
    universal_dir = os.path.join(save_file_dir, "prior_vs_word_length")
    os.makedirs(universal_dir, exist_ok=True)
    
    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(word_lengths, prior_values, alpha=0.7)
    plt.xlabel("Word Length")
    plt.ylabel("Word Prior Probability")
    plt.title("Word Length vs. Word Prior Probability (Universal)")
    plt.grid(True)
    plt.savefig(os.path.join(universal_dir, "scatter_word_length_vs_prior.png"))
    plt.close()
    
    # Linear regression plot
    slope, intercept, _, _, _ = linregress(word_lengths, prior_values)
    plt.figure(figsize=(8, 6))
    plt.scatter(word_lengths, prior_values, alpha=0.7, label="Data")
    plt.plot(word_lengths, np.array(word_lengths) * slope + intercept, color='red', label="Linear Fit")
    plt.xlabel("Word Length")
    plt.ylabel("Word Prior Probability")
    plt.title("Linear Regression: Word Length vs. Word Prior Probability")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(universal_dir, "linear_word_length_vs_prior.png"))
    plt.close()
    
    # Line chart connecting averages
    x_sorted, y_means = compute_average_fixations(word_lengths, prior_values)
    plt.figure(figsize=(8, 6))
    plt.plot(x_sorted, y_means, marker='o', linestyle='-', color='green')
    plt.xlabel("Word Length")
    plt.ylabel("Average Word Prior Probability")
    plt.title("Average Word Prior Probability vs. Word Length (Universal)")
    plt.grid(True)
    plt.savefig(os.path.join(universal_dir, "avg_word_length_vs_prior.png"))
    plt.close()
    
    print(f"Plots saved successfully in {save_file_dir}")

def analyze_accuracy(json_data, save_file_dir):
    # Ensure json_data is a parsed list, not a string
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    total_recognitions = 0
    correct_recognitions = 0
    
    for episode in json_data:
        if isinstance(episode, dict):  # Ensure it's a dictionary
            for fixation in episode.get("fixations", []):
                accurate_recognition = fixation.get("accurate_recognition")
                if accurate_recognition is not None:
                    total_recognitions += 1
                    if accurate_recognition:
                        correct_recognitions += 1
    
    accuracy = (correct_recognitions / total_recognitions) * 100 if total_recognitions > 0 else 0
    
    # Save results to a text file
    result = f"Total Recognitions: {total_recognitions}\nCorrect Recognitions: {correct_recognitions}\nAccuracy: {accuracy:.2f}%"
    save_file_dir = os.path.join(save_file_dir, "accuracy_results.txt")
    with open(save_file_dir, "w") as file:
        file.write(result)
    
    print(result)
    print(f"Accuracy results saved successfully in {save_file_dir}")

    return accuracy

