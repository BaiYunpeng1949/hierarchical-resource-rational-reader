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
    """Bins rows by frequency with additional statistics."""
    rows = []
    for class_label, (low, high) in bin_dict.items():
        mask = (df[freq_col] >= low) & (df[freq_col] < high)
        subset = df[mask]
        
        if len(subset) > 0:
            mean_log_freq = subset[logfreq_col].mean()
            mean_gaze = subset[gaze_col].mean()
            std_gaze = subset[gaze_col].std()
            n = len(subset)
            std_error = std_gaze / np.sqrt(n)
            
            rows.append({
                "class": class_label,
                "log_frequency": mean_log_freq,
                "average_gaze_duration": mean_gaze,
                "std_deviation": std_gaze,
                "n_samples": n,
                "std_error": std_error,
                "ci_95_lower": mean_gaze - 1.96 * std_error,
                "ci_95_upper": mean_gaze + 1.96 * std_error
            })
    
    return pd.DataFrame(rows)

def bin_and_aggregate_predictability(df, bin_dict, logit_col="logit_predictability", gaze_col="average_gaze_duration"):
    """Bins rows by predictability with additional statistics."""
    rows = []
    for class_label, (low, high) in bin_dict.items():
        mask = (df[logit_col] >= low) & (df[logit_col] < high)
        subset = df[mask]
        
        if len(subset) > 0:
            mean_logit = subset[logit_col].mean()
            mean_gaze = subset[gaze_col].mean()
            std_gaze = subset[gaze_col].std()
            n = len(subset)
            std_error = std_gaze / np.sqrt(n)
            
            rows.append({
                "class": class_label,
                "logit_predictability": mean_logit,
                "average_gaze_duration": mean_gaze,
                "std_deviation": std_gaze,
                "n_samples": n,
                "std_error": std_error,
                "ci_95_lower": mean_gaze - 1.96 * std_error,
                "ci_95_upper": mean_gaze + 1.96 * std_error
            })
    
    return pd.DataFrame(rows)

def analyze_priors_effect_on_gaze_duration(
        json_data, save_file_dir, csv_log_freq_file_path, csv_logit_pred_file_path, csv_binned_log_freq_file_path, csv_binned_logit_pred_file_path,                                
    ):
    """
    Generates gaze duration analysis plots for word frequency and predictability.
    """
    data = json.loads(json_data)
    os.makedirs(save_file_dir, exist_ok=True)
    
    # word_frequencies = []
    # word_predictabilities = []
    # word_predictabilities_clamped = []
    # gaze_durations = []

    freq_values = []
    freq_gaze_durations = []

    pred_values = []
    pred_gaze_durations = []

    for episode in data:
        prior_type = episode.get("prior_type", 0)

        # We'll always grab the final (done) fixation for gaze duration
        last_fixation = next(f for f in reversed(episode["fixations"]) if f["done"])
        gaze_duration = last_fixation["gaze_duration"]

        if prior_type == Constants.PRIOR_AS_FREQ:
            # Frequency prior: store 'occurance' as the raw frequency
            raw_freq = episode.get("occurance", 0.0)
            freq_values.append(raw_freq)
            freq_gaze_durations.append(gaze_duration)
        elif prior_type == Constants.PRIOR_AS_PRED:
            # Predictability prior: store 'word_prior_prob' as the predictability
            raw_pred = episode.get("word_prior_prob", 0.0)
            pred_values.append(raw_pred)
            pred_gaze_durations.append(gaze_duration)
        else:
            raise ValueError(f"Invalid prior type: {prior_type}")
    
    # ---------------
    # FREQUENCY PART
    # ---------------
    if len(freq_values) > 0:
        # Convert occurrences -> log frequency
        # If you have your own logic (like "pseudo_freq_to_raw"), apply it
        raw_freqs = freq_values
        log_freqs = [np.log10(f) for f in raw_freqs]

        df_log_freq = pd.DataFrame({
            "raw_data": raw_freqs,
            "log_frequency": log_freqs,
            "average_gaze_duration": freq_gaze_durations
        })
        df_log_freq.to_csv(csv_log_freq_file_path, index=False)

        # Binned frequency data
        df_log_freq_binned = bin_and_aggregate_frequency(df=df_log_freq, bin_dict=Constants.LOG_FREQ_BINS)
        df_log_freq_binned.to_csv(csv_binned_log_freq_file_path, index=False)
    else:
        print("No frequency-prior data found (prior_type=0). CSVs not written for frequency.")
    
    # ---------------------
    # PREDICTABILITY PART
    # ---------------------
    if len(pred_values) > 0:
        # Clamp and logit transform predictabilities
        clamped_preds = [clamp(p, Constants.PREDICTABILITY_MIN, Constants.PREDICTABILITY_MAX) for p in pred_values]
        logit_preds = [0.5 * np.log(p / (1.0 - p)) for p in clamped_preds]

        df_logit_pred = pd.DataFrame({
            "logit_predictability": logit_preds,
            "average_gaze_duration": pred_gaze_durations
        })
        df_logit_pred.to_csv(csv_logit_pred_file_path, index=False)

        # Binned predictability data
        df_logit_pred_binned = bin_and_aggregate_predictability(df_logit_pred, Constants.LOGIT_PRED_BINS)
        df_logit_pred_binned.to_csv(csv_binned_logit_pred_file_path, index=False)
    else:
        print("No predictability-prior data found (prior_type=1). CSVs not written for predictability.")

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

def compute_average_gaze_with_stats(word_lengths, gaze_durations):
    """Computes average gaze duration and statistics per word length."""
    stats = {}
    for w_len in sorted(set(word_lengths)):
        # Get all gaze durations for this word length
        gazes = [g for w, g in zip(word_lengths, gaze_durations) if w == w_len]
        stats[w_len] = {
            'average_gaze_duration': np.mean(gazes),
            'std_deviation': np.std(gazes),
            'n_samples': len(gazes),
            'std_error': np.std(gazes) / np.sqrt(len(gazes))  # Standard Error
        }
    return stats

def analyze_word_length_gaze_duration(json_data, save_file_dir, csv_file_path):
    """
    Generates gaze duration analysis plots for word length with statistics.
    """
    data = json.loads(json_data)
    os.makedirs(save_file_dir, exist_ok=True)
    
    # Group data by word length
    word_length_data = defaultdict(list)
    for episode in data:
        w_len = episode["word_len"]
        last_fixation = next(f for f in reversed(episode["fixations"]) if f["done"])
        gaze_duration = last_fixation["gaze_duration"]
        word_length_data[w_len].append(gaze_duration)
    
    # Calculate statistics for each word length
    stats_data = []
    for w_len in sorted(word_length_data.keys()):
        durations = word_length_data[w_len]
        mean_duration = np.mean(durations)
        std_dev = np.std(durations)
        n_samples = len(durations)
        std_error = std_dev / np.sqrt(n_samples)
        
        stats_data.append({
            'word_length': w_len,
            'average_gaze_duration': mean_duration,
            'std_deviation': std_dev,
            'n_samples': n_samples,
            'std_error': std_error,
            'ci_95_lower': mean_duration - 1.96 * std_error,
            'ci_95_upper': mean_duration + 1.96 * std_error
        })
    
    # Save to CSV
    df = pd.DataFrame(stats_data)
    df.to_csv(csv_file_path, index=False)

    # # Scatter plot
    # plt.figure(figsize=(8, 6))
    # plt.scatter(word_lengths, gaze_durations, alpha=0.7)
    # plt.xlabel("Word Length")
    # plt.ylabel("Gaze Duration (ms)")
    # plt.title("Gaze Duration vs. Word Length")
    # plt.grid(True)
    # plt.savefig(os.path.join(save_file_dir, "gaze_duration_vs_word_length.png"))
    # plt.close()
    
    # # Linear regression plot
    # slope, intercept, _, _, _ = linregress(word_lengths, gaze_durations)
    # plt.figure(figsize=(8, 6))
    # plt.scatter(word_lengths, gaze_durations, alpha=0.7, label="Data")
    # plt.plot(word_lengths, np.array(word_lengths) * slope + intercept, color='red', label="Linear Fit")
    # plt.xlabel("Word Length")
    # plt.ylabel("Gaze Duration (ms)")
    # plt.title("Linear Regression: Gaze Duration vs. Word Length")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(os.path.join(save_file_dir, "linear_gaze_duration_vs_word_length.png"))
    # plt.close()
    
    # # Mean value line chart
    # x_sorted, y_means = compute_average_fixations(word_lengths, gaze_durations)
    # plt.figure(figsize=(8, 6))
    # plt.plot(x_sorted, y_means, marker='o', linestyle='-', color='green')
    # plt.xlabel("Word Length")
    # plt.ylabel("Average Gaze Duration (ms)")
    # plt.title("Average Gaze Duration vs. Word Length")
    # plt.grid(True)
    # plt.savefig(os.path.join(save_file_dir, "avg_gaze_duration_vs_word_length.png"))
    # plt.close()
    
    # print(f"Gaze Duration Analysis Plots saved successfully in {save_file_dir}")

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
    
    # # Scatter plot
    # plt.figure(figsize=(8, 6))
    # plt.scatter(word_lengths, prior_values, alpha=0.7)
    # plt.xlabel("Word Length")
    # plt.ylabel("Word Prior Probability")
    # plt.title("Word Length vs. Word Prior Probability (Universal)")
    # plt.grid(True)
    # plt.savefig(os.path.join(universal_dir, "scatter_word_length_vs_prior.png"))
    # plt.close()
    
    # # Linear regression plot
    # slope, intercept, _, _, _ = linregress(word_lengths, prior_values)
    # plt.figure(figsize=(8, 6))
    # plt.scatter(word_lengths, prior_values, alpha=0.7, label="Data")
    # plt.plot(word_lengths, np.array(word_lengths) * slope + intercept, color='red', label="Linear Fit")
    # plt.xlabel("Word Length")
    # plt.ylabel("Word Prior Probability")
    # plt.title("Linear Regression: Word Length vs. Word Prior Probability")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(os.path.join(universal_dir, "linear_word_length_vs_prior.png"))
    # plt.close()
    
    # # Line chart connecting averages
    # x_sorted, y_means = compute_average_fixations(word_lengths, prior_values)
    # plt.figure(figsize=(8, 6))
    # plt.plot(x_sorted, y_means, marker='o', linestyle='-', color='green')
    # plt.xlabel("Word Length")
    # plt.ylabel("Average Word Prior Probability")
    # plt.title("Average Word Prior Probability vs. Word Length (Universal)")
    # plt.grid(True)
    # plt.savefig(os.path.join(universal_dir, "avg_word_length_vs_prior.png"))
    # plt.close()
    
    # print(f"Plots saved successfully in {save_file_dir}")

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