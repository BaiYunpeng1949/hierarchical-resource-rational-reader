import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Import time conditions from Constants.py
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Constants import TIME_CONDITIONS

# Path to the results folder (update if needed)
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'simulated_results',
    '0605_text_comprehension_under_time_pressure_v0606_2__rl_model_100000000_steps__1000'
)
RESULTS_FILE = os.path.join(RESULTS_DIR, 'simulated_episode_logs.json')

# Load the JSON data
with open(RESULTS_FILE, 'r') as f:
    episode_logs = json.load(f)

# Group metrics by time condition
grouped = defaultdict(list)
for ep in episode_logs:
    grouped[ep['time_condition']].append(ep)

def mean(values):
    return sum(values) / len(values) if values else 0

def std(values):
    return float(np.std(values)) if values else 0

# Prepare data for plotting
x_labels = list(TIME_CONDITIONS.keys())
num_regressions_means = [mean([ep['log_number_regressions'] for ep in grouped[tc]]) for tc in x_labels]
num_regressions_stds = [std([ep['log_number_regressions'] for ep in grouped[tc]]) for tc in x_labels]
regression_rate_num_read_means = [mean([ep['log_episodic_regression_rate_over_num_read_sentences'] for ep in grouped[tc]]) for tc in x_labels]
regression_rate_num_read_stds = [std([ep['log_episodic_regression_rate_over_num_read_sentences'] for ep in grouped[tc]]) for tc in x_labels]
regression_rate_steps_means = [mean([ep['log_episodic_regression_rate_over_steps'] for ep in grouped[tc]]) for tc in x_labels]
regression_rate_steps_stds = [std([ep['log_episodic_regression_rate_over_steps'] for ep in grouped[tc]]) for tc in x_labels]

def plot_bar(y, yerr, ylabel, title, filename):
    plt.figure(figsize=(7, 5))
    bars = plt.bar(x_labels, y, yerr=yerr, capsize=8, color='#4C72B0', alpha=0.85)
    plt.xlabel('Time Condition')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    # Annotate each bar with mean ± std
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, y, yerr)):
        plt.annotate(f'{mean_val:.2f}±{std_val:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 5),
                     textcoords='offset points',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.show()

# Plot 1: Number of regressions vs. time condition
plot_bar(num_regressions_means, num_regressions_stds, 'Average Number of Regressions', 'Number of Regressions vs. Time Condition', 'num_regressions_vs_time_condition.png')

# Plot 2: Regression rate over number of read sentences vs. time condition
plot_bar(regression_rate_num_read_means, regression_rate_num_read_stds, 'Avg. Regression Rate (per Read Sentences)', 'Regression Rate (per Read Sentences) vs. Time Condition', 'regression_rate_num_read_vs_time_condition.png')

# Plot 3: Regression rate over steps vs. time condition
plot_bar(regression_rate_steps_means, regression_rate_steps_stds, 'Avg. Regression Rate (per Steps)', 'Regression Rate (per Steps) vs. Time Condition', 'regression_rate_steps_vs_time_condition.png')