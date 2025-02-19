import os
import matplotlib.pyplot as plt
import json
import numpy as np

def analyze_fixations(json_data, save_file_dir, controlled_word_length=None):
    """
    Analyzes fixation data and generates:
      1. Scatter plots of Number of Fixations vs. (Word Length | Word Frequency | Word Predictability).
      2. Line plots of Average Fixations vs. (Word Length | Word Frequency | Word Predictability).
         - For Frequency and Predictability, you can optionally control for a specific word length
           via the 'controlled_word_length' parameter.
      3. A scatter plot of Fixation Position Analysis (color hue = fixation order).
      4. A bar plot showing the overall recognition accuracy (in %), based on "accurate_recognition"
         in the final fixation (the one with "done"=True).

    Saves the plots in the specified directory.
    """

    # Parse the JSON data
    data = json.loads(json_data)

    # Ensure the directory exists
    os.makedirs(save_file_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # PART A: EXTRACT FIXATION DATA FOR PLOTS
    # ------------------------------------------------------------------
    word_lengths_all = []
    word_frequencies_all = []
    word_predictabilities_all = []
    num_fixations_all = []
    fixation_positions = []

    # For recognition accuracy
    total_episodes_with_done = 0
    total_correct_recognitions = 0

    for episode in data:
        w_len  = episode["word_len"]
        w_freq = episode["word_frequency"]
        # Safely get predictability from logs
        w_pred = episode.get("Word predictability", 0.0)

        fixations = episode["fixations"]

        # (1) Count how many fixations have done=False
        num_fixations_not_done = sum(1 for f in fixations if not f["done"])

        # (2) Add to arrays for scatter/line plots
        word_lengths_all.append(w_len)
        word_frequencies_all.append(w_freq)
        word_predictabilities_all.append(w_pred)
        num_fixations_all.append(num_fixations_not_done)

        # (3) For position analysis, store all fixations
        for f in fixations:
            fixation_positions.append({
                "word_length": w_len,
                "steps": f["steps"],
                "position": f["action"],
                "done": f["done"]
            })

        # (4) Find the final done fixation to compute recognition accuracy
        #     Usually only one done fixation, but let's be safe:
        done_fixations = [fx for fx in fixations if fx["done"]]
        if len(done_fixations) > 0:
            final_fixation = done_fixations[-1]  # assume the last "done" is the final
            accurate = final_fixation.get("accurate_recognition", None)
            if accurate is not None:
                total_episodes_with_done += 1
                if accurate is True:
                    total_correct_recognitions += 1

    # ------------------------------------------------------------------
    # PART B: SCATTER PLOTS
    # ------------------------------------------------------------------

    # (1) Word Length vs. Fixations (scatter)
    plt.figure(figsize=(8, 6))
    plt.scatter(word_lengths_all, num_fixations_all, alpha=0.7)
    plt.xlabel("Word Length")
    plt.ylabel("Number of Fixations")
    plt.title("Number of Fixations vs. Word Length (Scatter)")
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "fixations_vs_word_length.png"))
    plt.close()

    # (2) Word Frequency vs. Fixations (scatter) -- can filter on word length
    if controlled_word_length is not None:
        freq_x = [wf for (wf, wl) in zip(word_frequencies_all, word_lengths_all) 
                  if wl == controlled_word_length]
        fix_y = [nf for (nf, wl) in zip(num_fixations_all, word_lengths_all) 
                 if wl == controlled_word_length]
    else:
        freq_x = word_frequencies_all
        fix_y  = num_fixations_all

    plt.figure(figsize=(8, 6))
    plt.scatter(freq_x, fix_y, alpha=0.7)
    plt.xlabel("Word Frequency")
    plt.ylabel("Number of Fixations")
    if controlled_word_length is not None:
        plt.title(f"Fixations vs. Word Frequency (Scatter) | Word Length = {controlled_word_length}")
    else:
        plt.title("Number of Fixations vs. Word Frequency (Scatter)")
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "fixations_vs_word_frequency.png"))
    plt.close()

    # (3) Word Predictability vs. Fixations (scatter)
    if controlled_word_length is not None:
        pred_x = [wp for (wp, wl) in zip(word_predictabilities_all, word_lengths_all)
                  if wl == controlled_word_length]
        fix_y_pred = [nf for (nf, wl) in zip(num_fixations_all, word_lengths_all)
                      if wl == controlled_word_length]
    else:
        pred_x     = word_predictabilities_all
        fix_y_pred = num_fixations_all

    plt.figure(figsize=(8, 6))
    plt.scatter(pred_x, fix_y_pred, alpha=0.7)
    plt.xlabel("Word Predictability")
    plt.ylabel("Number of Fixations")
    if controlled_word_length is not None:
        plt.title(f"Fixations vs. Word Predictability (Scatter) | Word Length = {controlled_word_length}")
    else:
        plt.title("Number of Fixations vs. Word Predictability (Scatter)")
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "fixations_vs_word_predictability.png"))
    plt.close()

    # ------------------------------------------------------------------
    # PART C: LINE PLOTS (average fixations by factor)
    # ------------------------------------------------------------------
    from collections import defaultdict

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

    # (1) Word Length vs. Average Fixations
    x_sorted_len, y_means_len = compute_average_fixations(word_lengths_all, num_fixations_all)
    plt.figure(figsize=(8, 6))
    plt.plot(x_sorted_len, y_means_len, marker='o', linestyle='-', color='orange')
    plt.xlabel("Word Length")
    plt.ylabel("Average Number of Fixations")
    plt.title("Average Fixations vs. Word Length (Line)")
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "fixations_vs_word_length_line.png"))
    plt.close()

    # (2) Word Frequency vs. Average Fixations
    if controlled_word_length is not None:
        freq_filtered, fix_filtered = [], []
        for (wf, wl, nf) in zip(word_frequencies_all, word_lengths_all, num_fixations_all):
            if wl == controlled_word_length:
                freq_filtered.append(wf)
                fix_filtered.append(nf)
    else:
        freq_filtered = word_frequencies_all
        fix_filtered  = num_fixations_all

    x_sorted_freq, y_means_freq = compute_average_fixations(freq_filtered, fix_filtered)
    plt.figure(figsize=(8, 6))
    plt.plot(x_sorted_freq, y_means_freq, marker='o', linestyle='-', color='green')
    plt.xlabel("Word Frequency")
    plt.ylabel("Average Number of Fixations")
    if controlled_word_length is not None:
        plt.title(f"Average Fixations vs. Word Frequency (Line)\nWord Length = {controlled_word_length}")
    else:
        plt.title("Average Fixations vs. Word Frequency (Line)")
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "fixations_vs_word_frequency_line.png"))
    plt.close()

    # (3) Word Predictability vs. Average Fixations
    if controlled_word_length is not None:
        pred_filtered, fix_filtered_pred = [], []
        for (wp, wl, nf) in zip(word_predictabilities_all, word_lengths_all, num_fixations_all):
            if wl == controlled_word_length:
                pred_filtered.append(wp)
                fix_filtered_pred.append(nf)
    else:
        pred_filtered = word_predictabilities_all
        fix_filtered_pred = num_fixations_all

    x_sorted_pred, y_means_pred = compute_average_fixations(pred_filtered, fix_filtered_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(x_sorted_pred, y_means_pred, marker='o', linestyle='-', color='purple')
    plt.xlabel("Word Predictability")
    plt.ylabel("Average Number of Fixations")
    if controlled_word_length is not None:
        plt.title(f"Average Fixations vs. Word Predictability (Line)\nWord Length = {controlled_word_length}")
    else:
        plt.title("Average Fixations vs. Word Predictability (Line)")
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "fixations_vs_word_predictability_line.png"))
    plt.close()

    # ------------------------------------------------------------------
    # PART D: FIXATION POSITION ANALYSIS (existing code)
    # ------------------------------------------------------------------
    fixation_x = [f["position"] for f in fixation_positions if not f["done"]]
    fixation_y = [f["word_length"] for f in fixation_positions if not f["done"]]
    fixation_steps = [f["steps"] for f in fixation_positions if not f["done"]]

    if fixation_steps:
        min_step, max_step = min(fixation_steps), max(fixation_steps)
        step_range = max(1, max_step - min_step)
        fixation_steps_normalized = (np.array(fixation_steps) - min_step) / step_range
    else:
        fixation_steps_normalized = np.zeros_like(fixation_steps)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        fixation_x,
        fixation_y,
        c=fixation_steps_normalized,
        cmap="Blues",
        alpha=0.7
    )
    plt.xlabel("Fixation Position (Action)")
    plt.ylabel("Word Length")
    plt.title("Fixation Position Analysis (Hue = Fixation Order)")
    plt.colorbar(scatter, label="Fixation Order (Earlier → Darker)")
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "fixation_positions_hue.png"))
    plt.close()

    # ------------------------------------------------------------------
    # PART E: OVERALL RECOGNITION ACCURACY
    # ------------------------------------------------------------------
    accuracy_pct = 0.0
    if total_episodes_with_done > 0:
        accuracy_pct = (total_correct_recognitions / total_episodes_with_done) * 100.0

    # We can plot it as a single bar or as text.
    plt.figure(figsize=(4, 6))
    plt.bar([0], [accuracy_pct], color='skyblue', width=0.5)
    plt.ylim([0, 100])
    plt.xticks([0], ["Overall"])  # Single category label on x-axis
    plt.ylabel("Recognition Accuracy (%)")
    plt.title("Overall Recognition Accuracy")
    # Optionally annotate the bar with the exact percentage
    plt.text(0, accuracy_pct + 1, f"{accuracy_pct:.2f}%", ha='center', va='bottom', fontsize=12)
    plt.grid(True, axis="y", alpha=0.5)
    plt.savefig(os.path.join(save_file_dir, "recognition_accuracy.png"))
    plt.close()

    print("Plots saved successfully in:", save_file_dir)
    print(f"Overall Recognition Accuracy: {accuracy_pct:.2f}%")
