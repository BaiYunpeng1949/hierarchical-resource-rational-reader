# import os
# import matplotlib.pyplot as plt
# import json
# import numpy as np

# # def analyze_fixations(json_data, save_file_dir):
# #     """
# #     Analyzes fixation data and generates three plots:
# #     1. Number of Fixations vs. Word Length
# #     2. Number of Fixations vs. Word Frequency
# #     3. Fixation Position Analysis
# #     Saves the plots in the specified directory.
# #     """

# #     # Parse the JSON data
# #     data = json.loads(json_data)

# #     # Ensure the directory exists
# #     os.makedirs(save_file_dir, exist_ok=True)

# #     # Extract data for analysis
# #     word_lengths = []
# #     word_frequencies = []
# #     num_fixations = []
# #     fixation_positions = []

# #     for episode in data:
# #         word_length = episode["word_len"]
# #         word_frequency = episode["word_frequency"]
# #         fixations = episode["fixations"]

# #         # Count the number of fixations where "done" is False
# #         num_fixations_not_done = sum(1 for f in fixations if not f["done"])

# #         # Store data for plotting
# #         word_lengths.append(word_length)
# #         word_frequencies.append(word_frequency)
# #         num_fixations.append(num_fixations_not_done)

# #         # Store fixation positions along with word length for visualization
# #         for f in fixations:
# #             fixation_positions.append({
# #                 "word_length": word_length,
# #                 "steps": f["steps"],
# #                 "position": f["action"],  # Assuming action represents fixation position
# #                 "done": f["done"]
# #             })

# #     # Plot Number of Fixations vs. Word Length
# #     plt.figure(figsize=(8, 6))
# #     plt.scatter(word_lengths, num_fixations, alpha=0.7)
# #     plt.xlabel("Word Length")
# #     plt.ylabel("Number of Fixations")
# #     plt.title("Number of Fixations vs. Word Length")
# #     plt.grid(True)
# #     plt.savefig(os.path.join(save_file_dir, "fixations_vs_word_length.png"))
# #     plt.close()

# #     # Plot Number of Fixations vs. Word Frequency
# #     plt.figure(figsize=(8, 6))
# #     plt.scatter(word_frequencies, num_fixations, alpha=0.7)
# #     plt.xlabel("Word Frequency")
# #     plt.ylabel("Number of Fixations")
# #     plt.title("Number of Fixations vs. Word Frequency")
# #     plt.grid(True)
# #     plt.savefig(os.path.join(save_file_dir, "fixations_vs_word_frequency.png"))
# #     plt.close()

# #     # Process fixation positions for visualization with color hues
# #     fixation_x = [f["position"] for f in fixation_positions if not f["done"]]
# #     fixation_y = [f["word_length"] for f in fixation_positions if not f["done"]]
# #     fixation_steps = [f["steps"] for f in fixation_positions if not f["done"]]

# #     # Normalize fixation steps to [0,1] range for colormap
# #     if fixation_steps:
# #         fixation_steps_normalized = (np.array(fixation_steps) - min(fixation_steps)) / max(1, (max(fixation_steps) - min(fixation_steps)))
# #     else:
# #         fixation_steps_normalized = np.zeros_like(fixation_steps)  # Handle empty case

# #     # Plot Fixation Positions with color hues
# #     plt.figure(figsize=(10, 6))
# #     scatter = plt.scatter(fixation_x, fixation_y, c=fixation_steps_normalized, cmap="Blues", alpha=0.7)
# #     plt.xlabel("Fixation Position (Action)")
# #     plt.ylabel("Word Length")
# #     plt.title("Fixation Position Analysis (Hue Encodes Fixation Order)")
# #     plt.colorbar(scatter, label="Fixation Order (Earlier → Darker)")
# #     plt.grid(True)
# #     plt.savefig(os.path.join(save_file_dir, "fixation_positions_hue.png"))
# #     plt.close()

# #     print(f"Plots saved successfully in {save_file_dir}")


# def analyze_fixations(json_data, save_file_dir):
#     """
#     Analyzes fixation data and generates four plots:
#     1. Number of Fixations vs. Word Length
#     2. Number of Fixations vs. Word Frequency
#     3. Number of Fixations vs. Word Predictability
#     4. Fixation Position Analysis (color hue = fixation order)
    
#     Saves the plots in the specified directory.
#     """

#     # Parse the JSON data
#     data = json.loads(json_data)

#     # Ensure the directory exists
#     os.makedirs(save_file_dir, exist_ok=True)

#     # Extract data for analysis
#     word_lengths = []
#     word_frequencies = []
#     word_predictabilities = []
#     num_fixations = []
#     fixation_positions = []

#     for episode in data:
#         word_length = episode["word_len"]
#         word_frequency = episode["word_frequency"]
        
#         # Some logs might label this differently; adjust based on how your data is stored
#         word_predictability = episode.get("Word predictability", 0.0)
        
#         fixations = episode["fixations"]

#         # Count the number of fixations where "done" is False
#         num_fixations_not_done = sum(1 for f in fixations if not f["done"])

#         # Store data for plotting
#         word_lengths.append(word_length)
#         word_frequencies.append(word_frequency)
#         word_predictabilities.append(word_predictability)
#         num_fixations.append(num_fixations_not_done)

#         # Store fixation positions along with word length for visualization
#         for f in fixations:
#             fixation_positions.append({
#                 "word_length": word_length,
#                 "steps": f["steps"],
#                 "position": f["action"],  # Assuming action represents fixation position
#                 "done": f["done"]
#             })

#     # 1. Number of Fixations vs. Word Length
#     plt.figure(figsize=(8, 6))
#     plt.scatter(word_lengths, num_fixations, alpha=0.7)
#     plt.xlabel("Word Length")
#     plt.ylabel("Number of Fixations")
#     plt.title("Number of Fixations vs. Word Length")
#     plt.grid(True)
#     plt.savefig(os.path.join(save_file_dir, "fixations_vs_word_length.png"))
#     plt.close()

#     # 2. Number of Fixations vs. Word Frequency
#     plt.figure(figsize=(8, 6))
#     plt.scatter(word_frequencies, num_fixations, alpha=0.7)
#     plt.xlabel("Word Frequency")
#     plt.ylabel("Number of Fixations")
#     plt.title("Number of Fixations vs. Word Frequency")
#     plt.grid(True)
#     plt.savefig(os.path.join(save_file_dir, "fixations_vs_word_frequency.png"))
#     plt.close()

#     # 3. Number of Fixations vs. Word Predictability
#     plt.figure(figsize=(8, 6))
#     plt.scatter(word_predictabilities, num_fixations, alpha=0.7)
#     plt.xlabel("Word Predictability")
#     plt.ylabel("Number of Fixations")
#     plt.title("Number of Fixations vs. Word Predictability")
#     plt.grid(True)
#     plt.savefig(os.path.join(save_file_dir, "fixations_vs_word_predictability.png"))
#     plt.close()

#     # 4. Fixation Position Analysis (Hue Encodes Fixation Order)
#     fixation_x = [f["position"] for f in fixation_positions if not f["done"]]
#     fixation_y = [f["word_length"] for f in fixation_positions if not f["done"]]
#     fixation_steps = [f["steps"] for f in fixation_positions if not f["done"]]

#     # Normalize fixation steps to [0,1] range for colormap
#     if fixation_steps:
#         min_step, max_step = min(fixation_steps), max(fixation_steps)
#         range_step = max(1, (max_step - min_step))
#         fixation_steps_normalized = (np.array(fixation_steps) - min_step) / range_step
#     else:
#         fixation_steps_normalized = np.zeros_like(fixation_steps)  # Handle empty case gracefully

#     plt.figure(figsize=(10, 6))
#     scatter = plt.scatter(
#         fixation_x,
#         fixation_y,
#         c=fixation_steps_normalized,
#         cmap="Blues",
#         alpha=0.7
#     )
#     plt.xlabel("Fixation Position (Action)")
#     plt.ylabel("Word Length")
#     plt.title("Fixation Position Analysis (Hue Encodes Fixation Order)")
#     plt.colorbar(scatter, label="Fixation Order (Earlier → Darker)")
#     plt.grid(True)
#     plt.savefig(os.path.join(save_file_dir, "fixation_positions_hue.png"))
#     plt.close()

#     print(f"Plots saved successfully in {save_file_dir}")

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

    Saves the plots in the specified directory.
    """

    # Parse the JSON data
    data = json.loads(json_data)

    # Ensure the directory exists
    os.makedirs(save_file_dir, exist_ok=True)

    # --- Extract data for analysis ---
    word_lengths_all = []
    word_frequencies_all = []
    word_predictabilities_all = []
    num_fixations_all = []         # parallel to the above lists
    fixation_positions = []

    for episode in data:
        w_len   = episode["word_len"]
        w_freq  = episode["word_frequency"]
        # Use the .get(...) method to handle missing keys gracefully
        w_pred  = episode.get("Word predictability", 0.0)

        fixations = episode["fixations"]

        # Count number of fixations where "done" is False
        num_fixations_not_done = sum(1 for f in fixations if not f["done"])

        # Store data
        word_lengths_all.append(w_len)
        word_frequencies_all.append(w_freq)
        word_predictabilities_all.append(w_pred)
        num_fixations_all.append(num_fixations_not_done)

        # Store fixation positions for the position analysis
        for f in fixations:
            fixation_positions.append({
                "word_length": w_len,
                "steps": f["steps"],
                "position": f["action"],  # or whatever your position coordinate is
                "done": f["done"]
            })

    # ------------------------------------------------------------------
    # 1. SCATTER PLOTS
    # ------------------------------------------------------------------

    # (a) Word Length vs. Fixations (scatter)
    plt.figure(figsize=(8, 6))
    plt.scatter(word_lengths_all, num_fixations_all, alpha=0.7)
    plt.xlabel("Word Length")
    plt.ylabel("Number of Fixations")
    plt.title("Number of Fixations vs. Word Length (Scatter)")
    plt.grid(True)
    scatter_path = os.path.join(save_file_dir, "fixations_vs_word_length.png")
    plt.savefig(scatter_path)
    plt.close()

    # (b) Word Frequency vs. Fixations (scatter)
    #     If controlled_word_length is set, filter the data
    if controlled_word_length is not None:
        freq_x = [
            wf for (wf, wl) in zip(word_frequencies_all, word_lengths_all)
            if wl == controlled_word_length
        ]
        fix_y = [
            nf for (nf, wl) in zip(num_fixations_all, word_lengths_all)
            if wl == controlled_word_length
        ]
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
    scatter_path = os.path.join(save_file_dir, "fixations_vs_word_frequency.png")
    plt.savefig(scatter_path)
    plt.close()

    # (c) Word Predictability vs. Fixations (scatter)
    if controlled_word_length is not None:
        pred_x = [
            wp for (wp, wl) in zip(word_predictabilities_all, word_lengths_all)
            if wl == controlled_word_length
        ]
        fix_y_pred = [
            nf for (nf, wl) in zip(num_fixations_all, word_lengths_all)
            if wl == controlled_word_length
        ]
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
    scatter_path = os.path.join(save_file_dir, "fixations_vs_word_predictability.png")
    plt.savefig(scatter_path)
    plt.close()

    # ------------------------------------------------------------------
    # 2. LINE PLOTS (average fixations by factor)
    # ------------------------------------------------------------------
    # Helper function: group by unique x-values & compute average fixations
    def compute_average_fixations(xs, ys):
        """
        Given parallel lists xs, ys (each index is one data point),
        group by unique x-values and return sorted (x_unique, y_mean).
        """
        from collections import defaultdict
        aggregator = defaultdict(list)
        for x_val, y_val in zip(xs, ys):
            aggregator[x_val].append(y_val)

        # Sort by x_val and compute means
        x_sorted = sorted(aggregator.keys())
        y_means  = [np.mean(aggregator[x_val]) for x_val in x_sorted]
        return x_sorted, y_means

    # (a) Word Length vs. Average Fixations
    x_sorted_len, y_means_len = compute_average_fixations(word_lengths_all, num_fixations_all)
    plt.figure(figsize=(8, 6))
    plt.plot(x_sorted_len, y_means_len, marker='o', linestyle='-', color='orange')
    plt.xlabel("Word Length")
    plt.ylabel("Average Number of Fixations")
    plt.title("Average Fixations vs. Word Length (Line)")
    plt.grid(True)
    line_path = os.path.join(save_file_dir, "fixations_vs_word_length_line.png")
    plt.savefig(line_path)
    plt.close()

    # (b) Word Frequency vs. Average Fixations
    #     If controlling for word length, filter first
    if controlled_word_length is not None:
        freq_filtered = []
        fix_filtered  = []
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
    line_path = os.path.join(save_file_dir, "fixations_vs_word_frequency_line.png")
    plt.savefig(line_path)
    plt.close()

    # (c) Word Predictability vs. Average Fixations
    if controlled_word_length is not None:
        pred_filtered = []
        fix_filtered_pred = []
        for (wp, wl, nf) in zip(word_predictabilities_all, word_lengths_all, num_fixations_all):
            if wl == controlled_word_length:
                pred_filtered.append(wp)
                fix_filtered_pred.append(nf)
    else:
        pred_filtered     = word_predictabilities_all
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
    line_path = os.path.join(save_file_dir, "fixations_vs_word_predictability_line.png")
    plt.savefig(line_path)
    plt.close()

    # ------------------------------------------------------------------
    # 3. FIXATION POSITION ANALYSIS (existing code)
    # ------------------------------------------------------------------
    fixation_x = [f["position"] for f in fixation_positions if not f["done"]]
    fixation_y = [f["word_length"] for f in fixation_positions if not f["done"]]
    fixation_steps = [f["steps"] for f in fixation_positions if not f["done"]]

    # Normalize fixation steps to [0,1] for color mapping
    if fixation_steps:
        min_step = min(fixation_steps)
        max_step = max(fixation_steps)
        step_range = max(1, max_step - min_step)
        fixation_steps_normalized = (np.array(fixation_steps) - min_step) / step_range
    else:
        fixation_steps_normalized = np.zeros_like(fixation_steps)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        fixation_x, fixation_y,
        c=fixation_steps_normalized, cmap="Blues", alpha=0.7
    )
    plt.xlabel("Fixation Position (Action)")
    plt.ylabel("Word Length")
    plt.title("Fixation Position Analysis (Hue Encodes Fixation Order)")
    plt.colorbar(scatter, label="Fixation Order (Earlier → Darker)")
    plt.grid(True)
    pos_path = os.path.join(save_file_dir, "fixation_positions_hue.png")
    plt.savefig(pos_path)
    plt.close()

    print(f"Plots saved successfully in {save_file_dir}")
