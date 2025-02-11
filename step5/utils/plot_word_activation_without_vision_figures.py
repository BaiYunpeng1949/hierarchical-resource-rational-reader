import os
import matplotlib.pyplot as plt
import json
import numpy as np

def analyze_fixations(json_data, save_file_dir):
    """
    Analyzes fixation data and generates three plots:
    1. Number of Fixations vs. Word Length
    2. Number of Fixations vs. Word Frequency
    3. Fixation Position Analysis
    Saves the plots in the specified directory.
    """

    # Parse the JSON data
    data = json.loads(json_data)

    # Ensure the directory exists
    os.makedirs(save_file_dir, exist_ok=True)

    # Extract data for analysis
    word_lengths = []
    word_frequencies = []
    num_fixations = []
    fixation_positions = []

    for episode in data:
        word_length = episode["word_len"]
        word_frequency = episode["word_frequency"]
        fixations = episode["fixations"]

        # Count the number of fixations where "done" is False
        num_fixations_not_done = sum(1 for f in fixations if not f["done"])

        # Store data for plotting
        word_lengths.append(word_length)
        word_frequencies.append(word_frequency)
        num_fixations.append(num_fixations_not_done)

        # Store fixation positions along with word length for visualization
        for f in fixations:
            fixation_positions.append({
                "word_length": word_length,
                "steps": f["steps"],
                "position": f["action"],  # Assuming action represents fixation position
                "done": f["done"]
            })

    # Plot Number of Fixations vs. Word Length
    plt.figure(figsize=(8, 6))
    plt.scatter(word_lengths, num_fixations, alpha=0.7)
    plt.xlabel("Word Length")
    plt.ylabel("Number of Fixations")
    plt.title("Number of Fixations vs. Word Length")
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "fixations_vs_word_length.png"))
    plt.close()

    # Plot Number of Fixations vs. Word Frequency
    plt.figure(figsize=(8, 6))
    plt.scatter(word_frequencies, num_fixations, alpha=0.7)
    plt.xlabel("Word Frequency")
    plt.ylabel("Number of Fixations")
    plt.title("Number of Fixations vs. Word Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "fixations_vs_word_frequency.png"))
    plt.close()

    # Process fixation positions for visualization with color hues
    fixation_x = [f["position"] for f in fixation_positions if not f["done"]]
    fixation_y = [f["word_length"] for f in fixation_positions if not f["done"]]
    fixation_steps = [f["steps"] for f in fixation_positions if not f["done"]]

    # Normalize fixation steps to [0,1] range for colormap
    if fixation_steps:
        fixation_steps_normalized = (np.array(fixation_steps) - min(fixation_steps)) / max(1, (max(fixation_steps) - min(fixation_steps)))
    else:
        fixation_steps_normalized = np.zeros_like(fixation_steps)  # Handle empty case

    # Plot Fixation Positions with color hues
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(fixation_x, fixation_y, c=fixation_steps_normalized, cmap="Blues", alpha=0.7)
    plt.xlabel("Fixation Position (Action)")
    plt.ylabel("Word Length")
    plt.title("Fixation Position Analysis (Hue Encodes Fixation Order)")
    plt.colorbar(scatter, label="Fixation Order (Earlier → Darker)")
    plt.grid(True)
    plt.savefig(os.path.join(save_file_dir, "fixation_positions_hue.png"))
    plt.close()

    print(f"Plots saved successfully in {save_file_dir}")
