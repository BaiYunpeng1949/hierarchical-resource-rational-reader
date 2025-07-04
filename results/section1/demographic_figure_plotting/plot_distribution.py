import matplotlib.pyplot as plt
import numpy as np
# from scipy.interpolate import interp1d
# from scipy.stats import norm

# Example data
words = [
    "reading word", 
    "adjacent word1", "adjacent word2", "adjacent word3",
    # "adjacent word3", "adjacent word4", "adjacent word5", 
    # "reading word", 
    # "adjacent word6", "adjacent word7", 
    # "adjacent word8", "adjacent word9"
]
# beliefs: Sharper Gaussian-like, highest at reading word, drops off quickly
# beliefs = [0.005, 0.01, 0.02, 0.04, 0.10, 0.65, 0.10, 0.04, 0.02, 0.005]  # Must sum to 1
# beliefs = [0.3, 0.6, 0.7, 0.1, 0.05]
beliefs = [0.9, 0.6, 0.7, 0.65]

# Define RGB colors (0-255)
bar_colors_rgb = [
    (0, 255, 0),
    (196, 196, 196), (196, 196, 196), (196, 196, 196),
    # (246, 198, 173), (246, 198, 173), (246, 198, 173),
    # (255, 0, 0),  # Red for reading word
    # (246, 198, 173),   # normal color for the initalization
    # (246, 198, 173), (246, 198, 173), 
    # (246, 198, 173), (246, 198, 173)
]

# Convert to 0-1 scale for matplotlib
bar_colors = [(r/255, g/255, b/255) for r, g, b in bar_colors_rgb]

# Plot
plt.figure(figsize=(2, 2))
ax = plt.gca()
bars = ax.bar(words, beliefs, color=bar_colors)

# Add black border to the central bar
central_bar_index = words.index("reading word")
bars[central_bar_index].set_edgecolor('black')
bars[central_bar_index].set_linewidth(2)

plt.ylabel("Sentence appraisal", fontsize=12)
# plt.title("Belief Distribution over Words")
plt.ylim(0, 1)

# Hide the entire x-axis (ticks and labels)
ax.get_xaxis().set_visible(False)

# Remove the top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig("belief_distribution.png", dpi=300, bbox_inches='tight')

# Move x-label to the end of the x-axis (optional, you can remove this if you want nothing at all)
# plt.text(len(words)-0.5, -0.08, "Parallelly Aggregated Words", ha='right', va='top', fontsize=12, transform=plt.gca().transAxes)

plt.show()
