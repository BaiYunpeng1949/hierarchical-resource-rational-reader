import matplotlib.pyplot as plt
import numpy as np

# Data from the table
# Format: [human_high_knowledge, sim_high_knowledge, human_low_knowledge, sim_low_knowledge]
high_coherence = [0.484, 0.963, 0.381, 0.552]  # [human_high, sim_high, human_low, sim_low]
low_coherence = [0.417, 0.815, 0.291, 0.529]   # [human_high, sim_high, human_low, sim_low]

# Set up the plot
plt.figure(figsize=(12, 6))

# Set width of bars
barWidth = 0.15

# Set positions of the bars on X axis
r1 = np.arange(2)  # 2 groups: high and low knowledge
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Create the bars with patterns
bars1 = plt.bar(r1, [high_coherence[0], low_coherence[0]], width=barWidth, 
        label='Human (High Knowledge)', color='blue', hatch='/')
bars2 = plt.bar(r2, [high_coherence[1], low_coherence[1]], width=barWidth, 
        label='Sim (High Knowledge)', color='green', hatch='/')
bars3 = plt.bar(r3, [high_coherence[2], low_coherence[2]], width=barWidth, 
        label='Human (Low Knowledge)', color='blue', hatch='.')
bars4 = plt.bar(r4, [high_coherence[3], low_coherence[3]], width=barWidth, 
        label='Sim (Low Knowledge)', color='green', hatch='.')

# Add value labels on top of each bar
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# Add labels and title
plt.xlabel('Coherence Level')
plt.ylabel('Similarity Score')
plt.title('Human vs Simulation Performance Comparison')
plt.xticks([r + barWidth*1.5 for r in range(2)], ['High Coherence', 'Low Coherence'])

# Add legend
plt.legend()

# Add grid for better readability
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('similarity_comparison.png')
plt.close()
