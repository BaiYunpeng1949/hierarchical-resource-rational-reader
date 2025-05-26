import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_appraisal_heatmap(json_file_path, episode_id=1):
    """
    Create a heatmap of sentence appraisal levels over time.
    
    Args:
        json_file_path (str): Path to the JSON file containing simulation results
        episode_id (int): Which episode to plot (default: 1)
    """
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Find the specified episode
    episode_data = None
    for episode in data:
        if episode['episode_id'] == episode_id:
            episode_data = episode
            break
    
    if episode_data is None:
        raise ValueError(f"Episode {episode_id} not found in the data")
    
    # Extract the step-wise logs, excluding terminated steps
    step_logs = [step for step in episode_data['step_wise_log'] if not step['terminate']]
    
    # Get the number of sentences
    num_sentences = episode_data['num_sentences']
    
    # Create a matrix to store appraisal scores
    # Initialize with -1 (unread sentences)
    appraisal_matrix = np.full((num_sentences, len(step_logs)), -1.0)
    
    # Fill the matrix with appraisal scores
    for step_idx, step in enumerate(step_logs):
        scores = step['already_read_sentences_appraisal_scores_distribution']
        for sent_idx, score in enumerate(scores):
            if sent_idx < num_sentences:  # Ensure we don't exceed the number of sentences
                appraisal_matrix[sent_idx, step_idx] = score
    
    # Create the heatmap
    plt.figure(figsize=(15, 8))
    
    # Create custom colormap: white for -1 (unread), blue to red for scores
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # Plot the heatmap with more prominent grid
    ax = sns.heatmap(appraisal_matrix, 
                    cmap=cmap,
                    vmin=-1, 
                    vmax=1,
                    center=0,
                    cbar_kws={'label': 'Appraisal Score'},
                    xticklabels=range(len(step_logs)),
                    yticklabels=range(num_sentences),
                    linewidths=1,  # Make grid lines more visible
                    linecolor='black')  # Black grid lines
    
    # Make grid lines more prominent
    ax.set_xticks(np.arange(len(step_logs)) + 0.5, minor=False)
    ax.set_yticks(np.arange(num_sentences) + 0.5, minor=False)
    ax.grid(True, which='major', color='black', linewidth=1)
    
    plt.title(f'Sentence Appraisal Levels Over Time (Episode {episode_id})')
    plt.xlabel('Reading Step')
    plt.ylabel('Sentence Index')
    
    # Save the figure
    plt.savefig('appraisal_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Example usage
    json_file_path = "/home/baiy4/reader-agent-zuco/step5/modules/rl_envs/text_comprehension_v0516/temp_sim_data/0520_text_comprehension_v0516_03_rl_model_100000000_steps/5ep/raw_sim_results.json"
    plot_appraisal_heatmap(json_file_path, episode_id=1)
