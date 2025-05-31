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
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                  gridspec_kw={'height_ratios': [4, 1]})
    
    # Create custom colormap: white for -1 (unread), blue to red for scores
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # Plot the main heatmap
    sns.heatmap(appraisal_matrix, 
                cmap=cmap,
                vmin=-1, 
                vmax=1,
                center=0,
                cbar_kws={'label': 'Appraisal Score'},
                xticklabels=range(len(step_logs)),
                yticklabels=range(num_sentences),
                linewidths=1,  # Keep cell borders
                linecolor='black',
                square=False,  # Don't force square cells
                ax=ax1)
    
    # Set up ticks for main heatmap
    ax1.set_xticks(np.arange(len(step_logs)) + 0.5, minor=False)
    ax1.set_yticks(np.arange(num_sentences) + 0.5, minor=False)
    # Remove grid lines
    ax1.grid(False)
    # Remove tick lines but keep labels
    ax1.tick_params(axis='both', which='both', length=0)
    ax1.set_title(f'Sentence Appraisal Levels Over Time (Episode {episode_id})')
    ax1.set_xlabel('')
    ax1.set_ylabel('Sentence Index')
    
    # Extract ongoing comprehension scores and create a single-row matrix
    comprehension_scores = []
    for step in step_logs:
        score = step['on_going_comprehension_log_scalar']
        if isinstance(score, list):
            score = score[0]  # Take the first value if it's a list
        comprehension_scores.append(score)
    
    comprehension_matrix = np.array(comprehension_scores).reshape(1, -1)
    
    # Plot the comprehension heatmap with controlled rectangle size
    sns.heatmap(comprehension_matrix,
                cmap='YlOrRd',  # Yellow to Orange to Red colormap
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Comprehension Score'},
                xticklabels=range(len(step_logs)),
                yticklabels=['Comprehension'],
                linewidths=1,  # Keep cell borders
                linecolor='black',
                square=False,  # Don't force square cells
                ax=ax2)
    
    # Set up ticks for comprehension heatmap
    ax2.set_xticks(np.arange(len(step_logs)) + 0.5, minor=False)
    ax2.set_yticks([0.5], minor=False)
    # Remove grid lines
    ax2.grid(False)
    # Remove tick lines but keep labels
    ax2.tick_params(axis='both', which='both', length=0)
    ax2.set_xlabel('Reading Step')
    ax2.set_ylabel('')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('appraisal_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Example usage
    # json_file_path = "/home/baiy4/reader-agent-zuco/step5/modules/rl_envs/text_comprehension_v0516/temp_sim_data/0520_text_comprehension_v0516_03_rl_model_100000000_steps/5ep/raw_sim_results.json"
    # json_file_path = "/home/baiy4/reader-agent-zuco/step5/modules/rl_envs/text_comprehension_v0516/temp_sim_data/0530_text_comprehension_v0516_04_rl_model_90000000_steps/5ep/raw_sim_results.json"
    json_file_path = "/home/baiy4/reader-agent-zuco/step5/modules/rl_envs/text_comprehension_v0516/temp_sim_data/0530_text_comprehension_v0516_05_rl_model_100000000_steps/5ep/raw_sim_results.json"
    plot_appraisal_heatmap(json_file_path, episode_id=2)
