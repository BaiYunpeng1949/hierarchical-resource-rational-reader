import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from pathlib import Path
from scipy import stats




if __name__ == "__main__":
    

    json_file_path = "/home/baiy4/reader-agent-zuco/step5/modules/rl_envs/text_comprehension_v0516/temp_sim_data/0708_text_comprehension_v0516_no_time_decay_softmin_reward_function_hierarchical_discrete_actions_limited_episodes_03_rl_model_40000000_steps/1000ep/raw_sim_results.json"
    
    # Load the simulation data
    json_path = Path(json_file_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    # Collect all appraisals and regressed sentence appraisals
    all_appraisals = []
    regressed_appraisals = []

    for episode in data:
        init_appraisals = episode["init_sentence_appraisal_scores_distribution"]
        # Filter out -1 values (invalid appraisals)
        valid_appraisals = [score for score in init_appraisals if score >= 0]
        all_appraisals.extend(valid_appraisals)

        for step in episode["step_wise_log"]:
            if step["is_regress"]:
                idx = step["actual_reading_sentence_index"]
                if idx is not None and idx < len(init_appraisals):
                    score = init_appraisals[idx]
                    if score >= 0:  # Only include valid scores
                        regressed_appraisals.append(score)

    # Convert to numpy arrays
    all_appraisals = np.array(all_appraisals)
    regressed_appraisals = np.array(regressed_appraisals)

    print(f"Total valid sentences: {len(all_appraisals)}")
    print(f"Total regressed sentences: {len(regressed_appraisals)}")
    print(f"Mean appraisal of all sentences: {np.mean(all_appraisals):.3f}")
    print(f"Mean appraisal of regressed sentences: {np.mean(regressed_appraisals):.3f}")
    
    # Statistical test
    if len(regressed_appraisals) > 0:
        t_stat, p_value = stats.ttest_ind(all_appraisals, regressed_appraisals)
        print(f"T-test p-value: {p_value:.6f}")
        print(f"Regressed sentences are significantly different: {p_value < 0.05}")

    # Create single plot for proportion
    plt.figure(figsize=(5, 3))

    # Calculate proportion of sentences regressed in each bin
    bins = np.linspace(0.0, 1.0, 21)  # 0.0 to 1.0 in 0.05 steps
    all_counts, _ = np.histogram(all_appraisals, bins=bins)
    regress_counts, _ = np.histogram(regressed_appraisals, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate proportion of sentences regressed in each bin
    proportions = np.zeros_like(all_counts, dtype=float)
    mask = all_counts > 0
    proportions[mask] = regress_counts[mask] / all_counts[mask]
    
    # Filter out bins with zero counts for plotting
    plot_mask = all_counts > 0
    plot_centers = bin_centers[plot_mask]
    plot_proportions = proportions[plot_mask]
    
    # Plot as line chart in green
    plt.plot(plot_centers, plot_proportions, color='green', linewidth=3, marker='o', markersize=6)
    plt.title("Proportion of Sentences Regressed by Appraisal Score", fontsize=14, fontweight='bold')
    plt.xlabel("Initial Appraisal Score", fontsize=12)
    plt.ylabel("Proportion Regressed", fontsize=12)
    plt.grid(True, alpha=0.3)
    # Removed ylim to let matplotlib auto-scale
    
    # # Add text annotation for key statistics
    # low_appraisal_threshold = 0.5
    # low_appraisal_mask = all_appraisals < low_appraisal_threshold
    # high_appraisal_mask = all_appraisals >= low_appraisal_threshold
    
    # low_regress_mask = regressed_appraisals < low_appraisal_threshold
    # high_regress_mask = regressed_appraisals >= low_appraisal_threshold
    
    # low_prop = np.sum(low_regress_mask) / np.sum(low_appraisal_mask) if np.sum(low_appraisal_mask) > 0 else 0
    # high_prop = np.sum(high_regress_mask) / np.sum(high_appraisal_mask) if np.sum(high_appraisal_mask) > 0 else 0
    
    # plt.text(0.05, 0.95, f'Low appraisal (<0.5): {low_prop:.3f}', 
    #          transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    # plt.text(0.05, 0.85, f'High appraisal (≥0.5): {high_prop:.3f}', 
    #          transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

    plt.tight_layout()
    plt.savefig("proportion_regressed_by_appraisal_score.png", dpi=300, bbox_inches='tight')
    print("Plot generated: 'proportion_regressed_by_appraisal_score.png'")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    # print(f"Low appraisal sentences (<0.5): {np.sum(low_appraisal_mask)} total, {np.sum(low_regress_mask)} regressed ({low_prop:.1%})")
    # print(f"High appraisal sentences (≥0.5): {np.sum(high_appraisal_mask)} total, {np.sum(high_regress_mask)} regressed ({high_prop:.1%})")
    # print(f"Ratio of regress rates: {low_prop/high_prop:.2f}x more likely to regress low-appraisal sentences")
    
    # Print detailed proportions for each bin
    print(f"\nDetailed proportions by appraisal score bin:")
    for i, (center, prop, count) in enumerate(zip(bin_centers, proportions, all_counts)):
        if count > 0:
            print(f"Bin {i+1} (center={center:.2f}): {count} total, {regress_counts[i]} regressed, proportion={prop:.3f}")
