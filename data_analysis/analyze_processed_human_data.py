import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import constants as const
import numpy as np

# Load the CSV file
filename = 'p1_to_p8_processed_metrics_incld_comprehension_with_outliers copy.csv'  # '10_27_21_42_processed_metrics.csv' # '10_23-p1-p32.csv' '10_27_21_42_processed_metrics.csv'
file_path = os.path.join(const.HUMAN_PROCESSED_DATA_DIR, filename)
df = pd.read_csv(file_path)

# Specify the directory where results will be saved
output_dir = os.path.join(
    const.HUMAN_ANALYZED_DATA_DIR,
    f"{datetime.datetime.now().strftime('%m_%d_%H_%M')}_analysis_results_from_{filename}"
)
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# File to save ANOVA results
results_file = os.path.join(output_dir, 'anova_results.txt')

# List of metrics to analyze
metrics = [
    const.FIX_COUNT,
    const.AVG_FIX_COUNT_PER_SEC,
    const.TOTAL_FIX_DUR,
    const.AVG_FIX_DUR_PER_SEC,
    const.AVG_FIX_DUR_PER_COUNT,
    const.SACCADE_COUNT,
    const.AVG_SACCADE_COUNT_PER_SEC,
    const.FIX_COUNT_PERCENTAGE,
    const.AVG_SACCADE_LENGTH_PX,
    const.AVG_SACCADE_LENGTH_DEG,
    const.AVG_SACCADE_VEL_PX,
    const.AVG_SACCADE_VEL_DEG,
    const.REGRESSION_FREQ_XY,
    const.REGRESSION_FREQ_X,
    const.MCQ_ACC,
    const.FREE_RECALL_SCORE,
    const.WORD_SKIP_PERCENTAGE_BY_READING_PROGRESS,
    const.WORD_SKIP_PERCENTAGE_BY_SACCADES,
    const.WORD_SKIP_PERCENTAGE_BY_SACCADES_V2,
    const.WORD_NOT_COVERED_PERCENTAGE,
    const.REVISIT_PERCENTAGE_BY_READING_PROGRESS,
    const.REVISIT_PERCENTAGE_BY_SACCADES,
    const.REVISIT_PERCENTAGE_BY_SACCADES_V2,
    const.READING_SPEED,
]

# Set up the plot style
sns.set(style="whitegrid")

# Function to sanitize file names
def sanitize_filename(filename):
    return filename.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')

def identify_outliers(data):
    # Calculate Q1 and Q3
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    # Define outlier criteria
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Identify outliers
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

# Open the results file for writing
with open(results_file, 'w') as f:
    # Analyze and plot each metric
    for metric in metrics:

        # Skip the metric if it's not in the dataframe
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in the dataset. Skipping this metric.")
            continue  # Skip to the next metric

        ### Create a box plot ###
        plt.figure(figsize=(10, 8))
        ax = sns.boxplot(x='Trial Condition', y=metric, data=df, showfliers=True)

        # Identify and annotate outliers
        for condition in df['Trial Condition'].unique():
            condition_data = df[df['Trial Condition'] == condition][metric]
            outliers = identify_outliers(condition_data)

            # Get the subset of the dataframe for current condition
            condition_df = df[df['Trial Condition'] == condition]

            # Get the numerical position of the condition on the x-axis
            x_positions = {cat: idx for idx, cat in enumerate(sorted(df['Trial Condition'].unique()))}
            x_pos = x_positions[condition]

            # Extract the positions of the fliers
            flier_data = []
            for artist in ax.artists + ax.lines:
                if artist.get_label() == '_nolegend_':
                    # Box and whisker elements
                    continue
                else:
                    # Fliers are plotted as Line2D objects
                    if hasattr(artist, 'get_xdata'):
                        x_data = artist.get_xdata()
                        y_data = artist.get_ydata()
                        for x, y in zip(x_data, y_data):
                            if np.isclose(x, x_pos) and y in outliers.values:
                                flier_data.append((x, y))

            # Annotate outliers on the plot
            for idx, outlier_value in outliers.items():
                participant_id = condition_df.loc[idx, 'Participant ID']
                stimulus_id = condition_df.loc[idx, 'Stimulus ID']
                label = f'ID:{participant_id}, Stim:{stimulus_id}'

                # Find the exact position of the outlier point
                x = x_pos
                y = outlier_value

                ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=(5, 5),  # Offset text by 5 points horizontally and vertically
                    textcoords='offset points',
                    ha='left',
                    va='bottom',
                    fontsize=8,
                    color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.5),
                    zorder=5  # Bring annotation to the front
                )

        plt.title(f'{metric.replace("_", " ").capitalize()} by Trial Condition - Box Plot')
        plt.xlabel('Trial Condition')
        plt.ylabel(metric.replace('_', ' ').capitalize())

        # Adjust layout to prevent clipping of annotations
        plt.tight_layout()

        # Save the box plot
        figure_name = sanitize_filename(f'{metric}_boxplot_by_condition.png')
        plt.savefig(os.path.join(output_dir, figure_name))
        plt.close()

        ### Create a bar chart with **standard deviation (SD)** ###
        plt.figure(figsize=(10, 8))
        ax = sns.barplot(
            x='Trial Condition',
            y=metric,
            data=df,
            errorbar='sd',  # Use 'ci' parameter for standard deviation
            capsize=0.1,  # Adds caps to the error bars
            err_kws={'color': 'black'},  # Set error bar color
            edgecolor='black'
        )

        # Add annotations for mean and standard deviation
        # Calculate mean and std per condition
        summary_df = df.groupby('Trial Condition')[metric].agg(['mean', 'std']).reset_index()

        # Loop over the bars
        for i, bar in enumerate(ax.patches):
            # Get the height of the bar (mean value)
            height = bar.get_height()
            # Get the mean and std from summary_df
            condition = summary_df['Trial Condition'][i]
            mean = summary_df['mean'][i]
            std = summary_df['std'][i]
            # Add text annotation
            ax.text(
                x=bar.get_x() + bar.get_width() / 2,
                y=height,
                s=f'Mean={mean:.2f}\nSD={std:.2f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

        plt.title(f'{metric.replace("_", " ").capitalize()} by Trial Condition - Mean ± SD')
        plt.xlabel('Trial Condition')
        plt.ylabel(metric.replace('_', ' ').capitalize())

        # Save the bar chart with standard deviation
        figure_name = sanitize_filename(f'{metric}_barchart_sd_by_condition.png')
        plt.savefig(os.path.join(output_dir, figure_name))
        plt.close()

        ### Perform ANOVA ###
        groups = [df[df['Trial Condition'] == condition][metric] for condition in df['Trial Condition'].unique()]
        anova_result = f_oneway(*groups)

        # Write ANOVA results to file
        f.write(f"ANOVA result for {metric.replace('_', ' ').capitalize()}:\n")
        f.write(f"F-statistic: {anova_result.statistic}, p-value: {anova_result.pvalue}\n\n")

        # Perform Tukey's HSD test
        tukey = pairwise_tukeyhsd(df[metric], df['Trial Condition'])
        f.write(str(tukey.summary()) + '\n\n')

print(f"Figures saved to {output_dir}")
print(f"ANOVA and Tukey HSD results saved to {results_file}")
