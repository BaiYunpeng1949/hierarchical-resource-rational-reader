import json
import numpy as np

def convert_human_metrics(raw_metrics_path, output_path):
    # Load raw human metrics
    with open(raw_metrics_path, 'r') as f:
        raw_metrics = json.load(f)
    
    # Initialize output structure
    analyzed_metrics = {}
    
    # Process each time condition (30s, 60s, 90s)
    for time_cond in ['30', '60', '90']:
        # Extract relevant metrics and their standard deviations
        reading_speed = raw_metrics[f'Average Reading Speed (wpm)_{time_cond}']
        reading_speed_std = raw_metrics[f'Average Reading Speed (wpm)_{time_cond}_std']
        
        skip_rate = raw_metrics[f'Word Skip Percentage by Saccades V2 With Word Index Correction_{time_cond}'] / 100
        skip_rate_std = raw_metrics[f'Word Skip Percentage by Saccades V2 With Word Index Correction_{time_cond}_std'] / 100
        
        regression_rate = raw_metrics[f'Revisit Percentage by Saccades V2 With Word Index Correction_{time_cond}'] / 100
        regression_rate_std = raw_metrics[f'Revisit Percentage by Saccades V2 With Word Index Correction_{time_cond}_std'] / 100
        
        # Create entry for this time condition
        analyzed_metrics[f'{time_cond}s'] = {
            "reading_speed_mean": reading_speed,
            "reading_speed_std": reading_speed_std,
            "skip_rate_mean": skip_rate,
            "skip_rate_std": skip_rate_std,
            "regression_rate_mean": regression_rate,
            "regression_rate_std": regression_rate_std,
            "num_episodes": 1  # Single measurement per condition
        }
    
    # Save converted metrics
    with open(output_path, 'w') as f:
        json.dump(analyzed_metrics, f, indent=4)

if __name__ == "__main__":
    raw_metrics_path = "raw_human_metrics.json"
    output_path = "analyzed_human_metrics.json"
    convert_human_metrics(raw_metrics_path, output_path)
