import json
import os
from pathlib import Path

def combine_human_data():
    # Directory containing the raw human data files
    data_dir = Path("_raw_human_data")
    
    # List to store all trials
    all_trials = []
    
    # Process each participant's file
    for file_path in data_dir.glob("*_reading_patterns.json"):
        # Extract participant ID from filename (e.g., "ZAB" from "ZAB_reading_patterns.json")
        participant_id = file_path.stem.split('_')[0]
        
        print(f"Processing {participant_id}...")
        
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Add participant_id to each trial
        for trial in data:
            if trial is not None:  # Skip None values
                # Create a new ordered dictionary with episode_id first
                new_trial = {'episode_id': len(all_trials)}  # Start from 0
                
                # Add all existing fields, inserting participant_id after sentence_id
                for key, value in trial.items():
                    new_trial[key] = value
                    if key == 'sentence_id':
                        new_trial['participant_id'] = participant_id
                
                all_trials.append(new_trial)
    
    # Save combined data
    output_file = os.path.join(os.path.dirname(__file__), "_raw_human_data", "combined_raw_human_data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_trials, f, indent=2, ensure_ascii=False)
    
    num_files = len(list(data_dir.glob('*_reading_patterns.json')))
    print(f"\nCombined {len(all_trials)} trials from {num_files} participants")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    combine_human_data() 