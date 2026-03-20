import os
import json
import sys

# Add parent directory to path to import simulator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import run_batch_simulations

def check_sentence_word_counts(results_file):
    """
    Check if the number of words in sentences matches between text reading logs and sentence reading summaries.
    
    Args:
        results_file (str): Path to the simulation results JSON file
    """
    # Load the results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Track mismatches
    mismatches = []
    
    # Iterate through each simulation result
    for sim_idx, sim_result in enumerate(results):
        text_reading_logs = sim_result["text_reading_logs"]
        
        # Iterate through each sentence in the text reading logs
        for sentence_idx, sentence_log in enumerate(text_reading_logs):
            # Get the sentence reading summary
            sentence_summary = sentence_log.get("sentence_reading_summary", {})
            
            # Get word counts from both sources
            text_log_words = sentence_log.get("num_words_in_sentence", None)
            summary_words = sentence_summary.get("number_words_in_sentence", None)
            
            # Check if both values exist and match
            if text_log_words is not None and summary_words is not None:
                if text_log_words != summary_words:
                    mismatches.append({
                        "simulation_index": sim_idx,
                        "sentence_index": sentence_idx,
                        "text_log_words": text_log_words,
                        "summary_words": summary_words,
                        "stimulus_id": sim_result["stimulus_index"],
                        "time_condition": sim_result["time_condition"]
                    })
            else:
                print(f"Warning: Missing word count data in simulation {sim_idx}, sentence {sentence_idx}")
                print(f"Text log words: {text_log_words}, Summary words: {summary_words}")
    
    # Print results
    if mismatches:
        print("\nFound mismatches in word counts:")
        for mismatch in mismatches:
            print(f"\nSimulation {mismatch['simulation_index']} (Stimulus {mismatch['stimulus_id']}, {mismatch['time_condition']}):")
            print(f"Sentence {mismatch['sentence_index']}:")
            print(f"  Text log words: {mismatch['text_log_words']}")
            print(f"  Summary words: {mismatch['summary_words']}")
        print(f"\nTotal mismatches found: {len(mismatches)}")
    else:
        print("\nNo mismatches found in word counts!")

if __name__ == "__main__":
    
    # File name
    file_name = "20250613_1652_trials5_stims9_conds3"

    # Get the current directory
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(current_dir, "simulated_results", file_name, "all_simulation_results.json")

    # Example usage
    check_sentence_word_counts(results_file)
