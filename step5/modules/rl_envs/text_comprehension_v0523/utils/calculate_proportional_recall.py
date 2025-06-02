import json
from typing import Dict, Tuple
import argparse
import os
from datetime import datetime

def load_propositions(file_path: str) -> list:
    """Load the organized propositions from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_proportional_recall(propositions: list, high_threshold: float, low_threshold: float) -> Tuple[float, float, float, float]:
    """Calculate proportional recall for both Fully Coherent and Minimally Coherent texts.
    
    Args:
        propositions: List of text entries with their propositions
        high_threshold: Global coherence threshold for high knowledge
        low_threshold: Global coherence threshold for low knowledge
        
    Returns:
        Tuple of (fully_coherent_high_ratio, fully_coherent_low_ratio, 
                 minimally_coherent_high_ratio, minimally_coherent_low_ratio)
    """
    fully_coherent_high_count = 0
    fully_coherent_low_count = 0
    fully_coherent_total = 0
    minimally_coherent_high_count = 0
    minimally_coherent_low_count = 0
    minimally_coherent_total = 0
    
    for text_entry in propositions:
        coherence = text_entry['coherence']
        props = text_entry['original_propositions']
        
        # Count propositions above thresholds
        high_above = sum(1 for p in props if p['global_coherence'] >= high_threshold)
        low_above = sum(1 for p in props if p['global_coherence'] >= low_threshold)
        
        if coherence == "Fully Coherent":
            fully_coherent_high_count += high_above
            fully_coherent_low_count += low_above
            fully_coherent_total += len(props)
        elif coherence == "Minimally Coherent":
            minimally_coherent_high_count += high_above
            minimally_coherent_low_count += low_above
            minimally_coherent_total += len(props)
    
    # Calculate ratios
    fully_coherent_high_ratio = fully_coherent_high_count / fully_coherent_total if fully_coherent_total > 0 else 0
    fully_coherent_low_ratio = fully_coherent_low_count / fully_coherent_total if fully_coherent_total > 0 else 0
    minimally_coherent_high_ratio = minimally_coherent_high_count / minimally_coherent_total if minimally_coherent_total > 0 else 0
    minimally_coherent_low_ratio = minimally_coherent_low_count / minimally_coherent_total if minimally_coherent_total > 0 else 0
    
    return (fully_coherent_high_ratio, fully_coherent_low_ratio,
            minimally_coherent_high_ratio, minimally_coherent_low_ratio)

def save_results(results: str, output_file: str):
    """Save results to a text file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(results)

def main():
    parser = argparse.ArgumentParser(description='Calculate proportional recall based on global coherence thresholds')
    parser.add_argument('--input_file', type=str, default='../assets/organized_example_propositions_v0527.json',
                      help='Path to the input JSON file')
    parser.add_argument('--high_threshold', type=float, default=0.85,
                      help='High knowledge global coherence threshold (default: 0.85)')
    parser.add_argument('--low_threshold', type=float, default=0.70,
                      help='Low knowledge global coherence threshold (default: 0.70)')
    
    args = parser.parse_args()
    
    # Load propositions
    propositions = load_propositions(args.input_file)
    
    # Calculate proportional recall
    (fully_coherent_high_ratio, fully_coherent_low_ratio,
     minimally_coherent_high_ratio, minimally_coherent_low_ratio) = calculate_proportional_recall(
        propositions, args.high_threshold, args.low_threshold)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory path
    results_dir = "../assets/proportional_recall_results"
    output_file = os.path.join(results_dir, f"recall_results_{timestamp}.txt")
    
    # Prepare results text
    results = f"""Proportional Recall Results
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Thresholds:
- High Knowledge Threshold: {args.high_threshold}
- Low Knowledge Threshold: {args.low_threshold}

Results:
Fully Coherent texts:
- High Knowledge (>={args.high_threshold}): {fully_coherent_high_ratio:.2%} of propositions
- Low Knowledge (>={args.low_threshold}): {fully_coherent_low_ratio:.2%} of propositions

Minimally Coherent texts:
- High Knowledge (>={args.high_threshold}): {minimally_coherent_high_ratio:.2%} of propositions
- Low Knowledge (>={args.low_threshold}): {minimally_coherent_low_ratio:.2%} of propositions
"""
    
    # Save results to file
    save_results(results, output_file)
    
    # Also print to console
    print(f"\nResults have been saved to: {output_file}")
    print(results)

if __name__ == "__main__":
    main()
