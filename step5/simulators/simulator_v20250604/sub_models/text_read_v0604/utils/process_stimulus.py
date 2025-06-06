import json
import os
from pathlib import Path
import random
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import softmax

# Add the parent directory to Python path to allow imports
sys.path.append(str(Path(__file__).parent.parent))
import Constants

class SentenceAppraiser:
    def __init__(self):
        self.model_name = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()
    
    def get_sentence_surprisal(self, prior_context, sentence):
        """Compute surprisal (log loss) of a sentence given prior context."""
        # Combine context and current sentence
        input_text = prior_context + " " + sentence if prior_context else sentence
        
        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs["input_ids"]

        # Set labels to input IDs for computing loss
        with torch.no_grad():
            outputs = self.model(**inputs, labels=input_ids)
            loss = outputs.loss.item()  # Cross-entropy loss over tokens
        
        # Get the ease rather than the difficulty (loss)
        ease = 1 / (1 + loss)
            
        return round(ease, 3)  # Lower is easier/more predictable

def load_stimulus_data(file_path):
    """Load the stimulus data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_sentences(stimulus_data):
    """Process sentences and generate appraisals using language model."""
    processed_data = []
    appraiser = SentenceAppraiser()
    
    for stimulus in stimulus_data:
        # Get sentence lengths and calculate reading times
        sentence_lengths = [s["word_count"] for s in stimulus["sentences"]]
        sentence_reading_times = [length * Constants.READING_SPEED for length in sentence_lengths]
        
        # Generate appraisal scores using language model
        num_sentences = len(stimulus["sentences"])
        sentence_appraisal_scores = []
        prior_context = ""

        # for sentence_info in stimulus["sentences"]:
        #     current_sentence = sentence_info["sentence"]
        #     score = appraiser.get_sentence_surprisal(prior_context, current_sentence)
        #     sentence_appraisal_scores.append(score)
            
        #     # Accumulate context for the next step
        #     prior_context += " " + current_sentence
        
        # Collect raw ease scores
        raw_scores = []
        for sentence_info in stimulus["sentences"]:
            current_sentence = sentence_info["sentence"]
            score = appraiser.get_sentence_surprisal(prior_context, current_sentence)
            raw_scores.append(score)
            prior_context += " " + current_sentence

        # Min-max normalize to [0, 1]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        if max_score == min_score:
            # Avoid divide-by-zero when all scores are equal
            norm_scores = [1.0 for _ in raw_scores]
        else:
            norm_scores = [(s - min_score) / (max_score - min_score) for s in raw_scores]

        sentence_appraisal_scores = [round(s, 3) for s in norm_scores]

        
        # Pad with -1 to maintain consistent length (20 sentences max)
        padded_scores = sentence_appraisal_scores + [-1] * (20 - num_sentences)
        
        # Create text entry in the same format as TextManager
        text_entry = {
            "stimulus_id": stimulus["stimulus_id"],
            "stimulus_source": "real stimuli",
            "text_content": stimulus["text"],
            "num_sentences": num_sentences,
            "sentence_appraisal_scores_distribution": padded_scores,
            "sentence_lengths": sentence_lengths,
            "sentence_reading_times": sentence_reading_times,
            "total_words": sum(sentence_lengths),
            "total_one_pass_reading_time": sum(sentence_reading_times)
        }
        
        processed_data.append(text_entry)
    
    return processed_data

def save_processed_data(data, output_path):
    """Save the processed data to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    # Get the current file's directory
    current_dir = Path(__file__).parent.parent
    
    # Define input and output paths
    input_file = current_dir / "assets" / "metadata_sentence_indeces.json"
    output_file = current_dir / "assets" / "processed_stimulus.json"
    
    # Create assets directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load and process data
    stimulus_data = load_stimulus_data(input_file)
    processed_data = process_sentences(stimulus_data)
    
    # Save processed data
    save_processed_data(processed_data, output_file)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    main()
