import json
import os
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load BAAI bge-base-en model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en")
model = AutoModel.from_pretrained("BAAI/bge-base-en")

def embed(texts):
    """Generate embeddings for a list of texts using the BGE model."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.numpy()

def calculate_coherence_scores(propositions: List[str], window_size: int = 5) -> tuple[List[float], List[float]]:
    """Calculate both global and local coherence scores for each proposition.
    
    Args:
        propositions: List of propositions to calculate coherence for
        window_size: Size of the sliding window for local coherence calculation
        
    Returns:
        Tuple of (global_scores, local_scores) where each is a list of float scores

        NOTE: these scores could be normalised to increase the parameter tuning range, but we don't do that here for now
    """
    if not propositions:
        return [], []
    
    # Get embeddings for all propositions
    embeddings = embed(propositions)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Calculate global coherence scores
    global_scores = []
    for i in range(len(propositions)):
        # Get similarities with all other propositions
        similarities = similarity_matrix[i]
        # Remove self-similarity
        similarities = np.delete(similarities, i)
        # Calculate average similarity
        avg_similarity = np.mean(similarities)
        global_scores.append(float(avg_similarity))
    
    # Calculate local coherence scores using sliding window
    local_scores = []
    for i in range(len(propositions)):
        # Define window boundaries
        window_start = max(0, i - window_size + 1)
        window_end = min(len(propositions), i + window_size)
        
        # Get similarities within the window
        window_similarities = similarity_matrix[i, window_start:window_end]
        # Remove self-similarity if it exists in the window
        if i >= window_start and i < window_end:
            window_similarities = np.delete(window_similarities, i - window_start)
        
        # Calculate average similarity within window
        if len(window_similarities) > 0:
            avg_window_similarity = np.mean(window_similarities)
        else:
            avg_window_similarity = 0.0
            
        local_scores.append(float(avg_window_similarity))
    
    return global_scores, local_scores

def load_example_texts(file_path: str) -> List[Dict[str, Any]]:
    """Load the example texts from the JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_propositions(texts: List[Dict[str, Any]], window_size: int = 5) -> List[Dict[str, Any]]:
    """Extract all propositions from the texts and organize them with coherence scores."""
    organized_texts = []
    current_coherence = None
    prop_counter = 1  # Counter for unique proposition IDs
    
    for text in texts:
        text_propositions = []
        coherence = text['coherence']
        
        # Reset counter if coherence level changes
        if coherence != current_coherence:
            prop_counter = 1
            current_coherence = coherence
        
        # Collect all propositions for this coherence level
        all_props = []
        for sentence in text['text_into_sentences']:
            all_props.extend(sentence['propositions'])
        
        # Calculate both global and local coherence scores
        global_scores, local_scores = calculate_coherence_scores(all_props, window_size)
        
        # Create proposition entries with their scores
        for prop, global_score, local_score in zip(all_props, global_scores, local_scores):
            text_propositions.append({
                'proposition_index': prop_counter,
                'proposition_text': prop,
                'global_coherence': global_score,
                'local_coherence': local_score
            })
            prop_counter += 1
        
        organized_texts.append({
            'text_title': text['text_title'],
            'coherence': text['coherence'],
            'original_propositions': text_propositions
        })
    
    return organized_texts

def organize_propositions(input_file: str, output_file: str, window_size: int = 5):
    """Organize propositions from example texts and save to a new JSON file."""
    # Load the example texts
    texts = load_example_texts(input_file)
    
    # Extract and organize propositions
    organized_texts = extract_propositions(texts, window_size)

    # NOTE: window size is an empirically tunable parameter, default as 5, within the STM limit
    
    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(organized_texts, f, indent=2)

if __name__ == "__main__":
    # Define file paths
    input_file = "../assets/example_texts_v0526_chatgpt_generated.json"
    output_file = "../assets/organized_example_propositions_v0527.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Organize propositions with default window size of 5
    organize_propositions(input_file, output_file)
