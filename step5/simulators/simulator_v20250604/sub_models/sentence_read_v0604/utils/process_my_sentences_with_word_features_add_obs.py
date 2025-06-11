import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
import json
import os
from tqdm import tqdm
import numpy as np

# Constants
LANGUAGE_MODEL_NAME = "bert-base-uncased"

def compute_integration_difficulty(tokenizer, model, context: list[str], word: str, full_sentence: list[str], current_word_idx: int) -> tuple[float, float, float, float]:
    """
    Compute integration difficulty for a word given its context using surprisal.
    Returns surprisal, normalized difficulty score, and the word's probability in context.
    """
    # Prepare input by replacing the target word with [MASK]
    masked_sentence = full_sentence.copy()
    masked_sentence[current_word_idx] = "[MASK]"
    masked_text = "[CLS] " + " ".join(masked_sentence) + " [SEP]"
    
    # Tokenize the masked input
    inputs = tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True)
    
    # Find the position of the [MASK] token
    mask_token_id = tokenizer.mask_token_id
    mask_position = (inputs["input_ids"][0] == mask_token_id).nonzero().item()
    
    # Get model predictions at mask position
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, mask_position]
        
    # Get the token ID for the actual word
    word_tokens = tokenizer(word, return_tensors="pt", padding=True, add_special_tokens=False)
    word_token_id = word_tokens["input_ids"][0, 0]
    
    # Compute probability distribution over vocabulary
    probs = torch.softmax(logits, dim=0)
    
    # Get probability of the actual word in this context
    word_probability = probs[word_token_id]
    
    # Compute surprisal: -log P(word | context)
    eps = 1e-10
    surprisal = -torch.log(word_probability + eps).item()
    
    # Convert surprisal to integration difficulty using sigmoid
    alpha = 0.5
    difficulty = torch.sigmoid(alpha * torch.tensor(surprisal)).item()
    
    # Get top 100 predictions and their probabilities
    top_k = 100
    top_probs, top_indices = torch.topk(probs, top_k)
    
    # Find the rank of the actual word
    word_rank = None
    for rank, (token_id, prob) in enumerate(zip(top_indices, top_probs)):
        if token_id == word_token_id:
            word_rank = rank + 1
            break
    
    # Compute ranked word integration probability based on rank
    if word_rank is None:
        ranked_word_integration_probability = 0.2
    elif word_rank <= 3:
        ranked_word_integration_probability = 1.0
    elif word_rank <= 10:
        ranked_word_integration_probability = 1.0
    elif word_rank <= 20:
        ranked_word_integration_probability = 1.0
    elif word_rank <= 50:
        ranked_word_integration_probability = 0.8
    else:
        ranked_word_integration_probability = 0.4
    
    return surprisal, difficulty, word_probability.item(), ranked_word_integration_probability

def compute_word_prediction(tokenizer, model, context: list[str], target_word: str, preview_letters: int = 2, integration_ranks: dict = None) -> tuple[str, float, list[tuple[str, float, float]]]:
    """
    Compute word prediction given context and preview information.
    """
    # Create input with [MASK] token after context
    input_text = " ".join(context + ["[MASK]"])
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    # Find position of [MASK] token
    mask_token_id = tokenizer.mask_token_id
    mask_position = (inputs.input_ids[0] == mask_token_id).nonzero().item()
    
    # Get model predictions for masked position
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, mask_position]
        
    # Get probabilities for all words
    probs = torch.softmax(logits, dim=0)
    
    # Get preview information with noise
    clear_preview = target_word[:preview_letters].lower()
    target_length = len(target_word)
    
    # Define similar-looking letters for noisy preview
    similar_letters = {
        'a': 'aeo', 'e': 'eao', 'o': 'oae',
        'i': 'il1', 'l': 'li1', '1': 'li1',
        'n': 'nm', 'm': 'mn',
        'h': 'hb', 'b': 'bh',
        'p': 'pq', 'q': 'qp',
        'u': 'un', 'n': 'nu',
        'c': 'ce', 'e': 'ec',
        'v': 'vw', 'w': 'wv',
        'r': 'rn', 'n': 'nr',
    }
    
    # Get top predictions initially
    top_k = 1000
    top_probs, top_indices = torch.topk(probs, top_k)
    
    # Get candidates and handle subword tokens
    filtered_predictions = []
    total_filtered_prob = 0.0
    seen_words = set()
    
    for token_id, prob in zip(top_indices, top_probs):
        word = tokenizer.decode([token_id]).strip().lower()
        
        if word in seen_words or word.startswith('[') or not word:
            continue
            
        seen_words.add(word)
        word_len = len(word)
        
        # Check length constraints
        min_length = max(1, target_length - 2)
        max_length = target_length + 2
        if not (min_length <= word_len <= max_length):
            continue

        # Check clear preview
        if not word.startswith(clear_preview):
            continue

        # For letters beyond preview_letters, use noisy matching
        matches_noisy_preview = True
        for i in range(preview_letters, min(len(target_word), len(word), preview_letters + 3)):
            target_char = target_word[i].lower()
            word_char = word[i].lower()
            
            noise_threshold = (i - preview_letters + 1) * 0.3
            
            chars_similar = (word_char in similar_letters.get(target_char, target_char))
            random_accept = torch.rand(1).item() < noise_threshold
            
            if not (chars_similar or random_accept):
                matches_noisy_preview = False
                break

        if matches_noisy_preview:
            word_rank = integration_ranks.get(word, 100) if integration_ranks else 100
            
            if word_rank <= 3:
                ranked_prob = 1.0
            elif word_rank <= 10:
                ranked_prob = 1.0
            elif word_rank <= 20:
                ranked_prob = 1.0
            elif word_rank <= 50:
                ranked_prob = 0.8
            else:
                ranked_prob = 0.4
                
            filtered_predictions.append((word, prob.item(), ranked_prob))
            total_filtered_prob += prob.item()
    
    # Sort and normalize predictions
    filtered_predictions.sort(key=lambda x: x[1], reverse=True)
    
    if filtered_predictions:
        filtered_predictions = [(w, p/total_filtered_prob, r) for w, p, r in filtered_predictions]
        top_predictions = filtered_predictions[:5]
        predicted_word, predictability, _ = top_predictions[0]
    else:
        predicted_word = "[UNKNOWN]"
        predictability = 0.0
        top_predictions = []
    
    return predicted_word, predictability, top_predictions

def process_sentences_with_features(input_path: str, output_path: str):
    """
    Process sentences to add word features, integration difficulty, and prediction information.
    """
    print("Loading dataset...")
    with open(input_path, 'r') as f:
        dataset = json.load(f)
        
    print("Loading language model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(LANGUAGE_MODEL_NAME)
    
    # Set preview letters
    preview_letters = 2
    
    print("Processing sentences...")
    for stimulus_id, stimulus_data in tqdm(dataset.items()):
        for sentence in stimulus_data['sentences']:
            # Get the original words from the input file
            original_words = [word_info['word'] for word_info in sentence['words']]
            words = sentence['sentence'].split()
            processed_words = []
            
            for i, (word, original_word) in enumerate(zip(words, original_words)):
                # Get context and next word
                context = words[:i]
                next_word = words[i + 1] if i + 1 < len(words) else None
                
                # Get existing word features
                existing_features = sentence['words'][i] if 'words' in sentence and i < len(sentence['words']) else {}
                
                # Compute integration difficulty
                surprisal, difficulty, word_prob, ranked_prob = compute_integration_difficulty(
                    tokenizer, model, context, word, words, i
                )
                
                # Get integration ranks
                masked_sentence = words.copy()
                masked_sentence[i] = "[MASK]"
                masked_text = "[CLS] " + " ".join(masked_sentence) + " [SEP]"
                inputs = tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True)
                mask_position = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero().item()
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits[0, mask_position]
                    probs = torch.softmax(logits, dim=0)
                    top_k = 100
                    top_probs, top_indices = torch.topk(probs, top_k)
                
                integration_ranks = {}
                for rank, (token_id, _) in enumerate(zip(top_indices, top_probs)):
                    word = tokenizer.decode([token_id]).strip().lower()
                    if word and not word.startswith('['):
                        integration_ranks[word] = rank + 1
                
                # Compute word prediction for the next word
                if next_word:
                    predicted_word, predictability, top_predictions = compute_word_prediction(
                        tokenizer, model, words[:i+1], next_word, preview_letters, integration_ranks
                    )
                else:
                    predicted_word = "[END]"
                    predictability = 0.0
                    top_predictions = []
                
                # Create word features, preserving existing ones
                word_features = {
                    'word_id': i,
                    'word': original_word,  # Use the original word from input file
                    'word_clean': original_word.lower().strip(),
                    'length': len(original_word),
                    # Preserve existing features
                    'frequency': existing_features.get('frequency', 0),
                    'log_frequency': existing_features.get('log_frequency', 0),
                    'difficulty': existing_features.get('difficulty', 0),
                    'predictability': existing_features.get('predictability', 0),
                    'logit_predictability': existing_features.get('logit_predictability', 0),
                    # Add new features
                    'word_integration_probability': word_prob,
                    'surprisal': surprisal,
                    'integration_difficulty': difficulty,
                    'ranked_word_integration_probability': ranked_prob,
                    'next_word_predicted': predicted_word,
                    'prediction_metadata': {
                        'preview_letters': preview_letters,
                        'clear_preview': next_word[:preview_letters].lower() if next_word else "",
                        'target_length': len(next_word) if next_word else 0,
                        'length_tolerance': {
                            'min': max(1, len(next_word) - 2) if next_word else 0,
                            'max': len(next_word) + 2 if next_word else 0
                        }
                    },
                    'prediction_candidates': [
                        {
                            'word': w,
                            'probability': p,
                            'ranked_word_integration_probability': r
                        }
                        for w, p, r in top_predictions
                    ]
                }
                
                processed_words.append(word_features)
            
            # Update sentence with processed words
            sentence['words'] = processed_words
    
    print("Saving processed dataset...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print("Done! Dataset processed and saved with word features.")

if __name__ == "__main__":
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    parent_dir = os.path.dirname(current_dir)
    
    # Setup paths
    input_file = os.path.join(parent_dir, "assets", "processed_my_stimulus_with_word_features.json")
    output_file = os.path.join(parent_dir, "assets", "processed_my_stimulus_with_observations.json")
    
    process_sentences_with_features(input_file, output_file) 