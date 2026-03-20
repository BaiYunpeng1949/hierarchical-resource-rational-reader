import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from . import Constants
# import Constants
import json
import os
from tqdm import tqdm
import math
import numpy as np


def calc_dynamic_sentence_comprehension_score(scores, mode="softmin", tau=0.1):
    """
    Compute the dynamic sentence comprehension score

    Args:
        scores: list[float] -- the appraisal scores of the words in the sentence
        mode: str
        tau: float

    Returns:
        float
    """
    if len(scores) == 0:
        return 0
    else:
        if mode == "geometric mean":
            log_sum = sum(math.log(max(s, 1e-9)) for s in scores)
            return math.exp(log_sum / len(scores))
        elif mode == "harmonic mean":
            return len(scores) / sum(1/s for s in scores)
        elif mode == "softmin":
            w = np.exp(-np.array(scores) / tau)
            return float((w * scores).sum() / w.sum())
        elif mode == "mean":
            return np.mean(scores)
        elif mode == "deterministic_aggregated_predictability":
            # Expected value of your ±1 Bernoulli experiment:
            #   P(correct)=p  →  +1
            #   P(incorrect)=1-p → -1
            #   E[value] = (1)(p) + (-1)(1-p) = 2p - 1
            values = [2 * p_i - 1 for p_i in scores]   # each ∈ [‑1, +1]
            return float(sum(values))
        elif mode == "stochastic_aggregated_predictability":
            # Stochastic explicit Bernoulli draw
            samples = [np.random.binomial(1, p_i) for p_i in scores]
            aggregated_predicted_values = sum([1 if sample == 1 else -1 for sample in samples])
            return aggregated_predicted_values
        else:
            raise ValueError(f"Invalid mode: {mode}")


def compute_integration_difficulty(tokenizer, model, context: list[str], word: str, full_sentence: list[str], current_word_idx: int) -> tuple[float, float, float, float]:
    """
    Compute integration difficulty for a word given its context using surprisal.
    Returns surprisal, normalized difficulty score, and the word's probability in context.
    
    Args:
        tokenizer: BERT tokenizer
        model: BERT masked language model
        context: List of words before the target word
        word: The target word to evaluate
        full_sentence: Complete sentence containing the target word
        current_word_idx: Position of the target word in the sentence
        
    Returns:
        tuple[float, float, float, float]: (surprisal, difficulty, word_probability, ranked_word_integration_probability)
        - surprisal: -log P(word | context), measures unexpectedness
        - difficulty: Sigmoid-scaled surprisal, normalized to [0,1]
        - word_integration_probability: P(word | context), measures expectedness [0,1]
        - ranked_word_integration_probability: Rank-based normalized probability [0,1]

    NOTE: check when doing the predictability whether needs to control the size of the context (STM size around 4)
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
        logits = outputs.logits[0, mask_position]  # Shape: [vocab_size]
        
    # Get the token ID for the actual word
    word_tokens = tokenizer(word, return_tensors="pt", padding=True, add_special_tokens=False)
    word_token_id = word_tokens["input_ids"][0, 0]  # Get first token ID
    
    # Compute probability distribution over vocabulary
    probs = torch.softmax(logits, dim=0)  # Shape: [vocab_size]
    
    # Get probability of the actual word in this context
    word_probability = probs[word_token_id]  # Keep as tensor for computation # P(word | context)
    
    # Compute surprisal: -log P(word | context)
    eps = 1e-10  # Small epsilon to avoid log(0)
    surprisal = -torch.log(word_probability + eps).item()
    
    # Convert surprisal to integration difficulty using sigmoid
    alpha = 0.5  # Scaling factor
    difficulty = torch.sigmoid(alpha * torch.tensor(surprisal)).item()
    
    # Get top 100 predictions and their probabilities
    top_k = 100
    top_probs, top_indices = torch.topk(probs, top_k)
    
    # Find the rank of the actual word
    word_rank = None
    for rank, (token_id, prob) in enumerate(zip(top_indices, top_probs)):
        if token_id == word_token_id:
            word_rank = rank + 1  # 1-based ranking
            break
    
    # Compute ranked word integration probability based on rank
    if word_rank is None:
        ranked_word_integration_probability = 0.2  # Default value for words not in top 100
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

def process_dataset_with_integration_difficulty(input_path: str, output_path: str):
    """
    Process the dataset to add integration difficulty scores for each word.
    
    Args:
        input_path: Path to input JSON dataset
        output_path: Path to save processed dataset
    """
    print("Loading dataset...")
    with open(input_path, 'r') as f:
        dataset = json.load(f)
        
    print("Loading language model and tokenizer...")
    model_name = Constants.LANGUAGE_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    print("Processing sentences...")
    for sentence_id, sentence_data in tqdm(dataset.items()):
        words = sentence_data["words"]
        sentence = [word_data["word"] for word_data in words]
        
        # Process each word in the sentence
        for i, word_data in enumerate(words):
            # Get context (all words before current word)
            context = sentence[:i]
            
            # Compute integration difficulty
            surprisal, difficulty, word_integration_probability_given_sentence_as_context, ranked_word_integration_probability = compute_integration_difficulty(
                tokenizer, 
                model, 
                context, 
                word_data["word"],
                sentence,
                i
            )
            
            # Add scores to word data
            word_data["word_integration_probability"] = word_integration_probability_given_sentence_as_context
            word_data["surprisal"] = surprisal
            word_data["integration_difficulty"] = difficulty
            word_data["ranked_word_integration_probability"] = ranked_word_integration_probability
    
    print("Saving processed dataset...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print("Done! Dataset processed and saved with integration difficulty scores.")

def compute_word_prediction(tokenizer, model, context: list[str], target_word: str, preview_letters: int = 2, integration_ranks: dict = None) -> tuple[str, float, list[tuple[str, float, float]]]:
    """
    Compute word prediction given context and preview information.
    Returns the predicted word, its probability, and top candidates.
    
    Args:
        tokenizer: BERT tokenizer
        model: BERT masked language model
        context: List of words before target word
        target_word: The actual word to predict
        preview_letters: Number of clear letters visible in preview (default 2)
        integration_ranks: Dictionary mapping words to their ranks in integration difficulty computation
        
    Returns:
        tuple[str, float, list[tuple[str, float, float]]]: (predicted_word, probability, top_candidates)
        where top_candidates is a list of (word, probability, ranked_word_integration_probability)
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
        logits = outputs.logits[0, mask_position]  # Shape: [vocab_size]
        
    # Get probabilities for all words
    probs = torch.softmax(logits, dim=0)
    
    # Get preview information with noise
    clear_preview = target_word[:preview_letters].lower()  # First 2 letters are clear
    target_length = len(target_word)
    
    # Define similar-looking letters for noisy preview
    similar_letters = {
        'a': 'aeo', 'e': 'eao', 'o': 'oae',  # Round letters
        'i': 'il1', 'l': 'li1', '1': 'li1',  # Vertical lines
        'n': 'nm', 'm': 'mn',                 # n/m confusion
        'h': 'hb', 'b': 'bh',                 # Ascending letters
        'p': 'pq', 'q': 'qp',                 # Descending letters
        'u': 'un', 'n': 'nu',                 # u/n confusion
        'c': 'ce', 'e': 'ec',                 # c/e confusion
        'v': 'vw', 'w': 'wv',                 # v/w confusion
        'r': 'rn', 'n': 'nr',                 # r/n confusion
    }
    
    # Get top predictions initially
    top_k = 1000
    top_probs, top_indices = torch.topk(probs, top_k)
    
    # Get candidates and handle subword tokens
    filtered_predictions = []
    total_filtered_prob = 0.0
    seen_words = set()  # To avoid duplicates from subword tokens
    
    for token_id, prob in zip(top_indices, top_probs):
        # Decode single token
        word = tokenizer.decode([token_id]).strip().lower()
        
        # Skip if we've seen this word or if it's a special token
        if word in seen_words or word.startswith('[') or not word:
            continue
            
        seen_words.add(word)
        word_len = len(word)
        
        # Check length constraints with blurry length estimation
        min_length = max(1, target_length - 2)
        max_length = target_length + 2
        if not (min_length <= word_len <= max_length):
            continue

        # Check clear preview (first 2 letters must match exactly)
        if not word.startswith(clear_preview):
            continue

        # For letters beyond preview_letters, use noisy matching
        matches_noisy_preview = True
        for i in range(preview_letters, min(len(target_word), len(word), preview_letters + 3)):
            target_char = target_word[i].lower()
            word_char = word[i].lower()
            
            # The further the letter, the more noise we allow
            noise_threshold = (i - preview_letters + 1) * 0.3  # Increases noise with distance
            
            # Check if chars are similar or randomly accept with increasing probability
            chars_similar = (word_char in similar_letters.get(target_char, target_char))
            random_accept = torch.rand(1).item() < noise_threshold
            
            if not (chars_similar or random_accept):
                matches_noisy_preview = False
                break

        if matches_noisy_preview:
            # Get the word's rank from integration difficulty computation
            word_rank = integration_ranks.get(word, 100) if integration_ranks else 100
            
            # Compute ranked_word_integration_probability based on rank
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
    
    # Sort filtered predictions by probability
    filtered_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Normalize probabilities
    if filtered_predictions:
        # Normalize probabilities for sampling
        filtered_predictions = [(w, p/total_filtered_prob, r) for w, p, r in filtered_predictions]
        
        # Get top 5 predictions
        top_predictions = filtered_predictions[:5]
        
        # Get most likely word and its probability
        predicted_word, predictability, _ = top_predictions[0]
    else:
        # Fallback if no predictions
        predicted_word = "[UNKNOWN]"
        predictability = 0.0
        top_predictions = []
    
    return predicted_word, predictability, top_predictions

def process_dataset_with_predictions(input_path: str, output_path: str):
    """
    Process the dataset to add word prediction information for each word.
    Builds on top of the integration difficulty data.
    
    Args:
        input_path: Path to input JSON dataset (with integration difficulty)
        output_path: Path to save processed dataset
    """
    print("Loading dataset...")
    with open(input_path, 'r') as f:
        dataset = json.load(f)
        
    print("Loading language model and tokenizer...")
    model_name = Constants.LANGUAGE_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    # Set preview letters (currently using 2)
    preview_letters = 2
    
    print("Processing sentences...")
    for sentence_id, sentence_data in tqdm(dataset.items()):
        words = sentence_data["words"]
        sentence = [word_data["word"] for word_data in words]
        
        # Process each word in the sentence
        for i, word_data in enumerate(words):
            # Get context (all words before current word)
            context = sentence[:i]
            target_word = word_data["word"]
            
            # Get preview information
            clear_preview = target_word[:preview_letters].lower()
            target_length = len(target_word)
            
            # Get length tolerance range
            min_length = max(1, target_length - 2)
            max_length = target_length + 2
            
            # Get top 100 predictions for integration difficulty ranking
            masked_sentence = sentence.copy()
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
            
            # Create dictionary mapping words to their ranks
            integration_ranks = {}
            for rank, (token_id, _) in enumerate(zip(top_indices, top_probs)):
                word = tokenizer.decode([token_id]).strip().lower()
                if word and not word.startswith('['):
                    integration_ranks[word] = rank + 1
            
            # Step 2: Compute word prediction with integration ranks
            predicted_word, predictability, top_predictions = compute_word_prediction(
                tokenizer,
                model,
                context,
                target_word,
                preview_letters,
                integration_ranks
            )
            
            # Add prediction information to word data
            word_data["next_word_predicted"] = predicted_word
            word_data["predictability"] = predictability
            word_data["prediction_metadata"] = {
                "preview_letters": preview_letters,
                "clear_preview": clear_preview,
                "target_length": target_length,
                "length_tolerance": {
                    "min": min_length,
                    "max": max_length
                }
            }
            word_data["prediction_candidates"] = [
                {
                    "word": word,
                    "probability": prob,
                    "ranked_word_integration_probability": rank_prob
                }
                for word, prob, rank_prob in top_predictions
            ]
    
    print("Saving processed dataset...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print("Done! Dataset processed and saved with word prediction information.")

def process_dataset(input_path: str, output_path: str):
    """
    Process the raw dataset to add both integration difficulty and word prediction information.
    This is a combined pipeline that processes the raw dataset in one go.
    
    Args:
        input_path: Path to raw JSON dataset
        output_path: Path to save processed dataset
    """
    print("Loading raw dataset...")
    with open(input_path, 'r') as f:
        dataset = json.load(f)
        
    print("Loading language model and tokenizer...")
    model_name = Constants.LANGUAGE_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    # Set preview letters (currently using 2)
    preview_letters = 2
    
    print("Processing sentences...")
    for sentence_id, sentence_data in tqdm(dataset.items()):
        words = sentence_data["words"]
        sentence = [word_data["word"] for word_data in words]
        
        # Process each word in the sentence
        for i, word_data in enumerate(words):
            # Get context (all words before current word)
            context = sentence[:i]
            target_word = word_data["word"]
            
            # Step 1: Compute integration difficulty
            surprisal, difficulty, word_integration_probability_given_sentence_as_context, ranked_word_integration_probability = compute_integration_difficulty(
                tokenizer, 
                model, 
                context, 
                target_word,
                sentence,
                i
            )
            
            # Add integration difficulty scores and word probability
            word_data["word_integration_probability"] = word_integration_probability_given_sentence_as_context  # P(word | context)
            word_data["surprisal"] = surprisal
            word_data["integration_difficulty"] = difficulty
            word_data["ranked_word_integration_probability"] = ranked_word_integration_probability
            
            # Get top 100 predictions for integration difficulty ranking
            masked_sentence = sentence.copy()
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
            
            # Create dictionary mapping words to their ranks
            integration_ranks = {}
            for rank, (token_id, _) in enumerate(zip(top_indices, top_probs)):
                word = tokenizer.decode([token_id]).strip().lower()
                if word and not word.startswith('['):
                    integration_ranks[word] = rank + 1
            
            # Step 2: Get preview information
            clear_preview = target_word[:preview_letters].lower()
            target_length = len(target_word)
            
            # Get length tolerance range
            min_length = max(1, target_length - 2)
            max_length = target_length + 2
            
            # Step 3: Compute word prediction with integration ranks
            predicted_word, _, top_predictions = compute_word_prediction(
                tokenizer,
                model,
                context,
                target_word,
                preview_letters,
                integration_ranks
            )
            
            # Add prediction information, using original predictability values
            word_data["next_word_predicted"] = predicted_word
            word_data["prediction_metadata"] = {
                "preview_letters": preview_letters,
                "clear_preview": clear_preview,
                "target_length": target_length,
                "length_tolerance": {
                    "min": min_length,
                    "max": max_length
                }
            }
            word_data["prediction_candidates"] = [
                {
                    "word": word,
                    "probability": prob,
                    "ranked_word_integration_probability": rank_prob
                }
                for word, prob, rank_prob in top_predictions
            ]
    
    print("Saving processed dataset...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print("Done! Dataset processed and saved with both integration difficulty and word prediction information.")

if __name__ == "__main__":
    # # Process the dataset
    # input_path = os.path.join(os.path.dirname(__file__), "assets", "raw_sentences_dataset.json")
    # output_path = os.path.join(os.path.dirname(__file__), "assets", "sentences_dataset_processed.json")
    # process_dataset(input_path, output_path) 

    scores = [0.30000000000000004,1.0,0.30000000000000004,1.0,0.30000000000000004,1.0,0.30000000000000004,1.0,0.30000000000000004,0.4,0.30000000000000004,1.0,0.30000000000000004,1.0,0.30000000000000004,1.0,0.6000000000000001,1.0]
    print(scores)
    print("--------------------------------")
    print(calc_dynamic_sentence_comprehension_score(scores, mode="softmin", tau=0.1))
    print(calc_dynamic_sentence_comprehension_score(scores, mode="geometric mean"))
    print(calc_dynamic_sentence_comprehension_score(scores, mode="harmonic mean"))
    print(calc_dynamic_sentence_comprehension_score(scores, mode="mean"))

    score_2 = [0.30000000000000004,1.0,0.30000000000000004,1.0,0.30000000000000004,1.0,0.30000000000000004,1.0,0.30000000000000004,0.4,0.30000000000000004,1.0,0.30000000000000004,1.0,0.30000000000000004,1.0]
    print("--------------------------------")
    print(calc_dynamic_sentence_comprehension_score(score_2, mode="softmin", tau=0.1))
    print(calc_dynamic_sentence_comprehension_score(score_2, mode="geometric mean"))
    print(calc_dynamic_sentence_comprehension_score(score_2, mode="harmonic mean"))
    print(calc_dynamic_sentence_comprehension_score(score_2, mode="mean"))

    score_3 = [0.30000000000000004,1.0,0.30000000000000004,1.0,0.30000000000000004,1.0,0.30000000000000004,1.0,0.30000000000000004,0.4,0.30000000000000004,1.0,0.30000000000000004,1.0,1.0,1.0]
    print("--------------------------------")
    print(calc_dynamic_sentence_comprehension_score(score_3, mode="softmin", tau=0.1))
    print(calc_dynamic_sentence_comprehension_score(score_3, mode="geometric mean"))
    print(calc_dynamic_sentence_comprehension_score(score_3, mode="harmonic mean"))
    print(calc_dynamic_sentence_comprehension_score(score_3, mode="mean"))