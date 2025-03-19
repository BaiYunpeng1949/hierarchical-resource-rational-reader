import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from . import Constants
import json
import os
from tqdm import tqdm


# class TransitionFunction():   # TODO change to offline calculation
#     """
#     Transition Function that models human-like reading behavior using neural language models
#     for tracking comprehension state and predicting integration difficulty.

#     The overall architecture works like this:
#     1. Language model converts words into embeddings
#     2. GRU tracks comprehension as we read
#     3. Uncertainty estimator tells us when we might need to regress or pay more attention
#     """

#     def __init__(self):
#         # Load language model and tokenizer
#         self.model_name = Constants.LANGUAGE_MODEL_NAME  # or any other suitable model
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
#         # Use AutoModel for embeddings and AutoModelForMaskedLM for masked predictions
#         self.language_model = AutoModel.from_pretrained(self.model_name)
#         self.masked_lm_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        
#         self.hidden_size = self.language_model.config.hidden_size  # Usually 768 for bert-base-uncased
#         self.batch_size = 1 
#         self.num_gru_layers = 2  # Number of GRU layers
        
#         # Comprehension tracking parameters
#         self._context_size = Constants.CONTEXT_SIZE  # NOTE: this is where the STM takes effect
#         self._integration_threshold = 0.7
        
#         # Neural components for comprehension tracking
#         self.cumulative_comprehension_tracker = nn.GRU(
#             input_size=self.hidden_size,    
#             hidden_size=self.hidden_size,
#             num_layers=self.num_gru_layers,   
#             batch_first=True
#         )
        
#         self._at_that_moment_comprehension_for_each_word_log = {}

#     def reset(self, sentence_words: list[str]) -> list[dict]:
#         """Initialize comprehension state for new sentence"""
#         self.sentence_words = sentence_words
#         self.word_embeddings = self._get_word_embeddings(sentence_words)

#         # Initialize comprehension log with None values for all possible indices
#         self._at_that_moment_comprehension_for_each_word_log = {i: None for i in range(len(sentence_words))}
        
#         # Initialize hidden state with correct shape: [num_layers, batch_size, hidden_size]
#         self.cumulative_comprehension_state = torch.zeros(
#             self.num_gru_layers, 
#             self.batch_size, 
#             self.hidden_size
#         )
        
#         # Store full comprehension potential for each word
#         self.all_words_full_comprehension_states = [None] * len(sentence_words)
#         self.word_states = [None] * len(sentence_words)
        
#         # Calculate full comprehension potential for each word in sequence
#         temp_comprehension_state = torch.zeros_like(self.cumulative_comprehension_state)
        
#         for word_idx in range(len(sentence_words)):
#             word_embedding = self.word_embeddings[:, word_idx]
#             word_embedding_expanded = word_embedding.unsqueeze(1)
            
#             # Process through GRU with accumulated context
#             output, hidden = self.cumulative_comprehension_tracker(
#                 word_embedding_expanded,
#                 temp_comprehension_state
#             )
            
#             # Store this word's full comprehension (with context)
#             self.all_words_full_comprehension_states[word_idx] = hidden.detach()
            
#             # Update temporary state for next word's context
#             temp_comprehension_state = hidden
        
#         return self.word_states

#     def _get_word_embeddings(self, words: list[str]) -> torch.Tensor:
#         """Get contextual embeddings for each word by averaging over subword tokens"""
#         encoding = self.tokenizer(words, return_tensors="pt", padding=True, return_offsets_mapping=True)
        
#         with torch.no_grad():
#             outputs = self.language_model(**{k: v for k, v in encoding.items() if k in ['input_ids', 'attention_mask']})
            
#         hidden_states = outputs.last_hidden_state
#         word_embeddings = torch.zeros(1, len(words), self.hidden_size)
        
#         word_ids = encoding.word_ids(0)
        
#         for word_idx in range(len(words)):
#             token_positions = [i for i, wid in enumerate(word_ids) if wid == word_idx]
#             if token_positions:
#                 word_embeddings[0, word_idx] = hidden_states[0, token_positions].mean(dim=0)
                
#         return word_embeddings

#     def _compute_integration_difficulty(self, context: list[str], word: str) -> float:
#         """
#         Compute how difficult it is to integrate a word into its context using surprisal.
#         For BERT, we use both left and right context since it's a bidirectional model.
#         Surprisal = -log P(word | context_left, context_right), converted to difficulty score.
        
#         Args:
#             context: List of words before the target word
#             word: The target word to evaluate
            
#         Returns:
#             float: Integration difficulty score in [0,1]
#             Higher values mean the word is harder to integrate with its context
#         """
#         # Find the current word's position in the full sentence
#         current_word_idx = len(context)
#         full_sentence = self.sentence_words  # Get the full sentence
        
#         # Prepare input by replacing the target word with [MASK]
#         masked_sentence = full_sentence.copy()
#         masked_sentence[current_word_idx] = "[MASK]"
#         masked_text = "[CLS] " + " ".join(masked_sentence) + " [SEP]"
        
#         # Tokenize the masked input
#         inputs = self.tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True)
        
#         # Find the position of the [MASK] token
#         mask_token_id = self.tokenizer.mask_token_id
#         mask_position = (inputs["input_ids"][0] == mask_token_id).nonzero().item()
        
#         # Get model predictions at mask position
#         with torch.no_grad():
#             outputs = self.masked_lm_model(**inputs)
#             logits = outputs.logits[0, mask_position]  # Shape: [vocab_size]
            
#         # Get the token ID for the actual word
#         word_tokens = self.tokenizer(word, return_tensors="pt", padding=True, add_special_tokens=False)
#         word_token_id = word_tokens["input_ids"][0, 0]  # Get first token ID
        
#         # Compute probability of the actual word
#         probs = torch.softmax(logits, dim=0)
#         word_prob = probs[word_token_id]  # Keep as tensor
        
#         # Compute surprisal: -log P(word | context)
#         eps = 1e-10  # Small epsilon to avoid log(0)
#         surprisal = -torch.log(word_prob + eps)
        
#         # Convert surprisal to integration difficulty using sigmoid
#         alpha = 0.5  # Scaling factor as suggested
#         difficulty = torch.sigmoid(alpha * surprisal).item()
        
#         # Add debug print
#         print(f"\nDebug - Integration Difficulty Calculation:")
#         print(f"Left context: {' '.join(context)}")
#         print(f"Word: {word}")
#         print(f"Right context: {' '.join(full_sentence[current_word_idx + 1:])}")
#         print(f"Masked text: {masked_text}")
#         print(f"Word probability: {word_prob.item():.4f}")
#         print(f"Surprisal (-log prob): {surprisal.item():.4f}")
#         print(f"Final difficulty (sigmoid): {difficulty:.4f}")
        
#         # Get top 5 predictions for debugging
#         top_5_probs, top_5_indices = torch.topk(probs, 5)
#         top_5_words = [self.tokenizer.decode([idx]) for idx in top_5_indices]
#         print("\nTop 5 predicted words:")
#         for word, prob in zip(top_5_words, top_5_probs):
#             print(f"{word}: {prob:.4f}")
        
#         return difficulty

def compute_integration_difficulty(tokenizer, model, context: list[str], word: str, full_sentence: list[str], current_word_idx: int) -> tuple[float, float]:
    """
    Compute integration difficulty for a word given its context using surprisal.
    Returns both surprisal and normalized difficulty score.
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
    
    # Compute probability of the actual word
    probs = torch.softmax(logits, dim=0)
    word_prob = probs[word_token_id]  # Keep as tensor
    
    # Compute surprisal: -log P(word | context)
    eps = 1e-10  # Small epsilon to avoid log(0)
    surprisal = -torch.log(word_prob + eps)
    
    # Convert surprisal to integration difficulty using sigmoid
    alpha = 0.5  # Scaling factor
    difficulty = torch.sigmoid(alpha * surprisal).item()
    
    return surprisal.item(), difficulty

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
            surprisal, difficulty = compute_integration_difficulty(
                tokenizer, 
                model, 
                context, 
                word_data["word"],
                sentence,
                i
            )
            
            # Add scores to word data
            word_data["surprisal"] = surprisal
            word_data["integration_difficulty"] = difficulty
    
    print("Saving processed dataset...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print("Done! Dataset processed and saved with integration difficulty scores.")

def compute_word_prediction(tokenizer, model, context: list[str], target_word: str, preview_letters: int = 2) -> tuple[str, float, list[tuple[str, float]]]:
    """
    Compute word prediction given context and preview information.
    Returns the predicted word, its probability, and top candidates.
    
    Args:
        tokenizer: BERT tokenizer
        model: BERT masked language model
        context: List of words before target word
        target_word: The actual word to predict
        preview_letters: Number of clear letters visible in preview (default 2)
        
    Returns:
        tuple[str, float, list[tuple[str, float]]]: (predicted_word, probability, top_candidates)
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
            filtered_predictions.append((word, prob.item()))
            total_filtered_prob += prob.item()
    
    # Sort filtered predictions by probability
    filtered_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Normalize probabilities
    if filtered_predictions:
        # Normalize probabilities for sampling
        filtered_predictions = [(w, p/total_filtered_prob) for w, p in filtered_predictions]
        
        # Get top 5 predictions
        top_predictions = filtered_predictions[:5]
        
        # Get most likely word and its probability
        predicted_word, predictability = top_predictions[0]
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
            
            # Compute word prediction
            predicted_word, predictability, top_predictions = compute_word_prediction(
                tokenizer,
                model,
                context,
                target_word,
                preview_letters
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
                {"word": word, "probability": prob}
                for word, prob in top_predictions
            ]
    
    print("Saving processed dataset...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print("Done! Dataset processed and saved with word prediction information.")

if __name__ == "__main__":
    # Process the dataset
    input_path = os.path.join(os.path.dirname(__file__), "assets", "sentences_dataset_with_integration.json")
    output_path = os.path.join(os.path.dirname(__file__), "assets", "sentences_dataset_with_predictions.json")
    process_dataset_with_predictions(input_path, output_path) 