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

if __name__ == "__main__":
    # Process the dataset
    input_path = os.path.join(os.path.dirname(__file__), "assets", "sentences_dataset.json")
    output_path = os.path.join(os.path.dirname(__file__), "assets", "sentences_dataset_with_integration.json")
    process_dataset_with_integration_difficulty(input_path, output_path) 