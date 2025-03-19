import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from . import Constants


class TransitionFunction():
    """
    Transition Function that models human-like reading behavior using neural language models
    for tracking comprehension state and predicting integration difficulty.

    The overall architecture works like this:
    1. Language model converts words into embeddings
    2. GRU tracks comprehension as we read
    3. Uncertainty estimator tells us when we might need to regress or pay more attention
    """

    def __init__(self):
        # Load language model and tokenizer
        self.model_name = Constants.LANGUAGE_MODEL_NAME  # or any other suitable model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Use AutoModel for embeddings and AutoModelForMaskedLM for masked predictions
        self.language_model = AutoModel.from_pretrained(self.model_name)
        self.masked_lm_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        
        self.hidden_size = self.language_model.config.hidden_size  # Usually 768 for bert-base-uncased
        self.batch_size = 1 
        self.num_gru_layers = 2  # Number of GRU layers
        
        # Comprehension tracking parameters
        self._context_size = Constants.CONTEXT_SIZE  # NOTE: this is where the STM takes effect
        self._integration_threshold = 0.7
        
        # Neural components for comprehension tracking
        self.cumulative_comprehension_tracker = nn.GRU(
            input_size=self.hidden_size,    
            hidden_size=self.hidden_size,
            num_layers=self.num_gru_layers,   
            batch_first=True
        )
        
        self._at_that_moment_comprehension_for_each_word_log = {}

    def reset(self, sentence_words: list[str]) -> list[dict]:
        """Initialize comprehension state for new sentence"""
        self.sentence_words = sentence_words
        self.word_embeddings = self._get_word_embeddings(sentence_words)

        # Initialize comprehension log with None values for all possible indices
        self._at_that_moment_comprehension_for_each_word_log = {i: None for i in range(len(sentence_words))}
        
        # Initialize hidden state with correct shape: [num_layers, batch_size, hidden_size]
        self.cumulative_comprehension_state = torch.zeros(
            self.num_gru_layers, 
            self.batch_size, 
            self.hidden_size
        )
        
        # Store full comprehension potential for each word
        self.all_words_full_comprehension_states = [None] * len(sentence_words)
        self.word_states = [None] * len(sentence_words)
        
        # Calculate full comprehension potential for each word in sequence
        temp_comprehension_state = torch.zeros_like(self.cumulative_comprehension_state)
        
        for word_idx in range(len(sentence_words)):
            word_embedding = self.word_embeddings[:, word_idx]
            word_embedding_expanded = word_embedding.unsqueeze(1)
            
            # Process through GRU with accumulated context
            output, hidden = self.cumulative_comprehension_tracker(
                word_embedding_expanded,
                temp_comprehension_state
            )
            
            # Store this word's full comprehension (with context)
            self.all_words_full_comprehension_states[word_idx] = hidden.detach()
            
            # Update temporary state for next word's context
            temp_comprehension_state = hidden
        
        return self.word_states

    def _get_word_embeddings(self, words: list[str]) -> torch.Tensor:
        """Get contextual embeddings for each word by averaging over subword tokens"""
        encoding = self.tokenizer(words, return_tensors="pt", padding=True, return_offsets_mapping=True)
        
        with torch.no_grad():
            outputs = self.language_model(**{k: v for k, v in encoding.items() if k in ['input_ids', 'attention_mask']})
            
        hidden_states = outputs.last_hidden_state
        word_embeddings = torch.zeros(1, len(words), self.hidden_size)
        
        word_ids = encoding.word_ids(0)
        
        for word_idx in range(len(words)):
            token_positions = [i for i, wid in enumerate(word_ids) if wid == word_idx]
            if token_positions:
                word_embeddings[0, word_idx] = hidden_states[0, token_positions].mean(dim=0)
                
        return word_embeddings

    def _compute_integration_difficulty(self, context: list[str], word: str) -> float:
        """
        Compute how difficult it is to integrate a word into its context.
        Uses a masked language model to predict the word given its context.
        
        Args:
            context: List of words before the target word
            word: The target word to evaluate
            
        Returns:
            float: Integration difficulty score in [0,1]
            Higher values mean the word is harder to integrate with its context
        """
        # Prepare the input with [MASK] token and special tokens
        masked_text = "[CLS] " + " ".join(context) + " [MASK] [SEP]"
        
        # Tokenize the input
        inputs = self.tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True)
        
        # Find the position of the [MASK] token
        mask_token_id = self.tokenizer.mask_token_id
        mask_position = (inputs["input_ids"][0] == mask_token_id).nonzero().item()
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.masked_lm_model(**inputs)
            predictions = outputs.logits[0, mask_position]  # Get predictions for masked position
            
        # Get the token ID for the actual word
        word_tokens = self.tokenizer(word, return_tensors="pt", padding=True, add_special_tokens=False)
        word_token_id = word_tokens["input_ids"][0, 0]  # Get first token ID
        
        # Filter out special tokens and punctuation
        valid_token_ids = []
        for i in range(len(self.tokenizer)):
            token = self.tokenizer.decode([i])
            # Skip special tokens and single punctuation marks
            if (token not in self.tokenizer.all_special_tokens and 
                not (len(token) == 1 and token in '.,!?;:') and
                not token.startswith('##')):  # Skip subword tokens
                valid_token_ids.append(i)
        
        # Apply mask to filter out invalid tokens
        mask = torch.zeros_like(predictions)
        mask[valid_token_ids] = 1
        filtered_predictions = predictions * mask
        
        # Get the probability of the actual word
        word_prob = torch.softmax(filtered_predictions, dim=0)[word_token_id].item()
        
        # Convert probability to difficulty score
        # Higher probability = easier to integrate = lower difficulty
        difficulty = 1.0 - word_prob
        
        # Add debug print
        print(f"\nDebug - Integration Difficulty Calculation:")
        print(f"Context: {' '.join(context)}")
        print(f"Word: {word}")
        print(f"Masked text: {masked_text}")
        print(f"Mask position: {mask_position}")
        print(f"Word token ID: {word_token_id}")
        print(f"Word probability: {word_prob:.4f}")
        print(f"Final difficulty: {difficulty:.4f}")
        
        # Get top 5 predictions for debugging (filtered)
        top_5_probs, top_5_indices = torch.topk(torch.softmax(filtered_predictions, dim=0), 5)
        top_5_words = [self.tokenizer.decode([idx]) for idx in top_5_indices]
        print("\nTop 5 predicted words:")
        for word, prob in zip(top_5_words, top_5_probs):
            print(f"{word}: {prob:.4f}")
        
        return difficulty

    def update_state_read_next_word(self, states: list[dict], current_word_idx: int, sentence_length: int) -> tuple[list[dict], int, bool]:
        """Update comprehension state when reading next word."""
        if current_word_idx >= sentence_length - 1:
            return states, current_word_idx, False
            
        current_word_idx += 1
        
        # Get word embedding and full comprehension
        word_embedding = self.word_embeddings[:, current_word_idx].detach()
        full_comprehension = self.all_words_full_comprehension_states[current_word_idx]
        
        # Get context words for integration difficulty calculation
        context_words = self.sentence_words[max(0, current_word_idx - self._context_size):current_word_idx]
        current_word = self.sentence_words[current_word_idx]
        
        # Compute integration difficulty
        integration_difficulty = self._compute_integration_difficulty(context_words, current_word)
        
        # Apply integration difficulty to the word embedding before processing
        diminished_word_embedding = word_embedding * (1 - integration_difficulty)
        
        # Update cumulative comprehension state with diminished word
        diminished_word_expanded = diminished_word_embedding.unsqueeze(1)
        output, hidden = self.cumulative_comprehension_tracker(
            diminished_word_expanded,
            self.cumulative_comprehension_state
        )
        self.cumulative_comprehension_state = hidden
        
        # Store state with actual comprehension after processing
        states[current_word_idx] = {
            'embedding': word_embedding.detach(),
            'comprehension': hidden.detach(),
            'difficulty': torch.tensor(integration_difficulty)
        }

        # Store the actual comprehension for this word
        self._at_that_moment_comprehension_for_each_word_log[current_word_idx] = hidden.detach()
        
        return states, current_word_idx, True

    def update_state_skip_next_word(self, states: list[dict], current_word_idx: int, sentence_length: int, predictability: float) -> tuple[list[dict], int, bool]:
        """Update state when skipping word."""
        if current_word_idx >= sentence_length - 2:
            return states, current_word_idx, False
            
        skipped_word_idx = current_word_idx + 1
        next_word_idx = skipped_word_idx + 1
        
        # Handle skipped word
        actual_word_embedding = self.word_embeddings[:, skipped_word_idx]
        predicted_word_embedding = self._predict_skipped_word_embedding(skipped_word_idx, predictability)
        uncertainty = self._compute_uncertainty(actual_word_embedding, predicted_word_embedding)

        # Apply uncertainty to the word embedding
        diminished_word_embedding = predicted_word_embedding * (1 - uncertainty.item())

        # Update cumulative comprehension state with diminished word
        output, hidden = self.cumulative_comprehension_tracker(
            diminished_word_embedding.unsqueeze(1),
            self.cumulative_comprehension_state
        )
        self.cumulative_comprehension_state = hidden
        
        # Update skipped word state
        states[skipped_word_idx] = {
            'embedding': predicted_word_embedding.detach(),
            'comprehension': hidden.detach(),
            'difficulty': uncertainty.detach()
        }

        # Store the actual comprehension for this word
        self._at_that_moment_comprehension_for_each_word_log[skipped_word_idx] = hidden.detach()
        
        # Handle next word (similar to read_next_word)
        current_word_idx = next_word_idx
        word_embedding = self.word_embeddings[:, current_word_idx].detach()
        full_comprehension = self.all_words_full_comprehension_states[current_word_idx]
        
        # Get context words for integration difficulty calculation
        context_words = self.sentence_words[max(0, current_word_idx - self._context_size):current_word_idx]
        current_word = self.sentence_words[current_word_idx]
        
        # Compute integration difficulty
        integration_difficulty = self._compute_integration_difficulty(context_words, current_word)
        
        # Apply integration difficulty to the word embedding before processing
        diminished_word_embedding = word_embedding * (1 - integration_difficulty)
        
        # Update cumulative comprehension state with diminished word
        diminished_word_expanded = diminished_word_embedding.unsqueeze(1)
        output, hidden = self.cumulative_comprehension_tracker(
            diminished_word_expanded,
            self.cumulative_comprehension_state
        )
        self.cumulative_comprehension_state = hidden
        
        # Store state with actual comprehension after processing
        states[current_word_idx] = {
            'embedding': word_embedding.detach(),
            'comprehension': hidden.detach(),
            'difficulty': torch.tensor(integration_difficulty)
        }

        # Store the actual comprehension for this word
        self._at_that_moment_comprehension_for_each_word_log[current_word_idx] = hidden.detach()
        
        return states, current_word_idx, True

    def update_state_regress(self, states: list[dict], current_word_idx: int) -> tuple[list[dict], int, bool]:
        """Update state during regression, using full comprehension."""
        if current_word_idx <= 0:
            return states, current_word_idx, False
            
        regression_from_idx = current_word_idx
        current_word_idx -= 1
        
        # Use full comprehension for regressed word
        full_comprehension = self.all_words_full_comprehension_states[current_word_idx]
        
        # Get comprehension state from 2 words back if it exists, otherwise use zeros
        if current_word_idx >= 2 and self._at_that_moment_comprehension_for_each_word_log[current_word_idx - 2] is not None:
            self.cumulative_comprehension_state = self._at_that_moment_comprehension_for_each_word_log[current_word_idx - 2]
        else:
            self.cumulative_comprehension_state = torch.zeros_like(self.cumulative_comprehension_state)

        # Since this is a regression, the full word embedding is used
        word_embedding = self.word_embeddings[:, current_word_idx].detach()
        word_embedding_expanded = word_embedding.unsqueeze(1)
        output, hidden = self.cumulative_comprehension_tracker(
            word_embedding_expanded,
            self.cumulative_comprehension_state
        )
        self.cumulative_comprehension_state = hidden

        # Update state with full comprehension
        states[current_word_idx] = {
            'embedding': word_embedding,
            'comprehension': hidden.detach(),
            'difficulty': torch.tensor(0.0)  # No difficulty when regressing
        }

        # Store the actual comprehension for this word
        self._at_that_moment_comprehension_for_each_word_log[current_word_idx] = full_comprehension.detach()
        # Then clear all the lateral words' comprehension states
        for i in range(current_word_idx + 1, len(self.sentence_words)):
            if self._at_that_moment_comprehension_for_each_word_log[i] is not None:
                self._at_that_moment_comprehension_for_each_word_log[i] = None
        
        return states, current_word_idx, True

    def _predict_skipped_word_embedding(self, skipped_word_idx: int, predictability: float) -> torch.Tensor:
        """Predict embedding for skipped word based on context."""
        context_window = self.word_embeddings[:, max(0, skipped_word_idx-self._context_size):skipped_word_idx]
        
        # Get weights for context window according to distance from skipped word
        weights = torch.softmax(torch.arange(context_window.size(1), dtype=torch.float), dim=0)
        weights = weights.unsqueeze(0).unsqueeze(-1)
        
        # Predict the next word's embedding based on context distance weights
        predicted_embedding = (context_window * weights).sum(dim=1)
        return predicted_embedding

    def _compute_uncertainty(self, actual_state: torch.Tensor, predicted_state: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty by comparing predicted state with actual word embedding."""
        eps = 1e-8
        
        # Check for zero vectors
        actual_zero = torch.all(actual_state == 0)
        predicted_zero = torch.all(predicted_state == 0)
        
        if actual_zero or predicted_zero:
            return torch.ones_like(actual_state[:, :1])
            
        # Normalize vectors for cosine similarity with numerical stability
        actual_norm = actual_state / (actual_state.norm(dim=-1, keepdim=True) + eps)
        predicted_norm = predicted_state / (predicted_state.norm(dim=-1, keepdim=True) + eps)
        
        # Cosine similarity is in [-1, 1]
        similarity = torch.sum(actual_norm * predicted_norm, dim=-1)
        similarity = torch.clamp(similarity, min=-1.0, max=1.0)

        # Uncertainty conversion to [0, 1]
        uncertainty = (1 - similarity) / 2

        return uncertainty.unsqueeze(-1)


if __name__ == "__main__":
    def test_integration_difficulty():
        """Test the integration difficulty calculation with different contexts"""
        transition_function = TransitionFunction()
        
        # Test cases with different contexts
        test_cases = [
            {
                "context": ["The", "cat", "is"],
                "word": "sleeping",
                "description": "Simple verb after 'The cat is'"
            },
            {
                "context": ["I", "went", "to", "the"],
                "word": "store",
                "description": "Common noun after 'I went to the'"
            },
            {
                "context": ["The", "book", "is", "on", "the"],
                "word": "table",
                "description": "Common noun after 'The book is on the'"
            },
            {
                "context": ["She", "likes", "to"],
                "word": "read",
                "description": "Common verb after 'She likes to'"
            },
            {
                "context": ["The", "weather", "is"],
                "word": "nice",
                "description": "Common adjective after 'The weather is'"
            }
        ]
        
        print("\nTesting Integration Difficulty Calculation:")
        print("=" * 50)
        
        for case in test_cases:
            difficulty = transition_function._compute_integration_difficulty(case["context"], case["word"])
            print(f"\nTest Case: {case['description']}")
            print(f"Context: {' '.join(case['context'])}")
            print(f"Word: {case['word']}")
            print(f"Integration Difficulty: {difficulty:.4f}")
            print("-" * 50)
    
    # Run the test
    test_integration_difficulty() 