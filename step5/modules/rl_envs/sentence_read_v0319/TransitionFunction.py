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
        Compute how difficult it is to integrate a word into its context using surprisal.
        For BERT, we use both left and right context since it's a bidirectional model.
        Surprisal = -log P(word | context_left, context_right), converted to difficulty score.
        
        Args:
            context: List of words before the target word
            word: The target word to evaluate
            
        Returns:
            float: Integration difficulty score in [0,1]
            Higher values mean the word is harder to integrate with its context
        """
        # Find the current word's position in the full sentence
        current_word_idx = len(context)
        full_sentence = self.sentence_words  # Get the full sentence
        
        # Prepare input by replacing the target word with [MASK]
        masked_sentence = full_sentence.copy()
        masked_sentence[current_word_idx] = "[MASK]"
        masked_text = "[CLS] " + " ".join(masked_sentence) + " [SEP]"
        
        # Tokenize the masked input
        inputs = self.tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True)
        
        # Find the position of the [MASK] token
        mask_token_id = self.tokenizer.mask_token_id
        mask_position = (inputs["input_ids"][0] == mask_token_id).nonzero().item()
        
        # Get model predictions at mask position
        with torch.no_grad():
            outputs = self.masked_lm_model(**inputs)
            logits = outputs.logits[0, mask_position]  # Shape: [vocab_size]
            
        # Get the token ID for the actual word
        word_tokens = self.tokenizer(word, return_tensors="pt", padding=True, add_special_tokens=False)
        word_token_id = word_tokens["input_ids"][0, 0]  # Get first token ID
        
        # Compute probability of the actual word
        probs = torch.softmax(logits, dim=0)
        word_prob = probs[word_token_id]  # Keep as tensor
        
        # Compute surprisal: -log P(word | context)
        eps = 1e-10  # Small epsilon to avoid log(0)
        surprisal = -torch.log(word_prob + eps)
        
        # Convert surprisal to integration difficulty using sigmoid
        alpha = 0.5  # Scaling factor as suggested
        difficulty = torch.sigmoid(alpha * surprisal).item()
        
        # Add debug print
        print(f"\nDebug - Integration Difficulty Calculation:")
        print(f"Left context: {' '.join(context)}")
        print(f"Word: {word}")
        print(f"Right context: {' '.join(full_sentence[current_word_idx + 1:])}")
        print(f"Masked text: {masked_text}")
        print(f"Word probability: {word_prob.item():.4f}")
        print(f"Surprisal (-log prob): {surprisal.item():.4f}")
        print(f"Final difficulty (sigmoid): {difficulty:.4f}")
        
        # Get top 5 predictions for debugging
        top_5_probs, top_5_indices = torch.topk(probs, 5)
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

    def _compute_word_predictability(self, context: list[str], target_word: str, preview_letters: int = 3) -> tuple[float, list[tuple[str, float]]]:
        """
        Compute predictability of a word given its context and parafoveal preview information.
        
        Args:
            context: List of words before the target word
            target_word: The actual word to predict
            preview_letters: Number of letters visible in parafoveal preview (default 3)
            
        Returns:
            tuple[float, list[tuple[str, float]]]: (predictability of target word, list of top predictions with probabilities)
        """
        # Prepare input text with context
        input_text = "[CLS] " + " ".join(context) + " [SEP]"
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        
        # Get model predictions for next word
        with torch.no_grad():
            outputs = self.masked_lm_model(**inputs)
            logits = outputs.logits[0, -2]  # Take predictions at the last real token
            
        # Get probabilities for all words
        probs = torch.softmax(logits, dim=0)
        
        # Get preview information
        preview = target_word[:preview_letters].lower()
        target_length = len(target_word)
        length_tolerance = 2  # Allow words within ±2 length of target
        
        # Get top 100 predictions initially
        top_k = 100
        top_probs, top_indices = torch.topk(probs, top_k)
        candidates = [self.tokenizer.decode([idx]).strip() for idx in top_indices]
        
        # Filter candidates by preview and length
        filtered_predictions = []
        total_filtered_prob = 0.0
        
        for word, prob in zip(candidates, top_probs):
            word = word.lower().strip()
            # Check if word starts with preview and is within length tolerance
            if (word.startswith(preview) and 
                abs(len(word) - target_length) <= length_tolerance):
                filtered_predictions.append((word, prob.item()))
                total_filtered_prob += prob.item()
        
        # Sort filtered predictions by probability
        filtered_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get target word probability
        target_prob = 0.0
        for word, prob in filtered_predictions:
            if word.lower() == target_word.lower():
                target_prob = prob
                break
        
        # Normalize probabilities if we have any filtered predictions
        if filtered_predictions:
            filtered_predictions = [(w, p/total_filtered_prob) for w, p in filtered_predictions]
            target_prob = target_prob/total_filtered_prob if target_prob > 0 else 0.0
        
        return target_prob, filtered_predictions[:5]  # Return top 5 filtered predictions


if __name__ == "__main__":
    def test_integration_difficulty():
        """Test the integration difficulty calculation with different contexts"""
        transition_function = TransitionFunction()
        
        # Test cases with different contexts and full sentences
        test_cases = [
            {
                "sentence": ["The", "cat", "is", "sleeping", "on", "the", "mat"],
                "word_idx": 3,
                "description": "Simple verb 'sleeping' in normal context"
            },
            {
                "sentence": ["I", "went", "to", "the", "store", "to", "buy", "food"],
                "word_idx": 4,
                "description": "Common noun 'store' in normal context"
            },
            {
                "sentence": ["The", "book", "is", "on", "the", "table", "in", "the", "room"],
                "word_idx": 5,
                "description": "Common noun 'table' in normal context"
            },
            {
                "sentence": ["The", "book", "is", "on", "the", "understanding", "of", "physics"],
                "word_idx": 5,
                "description": "Incongruent word 'understanding' in spatial context"
            },
            {
                "sentence": ["She", "likes", "to", "read", "books", "in", "the", "library"],
                "word_idx": 3,
                "description": "Common verb 'read' in normal context"
            },
            {
                "sentence": ["She", "likes", "to", "read", "encyclopedia", "in", "the", "library"],
                "word_idx": 4,
                "description": "Experimental case of 'encyclopedia' in normal context"
            }
        ]
        
        print("\nTesting Integration Difficulty Calculation:")
        print("=" * 50)
        
        for case in test_cases:
            # Initialize the sentence for this test case
            transition_function.reset(case["sentence"])
            
            # Get context and word
            word_idx = case["word_idx"]
            context = case["sentence"][:word_idx]
            word = case["sentence"][word_idx]
            
            # Calculate difficulty
            difficulty = transition_function._compute_integration_difficulty(context, word)
            
            # Print results
            print(f"\nTest Case: {case['description']}")
            print(f"Full sentence: {' '.join(case['sentence'])}")
            print(f"Left context: {' '.join(context)}")
            print(f"Word: {word}")
            print(f"Right context: {' '.join(case['sentence'][word_idx + 1:])}")
            print(f"Integration Difficulty: {difficulty:.4f}")
            print("-" * 50)
    
    def test_word_predictability():
        """Test the word predictability calculation with different contexts"""
        transition_function = TransitionFunction()
        
        # Test cases
        test_cases = [
            {
                "sentence": ["The", "cat", "likes", "to", "sleep", "on", "the", "mat"],
                "word_idx": 2,
                "description": "Predicting 'likes' after 'The cat'"
            },
            {
                "sentence": ["She", "went", "to", "the", "store", "to", "buy", "food"],
                "word_idx": 4,
                "description": "Predicting 'store' after 'She went to the'"
            },
            {
                "sentence": ["The", "book", "is", "interesting", "and", "fun", "to", "read"],
                "word_idx": 3,
                "description": "Predicting 'interesting' after 'The book is'"
            },
            {
                "sentence": ["I", "like", "to", "read", "books", "in", "the", "library"],
                "word_idx": 3,
                "description": "Predicting 'read' after 'I like to'"
            }
        ]
        
        print("\nTesting Word Predictability Calculation:")
        print("=" * 70)
        
        for case in test_cases:
            # Initialize the sentence
            transition_function.reset(case["sentence"])
            
            # Get context and target word
            word_idx = case["word_idx"]
            context = case["sentence"][:word_idx]
            target_word = case["sentence"][word_idx]
            
            # Get preview letters (first 3)
            preview_letters = target_word[:3].lower()
            
            # Calculate predictability
            predictability, top_predictions = transition_function._compute_word_predictability(
                context, 
                target_word
            )
            
            # Print results
            print(f"\nTest Case: {case['description']}")
            print(f"Full sentence: {' '.join(case['sentence'])}")
            print(f"Context: {' '.join(context)}")
            print(f"Target word: {target_word}")
            print(f"Preview letters: {preview_letters}")
            print(f"Word length: {len(target_word)}")
            print(f"Predictability: {predictability:.4f}")
            print("\nTop 5 predictions (with preview filter):")
            for word, prob in top_predictions:
                print(f"  {word}: {prob:.4f}")
            print("-" * 70)
    
    # Run both tests
    print("\nRunning integration difficulty test...")
    test_integration_difficulty()
    
    print("\nRunning word predictability test...")
    test_word_predictability() 