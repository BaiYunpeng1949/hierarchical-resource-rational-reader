import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
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
        self.language_model = AutoModel.from_pretrained(self.model_name)
        self.hidden_size = self.language_model.config.hidden_size  # Usually 768 for bert-base-uncased
        self.batch_size = 1 
        self.num_gru_layers = 2  # Number of GRU layers
        
        # Comprehension tracking parameters
        self._context_size = Constants.CONTEXT_SIZE  # NOTE: this is where the STM takes effect, but might just use prepositions instead of simple words, do this later when necessary.
        self._integration_threshold = 0.7
        
        # Neural components for comprehension tracking
        # Input: each word's embedding
        # Hidden state: maintains cumulative understanding of all words read so far
        # Each layer captures different levels of meaning:
        # - Layer 1: local word relationships and immediate context
        # - Layer 2: higher-level sentence meaning and global context
        # Objective: to track the cumulative understanding of the sentence as we read, 
        #   Represents deep understanding of sentence structure. Maintains memory of how words relate to each other
        self.cumulative_comprehension_tracker = nn.GRU(
            input_size=self.hidden_size,    
            hidden_size=self.hidden_size,
            num_layers=self.num_gru_layers,   
            batch_first=True
        )
        
        self._at_that_moment_comprehension_for_each_word_log: Dict[int, torch.Tensor] = {}
        # Comment out the neural network approach
        """
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        """

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
        # NOTE definition: full_comprehension_states[i] is the full comprehension potential of the i-th word
        #   which is the cumulative understanding of the sentence considering all previous words
        self.all_words_full_comprehension_states = [None] * len(sentence_words)
        self.word_states = [None] * len(sentence_words)
        
        # Calculate full comprehension potential for each word in sequence
        # This represents each word's contextual understanding considering all previous words
        temp_comprehension_state = torch.zeros_like(self.cumulative_comprehension_state)
        
        for word_idx in range(len(sentence_words)):
            word_embedding = self.word_embeddings[:, word_idx]
            word_embedding_expanded = word_embedding.unsqueeze(1)
            
            # Process through GRU with accumulated context
            output, hidden = self.cumulative_comprehension_tracker(
                word_embedding_expanded,
                temp_comprehension_state  # Use accumulated state for context
            )
            
            # Store this word's full comprehension (with context)
            self.all_words_full_comprehension_states[word_idx] = hidden.detach()
            
            # Update temporary state for next word's context
            temp_comprehension_state = hidden
        
        return self.word_states

    def _get_word_embeddings(self, words: list[str]) -> torch.Tensor:
        """Get contextual embeddings for each word by averaging over subword tokens"""
        # Tokenize with offset mapping to track which tokens belong to which word
        encoding = self.tokenizer(words, return_tensors="pt", padding=True, return_offsets_mapping=True)
        
        with torch.no_grad():
            outputs = self.language_model(**{k: v for k, v in encoding.items() if k in ['input_ids', 'attention_mask']})
            
        # Get the hidden states
        hidden_states = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]
        
        # Initialize tensor to store word-level embeddings
        word_embeddings = torch.zeros(1, len(words), self.hidden_size)
        
        # Get word IDs using the encoding object
        word_ids = encoding.word_ids(0)  # 0 is the batch index
        
        # Average embeddings for subwords belonging to the same word
        for word_idx in range(len(words)):
            # Find all token positions for this word
            token_positions = [i for i, wid in enumerate(word_ids) if wid == word_idx]
            if token_positions:
                # Average embeddings of subwords
                word_embeddings[0, word_idx] = hidden_states[0, token_positions].mean(dim=0)
                
        return word_embeddings
        
    def update_state_read_next_word(self, states: list[dict], current_word_idx: int, sentence_length: int) -> tuple[list[dict], int, bool]:
        """Update comprehension state when reading next word."""
        if current_word_idx >= sentence_length - 1:
            return states, current_word_idx, False
            
        current_word_idx += 1
        
        # Get word embedding and full comprehension
        word_embedding = self.word_embeddings[:, current_word_idx].detach()
        full_comprehension = self.all_words_full_comprehension_states[current_word_idx]
        
        # Get context state (comprehension without current word)
        context_state = self._get_context_state(end_word_idx=current_word_idx)
        
        # Compute integration difficulty
        integration_difficulty = self._compute_integration_difficulty(
            full_comprehension[-1].unsqueeze(0),
            context_state.unsqueeze(1)
        )
        
        # Apply integration difficulty to the word embedding before processing
        # This means difficult words have less impact on the comprehension state
        # Ensure integration_difficulty is a scalar
        difficulty_scalar = float(integration_difficulty.item())
        diminished_word_embedding = word_embedding * (1 - difficulty_scalar)
        
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
            'comprehension': hidden.detach(),  # Store actual comprehension after processing
            'difficulty': torch.tensor(difficulty_scalar)
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

        # TODO debug delete later
        print(f"The uncertainty is: {uncertainty.item()}")

        # Apply uncertainty to the word embedding
        diminished_word_embedding = predicted_word_embedding * (1 - uncertainty.item())

        # TODO: debug delete later
        print(f"The diminished word embedding for word skipping is: {diminished_word_embedding}")

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
        
        ######################################################################
        # Handle next word (similar to read_next_word)
        current_word_idx = next_word_idx
        word_embedding = self.word_embeddings[:, current_word_idx].detach()
        full_comprehension = self.all_words_full_comprehension_states[current_word_idx]
        
        # Compute integration difficulty
        context_state = self._get_context_state(end_word_idx=current_word_idx)
        integration_difficulty = self._compute_integration_difficulty(
            full_comprehension[-1].unsqueeze(0),
            context_state.unsqueeze(1)
        )

        # TODO debug delete later
        print(f"The integration difficulty is: {integration_difficulty.item()}")
        
        # Apply integration difficulty to the word embedding before processing
        # This means difficult words have less impact on the comprehension state
        diminished_word_embedding = word_embedding * (1 - integration_difficulty.item())
        
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
            'comprehension': hidden.detach(),  # Store actual comprehension after processing
            'difficulty': integration_difficulty.detach()
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
        
    def _get_context_state(self, end_word_idx: int) -> torch.Tensor:
        """
        Get contextual state from previous words (mean of previous words embeddings)
        end_word_idx: the index of the last word to include in the context
        Raw word embeddings averaged together, Represents surface-level context from nearby words, 
        No processing through GRU/comprehension
        NOTE: need to verify whether the integration needs comprehension here later.
        """
        context_start_word_idx = max(0, end_word_idx - self._context_size)
        
        # Handle empty context case (first word)
        if end_word_idx == 0:
            return torch.zeros_like(self.word_embeddings[:, 0])
            
        context_embeddings = self.word_embeddings[:, context_start_word_idx:end_word_idx]
        mean_context_embedding = context_embeddings.mean(dim=1)

        return mean_context_embedding
        
    def _compute_integration_difficulty(self, current: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Compute difficulty of integrating new information by measuring how much the word changes our understanding
        
        Args:
            current: The current word's comprehension state (with this word)
            context: The context state (without this word)
            
        Returns:
            Normalized difficulty score in [0,1] where:
            0 = easy to integrate (word fits naturally with context)
            1 = hard to integrate (word significantly changes understanding)
        """
        # For first word or empty context, return medium difficulty
        if torch.all(context == 0):
            return torch.tensor([[0.5]])
            
        # Compute how much the word changes our understanding
        # Larger change = harder to integrate
        raw_distance = torch.norm(current - context, dim=-1)
        
        # Normalize using sigmoid with scaling factor
        # The scaling factor (0.5) controls how sensitive the difficulty is to change
        # Larger values make it more sensitive to small changes
        # Smaller values make it less sensitive to small changes
        normalized_difficulty = torch.sigmoid(raw_distance * 0.5)
        
        return normalized_difficulty.unsqueeze(-1)  # Ensure correct shape
        
    def _predict_skipped_word_embedding(self, skipped_word_idx: int, predictability: float) -> torch.Tensor:
        """
        TODO: Enhanced word prediction with parafoveal preview -> realize this later, remove the predictability
        Future improvements:
        1. Parafoveal Information:
            - First n letters of skipped word
            - Estimated word length
            - Word shape (ascenders/descenders)
        
        2. Implementation Plan:
            a) Create preview representation:
               - Encode available letters using BERT subword tokens
               - Encode length as positional feature
               - Encode word shape features
            
            b) Enhanced prediction:
               - Combine context prediction with preview info
               - Weight by preview clarity/confidence
               - Consider frequency effects for words matching preview
        
        Example:
        If skipping "house" with preview "ho__e" (length=5):
        - Context suggests: home, horse, house
        - Preview "ho__e" filters candidates
        - Length=5 further constrains
        - Frequency effects favor "house"
        """
        # Current implementation (context-based only)
        context_window = self.word_embeddings[:, max(0, skipped_word_idx-self._context_size):skipped_word_idx]
        
        # Get weights for context window according to distance from skipped word
        weights = torch.softmax(torch.arange(context_window.size(1), dtype=torch.float), dim=0)
        weights = weights.unsqueeze(0).unsqueeze(-1)    # Reshape to [1, context_size, 1]
        
        # Predict the next word's embedding only based on context distance weights
        predicted_embedding = (context_window * weights).sum(dim=1)
        return predicted_embedding
        
    def _apply_memory_decay(self, states: list[dict], current_idx: int):
        """Apply decay to previous states based on distance"""
        # for i in range(current_idx):
        #     if states[i] is not None:
        #         distance = current_idx - i
        #         decay = 0.9 ** distance
        #         states[i]['comprehension'] *= decay
        #         states[i]['difficulty'] = min(1.0, states[i]['difficulty'] + (1 - decay))
        pass    # NOTE: do not decay memory first, see whether needed for realisitic regressions.

    def _compute_uncertainty(self, actual_state: torch.Tensor, predicted_state: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty by comparing predicted state with actual word embedding.
        This better reflects prediction accuracy than comparing with context.
        
        Args:
            actual_state: The actual word embedding we're trying to predict
            predicted_state: Our prediction based on context
            
        Returns:
            Uncertainty score in [0,1] where:
            0 = perfect prediction
            1 = completely different from actual
        """
        # Add small epsilon for numerical stability
        eps = 1e-8
        
        # Check for zero vectors
        actual_zero = torch.all(actual_state == 0)
        predicted_zero = torch.all(predicted_state == 0)
        
        if actual_zero or predicted_zero:
            # If either vector is zero, return maximum uncertainty
            return torch.ones_like(actual_state[:, :1])
            
        # Normalize vectors for cosine similarity with numerical stability
        actual_norm = actual_state / (actual_state.norm(dim=-1, keepdim=True) + eps)
        predicted_norm = predicted_state / (predicted_state.norm(dim=-1, keepdim=True) + eps)
        
        # Cosine similarity is in [-1, 1]
        similarity = torch.sum(actual_norm * predicted_norm, dim=-1)
        similarity = torch.clamp(similarity, min=-1.0, max=1.0)

        # Uncertainty conversion to [0, 1]
        uncertainty = (1 - similarity) / 2

        return uncertainty.unsqueeze(-1)  # Match original shape