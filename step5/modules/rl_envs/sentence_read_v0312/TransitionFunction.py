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
        self.comprehension_tracker = nn.GRU(
            input_size=self.hidden_size,    
            hidden_size=self.hidden_size,
            num_layers=self.num_gru_layers,   
            batch_first=True
        )
        
        # Comment out the neural network approach
        """
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        """

    def reset(self, sentence_words: list[str]):
        """Initialize comprehension state for new sentence"""
        self.sentence_words = sentence_words
        self.word_embeddings = self._get_word_embeddings(sentence_words)

        # TODO debug delete later
        print(f"------------- Reset -------------------")
        print(f"Sentence words: {self.sentence_words}")
        print(f"The length of the sentence words is: {len(self.sentence_words)}")
        print(f"Word embeddings shape: {self.word_embeddings.shape}")
        
        # Initialize hidden state with correct shape: [num_layers, batch_size, hidden_size]
        self.cumulative_comprehension_state = torch.zeros(
            self.num_gru_layers, 
            self.batch_size, 
            self.hidden_size
        )
        self.word_states = [None] * len(sentence_words)
        
        # Get contextual embeddings with proper shape handling
        encoding = self.tokenizer(
            sentence_words,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=True
        )
        
        with torch.no_grad():
            outputs = self.language_model(**{k: v for k, v in encoding.items() if k in ['input_ids', 'attention_mask']})
        
        # Get word-level embeddings (similar to _get_word_embeddings)
        hidden_states = outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]
        word_ids = encoding.word_ids(0)  # 0 is the batch index
        
        # Initialize tensor for context embeddings
        self.sentence_context_embeddings = torch.zeros(1, len(sentence_words), self.hidden_size)
        
        # Average embeddings for subwords belonging to the same word
        for word_idx in range(len(sentence_words)):
            token_positions = [i for i, wid in enumerate(word_ids) if wid == word_idx]
            if token_positions:
                self.sentence_context_embeddings[0, word_idx] = hidden_states[0, token_positions].mean(dim=0)
        
        print(f"Sentence context embeddings shape: {self.sentence_context_embeddings.shape}")
        # Should now be [1, sentence_length, hidden_size]
        
        return self.word_states
    
    def _get_word_embeddings(self, words):
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
        
    def update_state_read_next_word(self, states, current_word_idx, sentence_length):
        """
        Update comprehension state when reading next word.
        Uses neural representations to update our cumulative understanding of the sentence.
        """
        if current_word_idx >= sentence_length - 1:
            return states, current_word_idx, False
            
        current_word_idx += 1

        # TODO debug delete later
        # print(f"--------------- Update state: read next word -----------------")
        # print(f"Current word idx: {current_word_idx}, sentence length: {sentence_length}")
        
        # Get word embedding and update comprehension state
        word_embedding = self.word_embeddings[:, current_word_idx].detach()  # shape: [batch_size, hidden_size]
        # detach() is used to ensure that the word_embedding is not part of the computation graph, 
        # so that it is not updated during backpropagation
        # print(f"Word embedding shape: {word_embedding.shape}")
        
        # Add sequence length dimension for GRU input
        word_embedding_expanded = word_embedding.unsqueeze(1)  # shape: [batch_size, 1, hidden_size]
        # print(f"After unsqueeze shape: {word_embedding_expanded.shape}")
        
        # Update cumulative comprehension with new word
        output, hidden = self.comprehension_tracker(
            word_embedding_expanded,  # shape: [batch_size, 1, hidden_size]
            self.cumulative_comprehension_state  # shape: [num_layers, batch_size, hidden_size]
        )
        self.cumulative_comprehension_state = hidden  # Use hidden state for cumulative comprehension
        
        # Estimate integration difficulty
        context_state = self._get_context_state(end_word_idx=current_word_idx)
        # print(f"Context state shape: {context_state.shape}")
        # print(f"The shape of the cumulative comprehension state: {self.cumulative_comprehension_state.shape}")
        # print(f"The shape of the last layer's hidden state: {hidden[-1].shape}")
        integration_difficulty = self._compute_integration_difficulty(
            hidden[-1].unsqueeze(0),  # Use last layer's (global, higher-level comprehension) hidden state, shape: [1, 1, hidden_size]
            context_state.unsqueeze(1)  # Add sequence dimension, shape: [1, 1, hidden_size]
        )
        # print(f"Integration difficulty shape: {integration_difficulty.shape}")
        # print(f"The integration difficulty: {integration_difficulty}")
        
        # Update word state based on integration success
        states[current_word_idx] = {
            'embedding': word_embedding.detach(),
            'comprehension': hidden.detach(),  # Store detached hidden state
            'difficulty': integration_difficulty.detach()
        }
        
        # Decay previous states based on working memory limitations
        self._apply_memory_decay(states, current_word_idx)
        
        return states, current_word_idx, True
        
    def update_state_skip_next_word(self, states, current_word_idx, sentence_length, predictability):
        """
        Update state when skipping word, using predictability and context
        to estimate comprehension loss.
        Updates both:
        1. The skipped word state (current_word_idx + 1)
        2. The word being read (current_word_idx + 2)
        """
        if current_word_idx >= sentence_length - 2:  # Need space for both skip and read
            return states, current_word_idx, False
            
        # First handle the skipped word (current_word_idx + 1)
        skipped_word_idx = current_word_idx + 1
        
        # Estimate skipped word comprehension from context
        context_state = self._get_context_state(end_word_idx=skipped_word_idx)
        predicted_state = self._predict_skipped_meaning(
            skipped_word_idx=skipped_word_idx, 
            predictability=predictability
        )
        
        # Higher uncertainty for skipped words
        uncertainty = self._compute_uncertainty(context_state, predicted_state)
        
        # Process predicted state through GRU to maintain consistent comprehension tracking
        predicted_expanded = predicted_state.unsqueeze(1)  # Add sequence dimension [batch_size, 1, hidden_size]
        
        # Update comprehension with predicted state, weighted by uncertainty
        output, hidden = self.comprehension_tracker(
            predicted_expanded * (1 - uncertainty.squeeze()),  # Reduce input based on uncertainty
            self.cumulative_comprehension_state
        )
        self.cumulative_comprehension_state = hidden  # Update cumulative state with new hidden state
        
        # Update skipped word state
        states[skipped_word_idx] = {
            'embedding': predicted_state.detach(),
            'comprehension': self.cumulative_comprehension_state.detach(),  # Use GRU state instead of manual stacking
            'difficulty': uncertainty.detach()
        }

        # Now handle the word being read (current_word_idx + 2)
        current_word_idx += 1
        states, current_word_idx, success = self.update_state_read_next_word(states, current_word_idx, sentence_length)
        
        return states, current_word_idx, True
        
    def update_state_regress(self, states, current_word_idx):
        """
        Update state during regression, attempting to reduce uncertainty.
        When regressing, we:
        1. Include context up to where we regressed from
        2. Reprocess the word with both backward and forward context
        3. Strengthen its representation in the cumulative comprehension
        """
        if current_word_idx <= 0:
            return states, current_word_idx, False
            
        # Store the index we're regressing from for context
        regression_from_idx = current_word_idx
        current_word_idx -= 1
        
        # Re-read word with additional context up to where we regressed from
        word_embedding = self.word_embeddings[:, current_word_idx].detach()
        
        # Get context including words up to AND INCLUDING where we regressed from
        context_embeddings = self.word_embeddings[:, :regression_from_idx + 1]  # +1 to include the word we regressed from
        context_state = context_embeddings.mean(dim=1)  # Average all context including regression point
        
        # Add sequence dimension for GRU input
        word_embedding_expanded = word_embedding.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Strengthen the word's representation by combining with context
        strengthened_embedding = (word_embedding + context_state) / 2  # Average with context, do this for strengthening because GRU cannot handle bi-directional now.
        strengthened_embedding = strengthened_embedding.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Update comprehension using strengthened representation
        output, hidden = self.comprehension_tracker(
            strengthened_embedding,  # Use strengthened embedding
            self.cumulative_comprehension_state
        )
        
        # Update both local and cumulative states with strengthened comprehension
        self.cumulative_comprehension_state = hidden
        states[current_word_idx]['comprehension'] = hidden.detach()
        
        # Compute new integration difficulty with full context
        integration_difficulty = self._compute_integration_difficulty(
            hidden[-1].unsqueeze(0),  # Use updated comprehension
            context_state.unsqueeze(1)  # Compare against full context
        )
        states[current_word_idx]['difficulty'] = integration_difficulty.detach()
        
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
        context_embeddings = self.word_embeddings[:, context_start_word_idx:end_word_idx]
        mean_context_embedding = context_embeddings.mean(dim=1)

        # TODO debug delete later
        print(f"-------------- Get context state ------------------")
        print(f"Context embeddings shape: {context_embeddings.shape}")
        print(f"Mean context embedding shape: {mean_context_embedding.shape}")
        print(f"--------------------------------")
        return mean_context_embedding
        
    def _compute_integration_difficulty(self, current: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Compute difficulty of integrating new information
        current: The current word's comprehension state
        context: The context from previous words
        torch.norm(current - context, dim=-1): Calculates the Euclidean distance between current and context
        
        dim=-1 means to compute along the last dimension of the tensors.
        For example, if current and context are embeddings with shape [batch_size, embedding_dim],
        dim=-1 will compute the norm across the embedding_dim dimension.
        This is equivalent to dim=1 in this case, but using -1 makes it work regardless of the number of dimensions.

        norm gives a measure of the semanticdifference between the current and context embeddings.
        If the norm is large, it indicates that the current word is difficult to integrate with its local context.

        NOTE: By comparing the raw context against the processed comprehension, 
            we can detect when a word is unexpected or difficult to integrate with its local context;
            while cumulative comprehension (procesed by the GRU) captures processed, integrated understanding
        
        NOTE: we are comparing the global, higher-level comprehension against the local, surface-level context.
        The current approach might be more cognitively plausible because:
            1. Processing Stage Comparison:
                We're comparing how the sentence meaning AFTER processing the new word (through GRU) differs from what we expected based on previous context
                This aligns with theories of surprisal and prediction error in reading comprehension
                It's not just about word similarity, but about how the word changes our understanding
            2. Cognitive Processing Evidence:
                Reading research suggests that integration difficulty isn't just about raw word similarity
                It's about how well the new information fits into our existing mental model
                Example: "The man bit the dog" - words are semantically related but hard to integrate into typical understanding
            3. Predictive Processing:
                The brain constantly makes predictions based on context
                Integration difficulty is measured by how much our understanding needs to change after seeing the new word
                This is captured by comparing processed state vs. raw context
        """
        integration_difficulty = torch.norm(current - context, dim=-1)
        return integration_difficulty
        
    def _predict_skipped_meaning(self, skipped_word_idx: int, predictability: float) -> torch.Tensor:
        """
        TODO: Enhanced word prediction with parafoveal preview
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
        context_window = self.sentence_context_embeddings[:, max(0, skipped_word_idx-self._context_size):skipped_word_idx]
        
        # Get weights for context window according to distance from skipped word
        weights = torch.softmax(torch.arange(context_window.size(1), dtype=torch.float), dim=0)
        weights = weights.unsqueeze(0).unsqueeze(-1)    # Reshape to [1, context_size, 1]
        
        # Predict the next word's embedding only based on context distance weights
        predicted_embedding = (context_window * weights).sum(dim=1)
        return predicted_embedding * predictability
        
    def _apply_memory_decay(self, states: list[dict], current_idx: int):
        """Apply decay to previous states based on distance"""
        # for i in range(current_idx):
        #     if states[i] is not None:
        #         distance = current_idx - i
        #         decay = 0.9 ** distance
        #         states[i]['comprehension'] *= decay
        #         states[i]['difficulty'] = min(1.0, states[i]['difficulty'] + (1 - decay))
        pass    # NOTE: do not decay memory first, see whether needed for realisitic regressions.

    def _compute_uncertainty(self, context_state: torch.Tensor, predicted_state: torch.Tensor) -> torch.Tensor:
        """
        Simple uncertainty estimation based on cosine similarity
        between predicted state and actual word embedding.
        Handles edge cases:
        - Zero vectors: returns maximum uncertainty (1.0)
        - Numerical stability: adds small epsilon to norm
        """
        # Add small epsilon for numerical stability
        eps = 1e-8
        
        # Check for zero vectors
        context_zero = torch.all(context_state == 0)
        predicted_zero = torch.all(predicted_state == 0)
        
        if context_zero or predicted_zero:
            # If either vector is zero, return maximum uncertainty
            return torch.ones_like(context_state[:, :1])
            
        # Normalize vectors for cosine similarity with numerical stability
        context_norm = context_state / (context_state.norm(dim=-1, keepdim=True) + eps)
        predicted_norm = predicted_state / (predicted_state.norm(dim=-1, keepdim=True) + eps)
        
        # Cosine similarity is in [-1, 1]
        similarity = torch.sum(context_norm * predicted_norm, dim=-1)
        similarity = torch.clamp(similarity, min=-1.0, max=1.0)

        # Uncertainty conversion to [0, 1]
        uncertainty = (1 - similarity) / 2

        # Used later as confidence weight
        predicted_state * (1 - uncertainty.squeeze())

        return uncertainty.unsqueeze(-1)  # Match original shape