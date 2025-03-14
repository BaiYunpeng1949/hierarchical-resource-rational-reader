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
        
        # Comprehension tracking parameters
        self._context_size = Constants.CONTEXT_SIZE
        self._integration_threshold = 0.7
        
        # Neural components for comprehension tracking
        # Input: each word's embedding; 
        # Updates its hidden state based on both the new word and previous context
        # Helps track how well we understand the sentence as we read
        # The 2 layers allow it to capture both local and global sentence structure
        # Output: the comprehension state of the word
        self.comprehension_tracker = nn.GRU(# GRU: Gated Recurrent Unit, a type of RNN, it maintains a memory of what has been processed
            input_size=self.hidden_size,    # The hidden size serve as the resolution of the comprehension state
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        # Uncertainty estimation:  a feedforward nn that estimates the uncertainty of the comprehension state
        # Inputs: the concatenation of the current word's embedding and the context state, thus hidden_size * 2
        # Outputs: a scalar between 0 and 1, representing the uncertainty of the comprehension state
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

    def reset(self, sentence_words: list[str]):
        """Initialize comprehension state for new sentence"""
        self.sentence_words = sentence_words
        self.word_embeddings = self._get_word_embeddings(sentence_words) # A 3D tensor with shape (batch_size, sequence_length, hidden_size)
        self.current_word_comprehension_state = torch.zeros(1, 1, self.hidden_size) # A 3D tensor with shape (batch: one sentence at a time, sequence length:one word at a time, hidden_size); a blank state
        self.word_states = [None] * len(sentence_words)
        return self.word_states
    
    def _get_word_embeddings(self, words):
        """Get contextual embeddings for each word"""
        tokens = self.tokenizer(words, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.language_model(**tokens)
        return outputs.last_hidden_state
        
    def update_state_read_next_word(self, states, current_word_idx, sentence_length):
        """
        Update comprehension state when reading next word.
        Now uses neural representations and uncertainty estimation.
        """
        if current_word_idx >= sentence_length - 1:
            return states, current_word_idx, False
            
        current_word_idx += 1
        
        # Get word embedding and update comprehension state
        word_embedding = self.word_embeddings[:, current_word_idx]
        print(f"Word embedding shape: {word_embedding.shape}")
        
        word_embedding_expanded = word_embedding.unsqueeze(0)
        print(f"After unsqueeze shape: {word_embedding_expanded.shape}")
        
        self.current_word_comprehension_state, _ = self.comprehension_tracker(
            word_embedding_expanded, 
            self.current_word_comprehension_state
        )
        print(f"Comprehension state shape: {self.current_word_comprehension_state.shape}")
        
        # Estimate integration difficulty
        context_state = self._get_context_state(end_word_idx=current_word_idx)
        integration_difficulty = self._compute_integration_difficulty(
            self.current_word_comprehension_state, 
            context_state
        )
        
        # Update word state based on integration success
        states[current_word_idx] = {
            'embedding': word_embedding,
            'comprehension': self.current_word_comprehension_state,
            'difficulty': integration_difficulty
        }
        
        # Decay previous states based on working memory limitations
        self._apply_memory_decay(states, current_word_idx)
        
        return states, current_word_idx, True
        
    def update_state_skip_next_word(self, states, current_word_idx, sentence_length, predictability):
        """
        Update state when skipping word, using predictability and context
        to estimate comprehension loss
        """
        if current_word_idx >= sentence_length - 1:
            return states, current_word_idx, False
            
        # Skip to next word
        current_word_idx += 2
        
        # Estimate skipped word comprehension from context
        context_state = self._get_context_state(end_word_idx=current_word_idx - 1)
        print(f"Context state shape: {context_state.shape}")

        predicted_state = self._predict_skipped_meaning(
            context_state, 
            predictability
        )
        print(f"Predicted state shape: {predicted_state.shape}")
        
        # Higher uncertainty for skipped words
        uncertainty = self.uncertainty_estimator(
            torch.cat([context_state, predicted_state], dim=-1)
        )
        print(f"Uncertainty shape: {uncertainty.shape}")

        states[current_word_idx - 1] = {
            'embedding': predicted_state,
            'comprehension': predicted_state * (1 - uncertainty),
            'difficulty': uncertainty
            }
        
        # TODO shouldn't also update the current word's comprehension state?
            
        return states, current_word_idx, True
        
    def update_state_regress(self, states, current_word_idx):
        """
        Update state during regression, attempting to reduce uncertainty
        """
        if current_word_idx <= 0:
            return states, current_word_idx, False
            
        current_word_idx -= 1
        
        # Re-read word with additional context
        word_embedding = self.word_embeddings[:, current_word_idx]
        context_state = self._get_context_state(end_word_idx=current_word_idx)    # TODO my question: only one word as the context?
        
        # Update comprehension with new context
        new_comprehension, _ = self.comprehension_tracker(
            word_embedding.unsqueeze(0),
            context_state.unsqueeze(0)
        )
        
        # Update state with improved comprehension
        states[current_word_idx]['comprehension'] = new_comprehension
        states[current_word_idx]['difficulty'] *= 0.5  # Reduce uncertainty
        
        return states, current_word_idx, True
        
    def _get_context_state(self, end_word_idx):
        """Get contextual state from previous words"""
        context_start_word_idx = max(0, end_word_idx - self._context_size)
        context_embeddings = self.word_embeddings[:, context_start_word_idx:end_word_idx]
        print(f"Context embeddings shape: {context_embeddings.shape}")
        return context_embeddings.mean(dim=1)
        
    def _compute_integration_difficulty(self, current, context):
        """
        Compute difficulty of integrating new information
        current: The current word's comprehension state
        context: The context from previous words
        torch.norm(current - context, dim=-1): Calculates the Euclidean distance between current and context
        
        dim=-1 means to compute along the last dimension of the tensors.
        For example, if current and context are embeddings with shape [batch_size, embedding_dim],
        dim=-1 will compute the norm across the embedding_dim dimension.
        This is equivalent to dim=1 in this case, but using -1 makes it work regardless of the number of dimensions.
        """
        integration_difficulty = torch.norm(current - context, dim=-1)
        print(f"Integration difficulty shape: {integration_difficulty.shape}")
        return integration_difficulty
        
    def _predict_skipped_meaning(self, context, predictability):
        """
        Predict meaning of skipped word from context
        context: The context from previous words
        predictability: The predictability of the skipped word

        TODO: issue of the predictability, where should it come from? what is it used here?
        """
        # Get embeddings for context
        context_embedding = self.language_model(**self.tokenizer(
            self.sentence_words, 
            return_tensors="pt", 
            padding=True
        )).last_hidden_state
        print(f"Context embedding shape: {context_embedding.shape}")
        
        # Use attention to predict skipped word
        attention_weights = torch.softmax(
            torch.matmul(context_embedding, context_embedding.transpose(-1, -2)),
            dim=-1
        )
        print(f"Attention weights shape: {attention_weights.shape}")
        
        # Weighted sum of context embeddings
        predicted_embedding = torch.matmul(attention_weights, context_embedding)
        print(f"Predicted embedding shape: {predicted_embedding.shape}")
        
        # Scale by predictability
        return predicted_embedding * predictability
        
    def _apply_memory_decay(self, states, current_idx):
        """Apply decay to previous states based on distance"""
        # for i in range(current_idx):
        #     if states[i] is not None:
        #         distance = current_idx - i
        #         decay = 0.9 ** distance
        #         states[i]['comprehension'] *= decay
        #         states[i]['difficulty'] = min(1.0, states[i]['difficulty'] + (1 - decay))
        pass    # NOTE: do not decay memory first, see whether needed for realisitic regressions.