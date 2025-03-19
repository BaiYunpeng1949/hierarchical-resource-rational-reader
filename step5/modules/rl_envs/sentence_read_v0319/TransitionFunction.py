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
        # print(f"\nDebug - Integration Difficulty Calculation:")
        # print(f"Left context: {' '.join(context)}")
        # print(f"Word: {word}")
        # print(f"Right context: {' '.join(full_sentence[current_word_idx + 1:])}")
        # print(f"Masked text: {masked_text}")
        # print(f"Word probability: {word_prob.item():.4f}")
        # print(f"Surprisal (-log prob): {surprisal.item():.4f}")
        # print(f"Final difficulty (sigmoid): {difficulty:.4f}")
        
        # Get top 5 predictions for debugging
        top_5_probs, top_5_indices = torch.topk(probs, 5)
        top_5_words = [self.tokenizer.decode([idx]) for idx in top_5_indices]
        # print("\nTop 5 predicted words:")
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

    def _get_length_tolerance(self, word_length: int) -> tuple[int, int]:
        """
        Get the acceptable length range for a word based on its true length.
        Longer words have more uncertainty in length estimation.
        
        Args:
            word_length: True length of the word
            
        Returns:
            tuple[int, int]: (min_length, max_length) acceptable range
        """
        if word_length <= 2:
            # Very short words (1-2 letters) - exact length
            return word_length, word_length
        elif word_length <= 4:
            # Short words (3-4 letters) - ±1 letter
            return max(1, word_length - 1), word_length + 1
        elif word_length <= 6:
            # Medium words (5-6 letters) - ±2 letters
            return max(1, word_length - 2), word_length + 2
        elif word_length <= 8:
            # Longer words (7-8 letters) - ±3 letters
            return max(1, word_length - 3), word_length + 3
        else:
            # Very long words (9+ letters) - ±4 letters or more
            noise_range = word_length // 3  # More noise for longer words
            return max(1, word_length - noise_range), word_length + noise_range

    def _compute_word_predictability(self, context: list[str], target_word: str, preview_letters: int = 2) -> tuple[float, str, list[tuple[str, float]]]:
        """
        Compute predictability of a word given its context and parafoveal preview information.
        Uses BERT's masked language modeling to predict the next word.
        Returns subjective prediction (sampled from likely candidates) rather than target word probability.
        
        Args:
            context: List of words before the target word
            target_word: The actual word to predict
            preview_letters: Number of clear letters visible in parafoveal preview (default 2)
            
        Returns:
            tuple[float, str, list[tuple[str, float]]]: (predictability, predicted_word, top_predictions)
            - predictability: Probability of the predicted word (not target word)
            - predicted_word: The word we predict (may not be target word)
            - top_predictions: List of (word, prob) tuples for top candidates
        """
        # Create input with [MASK] token after context
        input_text = " ".join(context + ["[MASK]"])
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        
        # Find position of [MASK] token
        mask_token_id = self.tokenizer.mask_token_id
        mask_position = (inputs.input_ids[0] == mask_token_id).nonzero().item()
        
        # Get model predictions for masked position
        with torch.no_grad():
            outputs = self.masked_lm_model(**inputs)
            logits = outputs.logits[0, mask_position]  # Shape: [vocab_size]
            
        # Get probabilities for all words
        probs = torch.softmax(logits, dim=0)
        
        # Get preview information with noise
        clear_preview = target_word[:preview_letters].lower()  # First 2 letters are clear
        target_length = len(target_word)
        min_length, max_length = self._get_length_tolerance(target_length)

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
        
        # Get top predictions initially (increased from 100 to 1000 for better coverage)
        top_k = 1000
        top_probs, top_indices = torch.topk(probs, top_k)
        
        # Get candidates and handle subword tokens
        filtered_predictions = []
        total_filtered_prob = 0.0
        seen_words = set()  # To avoid duplicates from subword tokens
        
        for token_id, prob in zip(top_indices, top_probs):
            # Decode single token
            word = self.tokenizer.decode([token_id]).strip().lower()
            
            # Skip if we've seen this word or if it's a special token
            if word in seen_words or word.startswith('[') or not word:
                continue
                
            seen_words.add(word)
            word_len = len(word)
            
            # Check length constraints with blurry length estimation
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
            
            # Sample predicted word from filtered candidates using their probabilities
            words, probs = zip(*filtered_predictions)
            probs = torch.tensor(probs)
            
            # Sample from top 5 predictions (or all if less than 5)
            top_n = min(5, len(filtered_predictions))
            if top_n > 0:
                # Get top N predictions and renormalize their probabilities
                top_words = words[:top_n]
                top_probs = probs[:top_n]
                top_probs = top_probs / top_probs.sum()  # Renormalize
                
                # Add more randomness to sampling
                noise = torch.rand_like(top_probs) * 0.2  # 20% random noise
                noisy_probs = top_probs + noise
                noisy_probs = noisy_probs / noisy_probs.sum()  # Renormalize
                
                # Sample predicted word using the noisy probabilities
                predicted_idx = torch.multinomial(noisy_probs, 1).item()
                predicted_word = top_words[predicted_idx]
                predictability = top_probs[predicted_idx].item()
            else:
                # Fallback if no predictions
                predicted_word = ""
                predictability = 0.0
        else:
            predicted_word = ""
            predictability = 0.0
            
        # Debug information
        print("\nDebug - Word Prediction Details:")
        print(f"Input text: {input_text}")
        print(f"Clear preview (first {preview_letters}): {clear_preview}")
        print(f"Length constraint: {min_length}-{max_length} (true length: {target_length})")
        print(f"Total filtered candidates: {len(filtered_predictions)}")
        print(f"Predicted word: {predicted_word} (prob: {predictability:.4f})")
        print(f"Target word: {target_word}")
        print(f"Sum of filtered probabilities: {total_filtered_prob:.4f}")
        
        return predictability, predicted_word, filtered_predictions[:5]  # Return top 5 filtered predictions

    def _compute_semantic_difference(self, original_sentence: list[str], perceived_sentence: list[str]) -> dict:
        """
        Compute semantic difference between original and perceived sentences using BERT embeddings.
        Now considers word importance, meaning changes, and semantic relationships.
        
        Args:
            original_sentence: List of words in the original sentence
            perceived_sentence: List of words in the perceived sentence
            
        Returns:
            dict: Dictionary containing various semantic difference metrics
        """
        # Convert sentences to text
        original_text = " ".join(original_sentence)
        perceived_text = " ".join(perceived_sentence)
        
        # Get embeddings for both sentences
        original_encoding = self.tokenizer(original_text, return_tensors="pt", padding=True, truncation=True)
        perceived_encoding = self.tokenizer(perceived_text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            original_outputs = self.language_model(**original_encoding)
            perceived_outputs = self.language_model(**perceived_encoding)
        
        # Get [CLS] token embeddings (sentence-level representation)
        original_embedding = original_outputs.last_hidden_state[0, 0]  # [CLS] token
        perceived_embedding = perceived_outputs.last_hidden_state[0, 0]  # [CLS] token
        
        # Compute cosine similarity between embeddings
        similarity = torch.nn.functional.cosine_similarity(
            original_embedding.unsqueeze(0),
            perceived_embedding.unsqueeze(0)
        ).item()
        
        # Define word importance weights (nouns and verbs are more important)
        word_importance = {
            'noun': 1.0,
            'verb': 1.0,
            'adjective': 0.8,
            'adverb': 0.7,
            'article': 0.3,
            'preposition': 0.4,
            'pronoun': 0.5,
            'conjunction': 0.3
        }
        
        # Compute word-level differences with importance weights
        word_differences = []
        total_importance = 0
        weighted_similarity = 0
        
        for i, (orig_word, perc_word) in enumerate(zip(original_sentence, perceived_sentence)):
            if orig_word.lower() != perc_word.lower():
                # Get word embeddings
                orig_encoding = self.tokenizer(orig_word, return_tensors="pt", padding=True, add_special_tokens=False)
                perc_encoding = self.tokenizer(perc_word, return_tensors="pt", padding=True, add_special_tokens=False)
                
                with torch.no_grad():
                    orig_outputs = self.language_model(**orig_encoding)
                    perc_outputs = self.language_model(**perc_encoding)
                
                # Get word embeddings (average over subword tokens)
                orig_embedding = orig_outputs.last_hidden_state[0].mean(dim=0)
                perc_embedding = perc_outputs.last_hidden_state[0].mean(dim=0)
                
                # Compute word-level similarity
                word_similarity = torch.nn.functional.cosine_similarity(
                    orig_embedding.unsqueeze(0),
                    perc_embedding.unsqueeze(0)
                ).item()
                
                # Determine word type and importance
                word_type = 'noun' if i in [1, 5] else 'verb' if i == 2 else 'article' if i in [0, 4] else 'preposition' if i == 3 else 'other'
                importance = word_importance.get(word_type, 0.5)
                
                # Update weighted metrics
                total_importance += importance
                weighted_similarity += word_similarity * importance
                
                word_differences.append({
                    "position": i,
                    "original": orig_word,
                    "perceived": perc_word,
                    "similarity": word_similarity,
                    "importance": importance,
                    "type": word_type
                })
        
        # Compute weighted semantic preservation
        if total_importance > 0:
            weighted_semantic_preservation = weighted_similarity / total_importance
        else:
            weighted_semantic_preservation = 0.0
        
        # Analyze meaning changes
        meaning_changes = []
        for diff in word_differences:
            if diff['type'] == 'noun' and diff['similarity'] < 0.5:
                meaning_changes.append("Major object/subject change")
            elif diff['type'] == 'verb' and diff['similarity'] < 0.5:
                meaning_changes.append("Major action change")
            elif diff['type'] == 'article' and diff['original'] != diff['perceived']:
                meaning_changes.append("Specificity change")
        
        # Compute final semantic preservation score
        # Consider both sentence-level similarity and weighted word-level similarity
        final_semantic_preservation = (similarity + weighted_semantic_preservation) / 2
        
        return {
            "sentence_similarity": similarity,
            "semantic_preservation": final_semantic_preservation,
            "weighted_semantic_preservation": weighted_semantic_preservation,
            "word_differences": word_differences,
            "num_differences": len(word_differences),
            "meaning_changes": meaning_changes,
            "total_importance": total_importance
        }

    def _compute_sentence_comprehension(self, states: list[dict]) -> dict:
        """
        Compute sentence-level comprehension by aggregating word-level comprehension states.
        Uses multiple aggregation methods to provide different perspectives on comprehension.
        
        Args:
            states: List of word states, each containing 'embedding', 'comprehension', and 'difficulty'
            
        Returns:
            dict: Dictionary containing various sentence-level comprehension metrics
        """
        # Extract comprehension states and difficulties
        comprehension_states = [state['comprehension'] for state in states]
        difficulties = [state['difficulty'].item() for state in states]
        
        # Convert to tensor for computation
        comprehension_tensor = torch.stack(comprehension_states)
        difficulties_tensor = torch.tensor(difficulties)
        
        # 1. Simple average of comprehension states
        simple_avg = comprehension_tensor.mean(dim=0)
        
        # 2. Weighted average based on integration difficulty (inverse of difficulty)
        weights = 1 - difficulties_tensor
        weights = weights / weights.sum()  # Normalize weights
        weighted_avg = (comprehension_tensor * weights.unsqueeze(-1)).sum(dim=0)
        
        # 3. Compute comprehension confidence (how confident we are in our understanding)
        comprehension_confidence = 1 - torch.mean(difficulties_tensor)
        
        # 4. Compute comprehension stability (how consistent the comprehension is across words)
        comprehension_stability = 1 - torch.std(difficulties_tensor)
        
        # 5. Compute semantic coherence (how well the words fit together)
        word_embeddings = torch.stack([state['embedding'] for state in states])
        semantic_coherence = torch.mean(torch.nn.functional.cosine_similarity(
            word_embeddings[:-1],
            word_embeddings[1:],
            dim=-1
        ))
        
        return {
            "simple_average": simple_avg,
            "weighted_average": weighted_avg,
            "comprehension_confidence": comprehension_confidence.item(),
            "comprehension_stability": comprehension_stability.item(),
            "semantic_coherence": semantic_coherence.item(),
            "average_difficulty": torch.mean(difficulties_tensor).item(),
            "max_difficulty": torch.max(difficulties_tensor).item(),
            "min_difficulty": torch.min(difficulties_tensor).item()
        }

    def _compute_cloze_comprehension(self, states: list[dict], original_sentence: list[str]) -> dict:
        """
        Evaluate sentence comprehension using cloze-test style word recovery.
        For each word, test if the model can recover it from its comprehension state.
        
        Args:
            states: List of word states containing comprehension information
            original_sentence: List of original words
            
        Returns:
            dict: Dictionary containing cloze-test comprehension metrics
        """
        word_recovery_scores = []
        word_importance_weights = []
        word_analyses = []  # Store analyses for all words
        
        # Define word importance weights
        word_importance = {
            'noun': 1.0,
            'verb': 1.0,
            'adjective': 0.8,
            'adverb': 0.7,
            'article': 0.3,
            'preposition': 0.4,
            'pronoun': 0.5,
            'conjunction': 0.3
        }
        
        for i, (state, orig_word) in enumerate(zip(states, original_sentence)):
            # Get comprehension state for this word
            comprehension_state = state['comprehension']
            
            # Create masked input with [MASK] at current position
            masked_sentence = original_sentence.copy()
            masked_sentence[i] = "[MASK]"
            masked_text = " ".join(masked_sentence)
            
            # Get model predictions for masked position
            inputs = self.tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True)
            mask_token_id = self.tokenizer.mask_token_id
            mask_position = (inputs["input_ids"][0] == mask_token_id).nonzero().item()
            
            with torch.no_grad():
                outputs = self.masked_lm_model(**inputs)
                logits = outputs.logits[0, mask_position]
                probs = torch.softmax(logits, dim=0)
            
            # Get probability of original word
            word_tokens = self.tokenizer(orig_word, return_tensors="pt", padding=True, add_special_tokens=False)
            word_token_id = word_tokens["input_ids"][0, 0]
            word_prob = probs[word_token_id].item()
            
            # Determine word type and importance
            word_type = 'noun' if i in [1, 5] else 'verb' if i == 2 else 'article' if i in [0, 4] else 'preposition' if i == 3 else 'other'
            importance = word_importance.get(word_type, 0.5)
            
            # Store recovery score and importance
            word_recovery_scores.append(word_prob)
            word_importance_weights.append(importance)
            
            # Get top 5 predictions for analysis
            top_5_probs, top_5_indices = torch.topk(probs, 5)
            top_5_words = [self.tokenizer.decode([idx]) for idx in top_5_indices]
            
            # Store detailed word analysis
            word_analyses.append({
                "position": i,
                "word": orig_word,
                "recovery_probability": word_prob,
                "importance": importance,
                "type": word_type,
                "top_predictions": list(zip(top_5_words, top_5_probs.tolist()))
            })
        
        # Compute weighted average recovery score
        total_importance = sum(word_importance_weights)
        if total_importance > 0:
            weighted_recovery = sum(score * weight for score, weight in zip(word_recovery_scores, word_importance_weights)) / total_importance
        else:
            weighted_recovery = 0.0
        
        # Compute per-word-type recovery scores
        type_recovery = {}
        for word_type in word_importance.keys():
            type_scores = [score for score, analysis in zip(word_recovery_scores, word_analyses) if analysis['type'] == word_type]
            if type_scores:
                type_recovery[word_type] = sum(type_scores) / len(type_scores)
        
        # Analyze comprehension quality
        comprehension_quality = {
            "noun_recovery": type_recovery.get('noun', 0.0),
            "verb_recovery": type_recovery.get('verb', 0.0),
            "content_word_recovery": (type_recovery.get('noun', 0.0) + type_recovery.get('verb', 0.0)) / 2,
            "function_word_recovery": (type_recovery.get('article', 0.0) + type_recovery.get('preposition', 0.0)) / 2,
            "overall_recovery": weighted_recovery
        }
        
        return {
            "word_recovery_scores": word_recovery_scores,
            "weighted_recovery": weighted_recovery,
            "average_recovery": sum(word_recovery_scores) / len(word_recovery_scores),
            "type_recovery": type_recovery,
            "word_analyses": word_analyses,
            "total_importance": total_importance,
            "comprehension_quality": comprehension_quality
        }

    def test_semantic_difference(self):
        """Test the semantic difference evaluation with example sentences."""
        # Test case 1: "the cat sits on a mat" vs "the cah sit on the mad"
        test_case1 = {
            "original": ["the", "cat", "sits", "on", "a", "mat"],
            # "perceived": ["the", "cah", "sit", "on", "the", "mad"],
            "perceived": ["the", "cah", "sit"],
            "description": "Case 1: Minor spelling errors and tense change"
        }
        
        # Test case 2: "he likes eating apples" vs "he liked eating apple"
        test_case2 = {
            "original": ["he", "likes", "eating", "apples"],
            "perceived": ["he", "liked", "eating", "apple"],
            "description": "Case 2: Tense change and plural/singular difference"
        }
        
        print("\nTesting Semantic Difference Evaluation:")
        print("=" * 70)
        
        for case in [test_case1, test_case2]:
            print(f"\n{case['description']}")
            print(f"Original: {' '.join(case['original'])}")
            print(f"Perceived: {' '.join(case['perceived'])}")
            
            # Compute semantic differences
            differences = self._compute_semantic_difference(case['original'], case['perceived'])
            
            # Print results
            print(f"\nSemantic Analysis:")
            print(f"Sentence Similarity: {differences['sentence_similarity']:.4f}")
            print(f"Semantic Preservation: {differences['semantic_preservation']:.4f}")
            print(f"Number of word differences: {differences['num_differences']}")
            
            print("\nWord-level differences:")
            for diff in differences['word_differences']:
                print(f"Position {diff['position']}:")
                print(f"  Original: {diff['original']}")
                print(f"  Perceived: {diff['perceived']}")
                print(f"  Word Similarity: {diff['similarity']:.4f}")
            
            print("-" * 70)

    def test_sentence_comprehension(self):
        """Test sentence-level comprehension evaluation with example sentences."""
        # Test case 1: Simple sentence with minor errors
        test_case1 = {
            "sentence": ["the", "cat", "sits", "on", "a", "mat"],
            "perceived": ["the", "cah", "sit", "on", "the", "mad"],
            "description": "Case 1: Simple sentence with spelling errors"
        }
        
        # Test case 2: More complex sentence with tense changes
        test_case2 = {
            "sentence": ["he", "likes", "eating", "apples"],
            "perceived": ["he", "liked", "eating", "apple"],
            "description": "Case 2: Complex sentence with tense changes"
        }
        
        print("\nTesting Sentence-Level Comprehension Evaluation:")
        print("=" * 70)
        
        for case in [test_case1, test_case2]:
            print(f"\n{case['description']}")
            print(f"Original: {' '.join(case['sentence'])}")
            print(f"Perceived: {' '.join(case['perceived'])}")
            
            # Initialize the sentence
            self.reset(case['sentence'])
            
            # Process each word to get states
            states = [None] * len(case['sentence'])
            for i in range(len(case['sentence'])):
                context = case['sentence'][:i]
                word = case['sentence'][i]
                
                # Compute integration difficulty
                difficulty = self._compute_integration_difficulty(context, word)
                
                # Get word embedding
                word_embedding = self.word_embeddings[:, i].detach()
                
                # Update comprehension state
                word_embedding_expanded = word_embedding.unsqueeze(1)
                output, hidden = self.cumulative_comprehension_tracker(
                    word_embedding_expanded,
                    self.cumulative_comprehension_state
                )
                self.cumulative_comprehension_state = hidden
                
                # Store state with actual difficulty
                states[i] = {
                    'embedding': word_embedding,
                    'comprehension': hidden.detach(),
                    'difficulty': torch.tensor(difficulty)
                }
            
            # Compute sentence-level comprehension
            comprehension = self._compute_sentence_comprehension(states)
            
            # Compute cloze-test comprehension
            cloze_comprehension = self._compute_cloze_comprehension(states, case['sentence'])
            
            # Print results
            print("\nComprehension Analysis:")
            print(f"Comprehension Confidence: {comprehension['comprehension_confidence']:.4f}")
            print(f"Comprehension Stability: {comprehension['comprehension_stability']:.4f}")
            print(f"Semantic Coherence: {comprehension['semantic_coherence']:.4f}")
            print(f"Average Difficulty: {comprehension['average_difficulty']:.4f}")
            
            print("\nCloze-Test Comprehension:")
            print(f"Weighted Recovery Score: {cloze_comprehension['weighted_recovery']:.4f}")
            print(f"Average Recovery Score: {cloze_comprehension['average_recovery']:.4f}")
            
            print("\nPer-Word-Type Recovery:")
            for word_type, score in cloze_comprehension['type_recovery'].items():
                print(f"{word_type}: {score:.4f}")
            
            print("\nWord-Level Recovery Analysis:")
            for analysis in cloze_comprehension['word_analyses']:
                print(f"\nPosition {analysis['position']}:")
                print(f"Word: {analysis['word']}")
                print(f"Type: {analysis['type']}")
                print(f"Recovery Probability: {analysis['recovery_probability']:.4f}")
                print("Top 5 Predictions:")
                for word, prob in analysis['top_predictions']:
                    print(f"  {word}: {prob:.4f}")
            
            print("-" * 70)


if __name__ == "__main__":
    def test_length_tolerance():
        """Test the blurry length estimation"""
        transition_function = TransitionFunction()
        
        test_lengths = [2, 4, 6, 8, 10, 15]
        print("\nTesting Length Tolerance Estimation:")
        print("=" * 50)
        
        for length in test_lengths:
            min_len, max_len = transition_function._get_length_tolerance(length)
            print(f"Word length: {length}")
            print(f"Acceptable range: {min_len}-{max_len}")
            print(f"Tolerance: ±{max_len - length}")
            print("-" * 50)
    
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
        preview_letters = 3  # Define preview_letters here
        
        # Test cases with various scenarios
        test_cases = [
            # Common predictable cases
            {
                "sentence": ["The", "cat", "likes", "to", "sleep", "on", "the", "mat"],
                "word_idx": 2,
                "description": "Common verb 'likes' after 'The cat'"
            },
            {
                "sentence": ["She", "went", "to", "the", "store", "to", "buy", "food"],
                "word_idx": 4,
                "description": "Predictable noun 'store' after retail context"
            },
            # Weather-related predictions
            {
                "sentence": ["The", "weather", "is", "sunny", "and", "warm", "today"],
                "word_idx": 3,
                "description": "Highly predictable adjective 'sunny' after weather context"
            },
            # Action sequences
            {
                "sentence": ["He", "picked", "up", "the", "phone", "and", "dialed"],
                "word_idx": 4,
                "description": "Very predictable object 'phone' after action"
            },
            # Domain-specific contexts
            {
                "sentence": ["The", "scientist", "conducted", "an", "experiment", "in", "the", "lab"],
                "word_idx": 4,
                "description": "Domain-specific word 'experiment' in scientific context"
            },
            # Cooking context
            {
                "sentence": ["The", "chef", "carefully", "chopped", "the", "onions", "for", "the", "soup"],
                "word_idx": 5,
                "description": "Cooking-related word 'onions' in kitchen context"
            },
            # Short words with clear preview
            {
                "sentence": ["I", "am", "at", "the", "bus", "stop"],
                "word_idx": 4,
                "description": "Short word 'bus' with clear preview"
            },
            # Long words with uncertain length
            {
                "sentence": ["The", "extraordinary", "performance", "impressed", "everyone"],
                "word_idx": 1,
                "description": "Long word 'extraordinary' with uncertain length"
            },
            # Multiple plausible continuations
            {
                "sentence": ["The", "student", "opened", "the", "book", "and", "started", "to", "read"],
                "word_idx": 6,
                "description": "Word 'started' with multiple plausible continuations"
            },
            # Surprising but grammatical
            {
                "sentence": ["The", "old", "car", "suddenly", "transformed", "into", "a", "robot"],
                "word_idx": 4,
                "description": "Surprising but grammatical 'transformed'"
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
            preview = target_word[:preview_letters].lower()
            
            # Calculate predictability
            predictability, predicted_word, top_predictions = transition_function._compute_word_predictability(
                context, 
                target_word,
                preview_letters
            )
            
            # Print results
            print(f"\nTest Case: {case['description']}")
            print(f"Full sentence: {' '.join(case['sentence'])}")
            print(f"Context: {' '.join(context)}")
            print(f"Target word: {target_word}")
            print(f"Predicted word: {predicted_word}")
            print(f"Preview letters: {preview}")
            print(f"Word length: {len(target_word)}")
            print(f"Predictability: {predictability:.4f}")
            print("\nTop 5 predictions (with preview filter):")
            for word, prob in top_predictions:
                print(f"  {word}: {prob:.4f}")
            print("-" * 70)
    
    def test_all():
        """Run all tests including the new semantic difference and comprehension tests."""
        print("\nRunning semantic difference test...")
        transition_function = TransitionFunction()
        transition_function.test_semantic_difference()
        
        print("\nRunning sentence comprehension test...")
        transition_function.test_sentence_comprehension()
    
    # Run all tests
    test_all() 