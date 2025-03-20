import os
import json
import random

class SentencesManager():
    """
    Sentences Manager that handles loading and sampling sentences from a dataset.
    Each sentence comes with word-level information including predictability values.
    """

    def __init__(self):
        # Read all the sentences from the assets/sentences_dataset.json
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'sentences_dataset_processed.json'), 'r') as f:
            self._sentences_dataset = json.load(f)
            
        # Store dataset statistics
        self._num_sentences = len(self._sentences_dataset)
        
        # Get sentence length bounds
        all_lengths = [len(self._sentences_dataset[str(i)]["words"]) for i in range(self._num_sentences)]
        self._max_sent_len = max(all_lengths)
        self._min_sent_len = min(all_lengths)

    def reset(self, sentence_idx=None):
        """
        Get a sentence for reading
        Args:
            sentence_idx: Optional specific sentence index for controlled testing
        Returns:
            dict with sentence information including:
            - words: list of words in the sentence
            - word_contextual_predictabilities: list of predictability values
            - other word-level metadata
        """
        # Get sentence data either by index or randomly
        if sentence_idx is not None and 0 <= sentence_idx < self._num_sentences:
            sentence_data = self._sentences_dataset[str(sentence_idx)]
        else:
            random_idx = random.randint(0, self._num_sentences - 1)
            sentence_data = self._sentences_dataset[str(random_idx)]
            
        # Extract word-level information
        words_metadata = sentence_data["words"]
        
        # Format sentence information

        # Get word predictabilities from the dataset: get the largest
        word_predictabilities = []
        predicted_words = []
        for word in words_metadata:
            # Get highest probability from prediction candidates
            max_prob = 0
            max_prob_word = ""
            for candidate in word["prediction_candidates"]:
                if candidate["probability"] > max_prob:
                    max_prob = candidate["probability"]
                    max_prob_word = candidate["word"]
            word_predictabilities.append(max_prob)
            predicted_words.append(max_prob_word)

        sentence_info = {
            "words": [word["word"] for word in words_metadata],
            "sentence_len": len(words_metadata),
            "clean_words": [word["word_clean"] for word in words_metadata],
            "word_indices": [word["word_id"] for word in words_metadata],
            "word_lengths": [len(word["word"]) for word in words_metadata],
            # "word_difficulties": [word["difficulty"] for word in words_metadata],
            # "word_frequencies": [word["frequency"] for word in words_metadata],
            # "word_log_frequencies": [word["log_frequency"] for word in words_metadata],
            # "word_contextual_predictabilities": [word["predictability"] for word in words_metadata],
            # "word_logit_contextual_predictabilities": [word["logit_predictability"] for word in words_metadata],
            "words_ranked_word_integration_probabilities": [word["ranked_word_integration_probability"] for word in words_metadata],
            "words_predictabilities": word_predictabilities,
            "predicted_words": predicted_words
        }
        
        return sentence_info


if __name__ == "__main__":
    sentences_manager = SentencesManager()
    sentences_manager.reset()
        