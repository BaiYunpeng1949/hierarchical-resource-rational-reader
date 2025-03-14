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
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'sentences_dataset.json'), 'r') as f:
            self._sentences_dataset = json.load(f)
            
        # Store dataset statistics
        self._num_sentences = len(self._sentences_dataset)
        
        # Get sentence length bounds
        all_lengths = [len(self._sentences_dataset[str(i)]["words"]) for i in range(self._num_sentences)]
        self._max_sent_len = max(all_lengths)
        self._min_sent_len = min(all_lengths)

    def reset(self):
        """
        Reset the sentences manager by sampling a new sentence.
        Returns:
            - number_of_words: Length of sampled sentence
            - word_predictabilities: List of predictability values for each word
        """
        # Randomly sample a sentence index
        sentence_idx = str(random.randint(0, self._num_sentences - 1))
        sampled_sentence = self._sentences_dataset[sentence_idx]
        
        # Extract word-level information
        words_metadata = sampled_sentence["words"]
        number_of_words = len(words_metadata)
        word_indices = [word["word_id"] for word in words_metadata]
        words = [word["word"] for word in words_metadata]
        clean_words = [word["word_clean"] for word in words_metadata]
        word_lengths = [len(word["word"]) for word in words_metadata]
        word_difficulties = [word["difficulty"] for word in words_metadata]
        word_frequencies = [word["frequency"] for word in words_metadata]
        word_log_frequencies = [word["log_frequency"] for word in words_metadata]
        word_contextual_predictabilities = [word["predictability"] for word in words_metadata]
        word_logit_contextual_predictabilities = [word["logit_predictability"] for word in words_metadata]

        # forge these information into a dictionary
        sentence_info = {
            "word_indices": word_indices,
            "words": words,
            "clean_words": clean_words,
            "word_lengths": word_lengths,
            "word_difficulties": word_difficulties,
            "word_frequencies": word_frequencies,
            "word_log_frequencies": word_log_frequencies,
            "word_contextual_predictabilities": word_contextual_predictabilities,
            "word_logit_contextual_predictabilities": word_logit_contextual_predictabilities,
        }

        # Check if all the information are position-wise aligned
        assert len(word_indices) == len(word_lengths) == len(word_difficulties) == len(word_frequencies) == len(word_log_frequencies) == len(word_contextual_predictabilities) == len(word_logit_contextual_predictabilities) == number_of_words

        return sentence_info


if __name__ == "__main__":
    sentences_manager = SentencesManager()
    sentences_manager.reset()
        