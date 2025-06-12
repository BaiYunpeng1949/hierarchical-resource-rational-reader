import os
import json
import random

from . import Constants

class SentencesManager():
    """
    Sentences Manager that handles loading and sampling sentences from a dataset.
    Each sentence comes with word-level information including predictability values.
    Supports both ZuCo dataset and custom stimulus-based dataset.
    """

    def __init__(self, dataset="ZuCo1.0"):
        # Set the dataset
        self._dataset = dataset
        self._sentences_dataset = {}
        self._current_sentence_id = 0

        if dataset == "Ours":
            # Read all stimulus
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'processed_my_stimulus_with_observations.json'), 'r') as f:
                self._stimulus_dataset = json.load(f)
            
            # Parse to sentences across different stimulus
            self._sentences_dataset = self._parse_stimulus_to_sentences(self._stimulus_dataset)
            
        elif dataset == "ZuCo1.0":
            # Read all the sentences from the assets/sentences_dataset.json
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'sentences_dataset_processed.json'), 'r') as f:
                self._sentences_dataset = json.load(f)
            
        # Store dataset statistics
        self._num_sentences = len(self._sentences_dataset)
        
        # Get sentence length bounds
        all_lengths = [len(self._sentences_dataset[str(i)]["words"]) for i in range(self._num_sentences)]
        self._max_sent_len = max(all_lengths)
        self._min_sent_len = min(all_lengths)

    def reset(self, sentence_idx=None, inputs: dict=None):
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
        # if sentence_idx is not None and 0 <= sentence_idx < self._num_sentences:
        #     sentence_data = self._sentences_dataset[str(sentence_idx)]
        # else:
        if inputs is None:
            random_idx = random.randint(0, self._num_sentences - 1)
            sentence_idx = random_idx
            sentence_data = self._sentences_dataset[str(random_idx)]
        else:
            assert self._dataset == "Ours", "Only Ours dataset supports the feature of intact simulations."
            stimulus_id = inputs["stimulus_id"]
            sentence_idx = inputs["sentence_id"]
            sentence_data = self._stimulus_dataset[str(stimulus_id)]["sentences"][sentence_idx]
            
        # Extract word-level information
        words_metadata = sentence_data["words"]
        
        # Get word predictabilities from the dataset: get the largest
        word_predictabilities = []
        predicted_words = []
        predicted_words_ranked_integration_probabilities = []
        for word in words_metadata:
            # Get highest probability from prediction candidates
            max_prob = 0
            max_prob_word = ""
            max_prob_word_ranked_integration_prob = 1e-4
            for candidate in word["prediction_candidates"]:
                if candidate["probability"] > max_prob:
                    max_prob = candidate["probability"]
                    max_prob_word = candidate["word"]
                    max_prob_word_ranked_integration_prob = candidate["ranked_word_integration_probability"]
            word_predictabilities.append(max_prob)
            predicted_words.append(max_prob_word)
            predicted_words_ranked_integration_probabilities.append(max_prob_word_ranked_integration_prob)
        
        # Get the sentence information
        sentence_info = {
            "sentence_id": sentence_idx,
            "participant_id": "SIM",
            "sentence_content": sentence_data["sentence_content"] if "sentence_content" in sentence_data else sentence_data["sentence"],
            "sentence_len": len(words_metadata),
            "words": [word["word"] for word in words_metadata],
            "word_cleans": [word["word_clean"] for word in words_metadata],
            "word_ids": [word["word_id"] for word in words_metadata],
            "word_lengths_for_analysis": [len(word["word"]) for word in words_metadata],
            "word_frequencies_per_million_for_analysis": [word["frequency"] for word in words_metadata],
            "word_log_frequencies_per_million_for_analysis": [word["log_frequency"] for word in words_metadata],
            "word_difficulties_for_analysis": [word["difficulty"] for word in words_metadata],
            "word_predictabilities_for_analysis": [word["predictability"] for word in words_metadata],
            "word_logit_predictabilities_for_analysis": [word["logit_predictability"] for word in words_metadata],
            "words_ranked_word_integration_probabilities_for_running_model": [word["ranked_word_integration_probability"] for word in words_metadata],
            "words_predictabilities_for_running_model": word_predictabilities,
            "predicted_words_for_running_model": predicted_words,
            "predicted_words_ranked_integration_probabilities_for_running_model": predicted_words_ranked_integration_probabilities,
            "individual_word_reading_time": Constants.READING_SPEED       # seconds/word
        }
        
        return sentence_info
    
    def _parse_stimulus_to_sentences(self, stimulus_dataset):
        """
        Parse the stimulus dataset to sentences across different stimulus
        Args:
            stimulus_dataset: Dictionary containing stimulus data with sentences
        Returns:
            Dictionary of sentences with sequential IDs
        """
        parsed_sentences = {}
        current_sentence_id = 0

        for stimulus_id, stimulus_data in stimulus_dataset.items():
            for sentence in stimulus_data["sentences"]:
                # Create a new sentence entry with sequential ID
                parsed_sentences[str(current_sentence_id)] = {
                    "sentence_id": current_sentence_id,
                    "stimulus_id": stimulus_id,
                    "sentence_content": sentence["sentence"],
                    "words": sentence["words"]
                }
                current_sentence_id += 1

        return parsed_sentences


if __name__ == "__main__":
    # Test with both datasets
    print("Testing ZuCo dataset...")
    sentences_manager_zuco = SentencesManager(dataset="ZuCo1.0")
    sentence_info_zuco = sentences_manager_zuco.reset()
    print(f"ZuCo sentence info: {sentence_info_zuco['sentence_id']}")

    print("\nTesting Our dataset...")
    sentences_manager_ours = SentencesManager(dataset="Ours")
    sentence_info_ours = sentences_manager_ours.reset()
    print(f"Our sentence info: {sentence_info_ours['sentence_id']}")
        