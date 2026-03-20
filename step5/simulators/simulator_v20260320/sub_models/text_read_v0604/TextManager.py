import os
import json
import random
from . import Constants
# import Constants


class TextManager():
    """
    Text Manager that handles loading and sampling text from a dataset.
    """

    def __init__(self, data_source=Constants.DATA_SOURCE["real_stimuli"], min_num_sentences=Constants.MIN_NUM_SENTENCES, max_num_sentences=Constants.MAX_NUM_SENTENCES):
        self.min_num_sentences = min_num_sentences
        self.max_num_sentences = max_num_sentences
        self._text_id_counter = 0
        self._data_source = data_source

    def reset(self, inputs: dict=None):
        if self._data_source == Constants.DATA_SOURCE["real_stimuli"]:
            return self._reset_real_stimuli(inputs=inputs)
        elif self._data_source == Constants.DATA_SOURCE["generated_stimuli"]:
            return self._reset_generated_stimuli()
        else:
            raise ValueError(f"Invalid data source: {self._data_source}")

    def _reset_real_stimuli(self, inputs=None):

        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_file_path = os.path.join(current_dir, "assets", "processed_stimulus.json")
        
        with open(json_file_path, "r") as f:
            data = json.load(f)
        
        if inputs is None:
            # Randomly sample a text from the data
            sampled_text = random.choice(data)
        else:
            stimulus_id = inputs["stimulus_id"]
            sampled_text = data[stimulus_id]

        return sampled_text

    def _reset_generated_stimuli(self):
        # First, determine number of sentences within bounds
        num_sentences = random.randint(self.min_num_sentences, self.max_num_sentences)
        
        # Calculate target total words based on text length constraints
        target_total_words = random.randint(Constants.MIN_TEXT_LENGTH, Constants.MAX_TEXT_LENGTH)
        
        # Calculate average words per sentence
        avg_words_per_sentence = target_total_words // num_sentences
        
        # Generate sentence lengths that sum up to target_total_words
        remaining_words = target_total_words
        sentence_lengths = []
        
        for i in range(num_sentences):
            if i == num_sentences - 1:
                # Last sentence gets remaining words
                sentence_length = remaining_words
            else:
                # Calculate min and max possible length for this sentence
                min_possible = max(Constants.MIN_SENTENCE_LENGTH, 
                                 remaining_words - (Constants.MAX_SENTENCE_LENGTH * (num_sentences - i - 1)))
                max_possible = min(Constants.MAX_SENTENCE_LENGTH, 
                                 remaining_words - (Constants.MIN_SENTENCE_LENGTH * (num_sentences - i - 1)))
                
                # Generate length within constraints
                sentence_length = random.randint(min_possible, max_possible)
                remaining_words -= sentence_length
            
            sentence_lengths.append(sentence_length)
        
        # Generate appraisal scores from Gaussian distribution centered at 0.6
        sentence_appraisal_scores = []
        for _ in range(num_sentences):
            # score = random.gauss(0.6, 0.2)
            score = random.uniform(0.0, 1.0)
            score = max(0.0, min(1.0, score))
            sentence_appraisal_scores.append(round(score, 1))
        
        # Pad with -1 to maintain consistent length (20 sentences max)
        padded_scores = sentence_appraisal_scores + [-1] * (20 - num_sentences)
        
        # Calculate sentence reading times
        sentence_reading_times = self._sample_sentences_reading_times(sentence_lengths)
        
        # Create new text entry
        text_entry = {
            "stimulus_id": self._text_id_counter,
            "stimulus_source": "generated",
            "text_content": "",  # Empty content as per original
            "num_sentences": num_sentences,
            "sentence_appraisal_scores_distribution": padded_scores,
            "sentence_lengths": sentence_lengths,
            "sentence_reading_times": sentence_reading_times,
            "total_words": sum(sentence_lengths),  # Added for verification
            "total_one_pass_reading_time": sum(sentence_reading_times)
        }
        
        self._text_id_counter += 1

        return text_entry
    
    def _sample_sentences_reading_times(self, sentence_lengths):
        """
        Sample noisy sentences reading times for generalizability when dealing with different lower-level agents.
        """
        lower_bound_coefficient = 0.8
        upper_bound_coefficient = 1.2
        
        # Sample noisy reading times from a Gaussian distribution
        noisy_reading_times = []
        for sentence_length in sentence_lengths:
            regular_reading_time = sentence_length * Constants.READING_SPEED
            lower_bound = regular_reading_time * (Constants.TIME_CONDITIONS['30s'] / Constants.TIME_CONDITIONS['60s']) * lower_bound_coefficient
            upper_bound = regular_reading_time * (Constants.TIME_CONDITIONS['90s'] / Constants.TIME_CONDITIONS['60s']) * upper_bound_coefficient
            noisy_reading_time = random.uniform(lower_bound, upper_bound)
            noisy_reading_times.append(noisy_reading_time)
        return noisy_reading_times


if __name__ == "__main__":
    text_manager = TextManager()
    text_entry = text_manager.reset()
    print(text_entry)