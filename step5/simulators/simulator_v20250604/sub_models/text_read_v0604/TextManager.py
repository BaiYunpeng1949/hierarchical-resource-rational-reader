import os
import json
import random
from . import Constants
# import Constants


class TextManager():
    """
    Text Manager that handles loading and sampling text from a dataset.
    """

    def __init__(self, min_num_sentences=Constants.MIN_NUM_SENTENCES, max_num_sentences=Constants.MAX_NUM_SENTENCES):
        self.min_num_sentences = min_num_sentences
        self.max_num_sentences = max_num_sentences
        self._text_id_counter = 0

    def reset(self):
        # Generate random number of sentences
        num_sentences = random.randint(self.min_num_sentences, self.max_num_sentences)
        
        # Generate appraisal scores from Gaussian distribution centered at 0.6
        # Using std dev of 0.2 to keep most values within reasonable range
        sentence_appraisal_scores = []
        for _ in range(num_sentences):
            score = random.gauss(0.6, 0.2)  # TODO change to the real sentences' appraisal values (need to simulate the first-pass appraisals and revised appraisals)
            # Clamp values between 0 and 1
            score = max(0.0, min(1.0, score))
            sentence_appraisal_scores.append(round(score, 1))
        
        # Pad with -1 to maintain consistent length (20 sentences max)
        padded_scores = sentence_appraisal_scores + [-1] * (20 - num_sentences)
        
        # Sentence lenghts
        sentence_lengths = []
        # Sentence reading times (in seconds)
        sentence_reading_times = []

        # Generate random number of words per sentence
        for sentence_idx in range(num_sentences):
            num_words_per_sentence = random.randint(Constants.MIN_SENTENCE_LENGTH, Constants.MAX_SENTENCE_LENGTH)
            sentence_lengths.append(num_words_per_sentence)
            sentence_reading_times.append(num_words_per_sentence * Constants.READING_SPEED)
        
        # Create new text entry
        text_entry = {
            "text_id": self._text_id_counter,
            "text_content": "",  # Empty content as per original
            "num_sentences": num_sentences,
            "sentence_appraisal_scores_distribution": padded_scores,
            "sentence_lengths": sentence_lengths,
            "sentence_reading_times": sentence_reading_times
        }
        
        self._text_id_counter += 1
        return text_entry



if __name__ == "__main__":
    text_manager = TextManager()
    text_entry = text_manager.reset()
    print(text_entry)