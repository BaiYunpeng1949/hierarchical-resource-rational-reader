import os
import json
import random
from . import Constants


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
        
        # Generate random appraisal scores for each sentence, rounded to 1 decimal
        sentence_appraisal_scores = [round(random.random(), 1) for _ in range(num_sentences)]
        
        # Pad with -1 to maintain consistent length (20 sentences max)
        padded_scores = sentence_appraisal_scores + [-1] * (20 - num_sentences)
        
        # Create new text entry
        text_entry = {
            "text_id": self._text_id_counter,
            "text_content": "",  # Empty content as per original
            "num_sentences": num_sentences,
            "sentence_appraisal_scores_distribution": padded_scores
        }
        
        self._text_id_counter += 1
        return text_entry



if __name__ == "__main__":
    text_manager = TextManager()
    text_entry = text_manager.reset()
    print(text_entry)