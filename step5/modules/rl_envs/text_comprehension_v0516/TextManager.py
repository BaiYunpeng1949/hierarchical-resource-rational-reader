import os
import json
import random


class TextManager():
    """
    Text Manager that handles loading and sampling text from a dataset.
    """

    def __init__(self):
        
        # A psuedo text dataset
        text_dataset = [
            {
                "text_id": 0,
                "text_content":"",
                "num_sentences": 10,
                "sentence_appraisal_scores_distribution": [0.3, 0.8, 0.8, 0.2, 0.4, 0.7, 0.6, 0.4, 0.6, 0.5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            }, 
            {
                "text_id": 1,
                "text_content":"",
                "num_sentences": 15,
                "sentence_appraisal_scores_distribution": [0.5, 0.3, 0.0, 0.3, 0.8, 0.1, 0.4, 0.7, 0.9, 0.1, 1.0, 0.9, 0.8, 0.3, 0.1, -1, -1, -1, -1, -1]
            },
            {
                "text_id": 2,
                "text_content":"",
                "num_sentences": 20,
                "sentence_appraisal_scores_distribution": [1.0, 1.0, 0.6, 0.0, 0.3, 0.9, 1.0, 0.9, 0.4, 0.9, 0.3, 0.5, 0.5, 0.1, 1.0, 0.4, 1.0, 0.7, 0.8, 0.2]
            }
        ]

        self._text_dataset = text_dataset
        self._num_texts = len(text_dataset)


    def reset(self):
        return random.choice(self._text_dataset)