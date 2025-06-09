import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from . import Constants
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import json
import os
from tqdm import tqdm
import yaml

# Suppress non-critical warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress resume_download warning
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*")  # Suppress unused weights warning

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
        """Initialize transition function with simplified state management"""
        self._config = self._load_config()
        
    def _load_config(self):
        """Load configuration from yaml file"""
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    
    def update_state_regress(self, current_word_index, sentence_len):
        """Update state when regressing to previous word
        
        Args:
            current_word_index: Current word position
            sentence_len: Total length of sentence
            
        Returns:
            tuple: (new_word_index, action_validity)
        """
        if current_word_index <= 0:
            return current_word_index, False
            
        new_word_index = current_word_index - 1
        return new_word_index, True
    
    def update_state_read_next_word(self, current_word_index, sentence_len):
        """Update state when reading next word
        
        Args:
            current_word_index: Current word position
            sentence_len: Total length of sentence
            
        Returns:
            tuple: (new_word_index, action_validity)
        """
        if current_word_index >= sentence_len - 1:
            return current_word_index, False
            
        new_word_index = current_word_index + 1
        return new_word_index, True
    
    def update_state_skip_next_word(self, current_word_index, sentence_len):
        """Update state when skipping next word
        
        Args:
            current_word_index: Current word position
            sentence_len: Total length of sentence
            
        Returns:
            tuple: (new_word_index, action_validity)
        """
        if current_word_index >= sentence_len - 2:  # Need at least 2 words ahead to skip
            return current_word_index, False
            
        new_word_index = current_word_index + 2
        return new_word_index, True
    
    def update_state_time(self, elapsed_time, expected_sentence_reading_time, word_reading_time):
        """Update state when reading next word
        
        Args:
            elapsed_time: Elapsed time
            expected_sentence_reading_time: Expected reading time for the current sentence
            word_reading_time: Reading time for the current word
        """
        updated_elapsed_time = elapsed_time + word_reading_time
        updated_remaining_time = expected_sentence_reading_time - updated_elapsed_time
        return updated_elapsed_time, updated_remaining_time

    def reset(self, sentence_words: list[str]) -> list[dict]:
        """Initialize comprehension state for new sentence"""
        self.sentence_words = sentence_words
        
        return None


if __name__ == "__main__":
    pass