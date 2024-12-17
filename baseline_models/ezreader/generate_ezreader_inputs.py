import re
import json
import nltk
from nltk.corpus import brown
from nltk.data import find
from collections import Counter
from step5.utils import constants as step5_constants
from data_analysis import constants as data_analysis_constants
import math

corpus_dir_path = "/home/baiy4/reading-model/step5/data/assets/corpus/corpus_10_27.txt"

# Define the WordFrequencyCalculator class
class WordFrequencyCalculator:
    def __init__(self):
        """
        Initialize the Word Frequency.
        """
        # Check if the Brown corpus is available
        try:
            find('corpora/brown')
        except LookupError:
            print(f"Downloading the Brown corpus...")
            nltk.download('brown', quiet=True)
        
        # Load the Brown corpus
        words = brown.words()
        
        # Calculate word frequencies
        word_counts = Counter(word.lower() for word in words)
        total_words = sum(word_counts.values())
        
        # Normalize frequencies to probabilities
        self.word_probabilities = {word: count / total_words for word, count in word_counts.items()}
        
        # Store min and max log probabilities for normalization
        probabilities = [prob for prob in self.word_probabilities.values() if prob > 0]
        self.min_log_prob = math.log(min(probabilities))
        self.max_log_prob = math.log(max(probabilities))
        
    def get_probability(self, word: str) -> float:
        """
        Get the probability of a word.
        :param word: the word
        :return: the probability of the word
        """
        return self.word_probabilities.get(word.lower(), 0.0)
    
    def get_log_frequency(self, word: str) -> float:
        """
        Get the normalized log frequency of a word between 0 and 1.
        """
        probability = self.get_probability(word)
        if probability == 0:
            probability = 1e-9  # Assign a small default probability
        log_prob = math.log(probability)
        # Normalize
        normalized_frequency = (log_prob - self.min_log_prob) / (self.max_log_prob - self.min_log_prob)
        return normalized_frequency

# Initialize the WordFrequencyCalculator
word_frequency_calculator = WordFrequencyCalculator()

# Replace the get_word_frequency function
def get_word_frequency(word):
    # Get the probability from the calculator
    probability = word_frequency_calculator.get_probability(word)
    # Convert probability to frequency per million words
    frequency_per_million = probability * 1e06  # Frequencies per million words
    # Ensure frequency is not zero to avoid issues in the model
    if frequency_per_million == 0:
        frequency_per_million = 0.01  # Assign a small default frequency per million
    return frequency_per_million

def get_integration_time(word):
    base_time = 25  # Integration time in ms, standard in E-Z Reader
    additional_time = 1 * len(word)  # Add 1 ms per character (optional)
    time = base_time + additional_time
    return time

def get_integration_failure(word):
    base_failure = 0.01  # Base failure probability
    failure = base_failure + (0.01 if len(word) > 7 else 0)
    return min(failure, 1.0)  # Ensure it doesn't exceed 1.0

# Tokenization function
def tokenize(sentence):
    # Keep punctuation as separate tokens if needed
    tokens = re.findall(r"\b\w+\b", sentence.lower())
    return tokens

# Default predictability
default_predictability = 0.1

# Read the text file
with open(corpus_dir_path, 'r') as file:
    lines = file.readlines()

stimuli = []

for stimulus_index, line in enumerate(lines):
    sentence = line.strip()
    words = tokenize(sentence)
    word_data = []
    
    for idx, word in enumerate(words):  # Use enumerate to get the word index
        token = word
        frequency = get_word_frequency(word)
        predictability = default_predictability
        integration_time = get_integration_time(word)
        integration_failure = get_integration_failure(word)
        
        word_entry = {
            "word_index": idx,  # Add word index here
            "token": token,
            "frequency": frequency,
            "predictability": predictability,
            "integration_time": integration_time,
            "integration_failure": integration_failure
        }
        word_data.append(word_entry)
    
    stimulus_entry = {
        "stimulus_index": stimulus_index,
        "participant_index": -1,
        "time_constraint": 60,  # For comparison
        "baseline_model_name": data_analysis_constants.EZREADER,
        "words": word_data
    }
    stimuli.append(stimulus_entry)

# Output the JSON data
with open('ezreader_input_data.json', 'w') as json_file:
    json.dump(stimuli, json_file, indent=4)
