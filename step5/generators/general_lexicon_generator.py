import os
import json
import string
import random
from collections import defaultdict


def generate_random_combinations(characters, length_threshold, max_num_unique_words, max_attempts_per_length=10000):
    """
    Generate up to max_num_unique_words unique random words for each length up to the length_threshold.
    :param characters: A string containing the characters to combine.
    :param length_threshold: The maximum length of words to include.
    :param max_num_unique_words: The maximum number of unique words for each length.
    :param max_attempts_per_length: The maximum number of attempts to generate unique words for each length.
    :return: A dictionary of words sorted by their length.
    """
    word_length_dict = defaultdict(set)

    for length in range(1, length_threshold + 1):
        attempts = 0
        while len(word_length_dict[length]) < max_num_unique_words and attempts < max_attempts_per_length:
            word = ''.join(random.choices(characters, k=length))
            word_length_dict[length].add(word)
            attempts += 1

        if len(word_length_dict[length]) < max_num_unique_words:
            print(f"Warning: Could only generate {len(word_length_dict[length])} unique words of length {length} after {max_attempts_per_length} attempts.")

    return {k: list(v) for k, v in word_length_dict.items()}


def save_words_to_file(words, output_file_path):
    """
    Save the words to a text file, one word per line.
    :param words: A list of words to save.
    :param output_file_path: The path to the output text file.
    """
    with open(output_file_path, 'w') as output_file:
        for word in words:
            output_file.write(word + '\n')


def save_word_length_dict_to_json(word_length_dict, output_json_path):
    """
    Save the word length dictionary to a JSON file.
    :param word_length_dict: A dictionary of words sorted by their length.
    :param output_json_path: The path to the output JSON file.
    """
    with open(output_json_path, 'w') as json_file:
        json.dump(word_length_dict, json_file, indent=4)


if __name__ == "__main__":
    characters = list(string.printable.strip())  # Get all printable ASCII characters and remove whitespace characters
    random.shuffle(characters)  # Shuffle the characters to randomize the order
    characters = ''.join(characters)

    length_threshold = 16  # Example length threshold, change as needed
    max_num_unique_words = 1000  # Example max number of unique words per length, change as needed

    # Generate unique random words for each length
    word_length_dict = generate_random_combinations(characters, length_threshold, max_num_unique_words)

    # Flatten the dictionary for saving to a file
    selected_words = [word for length in word_length_dict for word in word_length_dict[length]]

    # Get the directory and file paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lexicon_file_name = 'generalized_lexicon'

    # Save the words to a text file
    output_file_path = os.path.join(root_dir, 'data', 'assets', lexicon_file_name + f'_under_{length_threshold}.txt')
    save_words_to_file(selected_words, output_file_path)

    # Save the word length dictionary to a JSON file
    output_json_path = output_file_path.replace('.txt', '.json')
    save_word_length_dict_to_json(word_length_dict, output_json_path)
    print(f'Words saved to {output_file_path}')
