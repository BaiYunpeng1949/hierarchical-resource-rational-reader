import os
import json
from collections import Counter

import step5.utils.constants as const


def filter_words(input_file_path, output_file_path, length_threshold):
    """
    Filter words from the input file that are shorter than or equal to the length_threshold
    :param input_file_path:
    :param output_file_path:
    :param length_threshold:
    :return:
    """

    with open(input_file_path, 'r') as input_file:
        words = input_file.readlines()  # Read all lines/words from the file

    # Print the number of words in the corpus, and the number of unique words
    # Remove the duplicates from the list of words
    words = [word.strip() for word in words]
    if len(words) == len(set(words)):
        print(f"Number of words in the corpus: {len(words)}, {len(set(words))} unique words")
    else:
        print("There are duplicate words in the corpus.")

    # Make sure all words are english letters, and they are in lowercase
    words = [word.strip().lower() for word in words if word.isalpha()]

    # Print the table of the number of words of each length
    word_lengths = [len(word) for word in words]
    word_length_counts = Counter(word_lengths)
    print("Length | Count")
    print("-------|------")
    for length, count in sorted(word_length_counts.items()):
        print(f"{length:6} | {count:5}")

    # Filter words that are shorter than or equal to the length_threshold
    words = [word for word in words if len(word.strip()) <= length_threshold]

    # Write the filtered words to a new file, one line a word
    with open(output_file_path, 'w') as output_file:
        for word in words:
            output_file.write(word + '\n')

    # Assure the max length is correct
    assert max([len(word) for word in words]) <= length_threshold

    # Generate a json file that sort words by length, in the format of word length: [words]
    word_length_dict = {}
    for word in words:
        word_len = len(word)
        if word_len not in word_length_dict:
            word_length_dict[word_len] = []
        word_length_dict[word_len].append(word)

    # Write the word length dictionary to a new json file -- make sure they have the same file name
    with open(output_file_path.replace('.txt', '.json'), 'w') as json_file:
        json.dump(word_length_dict, json_file, indent=4)


if __name__ == "__main__":

    # Get the directory and file paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    lexicon_file_name = 'bai_lexicon'
    filter_words(
        input_file_path=os.path.join(root_dir, 'data', 'assets', lexicon_file_name+'.txt'),       # 'dwyl_words_alpha.txt' 'oxford_lexicon.txt'
        output_file_path=os.path.join(root_dir, 'data', 'assets', lexicon_file_name+f'_under_{const.MAX_WORD_LEN}.txt'), # f'lexicon_under_{cons.MAX_WORD_LEN}.txt'
        length_threshold=const.MAX_WORD_LEN
    )
