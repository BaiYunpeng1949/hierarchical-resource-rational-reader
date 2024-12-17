from collections import Counter
from PIL import Image
import os
import random
import csv
import itertools
import json

from memory_profiler import profile

import step5.utils.constants as constants


def encode_characters_in_word(word):

    """Encode the word characters into a 16-dim vector from 0 to 1 """
    encoded_word_characters = [-1] * constants.MAX_WORD_LEN

    # Encode every letter in the word using the full ASCII table from 0 to 127
    for idx, character in enumerate(word):
        if idx >= constants.MAX_WORD_LEN:
            break
        encoded_word_characters[idx] = (ord(character) - 32) / (126 - 32)
        # Normalize to range [0, 1] using printable ASCII range

    return encoded_word_characters

@profile
def load_oculomotor_controller_dataset(config):
        """
        Load the Oculomotor Controller dataset.
        """
        # Load the dataset
        # Read image environments -- determines which word to be identified
        image_envs_filename = config["resources"]["img_env_dir"]
        # --------------------------------------------------------------------------
        # Determine the dataset's mode -- using the training or testing dataset
        dataset_mode = None
        if config["rl"]["mode"] == constants.TRAIN or config["rl"]["mode"] == constants.CONTINUAL_TRAIN:
            dataset_mode = constants.TRAIN
        elif config["rl"]["mode"] == constants.SIMULATE:
            # self.dataset_mode = cons.TRAIN if self._config["rl"]["deploy"]["use_training_dataset"] is True else cons.TEST
            dataset_mode = constants.SIMULATE
        else:
            # TODO enable to test on simulated data as well
            if config["rl"]["test"]["dataset_for_testing"] == constants.TRAIN:
                dataset_mode = constants.TRAIN
            elif config["rl"]["test"]["dataset_for_testing"] == constants.TEST:
                dataset_mode = constants.TEST
            elif config["rl"]["test"]["dataset_for_testing"] == constants.SIMULATE:
                dataset_mode = constants.SIMULATE
            else:
                raise ValueError(f"Invalid mode: {config['rl']['test']['dataset_for_testing']}, must be one of {constants.TRAIN}, {constants.TEST}, {constants.SIMULATE}")
        # --------------------------------------------------------------------------
        image_envs_dir = os.path.join("data", "gen_envs", image_envs_filename, dataset_mode)
        # self._dataset_mode = dataset_mode
        # Read the JSON file (metadata)
        with open(os.path.join(image_envs_dir, constants.MD_FILE_NAME), "r") as file:
            metadata_of_stimuli = json.load(file)
        print(f"{constants.LV_THREE_DASHES}The metadata is loaded from {image_envs_dir}/{constants.MD_FILE_NAME}")
        # --------------------------------------------------------------------------
        lexicon_dir = os.path.join("data", "assets", config["resources"]["lexicon_filename"] + '.txt')
        lexicon = sorted(list(set(load_txt(lexicon_dir))))
        # Get a dictionary of the lexicon encoded in a 16-dim vector
        encoded_lexicon = {}
        for word in lexicon:
            encoded_lexicon[word] = encode_characters_in_word(word)
        # Output the loading information
        print(f"{constants.LV_THREE_DASHES}The lexicon is loaded from {lexicon_dir}")
        
        return metadata_of_stimuli, encoded_lexicon, dataset_mode 

def load_txt(filename):
    with open(filename, 'r', encoding='utf-8-sig') as file:
        words = file.readlines()

    # Remove newline characters and filter out words with non-standard UTF-8 characters
    valid_words = []
    for word in words:
        word = word.strip()

        try:
            # Attempt to encode the word in UTF-8
            word.encode('utf-8')
            if ' ' not in word:
                valid_words.append(word)
        except UnicodeEncodeError:
            # Skip words that can't be encoded in UTF-8
            pass

    return valid_words


def load_and_merge_corpus(corpus_file, corpus_zero_class_file):
    # Load the main corpus and zero-class corpus
    main_corpus = load_txt(corpus_file)
    zero_class_corpus = load_txt(corpus_zero_class_file)

    # Merge the two lists, while removing duplicates
    merged_corpus = list(set(main_corpus + zero_class_corpus))

    return merged_corpus


def generate_ngram_tokens(word, sliding_win_len=4):
    """Generate n-grams for a given word using padding."""

    ngrams = set()  # Use a set to avoid duplicates
    for i in range(len(word)):
        fix_idx = i
        ideal_start_idx = i - sliding_win_len // 2
        ideal_end_idx = ideal_start_idx + sliding_win_len
        start_idx = max(0, ideal_start_idx)
        end_idx = min(len(word), ideal_end_idx)
        token = word[start_idx:end_idx]
        # if start_idx <= 0:
        #     token = "_" + token
        # if end_idx >= len(word):
        #     token = token + "_"
        ngrams.add(token)

    return list(ngrams)


def generate_ngram_tokens_for_training_lexicon(_lexicon, sliding_win_len=4):
    # """Generate n-grams for all words in the corpus."""
    # all_ngrams = []
    # for word in _corpus:
    #     all_ngrams.extend(generate_ngram_tokens(word, n))
    # return all_ngrams

    """Generate all combinations of n-grams for each word in the corpus."""
    all_combinations = []
    for word in _lexicon:
        ngrams = generate_ngram_tokens(word, sliding_win_len)
        for i in range(1, len(ngrams) + 1):
            for combo in itertools.combinations(ngrams, i):
                all_combinations.append((word, str(len(all_combinations)), ', '.join(combo)))
    return all_combinations


def write_to_csv(data, filename):
    """Write the data to a CSV file."""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['word', 'combination_index', 'combinations'])
        for row in data:
            writer.writerow(row)


def is_valid_ngram(ngram):
    return set(ngram).issubset(set('abcdefghijklmnopqrstuvwxyz_'))


def store_lists_to_txt_file(contents, filename):
    """Store the list of corpus or n-grams into a text file."""
    with open(filename, 'w', encoding='utf-8') as file:
        for content in contents:
            file.write(content + '\n')


def normalise(x, x_min, x_max, a, b):
    # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
    return (b - a) * ((x - x_min) / (x_max - x_min)) + a


def find_duplicates(list_of_words):
    word_counts = Counter(list_of_words)
    duplicates = {word: count for word, count in word_counts.items() if count > 1}

    if duplicates:
        for word, count in duplicates.items():
            print(f"'{word}' appears {count} times.")
    else:
        print("There are no duplicates in the corpus.")


def split_corpus(input_file, percent=0.75, seed=42):
    """Split the content of the input file into two parts with a given ratio."""
    # Load the text from the input file
    all_words = load_txt(input_file)

    # Set the seed for reproducibility
    random.seed(seed)

    # Shuffle the list to ensure randomness
    random.shuffle(all_words)

    # Split the list according to the ratio
    split_index = int(len(all_words) * percent)
    return all_words[:split_index], all_words[:]


def read_images(directory):
    images = []
    filenames = []  # List to store filenames

    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return images, filenames  # Return both lists

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):  # Check for JPEG and PNG files
            filepath = os.path.join(directory, filename)
            try:
                img = Image.open(filepath)
                images.append(img)
                filenames.append(filename)  # Store the filename
                # print(f"Loaded image: {filename}")
            except IOError:
                print(f"Failed to open {filename}")

    return images, filenames  # Return both lists

def read_images_filenames(directory):
    pass


def filter_words(input_file_path, output_file_path, length_threshold):
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

    filter_words(
        input_file_path=os.path.join(root_dir, 'data', 'assets', 'dwyl_words_alpha.txt'),       # 'dwyl_words_alpha.txt' 'oxford_lexicon.txt'
        output_file_path=os.path.join(root_dir, 'data', 'assets', f'lexicon_under_{constants.MAX_WORD_LEN}.txt'), # f'lexicon_under_{cons.MAX_WORD_LEN}.txt'
        length_threshold=constants.MAX_WORD_LEN
    )

    # training_lexicon_dir = os.path.join(root_dir, 'viewer_models', 'recognize_word', 'assets', training_lexicon_name)
    # training_lexicon_file = os.path.join(training_lexicon_dir, 'data', f'{training_lexicon_name}.txt')
    #
    # # Split the input file content
    # _, training_lexicon = split_corpus(input_file=training_lexicon_file, percent=0.75, seed=42)
    #
    # # Define the output file paths
    # training_lexicon_output_file = os.path.join(training_lexicon_dir, 'training_lexicon.txt')
    #
    # # Store the split parts into their respective text files
    # store_lists_to_txt_file(training_lexicon, training_lexicon_output_file)
    #
    # # Generate the ngram tokens for the corpus
    # n = 7
    #
    # file_name = f'training_lexicon_{n}gram_combinations.csv'
    # tokens_output_file = os.path.join(training_lexicon_dir, file_name)
    # # store_lists_to_txt_file(training_lexicon_ngrams, tokens_output_file)
    # # print(f"{n}grams have been saved to {tokens_output_file}")
    #
    # all_combs = generate_ngram_tokens_for_training_lexicon(training_lexicon, sliding_win_len=n)
    # write_to_csv(all_combs, tokens_output_file)

    # TODO read in original dataset, get the max number of words in this dataset,
    #  and then create random words with different lengths specified by the const file, do this after my lunch
