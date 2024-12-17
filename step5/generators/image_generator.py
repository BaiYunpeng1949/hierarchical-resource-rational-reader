import os
import re
import json
import yaml
import random
import datetime
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from step5.utils import auxiliaries as aux
from step5.utils import constants as cons


def pull_words():
    """Pull words from the lexicon file, filter out non-alphabetic characters, and index them."""
    # Get the current root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Get the mode from the config yaml file
    with open(os.path.join(root_dir, "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    lexicon_text_file_name = config["resources"]["lexicon_filename"]+'.txt'
    lexicon_json_file_name = config["resources"]["lexicon_filename"] + '.json'
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lexicon_json_dir = os.path.join(root_dir, "data", "assets", lexicon_json_file_name)
    lexicon_text_dir = os.path.join(root_dir, "data", "assets", lexicon_text_file_name)
    lexicon_text = sorted(list(set(aux.load_txt(lexicon_text_dir))))
    # Read this json file into a python dictionary
    with open(lexicon_json_dir, 'r') as f:
        lexicon_json = json.load(f)

    # Filter out words with non-alphabetic characters and convert to lowercase
    lexicon_text = [word.lower() for word in lexicon_text if word.isalpha()]
    # # Define regex pattern to include common punctuation symbols
    # regex_pattern = r'^[a-zA-Z,.!?;:\"\']+$'
    # lexicon_text = [word.lower() for word in lexicon_text if word.isalpha() or (config["rl"]["mode"] == cons.SIMULATE and re.match(regex_pattern, word))]

    # Assign an index to each word and store it in a dictionary
    word_index_dict = {word: index for index, word in enumerate(lexicon_text)}

    return lexicon_json_file_name, lexicon_text, word_index_dict, lexicon_json, root_dir


def get_sentences() -> (dict, dict, str, str, str, str, str, str):
    """
    Get the sentences from the corpus txt file.
    Transfer new words to the lexicon txt and json files.
    But also reserve words with symbols for the simulation mode.
        1. For human users to read.
        2. For the model to read in chunks (separated by comma and period).
    :return: Two dictionaries containing the sentences (with and without symbols) and related file paths.
    """
    # Get the corpus path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Get the mode from the config yaml file
    with open(os.path.join(root_dir, "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    corpus_text_file_name = config["resources"]["corpus_filename"]+'.txt'
    corpus_json_file_name = config["resources"]["corpus_filename"] + '.json'
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus_json_dir = os.path.join(root_dir, "data", "assets", "corpus", corpus_json_file_name)
    corpus_text_dir = os.path.join(root_dir, "data", "assets", "corpus", corpus_text_file_name)

    # Read the sentences from the corpus file
    with open(corpus_text_dir, 'r') as f:
        sentences = f.readlines()

    # Define regex pattern to include common punctuation symbols
    regex_pattern = r'[^a-zA-Z\s,.!?;:\"\']'

    # Convert to dictionary and clean lines
    full_sentences_wo_punctuation_marks = {}
    full_sentences_w_punctuation_marks = {}
    for i, sentence in enumerate(sentences, 1):
        # Note: no need to transfer them to lower-cases because the Oculomotor Controller could deal with both cases
        clean_sentence_wo_symbols = re.sub(r'[^a-zA-Z\s]', '', sentence).strip().lower()
        clean_sentence_w_symbols = re.sub(regex_pattern, '', sentence).strip()
        full_sentences_wo_punctuation_marks[i] = clean_sentence_wo_symbols
        full_sentences_w_punctuation_marks[i] = clean_sentence_w_symbols

    # Check whether they are in the lexicon
    lexicon_text_file_name = config["resources"]["lexicon_filename"]+'.txt'
    lexicon_json_file_name = config["resources"]["lexicon_filename"] + '.json'
    lexicon_json_dir = os.path.join(root_dir, "data", "assets", lexicon_json_file_name)
    lexicon_text_dir = os.path.join(root_dir, "data", "assets", lexicon_text_file_name)
    with open(lexicon_text_dir, 'r') as file:
        lexicon = {word.strip().lower() for word in file.readlines()}

    # Check for words not in the lexicon --> write as a function
    def check_words_not_in_lexicon(full_sentences, lexicon):
        not_in_lexicon = set()
        for sentence in full_sentences.values():
            words = sentence.split()
            not_in_lexicon.update(word for word in words if word not in lexicon)
        return not_in_lexicon

    # Check for words not in the lexicon from full sentences without symbols --> replace with the actual OCR later
    not_in_lexicon = check_words_not_in_lexicon(full_sentences_wo_punctuation_marks, lexicon)

    # Determine to continue or not: only continue if there are words not in the lexicon
    if not_in_lexicon == set():
        print("All words are in the lexicon.")
        corpus_text_file_wo_punctuation_marks = corpus_text_file_name.replace('.txt',
                                                                              f'_wo{cons.PUNCTUATION_MARKS}.txt')
        corpus_text_dir_wo_punctuation_marks = os.path.join(root_dir, "data", "assets", "corpus",
                                                            corpus_text_file_wo_punctuation_marks)
        corpus_text_file_w_punctuation_marks = corpus_text_file_name.replace('.txt', f'_w{cons.PUNCTUATION_MARKS}.txt')
        corpus_text_dir_w_punctuation_marks = os.path.join(root_dir, "data", "assets", "corpus",
                                                           corpus_text_file_w_punctuation_marks)
        corpus_json_file_wo_punctuation_marks = corpus_json_file_name.replace('.json',
                                                                              f'_wo{cons.PUNCTUATION_MARKS}.json')
        corpus_json_dir_wo_punctuation_marks = os.path.join(root_dir, "data", "assets", "corpus",
                                                            corpus_json_file_wo_punctuation_marks)
        corpus_json_file_w_punctuation_marks = corpus_json_file_name.replace('.json',
                                                                             f'_w{cons.PUNCTUATION_MARKS}.json')
        corpus_json_dir_w_punctuation_marks = os.path.join(root_dir, "data", "assets", "corpus",
                                                           corpus_json_file_w_punctuation_marks)
        return (
            full_sentences_wo_punctuation_marks, full_sentences_w_punctuation_marks, corpus_text_file_name, corpus_json_file_name,
            corpus_json_dir, corpus_text_dir_wo_punctuation_marks, corpus_json_dir_wo_punctuation_marks, corpus_text_dir_w_punctuation_marks,
            corpus_json_dir_w_punctuation_marks
        )
    else:
        print(f"Words not in the lexicon: {not_in_lexicon}")
        # Display words that are not in the lexicon, remind modelers to add them
        while not_in_lexicon != set():
            for word in not_in_lexicon:
                print(f"In the process of adding the word '{word}' to the lexicon.")
                # Add the word to the lexicon lexicon_text_dir
                with open(lexicon_text_dir, 'a') as file:
                    file.write(f"{word}\n")
                # Add the word to the lexicon lexicon_json_dir according to different word lengths
                with open(lexicon_json_dir, 'r') as file:
                    lexicon_json = json.load(file)
                word_length = len(word)
                if word_length > cons.MAX_WORD_LEN:
                    raise ValueError(f"Word '{word}' is too long. The maximum word length is {cons.MAX_WORD_LEN}.")
                lexicon_json[str(word_length)].append(word)
                # Re-write the updated lexicon_json to the json file
                with open(lexicon_json_dir, 'w') as file:
                    json.dump(lexicon_json, file, indent=4)
                # Re-get the lexicon
                with open(lexicon_text_dir, 'r') as file:
                    lexicon = {word.strip().lower() for word in file.readlines()}

            # Check for words not in the lexicon again
            not_in_lexicon = check_words_not_in_lexicon(full_sentences_wo_punctuation_marks, lexicon)
            if not_in_lexicon == set():
                print("All words have been added to the lexicon now.")
                break

        # Make sure the lexicon text file all words are unique, does not have duplicates, remove duplicates
        with open(lexicon_text_dir, 'r') as file:
            lexicon_text = file.readlines()
        lexicon_text = list(set(lexicon_text))
        with open(lexicon_text_dir, 'w') as file:
            file.writelines(lexicon_text)

        # Write the updated lexicon_json to the json file
        with open(lexicon_json_dir, 'w') as file:
            json.dump(lexicon_json, file, indent=4)

        # Write the sentences without symbols to a new txt file
        corpus_text_file_wo_punctuation_marks = corpus_text_file_name.replace('.txt', f'_wo{cons.PUNCTUATION_MARKS}.txt')
        corpus_text_dir_wo_punctuation_marks = os.path.join(root_dir, "data", "assets", "corpus", corpus_text_file_wo_punctuation_marks)
        with open(corpus_text_dir_wo_punctuation_marks, 'w') as file:
            for sentence in full_sentences_wo_punctuation_marks.values():
                file.write(sentence + '\n')

        # Write the sentences with symbols to a new txt file
        corpus_text_file_w_punctuation_marks = corpus_text_file_name.replace('.txt', f'_w{cons.PUNCTUATION_MARKS}.txt')
        corpus_text_dir_w_punctuation_marks = os.path.join(root_dir, "data", "assets", "corpus", corpus_text_file_w_punctuation_marks)
        with open(corpus_text_dir_w_punctuation_marks, 'w') as file:
            for sentence in full_sentences_w_punctuation_marks.values():
                file.write(sentence + '\n')

        # Write the sentences without symbols to a new json file
        corpus_json_file_wo_punctuation_marks = corpus_json_file_name.replace('.json', f'_wo{cons.PUNCTUATION_MARKS}.json')
        corpus_json_dir_wo_punctuation_marks = os.path.join(root_dir, "data", "assets", "corpus", corpus_json_file_wo_punctuation_marks)
        with open(corpus_json_dir_wo_punctuation_marks, 'w') as file:
            json.dump(full_sentences_wo_punctuation_marks, file, indent=4)

        # Write the sentences with symbols to a new json file
        corpus_json_file_w_punctuation_marks = corpus_json_file_name.replace('.json', f'_w{cons.PUNCTUATION_MARKS}.json')
        corpus_json_dir_w_punctuation_marks = os.path.join(root_dir, "data", "assets", "corpus", corpus_json_file_w_punctuation_marks)
        with open(corpus_json_dir_w_punctuation_marks, 'w') as file:
            json.dump(full_sentences_w_punctuation_marks, file, indent=4)

        return (
            full_sentences_wo_punctuation_marks, full_sentences_w_punctuation_marks, corpus_text_file_name, corpus_json_file_name,
            corpus_json_dir, corpus_text_dir_wo_punctuation_marks, corpus_json_dir_wo_punctuation_marks, corpus_text_dir_w_punctuation_marks,
            corpus_json_dir_w_punctuation_marks
        )


def getsize(font, text):
    left, top, right, bottom = font.getbbox(text)
    width = right - left
    height = bottom - top
    return width, height, left, top, right, bottom


class ImageGenerator:
    def __init__(
            self,
            lexicon_file_name,
            image_size=cons.config["concrete_configs"]["img_size"],
            foveal_size=cons.config["concrete_configs"]["foveal_size"],
            parafoveal_size=cons.config["concrete_configs"]["parafoveal_size"],
            peripheral_size=cons.config["concrete_configs"]["peripheral_size"],
            word_size=cons.config["concrete_configs"]["word_size"],
            word_color="black",
            background_color="white",
            words: list = None,
            words_index_dict=None,
            words_dict: dict = cons.config["generator"]["dft_words"],
            num_words=cons.config["concrete_configs"]["num_words"],
            set_random_num_words=cons.config["concrete_configs"]["random num_words"],
            pos_sentences=cons.config["positions"]["sentence_center"],
            imgs_dir=None,
            json_dir=None,
            num_images=cons.config["concrete_configs"]["num_images"],
            test_size=cons.config["generator"]["test_size"],
            corpus_sentences_dict: dict = None,
            corpus_sentences_file_name: str = None,
            set_full_sentences: bool = False,
    ):
        self._lexicon_file_name = lexicon_file_name
        self._image_size = image_size
        self._foveal_size = foveal_size
        self._parafoveal_size = parafoveal_size
        self._peripheral_size = peripheral_size
        self._word_size = word_size
        self._word_color = word_color
        self._background_color = background_color
        self._words = words
        self._words_index_dict = words_index_dict
        self._words_dict = words_dict
        self._num_words = num_words
        self._set_random_num_words = set_random_num_words
        self._num_words_range = [cons.MIN_NUM_WORDS, cons.MAX_NUM_WORDS]
        self._pos_sentences = pos_sentences
        self._imgs_dir = imgs_dir
        self._json_dir = json_dir
        self._num_images = num_images
        self._test_size = test_size
        self._font_path = cons.FONT_PATH
        self._font = ImageFont.truetype(self._font_path, self._word_size)
        self._space_width = getsize(font=self._font, text=" ")[0]
        self._dict_english_letters = self._init_letter_size(font=self._font)
        self._corpus_sentences_dict = corpus_sentences_dict
        self._corpus_sentences_file_name = corpus_sentences_file_name
        self._set_full_sentences = set_full_sentences
        # Create a dictionary as a logger, which is going to store occurrences of words with different lengths
        self.gen_word_lengths_dist_dict = {str(i): [] for i in range(1, cons.MAX_WORD_LEN + 1)}

    def _create_directories(self, base_dir):
        current_date_time = datetime.datetime.now().strftime("%m-%d-%H-%M")
        folder_name = f'{current_date_time}_{self._num_images}_images_W{self._image_size[0]}H{self._image_size[1]}WS{self._word_size}_{self._lexicon_file_name.split(".")[0]}'
        save_path = os.path.join(base_dir, folder_name)
        train_save_path = os.path.join(save_path, cons.TRAIN)
        os.makedirs(train_save_path, exist_ok=True)
        test_save_path = os.path.join(save_path, cons.TEST)
        os.makedirs(test_save_path, exist_ok=True)
        simulate_save_path = os.path.join(save_path, cons.SIMULATE)
        os.makedirs(simulate_save_path, exist_ok=True)
        return save_path, train_save_path, test_save_path, simulate_save_path

    @staticmethod
    def getsize(font, text):
        left, top, right, bottom = font.getbbox(text)
        width = right - left
        height = bottom - top
        return width, height, left, top, right, bottom

    # def _init_letter_size(self, font, lowercase=True):
    #     """Get the size of the letter."""
    #     dict_letter_size = {}
    #     if lowercase:
    #         for letter in 'abcdefghijklmnopqrstuvwxyz':  # 'string.ascii_letters' generates 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    #             letter_width, letter_height, left, top, right, bottom = self.getsize(font, letter)
    #             dict_letter_size[letter] = (letter_width, letter_height, left, top, right, bottom)
    #     else:
    #         raise NotImplementedError
    #
    #     return dict_letter_size

    def _init_letter_size(self, font):
        """Get the size of the letter."""
        dict_letter_size = {}
        for letter in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
            letter_width, letter_height, left, top, right, bottom = self.getsize(font, letter)
            dict_letter_size[letter] = (letter_width, letter_height, left, top, right, bottom)
        return dict_letter_size

    @staticmethod
    def split_text_into_lines(selected_words, draw, font, image_width, image_height):
        # words = concatenated_words.split(' ')
        words = selected_words
        lines = []
        line_height = 0
        current_line = ''

        for word in words:
            # Check the width of the current line with the new word added
            test_line = ' '.join([current_line, word]).strip()
            # line_width, line_height = draw.textsize(test_line, font=font)
            line_width, line_height, left, top, right, bottom = getsize(font, test_line)

            # If the line is too wide, start a new line
            if line_width >= image_width:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line

        # Add the last line
        lines.append(current_line)

        # Calculate the total height of the text block
        total_height = len(lines) * max([getsize(font, line)[1] for line in lines])

        return lines, line_height, len(lines), total_height

    def _get_letter_boxes(self, word, font, start_x, start_y):
        """Get bounding boxes for each letter in the given word, considering the relative positions."""
        boxes = []
        letters = list(word)
        current_x = start_x
        for character in word:  # Character includes both letters and symbols
            # First, tell whether the word contains a symbol/punctuation mark
            if character in cons.SYMBOLS:
                width, height, left, top, right, bottom = getsize(font, character)
                current_x += width  # Move to the start of the next letter
            elif character in self._dict_english_letters:    # If the letter is one of the lowercase letters
                # width, height, left, top, right, bottom = self.getsize(font, letter)
                width, height, left, top, right, bottom = self._dict_english_letters[character]

                # Adjust the box position based on the relative top and bottom from getsize
                box = (current_x + left, start_y + top, current_x + right, start_y + bottom)

                boxes.append({
                    cons.md["letter width"]: width,
                    cons.md["letter height"]: height,
                    cons.md["letter left"]: left,
                    cons.md["letter top"]: top,
                    cons.md["letter right"]: right,
                    cons.md["letter bottom"]: bottom,
                    cons.md["letter box left"]: current_x + left,
                    cons.md["letter box top"]: start_y + top,
                    cons.md["letter box right"]: current_x + right,
                    cons.md["letter box bottom"]: start_y + bottom,
                })
                current_x += width  # Move to the start of the next letter
        return letters, boxes

    def _generate_images_and_metadata(
            self,
            imgs_save_path,
            json_file,
            mode: str = cons.TRAIN,   # 'train', 'test', 'simulate'
    ):
        """Generate images and metadata."""

        if mode == cons.TRAIN:
            num_images = int(self._num_images * (1 - self._test_size))
            metadata_flag = cons.TRAIN
        elif mode == cons.TEST:
            num_images = int(self._num_images * self._test_size)
            metadata_flag = cons.TEST
        elif mode == cons.SIMULATE:
            num_images = len(self._corpus_sentences_dict)
            metadata_flag = cons.SIMULATE
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        metadata = {
            cons.md["config"]: {
                cons.md["domain"]: metadata_flag,
                cons.md["lexicon file name"]: self._lexicon_file_name,
                cons.md["corpus file name"]: self._corpus_sentences_file_name,
                cons.md["corpus"]: self._corpus_sentences_dict,
                cons.md["img size"]: self._image_size,
                cons.md["foveal size"]: self._foveal_size,
                cons.md["parafoveal size"]: self._parafoveal_size,
                cons.md["peripheral size"]: self._peripheral_size,
                cons.md["word size"]: self._word_size,
                cons.md["word color"]: self._word_color,
                cons.md["background color"]: self._background_color,
                cons.md["words"]: [],  # self._words,
                cons.md["num words"]: f"fixed to {self._num_words}" if self._set_random_num_words is False else "random",
                cons.md["pos sentences"]: self._pos_sentences,
                cons.md["num images"]: num_images,
            },
            cons.md["images"]: []
        }

        total_words_count = len(self._words)

        font = ImageFont.truetype(self._font_path, self._word_size)

        for i in range(num_images):
            img = Image.new("RGB", self._image_size, self._background_color)
            draw = ImageDraw.Draw(img)

            selected_words = []

            # Configure image environment according to the mode
            if mode == cons.TRAIN or mode == cons.TEST:
                # Randomize the number of words on the image
                if self._set_random_num_words:
                    self._num_words = random.randint(*self._num_words_range)
                else:
                    if self._set_full_sentences:
                        self._num_words = len(self._words)
            elif mode == cons.SIMULATE:
                # Get the sentence from the corpus
                sentence = self._corpus_sentences_dict[i + 1]  # The index starts from 1
                # Get the words from the sentence
                selected_words = sentence.split()
                # Get the number of words in the sentence
                self._num_words = len(selected_words)
            else:
                raise ValueError(f"Unknown mode '{mode}'")

            # Randomly select 'num_words' words and concatenate them with spaces
            for word_idx in range(self._num_words):

                if mode == cons.SIMULATE:
                    # Output words one by one
                    word = selected_words[word_idx]
                    word_length = len(word)
                else:
                    # Randomly sample the word length first
                    word_length = random.choice(list(self._words_dict.keys()))
                    # Randomly sample a word from the words of the selected length
                    word = random.choice(self._words_dict[word_length])
                    # Guarantee that the word is not already in the selected words
                    break_counter = 0   # Prevent infinite resampling when there are too many words on the image
                    while word in selected_words:
                        if break_counter >= cons.TEN:
                            print(f"Breaking the loop after {break_counter} iterations.")
                            break
                        print(f"Word '{word}' already in selected words. Resampling...")
                        word = random.choice(self._words_dict[word_length])
                        break_counter += 1

                    # Append the word to the list of selected words and the dictionary of selected words
                    selected_words.append(word)

                self.gen_word_lengths_dist_dict[str(word_length)].append(word)

            # selected_words = [random.choice(list(self._words_dict.keys())) for _ in range(self._num_words)]
            selected_words_indexes = []
            selected_words_norm_indexes = []

            # Split the text into lines and get the top position to start drawing
            lines, line_height, num_lines, total_height = self.split_text_into_lines(
                selected_words,
                draw,
                font,
                self._image_size[0],
                self._image_size[1],
            )

            # Get the maximum line width
            line_widths = [getsize(font, line)[0] for line in lines]
            max_line_width = max(line_widths)
            # Initialize the x and y positions for the first word by traversing all lines first
            max_x = self._image_size[0] - max_line_width
            max_y = self._image_size[1] - total_height
            if self._pos_sentences == cons.config["positions"]["sentence_center"]:
                x_norm = 0.5
                y_norm = 0.5
            elif self._pos_sentences == cons.config["positions"]["sentence_random"]:
                x_norm = random.uniform(0, 1)
                y_norm = random.uniform(0, 1)
            else:
                raise ValueError(f"Unknown position '{self._pos_sentences}'")
            x_init = x_norm * max_x
            y_init = y_norm * max_y

            # Get the max offsets for the letters for both x and y directions across all words and letters
            max_font_offset_x = max([getsize(font, word)[2] for word in selected_words])
            max_font_offset_y = max([getsize(font, word)[3] for word in selected_words])
            max_font_height = max([getsize(font, word)[1] for word in selected_words])

            words_metadata = []  # This will store the metadata for each word
            for idx, line in enumerate(lines):
                words_in_line = line.split(' ')
                line_width, line_height, _, _, _, _ = getsize(font, line)
                x = x_init - max_font_offset_x  # Initial x position for the line
                # y = y_init + idx * max_font_height - max_font_offset_y  # idx = num_lines - 1
                y = y_init + idx * max_font_height

                for word in words_in_line:      # These words are possibly with punctuation marks
                    word_width, word_height, left, top, right, bottom = getsize(font, word)
                    # Draw the word and update x position for the next word
                    draw.text((x, y), word, fill=self._word_color, font=font)

                    # Get the word without punctuation marks and in the lower-case format
                    _word = re.sub(r'[^a-zA-Z\s]', '', word).strip().lower()
                    if word != _word:
                        print(f"The word is '{word}' and the word without punctuation marks or has a different case is '{_word}'.")

                    # Calculate the normalized index for the word
                    index = self._words_index_dict[_word]  # Get the index of the word
                    normalized_index = aux.normalise(index, 0, total_words_count-1, -1, 1)  # Normalize the index
                    selected_words_indexes.append(index)  # Store the index
                    selected_words_norm_indexes.append(normalized_index)  # Store the normalized index

                    # Get each letter's bounding box (positions)
                    letters_metadata = []  # This will store the metadata for each letter
                    letters, letter_boxes = self._get_letter_boxes(word, font, x, y)

                    for i_letter, letter_box in enumerate(letter_boxes):
                        letters_metadata.append({
                            cons.md["letter index"]: i_letter,
                            cons.md["letters"]: letters[i_letter],
                            cons.md["letter boxes"]: letter_boxes[i_letter],
                        })

                    words_metadata.append({
                        cons.md["word"]: word,
                        cons.md["word length"]: len(word),
                        cons.md["visuospatial info"]: {
                            cons.md["line index"]: idx,
                            cons.md["lines number"]: num_lines,
                            cons.md["word index in line"]: words_in_line.index(word),
                            cons.md["words number in line"]: len(words_in_line),
                        },
                        cons.md["position"]: {
                            "x": x, "y": y, "x_norm": x_norm, "y_norm": y_norm, "x_init": x_init, "y_init": y_init,
                            "max_x_offset": max_font_offset_x, "max_y_offset": max_font_offset_y,
                            "word_width": word_width, "word_height": word_height,
                            # "line_height": line_height,
                            # "y_lines": idx * line_height,
                            "line_height": max_font_height,
                            "y_lines": idx * max_font_height,
                        },
                        cons.md["index"]: index,
                        cons.md["normalized index"]: normalized_index,
                        cons.md["letters metadata"]: letters_metadata,
                    })
                    x += word_width + self._space_width  # Add a space after each word

            # Save the image
            image_filename = f"image_{i}.jpg"
            save_filename = os.path.join(imgs_save_path, image_filename)
            img.save(save_filename)

            # Add image information to metadata
            metadata['images'].append({
                cons.md["image index"]: i,
                cons.md["num words"]: self._num_words,
                cons.md["filename"]: image_filename,
                cons.md["words metadata"]: words_metadata,
                cons.md["selected words"]: selected_words,
                cons.md["selected words indexes"]: selected_words_indexes,
                cons.md["selected words norm indexes"]: selected_words_norm_indexes,
            })

            print(f"Image saved to {save_filename}")

            # Save metadata to JSON file
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"Metadata saved to {json_file}")

    @staticmethod
    def _compare_datasets(
            json_train_file,
            json_test_file,
            train_logger,
            test_logger,
            comparison_file_path,
    ):
        """Compare the training and testing datasets and report the differences in a text file."""
        # json_test_file = r"D:\Users\91584\PycharmProjects\reading-model\step5\data\gen_envs\03-21_11-54_1000_images\metadata.json"
        # json_train_file = r"D:\Users\91584\PycharmProjects\reading-model\step6\data\gen_envs\03-25_15-20_2000_images\train\metadata.json"

        # Open the JSON files
        with open(json_train_file, 'r') as f:
            train_data = json.load(f)
        with open(json_test_file, 'r') as f:
            test_data = json.load(f)

        # Compare the training and testing datasets
        # Function to extract three-word combinations as tuples
        def extract_combinations(data):
            combinations = set()
            for image in data.get(cons.md["images"], []):
                selected_words = image.get(cons.md["selected words"], [])
                if len(selected_words) >= 3:  # Ensure there are at least 3 words
                    # Create a tuple of the three-word combination and add to the set
                    combinations.add(tuple(selected_words[:3]))
            return combinations

        # Extract three-word combinations from both datasets
        train_combinations = extract_combinations(train_data)
        test_combinations = extract_combinations(test_data)

        # Calculate intersection of three-word combinations in both datasets
        common_combinations = train_combinations.intersection(test_combinations)

        # From the logger output the occurrences of words with different lengths
        train_logger_occurrences = {k: len(v) for k, v in train_logger.items()}
        test_logger_occurrences = {k: len(v) for k, v in test_logger.items()}

        # Write the comparison to a text file
        comparison_file = os.path.join(comparison_file_path, "train_test_comparison.txt")
        with open(comparison_file, 'w') as f:
            f.write(f"Number of unique three-word combinations in training dataset: {len(train_combinations)}\n")
            f.write(f"Number of unique three-word combinations in testing dataset: {len(test_combinations)}\n")
            f.write(f"Number of common three-word combinations in both datasets: {len(common_combinations)}\n")
            f.write(f"Common three-word combinations: {common_combinations}\n")
            f.write(f"Occurrences of words with different lengths in the training dataset: {train_logger_occurrences}\n")
            f.write(f"Occurrences of words with different lengths in the testing dataset: {test_logger_occurrences}\n")
            f.write(f"Train logger: {train_logger}\n")
            f.write(f"Test logger: {test_logger}\n")

    def generate_data(self):
        """Generate images."""
        _, imgs_train_save_path, imgs_test_save_path, imgs_sim_save_path = self._create_directories(self._imgs_dir)
        save_path, json_train_save_path, json_test_save_path, json_sim_save_path = self._create_directories(self._json_dir)

        json_train_file = os.path.join(json_train_save_path, cons.MD_FILE_NAME)
        json_test_file = os.path.join(json_test_save_path, cons.MD_FILE_NAME)
        json_sim_file = os.path.join(json_sim_save_path, cons.MD_FILE_NAME)

        self._generate_images_and_metadata(imgs_train_save_path, json_train_file, mode=cons.TRAIN)
        train_logger = self.gen_word_lengths_dist_dict.copy()
        self._generate_images_and_metadata(imgs_test_save_path, json_test_file, mode=cons.TEST)
        test_logger = self.gen_word_lengths_dist_dict.copy()
        self._generate_images_and_metadata(imgs_sim_save_path, json_sim_file, mode=cons.SIMULATE)

        # Compare the training and testing datasets and report the differences in a text file
        self._compare_datasets(
            json_train_file=json_train_file,
            json_test_file=json_test_file,
            train_logger=train_logger,
            test_logger=test_logger,
            comparison_file_path=save_path
        )


if __name__ == "__main__":
    # Pull words from the lexicon file to initialize the file first if there are no lexicon files
    _, _, _, _, _ = pull_words()

    # Get reading materials
    (_sentences_wo_punctuation_marks, _sentences_w_punctuation_marks_and_upper_cases, _corpus_filename, _corpus_json_file_name, _corpus_json_file_dir,
     _corpus_text_dir_wo_punctuation_marks, _corpus_json_dir_wo_punctuation_marks, _corpus_text_dir_w_punctuation_marks, _corpus_json_dir_w_punctuation_marks) = get_sentences()

    # Refresh the lexicon file after update by the sentences in the corpus
    _lex_file_name, _words, _words_index_dict, _words_dict, _root_dir = pull_words()

    # Get path dirs
    _img_dir = os.path.join(_root_dir, "data", "gen_envs")
    _json_dir = os.path.join(_root_dir, "data", "gen_envs")

    # Initialize the generator
    generator = ImageGenerator(
        lexicon_file_name=_lex_file_name,
        image_size=cons.config["concrete_configs"]["img_size"],
        word_size=cons.config["concrete_configs"]["word_size"],
        foveal_size=cons.config["concrete_configs"]["foveal_size"],
        parafoveal_size=cons.config["concrete_configs"]["parafoveal_size"],
        peripheral_size=cons.config["concrete_configs"]["peripheral_size"],
        words=cons.config['generator']['dft_words'][1],  # The default words, full sentences  #_words, # Random words
        words_index_dict=_words_index_dict,
        words_dict=_words_dict,
        num_words=cons.config["concrete_configs"]["num_words"],
        set_random_num_words=cons.config["concrete_configs"]["random num_words"],
        pos_sentences=cons.config["positions"]["sentence_center"],  # sentence_random or sentence_center
        imgs_dir=_img_dir,
        json_dir=_json_dir,
        num_images=cons.config["concrete_configs"]["num_images"],
        test_size=cons.config["generator"]["test_size"],
        corpus_sentences_dict=_sentences_w_punctuation_marks_and_upper_cases,
        corpus_sentences_file_name=_corpus_filename,
        set_full_sentences=cons.config["concrete_configs"]["corpus"],
    )
    # Generate images
    generator.generate_data()
