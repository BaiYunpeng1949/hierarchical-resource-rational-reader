import os
import re
import cv2
import json
import yaml
import random
import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from matplotlib import pyplot as plt
# from psychopy import visual, core

from step5.utils import auxiliaries as aux
from step5.utils import constants as cons


def pull_words():
    """Pull words from the lexicon file and index them."""
    # Get the current root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Get the mode from the config yaml file
    with open(os.path.join(root_dir, "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    lexicon_text_file_name = config["resources"]["lexicon_filename"] + '.txt'
    lexicon_json_file_name = config["resources"]["lexicon_filename"] + '.json'
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lexicon_json_dir = os.path.join(root_dir, "data", "assets", lexicon_json_file_name)
    lexicon_text_dir = os.path.join(root_dir, "data", "assets", lexicon_text_file_name)
    lexicon_text = sorted(list(set(aux.load_txt(lexicon_text_dir))))
    # Read this json file into a python dictionary
    with open(lexicon_json_dir, 'r') as f:
        lexicon_json = json.load(f)

    # No filtering out words; allow any characters
    # Assign an index to each word and store it in a dictionary
    word_index_dict = {word: index for index, word in enumerate(lexicon_text)}

    return lexicon_json_file_name, lexicon_text, word_index_dict, lexicon_json, root_dir


def get_sentences() -> (dict, str, str, str, str, str):
    """
    Get the sentences from the corpus txt file.
    Transfer new words to the lexicon txt and json files.
    Reserve words with symbols for the simulation mode:
        1. For human users to read.
        2. For the model to read in chunks (separated by comma and period).
    :return: A dictionary containing the sentences with symbols and related file paths.
    """
    # Get the corpus path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Get the mode from the config yaml file
    with open(os.path.join(root_dir, "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    corpus_text_file_name = config["resources"]["corpus_filename"] + '.txt'
    corpus_json_file_name = config["resources"]["corpus_filename"] + '.json'
    corpus_json_dir = os.path.join(root_dir, "data", "assets", "corpus", corpus_json_file_name)
    corpus_text_dir = os.path.join(root_dir, "data", "assets", "corpus", corpus_text_file_name)

    # Read the sentences from the corpus file
    # with open(corpus_text_dir, 'r') as f:
    with open(corpus_text_dir, encoding="utf-8") as f:
        sentences = f.readlines()

    # Define regex pattern to include all ASCII characters including punctuation
    regex_pattern_all_ascii = r'[^\x20-\x7E]'

    # Convert to dictionary and clean lines
    full_sentences_w_punctuation_marks = {}
    for i, sentence in enumerate(sentences, 1):
        clean_sentence_w_symbols = re.sub(regex_pattern_all_ascii, '',
                                          sentence).strip()  # Allows all printable ASCII characters including spaces
        full_sentences_w_punctuation_marks[i] = clean_sentence_w_symbols

    # Check whether they are in the lexicon
    lexicon_text_file_name = config["resources"]["lexicon_filename"] + '.txt'
    lexicon_json_file_name = config["resources"]["lexicon_filename"] + '.json'
    lexicon_json_dir = os.path.join(root_dir, "data", "assets", lexicon_json_file_name)
    lexicon_text_dir = os.path.join(root_dir, "data", "assets", lexicon_text_file_name)
    with open(lexicon_text_dir, 'r') as file:
        lexicon = {word.strip() for word in file.readlines()}

    # Check for words not in the lexicon --> write as a function
    def check_words_not_in_lexicon(full_sentences, lexicon):
        not_in_lexicon = set()
        for sentence in full_sentences.values():
            words = sentence.split()
            not_in_lexicon.update(word for word in words if word not in lexicon)
        return not_in_lexicon

    # Check for words not in the lexicon from full sentences with symbols
    not_in_lexicon = check_words_not_in_lexicon(full_sentences_w_punctuation_marks, lexicon)

    # Determine to continue or not: only continue if there are words not in the lexicon
    if not_in_lexicon == set():
        print("All words are in the lexicon.")
        corpus_text_file_w_punctuation_marks = corpus_text_file_name.replace('.txt', f'_w{cons.PUNCTUATION_MARKS}.txt')
        corpus_text_dir_w_punctuation_marks = os.path.join(root_dir, "data", "assets", "corpus",
                                                           corpus_text_file_w_punctuation_marks)
        corpus_json_file_w_punctuation_marks = corpus_json_file_name.replace('.json',
                                                                             f'_w{cons.PUNCTUATION_MARKS}.json')
        corpus_json_dir_w_punctuation_marks = os.path.join(root_dir, "data", "assets", "corpus",
                                                           corpus_json_file_w_punctuation_marks)
        return (
            full_sentences_w_punctuation_marks, corpus_text_file_name, corpus_json_file_name,
            corpus_json_dir, corpus_text_dir_w_punctuation_marks, corpus_json_dir_w_punctuation_marks
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
                    lexicon = {word.strip() for word in file.readlines()}

            # Check for words not in the lexicon again
            not_in_lexicon = check_words_not_in_lexicon(full_sentences_w_punctuation_marks, lexicon)
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

        # Write the sentences with symbols to a new txt file
        corpus_text_file_w_punctuation_marks = corpus_text_file_name.replace('.txt', f'_w{cons.PUNCTUATION_MARKS}.txt')
        corpus_text_dir_w_punctuation_marks = os.path.join(root_dir, "data", "assets", "corpus",
                                                           corpus_text_file_w_punctuation_marks)
        with open(corpus_text_dir_w_punctuation_marks, 'w') as file:
            for sentence in full_sentences_w_punctuation_marks.values():
                file.write(sentence + '\n')

        # Write the sentences with symbols to a new json file
        corpus_json_file_w_punctuation_marks = corpus_json_file_name.replace('.json',
                                                                             f'_w{cons.PUNCTUATION_MARKS}.json')
        corpus_json_dir_w_punctuation_marks = os.path.join(root_dir, "data", "assets", "corpus",
                                                           corpus_json_file_w_punctuation_marks)
        with open(corpus_json_dir_w_punctuation_marks, 'w') as file:
            json.dump(full_sentences_w_punctuation_marks, file, indent=4)

        return (
            full_sentences_w_punctuation_marks, corpus_text_file_name, corpus_json_file_name,
            corpus_json_dir, corpus_text_dir_w_punctuation_marks, corpus_json_dir_w_punctuation_marks
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
            training_foveal_and_peripheral_size=cons.config["concrete_configs"]["training_foveal_and_peripheral_size"],
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
        self._training_foveal_and_peripheral_size = training_foveal_and_peripheral_size
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
        self._corpus_stimulus_sentences_dict = corpus_sentences_dict
        self._corpus_sentences_file_name = corpus_sentences_file_name
        self._set_full_sentences = set_full_sentences
        # Create a dictionary as a logger, which is going to store occurrences of words with different lengths
        self.gen_word_lengths_dist_dict = {str(i): [] for i in range(1, cons.MAX_WORD_LEN + 1)}

    def _create_directories(self, base_dir):
        current_date_time = datetime.datetime.now().strftime("%m_%d_%H_%M")
        # folder_name = f'{current_date_time}_{self._num_images}_images_W{self._image_size[0]}H{self._image_size[1]}WS{self._word_size}_{self._lexicon_file_name.split(".")[0]}'
        folder_name = f'{current_date_time}_{self._num_images}_images_W{self._image_size[0]}H{self._image_size[1]}WS{self._word_size}_LS{int(cons.LINE_SPACING)}_MARGIN{cons.LEFT_MARGIN}'
        save_path = os.path.join(base_dir, folder_name)
        train_save_path = os.path.join(save_path, cons.TRAIN)
        os.makedirs(train_save_path, exist_ok=True)
        test_save_path = os.path.join(save_path, cons.TEST)
        os.makedirs(test_save_path, exist_ok=True)
        simulate_save_path = os.path.join(save_path, cons.SIMULATE)
        os.makedirs(simulate_save_path, exist_ok=True)
        bounding_boxed_stimulus_save_path = os.path.join(save_path, cons.BBOX_STIM)
        os.makedirs(bounding_boxed_stimulus_save_path, exist_ok=True)
        return save_path, train_save_path, test_save_path, simulate_save_path, bounding_boxed_stimulus_save_path

    @staticmethod
    def getsize(font, text):
        left, top, right, bottom = font.getbbox(text)
        width = right - left
        height = bottom - top
        return width, height, left, top, right, bottom

    @staticmethod
    def _generate_masked_and_downsampled_peripheral_view(
            original_stimuli_gray_pixels,
            bbox,
            downsample_image_width,
            downsample_image_height,
    ):
        """
        Generate a masked and downsampled peripheral view and store it in the metadata.
        """
        # Create a copy of the original image and apply the mask
        masked_image = original_stimuli_gray_pixels.copy()

        # Extract and convert bounding box coordinates to integers
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = map(int, bbox)

        # Set all pixels inside the bounding box to black (pixel value 0)
        masked_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0

        # Downsample the masked peripheral view
        downsampled_peripheral_view = cv2.resize(
            masked_image,
            (downsample_image_width, downsample_image_height),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalise the downsampled peripheral view to the range [-1, 1]
        normalised_downsampled_peripheral_view = aux.normalise(downsampled_peripheral_view, 0, 255,  -1, 1)

        return normalised_downsampled_peripheral_view

    @staticmethod
    def _generate_masked_and_downsampled_plain_peripheral_view(
            original_stimuli_gray_pixels,
            bbox,
            downsample_image_width,
            downsample_image_height,
    ):
        """
        Generate a downsampled peripheral view with a solid black box on a plain white background.
        """
        # Create a all 255 (white) image with the same size as the original image
        plain_image = np.full_like(original_stimuli_gray_pixels, 255)

        # Extract and convert bounding box coordinates to integers
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = map(int, bbox)

        # Set all pixels inside the bounding box to black (pixel value 0)
        plain_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0

        # Downsample the masked peripheral view
        downsampled_peripheral_view = cv2.resize(
            plain_image,
            (downsample_image_width, downsample_image_height),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalise the downsampled peripheral view to the range [-1, 1]
        normalised_downsampled_peripheral_view = aux.normalise(downsampled_peripheral_view, 0, 255, -1, 1)

        return normalised_downsampled_peripheral_view

    def _generate_foveal_patches(self, img, word_bbox, offset=5):
        """
        Generate foveal patches for each word in the image and return the normalized patch
        and the relative bounding box.

        :param img: The PIL image object containing the text.
        :param word_bbox: The bounding box of the word in the original image.
        :param offset: The number of pixels to expand the foveal patch in each direction.
        :return: A tuple containing the normalized foveal patch and the relative bounding box.
        """
        img_np = np.array(img)

        # Calculate foveal patch coordinates with an offset
        foveal_left = max(0, int(word_bbox[0] - self._foveal_size[0] / 2) - offset)
        foveal_right = min(img_np.shape[1], int(word_bbox[2] + self._foveal_size[0] / 2) + offset)
        foveal_top = max(0, int(word_bbox[1] - self._foveal_size[1] / 2) - offset)
        foveal_bottom = min(img_np.shape[0], int(word_bbox[3] + self._foveal_size[1] / 2) + offset)

        # Extract the foveal patch
        foveal_patch = img_np[foveal_top:foveal_bottom, foveal_left:foveal_right]

        # Normalize the foveal patch to the range [-1, 1]
        normalized_foveal_patch = aux.normalise(foveal_patch, 0, 255, -1, 1)

        # Calculate the relative bounding box within the foveal patch
        relative_left = word_bbox[0] - foveal_left
        relative_top = word_bbox[1] - foveal_top
        relative_right = word_bbox[2] - foveal_left
        relative_bottom = word_bbox[3] - foveal_top

        relative_bbox = (relative_left, relative_top, relative_right, relative_bottom)

        # Return the normalized foveal patch and the relative bounding box
        return normalized_foveal_patch, relative_bbox

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
            width, height, left, top, right, bottom = getsize(font, character)
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

    @staticmethod
    def _get_total_height(max_height, num_lines):
        return num_lines * (max_height + cons.LINE_SPACING) - cons.LINE_SPACING

    def _get_bounding_boxes(self, word_width, line_height, word_index_in_line, max_word_index_in_line, x, y):

        if word_index_in_line == 0:
            left_top_x = x
            left_top_y = y - cons.LINE_SPACING / 2
        else:
            left_top_x = x - self._space_width / 2
            left_top_y = y - cons.LINE_SPACING / 2

        if word_index_in_line == max_word_index_in_line:
            right_bottom_x = x + word_width
            right_bottom_y = y + line_height + cons.LINE_SPACING / 2
        else:
            right_bottom_x = x + word_width + self._space_width / 2
            right_bottom_y = y + line_height + cons.LINE_SPACING / 2

        return [left_top_x, left_top_y, right_bottom_x, right_bottom_y]  # left, top, right, bottom

    def _generate_images_and_metadata(
            self,
            imgs_save_path: str = None,
            bbox_stim_save_path: str = None,
            json_file: str = None,
            mode: str = cons.TRAIN,  # 'train', 'test', 'simulate'
    ):
        """Generate images and metadata."""

        if mode == cons.TRAIN:
            num_images = int(self._num_images * (1 - self._test_size))
            metadata_flag = cons.TRAIN
        elif mode == cons.TEST:
            num_images = int(self._num_images * self._test_size)
            metadata_flag = cons.TEST
        elif mode == cons.SIMULATE:
            num_images = len(self._corpus_stimulus_sentences_dict)
            metadata_flag = cons.SIMULATE
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        metadata = {
            cons.md["config"]: {
                cons.md["domain"]: metadata_flag,
                cons.md["lexicon file name"]: self._lexicon_file_name,
                cons.md["corpus file name"]: self._corpus_sentences_file_name,
                cons.md["corpus"]: self._corpus_stimulus_sentences_dict,
                cons.md["img size"]: self._image_size,
                cons.md["foveal size"]: self._foveal_size,
                cons.md["parafoveal size"]: self._parafoveal_size,
                cons.md["peripheral size"]: self._peripheral_size,
                cons.md["training foveal and peripheral size"]: self._training_foveal_and_peripheral_size,
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
            # Create an 8-bit grayscale image (to reduce anti-aliasing)
            img = Image.new("L", self._image_size, 255)  # 'L' mode is 8-bit pixels, black and white
            draw = ImageDraw.Draw(img)
            words_in_stimulus = []

            if mode == cons.SIMULATE:
                # img_bbox = Image.new("L", self._image_size, 255)  # 'L' mode is 8-bit pixels, black and white
                img_bbox = Image.new("RGB", self._image_size, "white")  # 'RGB' mode to support colored drawing
                draw_bbox = ImageDraw.Draw(img_bbox)
            else:
                img_bbox = None
                draw_bbox = None

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
                stimulus_sentence = self._corpus_stimulus_sentences_dict[i + 1]  # The index starts from 1
                # Get the words from the sentence
                words_in_stimulus = stimulus_sentence.split()
                # Get the number of words in the sentence
                self._num_words = len(words_in_stimulus)
            else:
                raise ValueError(f"Unknown mode '{mode}'")

            # Randomly select 'num_words' words and concatenate them with spaces
            for word_idx_in_line in range(self._num_words):
                if mode == cons.SIMULATE:
                    # Output words one by one
                    word = words_in_stimulus[word_idx_in_line]
                    word_length = len(word)
                else:
                    # Randomly sample the word length first
                    word_length = random.choice(list(self._words_dict.keys()))
                    # Randomly sample a word from the words of the selected length
                    word = random.choice(self._words_dict[word_length])
                    # Guarantee that the word is not already in the selected words
                    break_counter = 0  # Prevent infinite resampling when there are too many words on the image
                    while word in words_in_stimulus:  # Ensure word is valid
                        if break_counter >= cons.TEN:
                            print(f"Breaking the loop after {break_counter} iterations.")
                            break
                        print(f"Word '{word}' already in selected words. Resampling...")
                        word = random.choice(self._words_dict[word_length])
                        break_counter += 1

                    # Append the word to the list of selected words and the dictionary of selected words
                    words_in_stimulus.append(word)

                self.gen_word_lengths_dist_dict[str(word_length)].append(word)

            selected_words_indexes = []
            selected_words_norm_indexes = []

            # Get the drawable width on the stimulus image
            drawable_width = self._image_size[0] - cons.LEFT_MARGIN - cons.RIGHT_MARGIN

            # Split the text into lines and get the top position to start drawing
            lines, line_height, num_lines, total_net_height = self.split_text_into_lines(
                words_in_stimulus, draw, font, drawable_width, self._image_size[1],
            )

            # Get the maximum line width
            line_widths = [getsize(font, line)[0] for line in lines]
            max_line_width = max(line_widths)

            # Get the max offsets for the letters for both x and y directions across all words and letters
            max_font_offset_x = max([getsize(font, word)[2] for word in words_in_stimulus])
            max_font_offset_y = max([getsize(font, word)[3] for word in words_in_stimulus])
            max_font_height = max([getsize(font, word)[1] for word in words_in_stimulus])

            # Center the text horizontally based on the drawable area
            x_norm, y_norm = 0.5, 0.5
            total_height = self._get_total_height(max_height=max_font_height, num_lines=num_lines)
            x_init = cons.LEFT_MARGIN + (drawable_width - max_line_width) / 2
            y_init = (self._image_size[1] - total_height) / 2  # Center vertically

            words_metadata = []  # This will store the metadata for each word

            # Drawing lines with margins
            for line_idx, line in enumerate(lines):
                words_in_line = line.split(' ')
                x = x_init
                y = y_init + line_idx * (line_height + cons.LINE_SPACING)

                for word_idx_in_line, word in enumerate(words_in_line):
                    # for word in words_in_line:
                    draw.text((x, y), word, fill=0, font=font)  # Drawing text in black (0)
                    word_width, word_height, left, top, right, bottom = getsize(font, word)

                    # Get each word's bounding box (positions)
                    word_bounding_box = self._get_bounding_boxes(
                        word_width, line_height, word_idx_in_line, len(words_in_line) - 1, x, y
                    )

                    if mode == cons.SIMULATE:
                        # Draw bounding boxes for the words
                        draw_bbox.rectangle(
                            xy=[(word_bounding_box[0], word_bounding_box[1]),
                                (word_bounding_box[2], word_bounding_box[3])],
                            outline="red", width=1
                        )
                        draw_bbox.text((x, y), word, fill=0, font=font)  # Drawing text in black (0)

                    # Get the word's index without filtering
                    if word in self._words_index_dict:
                        index = self._words_index_dict[word]  # Get the index of the word
                        normalized_index = aux.normalise(index, 0, total_words_count - 1, -1, 1)  # Normalize the index
                        selected_words_indexes.append(index)  # Store the index
                        selected_words_norm_indexes.append(normalized_index)  # Store the normalized index

                    # Get each letter's bounding box (positions)
                    letters_metadata = []  # This will store the metadata for each letter
                    letters, letter_boxes = self._get_letter_boxes(word, font, x, y)

                    # Advance for the next word
                    x += word_width + self._space_width

                    for i_letter, letter_box in enumerate(letter_boxes):
                        letters_metadata.append({
                            cons.md["letter index"]: i_letter,
                            cons.md["letters"]: letters[i_letter],
                            cons.md["letter boxes"]: letter_boxes[i_letter],
                        })

                    words_metadata.append({
                        cons.md["word"]: word,
                        cons.md["word_bbox"]: word_bounding_box,
                        # cons.md["normalised_masked_downsampled_peripheral_view"]: masked_peripheral_view.tolist(),  # Store as list in metadata
                        cons.md["word length"]: len(word),
                        cons.md["visuospatial info"]: {
                            cons.md["line index"]: line_idx,
                            cons.md["lines number"]: num_lines,
                            cons.md["word index in line"]: words_in_line.index(word),
                            cons.md["words number in line"]: len(words_in_line),
                        },
                        cons.md["position"]: {
                            "x": x, "y": y, "x_norm": x_norm, "y_norm": y_norm, "x_init": x_init, "y_init": y_init,
                            "max_x_offset": max_font_offset_x, "max_y_offset": max_font_offset_y,
                            "word_width": word_width, "word_height": word_height,
                            "line_height": max_font_height,
                            "y_lines": line_idx * max_font_height,
                        },
                        cons.md["index"]: index if word in self._words_index_dict else None,
                        cons.md["normalized index"]: normalized_index if word in self._words_index_dict else None,
                        cons.md["letters metadata"]: letters_metadata,
                    })

            # Generate and store foveal patch and masked and downsampled peripheral view for each word after all words have been drawn
            for word_meta in words_metadata:
                word_bbox = word_meta[cons.md["word_bbox"]]
                # Generate and store foveal patch for this word
                foveal_patch, relative_bbox = self._generate_foveal_patches(img, word_bbox)
                word_meta[cons.md["normalised_foveal_patch"]] = foveal_patch.tolist()
                word_meta[cons.md["relative_bbox_foveal_patch"]] = list(relative_bbox)
                # Generate and store masked and downsampled peripheral view for this word
                masked_peripheral_view = self._generate_masked_and_downsampled_peripheral_view(
                    np.array(img), word_bbox, self._training_foveal_and_peripheral_size[0],
                    self._training_foveal_and_peripheral_size[1]
                )
                masked_peripheral_view = self._generate_masked_and_downsampled_plain_peripheral_view(
                    np.array(img), word_bbox, self._training_foveal_and_peripheral_size[0],
                    self._training_foveal_and_peripheral_size[1]
                )
                word_meta[cons.md["normalised_masked_downsampled_peripheral_view"]] = masked_peripheral_view.tolist()

            # Save the image
            # Apply color to the image while still in 'L' mode
            img_rgb = ImageOps.colorize(img, black=self._word_color, white=self._background_color)
            # Convert the image to 'RGB' mode for saving
            img_rgb = img_rgb.convert("RGB")
            image_filename = f"image_{i}.png"
            save_filename = os.path.join(imgs_save_path, image_filename)
            img_rgb.save(save_filename)

            if mode == cons.SIMULATE:
                # Save the image with bounding boxes
                # img_bbox = ImageOps.colorize(img_bbox, black=self._word_color, white=self._background_color)
                # img_bbox = img_bbox.convert("RGB")
                image_bbox_filename = f"image_{i}_bbox.png"
                save_bbox_filename = os.path.join(bbox_stim_save_path, image_bbox_filename)
                img_bbox.save(save_bbox_filename)

            # Store the original image pixels in the metadata
            normalized_image_pixels = aux.normalise(np.array(img), 0, 255, -1, 1).tolist()  # Normalize the image pixels

            # # TODO debug delete later
            # # original_image_pixels = np.array(img).tolist()  # Convert image to list and store in metadata
            # print(f"The original image pixels are: {normalized_image_pixels}\n")
            # print(f"The shape of that is: {np.array(img).shape}\n")
            # return

            # Add image information to metadata
            metadata['images'].append({
                cons.md["image index"]: i,
                # cons.md["normalised_original_image_pixels"]: normalized_image_pixels,  # Store original image pixels
                cons.md["normalised_original_image_pixels"]: [],  # Save space by not storing the image pixels
                cons.md["num words"]: self._num_words,
                cons.md["filename"]: image_filename,
                cons.md["words metadata"]: words_metadata,
                cons.md["selected words"]: words_in_stimulus,
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
            f.write(
                f"Occurrences of words with different lengths in the training dataset: {train_logger_occurrences}\n")
            f.write(f"Occurrences of words with different lengths in the testing dataset: {test_logger_occurrences}\n")
            f.write(f"Train logger: {train_logger}\n")
            f.write(f"Test logger: {test_logger}\n")

    def generate_data(self):
        """Generate images."""
        _, imgs_train_save_path, imgs_test_save_path, imgs_sim_save_path, imgs_bbox_stim_save_path = self._create_directories(
            self._imgs_dir)
        save_path, json_train_save_path, json_test_save_path, json_sim_save_path, json_bbox_stim_save_path = self._create_directories(
            self._json_dir)

        json_train_file = os.path.join(json_train_save_path, cons.MD_FILE_NAME)
        json_test_file = os.path.join(json_test_save_path, cons.MD_FILE_NAME)
        json_sim_file = os.path.join(json_sim_save_path, cons.MD_FILE_NAME)

        self._generate_images_and_metadata(
            imgs_save_path=imgs_train_save_path,
            json_file=json_train_file,
            mode=cons.TRAIN
        )
        train_logger = self.gen_word_lengths_dist_dict.copy()
        self._generate_images_and_metadata(
            imgs_save_path=imgs_test_save_path,
            json_file=json_test_file,
            mode=cons.TEST
        )
        test_logger = self.gen_word_lengths_dist_dict.copy()
        self._generate_images_and_metadata(
            imgs_sim_save_path,
            imgs_bbox_stim_save_path,
            json_sim_file,
            mode=cons.SIMULATE
        )

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
    (_sentences_w_punctuation_marks_and_upper_cases, _corpus_filename, _corpus_json_file_name, _corpus_json_file_dir,
     _corpus_text_dir_w_punctuation_marks, _corpus_json_dir_w_punctuation_marks) = get_sentences()

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
        training_foveal_and_peripheral_size=cons.config["concrete_configs"]["training_foveal_and_peripheral_size"],
        words=cons.config['generator']['dft_words'][1],  # The default words, full sentences  #_words, # Random words
        words_index_dict=_words_index_dict,
        words_dict=_words_dict,
        num_words=cons.config["concrete_configs"]["num_words"],
        set_random_num_words=cons.config["concrete_configs"]["random num_words"],
        pos_sentences=cons.config["positions"]["sentence_top_center"],  # sentence_random or sentence_center
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
