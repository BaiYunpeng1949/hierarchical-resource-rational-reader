import os.path

import json

from PIL import Image, ImageDraw, ImageFont
from step5.utils import auxiliaries
from step3.gens import image_generator as gens
from step5.utils import constants as cons
import math
import time
import numpy as np


def get_img_index(filename):
    """Extract the word from the filename."""
    # Assuming the format is always word_number.jpg
    _, _index_with_suffix = filename.split('_')
    _index = _index_with_suffix.split('.')[0]
    return int(_index)


class PseudoOfflineOCRModel:

    def __init__(
            self,
            font,
            image_metadata=None,
            debug=False
    ):

        # # Initialize the model with the font and the dictionary of lowercase letters
        # self.dict_lower_letters = self.init_letter_size(font)

        self._metadata = None
        self._image_metadata = image_metadata
        self._debug = debug

    @staticmethod
    def _get_letter_boxes(word_metadata, letters_metadata):
        """Get bounding boxes for each letter in the word, considering the relative positions."""
        # Initialize the list to store the letter boxes
        letters_boxes = []
        word_length = int(word_metadata[cons.md["word length"]])
        for letter_metadata in letters_metadata:
            box_left = float(letter_metadata[cons.md["letter boxes"]][cons.md["letter box left"]])
            box_top = float(letter_metadata[cons.md["letter boxes"]][cons.md["letter box top"]])
            box_right = float(letter_metadata[cons.md["letter boxes"]][cons.md["letter box right"]])
            box_bottom = float(letter_metadata[cons.md["letter boxes"]][cons.md["letter box bottom"]])

            letters_boxes.append((box_left, box_top, box_right, box_bottom))
        return letters_boxes

    @staticmethod
    def circle_letter_overlap(box, circle_center, circle_radius, resolution=10):
        """Calculate the overlap area of a letter box and a circle using a grid-based approximation method."""
        box_x1, box_y1, box_x2, box_y2 = box
        rect_width = box_x2 - box_x1
        rect_height = box_y2 - box_y1
        rect_area = rect_width * rect_height

        # Determine the size of each small square in the grid
        dx = rect_width / resolution
        dy = rect_height / resolution

        overlap_area = 0
        # Iterate over the grid of points within the rectangle
        for i in range(resolution):
            for j in range(resolution):
                # Center point of the current square
                x = box_x1 + (i + 0.5) * dx
                y = box_y1 + (j + 0.5) * dy
                # Check if the center point is inside the circle
                if (x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2 <= circle_radius ** 2:
                    overlap_area += dx * dy

        # Calculate the percentage of the rectangle's area that is overlapped
        percent_overlap = (overlap_area / rect_area) * 100

        return percent_overlap

    @staticmethod
    def rectangle_letter_overlap(box, rect_top_left, rect_bottom_right):
        """Calculate the overlap area of a letter box and a rectangle using a grid-based approximation method."""
        box_x1, box_y1, box_x2, box_y2 = box
        rect_x1, rect_y1 = rect_top_left
        rect_x2, rect_y2 = rect_bottom_right

        # Find the overlapping region
        overlap_x1 = max(box_x1, rect_x1)
        overlap_y1 = max(box_y1, rect_y1)
        overlap_x2 = min(box_x2, rect_x2)
        overlap_y2 = min(box_y2, rect_y2)

        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        else:
            overlap_area = 0

        rect_area = (box_x2 - box_x1) * (box_y2 - box_y1)
        percent_overlap = (overlap_area / rect_area) * 100 if rect_area != 0 else 0

        return percent_overlap

    def get_covered_letters(self, rect_top_left, rect_bottom_right, boxes, word,
                            coverage_threshold=cons.OCR_COVERAGE_THRESHOLD):
        """Determine which letters in the word are sufficiently covered by the rectangle, ensuring continuity."""
        covered_letters_indices = []

        rect_x1, rect_y1 = rect_top_left
        rect_x2, rect_y2 = rect_bottom_right

        for i, box in enumerate(boxes):
            box_x1, box_y1, box_x2, box_y2 = box

            # Check if the box intersects with the rectangle (partial or full)
            if box_x2 < rect_x1 or box_x1 > rect_x2 or box_y2 < rect_y1 or box_y1 > rect_y2:
                continue  # Skip boxes that are completely outside the rectangle

            # Calculate the overlap percentage for the box
            overlap_percentage = self.rectangle_letter_overlap(box, rect_top_left, rect_bottom_right)
            if overlap_percentage >= coverage_threshold:
                covered_letters_indices.append(i)

            if self._debug:
                print(f"Letter '{word[i]}' overlap percentage: {overlap_percentage}")

        # Check for continuity in the covered letters
        if covered_letters_indices:
            continuous_indices = [covered_letters_indices[0]]  # Start with the first covered letter
            for i in range(1, len(covered_letters_indices)):
                if covered_letters_indices[i] - continuous_indices[-1] > 1:
                    # Fill the gap if the next covered letter is not adjacent
                    continuous_indices.extend(range(continuous_indices[-1] + 1, covered_letters_indices[i] + 1))
                else:
                    continuous_indices.append(covered_letters_indices[i])

            covered_letters = [word[i] for i in continuous_indices]
        else:
            covered_letters = [cons.NA]

        return covered_letters

    # def test(
    #         self,
    #         image=None,
    #         fixation_center: tuple = None,
    #         foveal_size: tuple = None,
    #         parafoveal_size: tuple = None,      # TODO finish later
    #         target_word_idx: int = None,
    #         surrounding_num_words_buffer: int = cons.SURROUNDING_NUM_WORDS,    # Only focus on the target word and the surrounding few words to increase the search efficiency
    #         draw_image: bool = False
    # ):
    #     """Simulate OCR considering a rectangle around the fixation position."""
    #
    #     # Calculate the top-left and bottom-right corners of the rectangle around the fixation position
    #     foveal_rect_top_left = (fixation_center[0] - foveal_size[0] / 2, fixation_center[1] - foveal_size[1] / 2)
    #     foveal_rect_bottom_right = (fixation_center[0] + foveal_size[0] / 2, fixation_center[1] + foveal_size[1] / 2)
    #
    #     # Calculate the top-left and bottom-right corners of the parafoveal rectangle
    #     parafoveal_rect_top_left = (foveal_rect_top_left[0], foveal_rect_top_left[1] - parafoveal_size[1] / 2)
    #     parafoveal_rect_bottom_right = (foveal_rect_top_left[0] + parafoveal_size[0], foveal_rect_top_left[1] + parafoveal_size[1] / 2)
    #
    #     draw = None
    #     if draw_image:
    #         draw = ImageDraw.Draw(image)
    #         # Draw the rectangle
    #         draw.rectangle([foveal_rect_top_left, foveal_rect_bottom_right], outline="red", width=2)
    #
    #     # Initialize the list with NA values for all words
    #     num_words_on_img = self._image_metadata[cons.md['num words']]
    #     covered_letters_across_words = [[cons.NA] for _ in range(num_words_on_img)]
    #
    #     # Calculate the index range of the target word and its surrounding words
    #     index_bottom_limit = max(target_word_idx - surrounding_num_words_buffer, 0)
    #     index_top_limit = min(target_word_idx + surrounding_num_words_buffer, num_words_on_img - 1)
    #     index_list = range(index_bottom_limit, index_top_limit + 1)
    #
    #     # Process only the words within the relevant indices
    #     for index in index_list:
    #         word_metadata = self._image_metadata[cons.md["words metadata"]][index]
    #
    #         # Get the metadata of the letters of the current word
    #         letters_metadata = word_metadata[cons.md["letters metadata"]]
    #
    #         # Get the letter boxes
    #         boxes = self._get_letter_boxes(
    #             word_metadata=word_metadata,
    #             letters_metadata=letters_metadata,
    #         )
    #
    #         if draw_image:
    #             # Draw boxes around letters
    #             for box in boxes:
    #                 draw.rectangle(box, outline="blue", width=1)
    #
    #             image.show()
    #
    #         word = word_metadata[cons.md["word"]]
    #         covered_letters_for_one_word = self.get_covered_letters(foveal_rect_top_left, foveal_rect_bottom_right,
    #                                                                 boxes, word)
    #
    #         # Replace the NA value at the current index with the actual covered letters
    #         covered_letters_across_words[index] = covered_letters_for_one_word
    #
    #     return covered_letters_across_words

    def test(
            self,
            image=None,
            fixation_center: tuple = None,
            foveal_size: tuple = None,
            parafoveal_size: tuple = None,  # Process parafoveal view
            target_word_idx: int = None,
            surrounding_num_words_buffer: int = cons.SURROUNDING_NUM_WORDS,
            draw_image: bool = False
    ) -> (list, list):
        """Simulate OCR considering rectangles around the fixation position."""

        # Calculate the top-left and bottom-right corners of the foveal rectangle around the fixation position
        foveal_rect_top_left = (fixation_center[0] - foveal_size[0] / 2, fixation_center[1] - foveal_size[1] / 2)
        foveal_rect_bottom_right = (fixation_center[0] + foveal_size[0] / 2, fixation_center[1] + foveal_size[1] / 2)

        # Calculate the top-left and bottom-right corners of the parafoveal rectangle
        parafoveal_rect_top_left = (foveal_rect_top_left[0], foveal_rect_top_left[1])
        parafoveal_rect_bottom_right = (
        foveal_rect_top_left[0] + parafoveal_size[0], foveal_rect_top_left[1] + parafoveal_size[1])

        draw = None
        if draw_image:
            draw = ImageDraw.Draw(image)
            # Draw the foveal rectangle
            draw.rectangle([foveal_rect_top_left, foveal_rect_bottom_right], outline="red", width=3)
            # Draw the parafoveal rectangle
            draw.rectangle([parafoveal_rect_top_left, parafoveal_rect_bottom_right], outline="blue", width=1)

        # Initialize the list with NA values for all words
        num_words_on_img = self._image_metadata[cons.md['num words']]
        covered_letters_across_words_foveal = [[cons.NA] for _ in range(num_words_on_img)]
        covered_letters_across_words_parafoveal = [[cons.NA] for _ in range(num_words_on_img)]

        # Calculate the index range of the target word and its surrounding words
        index_bottom_limit = max(target_word_idx - surrounding_num_words_buffer, 0)
        index_top_limit = min(target_word_idx + surrounding_num_words_buffer, num_words_on_img - 1)
        index_list = range(index_bottom_limit, index_top_limit + 1)

        # Process only the words within the relevant indices
        for index in index_list:
            word_metadata = self._image_metadata[cons.md["words metadata"]][index]

            # Get the metadata of the letters of the current word
            letters_metadata = word_metadata[cons.md["letters metadata"]]

            # Get the letter boxes
            boxes = self._get_letter_boxes(
                word_metadata=word_metadata,
                letters_metadata=letters_metadata,
            )

            if draw_image:
                # Draw boxes around letters
                for box in boxes:
                    draw.rectangle(box, outline="blue", width=1)
                image.show()

            # Process foveal region
            word = word_metadata[cons.md["word"]]
            covered_letters_foveal = self.get_covered_letters(foveal_rect_top_left, foveal_rect_bottom_right,
                                                              boxes, word)

            # Process parafoveal region
            covered_letters_parafoveal = self.get_covered_letters(parafoveal_rect_top_left,
                                                                  parafoveal_rect_bottom_right,
                                                                  boxes, word)

            covered_letters_across_words_foveal[index] = covered_letters_foveal
            covered_letters_across_words_parafoveal[index] = covered_letters_parafoveal

        return covered_letters_across_words_foveal, covered_letters_across_words_parafoveal

    @staticmethod
    def getsize(font, text):
        left, top, right, bottom = font.getbbox(text)
        width = right - left
        height = bottom - top
        return width, height, left, top, right, bottom

    def init_letter_size(self, font, lowercase=True):
        """Get the size of the letter."""
        dict_letter_size = {}
        if lowercase:
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                # 'string.ascii_letters' generates 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                letter_width, letter_height, left, top, right, bottom = self.getsize(font, letter)
                dict_letter_size[letter] = (letter_width, letter_height, left, top, right, bottom)
        else:
            raise NotImplementedError

        return dict_letter_size


if __name__ == "__main__":

    start_time = time.time()

    # For debugging and testings
    foldername = r'D:\Users\91584\PycharmProjects\reading-model\step5\data\gen_envs\08_15_16_50_100_images_W1500H1000WS30_generalized_lexicon_under_20\simulate'
    filename = 'image_3.jpg'  # Provide the correct path to the image
    font_path = cons.FONT_PATH  # Replace with your font file path
    font_size = cons.config["concrete_configs"]["word_size"]
    foveal_rect_width = cons.config["concrete_configs"]["foveal_size"][0]
    foveal_rect_height = cons.config["concrete_configs"]["foveal_size"][1]
    parafoveal_rect_width = cons.config["concrete_configs"]["parafoveal_size"][0]
    parafoveal_rect_height = cons.config["concrete_configs"]["parafoveal_size"][1]
    draw_image = Image.open(os.path.join(foldername, filename))
    font = ImageFont.truetype(font_path, font_size)

    fixation_position = (495, 450)  # Example circle center, adjust as needed
    target_word_idx = 79
    # foveal_radius = auxiliaries.FOVEA_FACTOR * font_size  # Example circle radius, adjust as needed
    foveal_rect_top_left = (fixation_position[0] - foveal_rect_width / 2, fixation_position[1] - foveal_rect_height / 2)
    foveal_rect_bottom_right = (fixation_position[0] + foveal_rect_width / 2, fixation_position[1] + foveal_rect_height / 2)

    # Get the image index
    _index = get_img_index(filename)
    # Read this image's JSON file (metadata)
    with open(os.path.join(foldername, cons.MD_FILE_NAME), "r") as file:
        metadata = json.load(file)

    _image_metadata = metadata[cons.md["images"]][_index]

    # Initialize the pseudo-OCR model
    model = PseudoOfflineOCRModel(font, _image_metadata, debug=True)

    # _covered_letters_across_words = model.test(draw_image, fixation_position, foveal_radius, draw_image=True)
    _covered_letters_across_words_foveal, _covered_letters_across_words_parafoveal = model.test(
        image=draw_image,
        fixation_center=fixation_position,
        foveal_size=(foveal_rect_width, foveal_rect_height),
        parafoveal_size=(parafoveal_rect_width, parafoveal_rect_height),
        target_word_idx=target_word_idx,
        draw_image=True
    )
    print("Covered letters across words on foveal:", _covered_letters_across_words_foveal, "\n")
    print("Covered letters across words on parafoveal:", _covered_letters_across_words_parafoveal, "\n")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds.")
