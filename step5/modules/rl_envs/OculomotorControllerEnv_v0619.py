import shutil

import numpy as np
import yaml
import json
import os
import re
import math
import matplotlib.pyplot as plt

from gymnasium import Env
from gymnasium.spaces import Box, Dict
from annoy import AnnoyIndex
from PIL import Image, ImageDraw, ImageFont

import pandas as pd

from step5.utils import auxiliaries as aux
from step5.utils import pseudo_offline_ocr_model as ocr
from step3.gens import image_generator as gens
from step5.utils import constants as cons


class OculomotorControllerEnv(Env):

    def __init__(self):

        """
        Created on 19 June 2024.
        This is the lowest-level environment for the reading model, where the oculomotor control is done by an RL agent.
        Author: Bai Yunpeng
        """

        # Get the current root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Get the mode from the config yaml file
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        print(f"{cons.LV_TWO_DASHES}Oculomotor Controller Environment -- Deploying the environment in the {self._config['rl']['mode']} mode.")

        # Define constants
        self.MAX_WORD_LEN = cons.MAX_WORD_LEN
        self.MIN_WORD_LEN = cons.MIN_WORD_LEN
        self._WORD_PIXEL_SIZE = cons.config["concrete_configs"]["word_size"]
        self._FOVEA_RADIUS = cons.FOVEA_FACTOR * self._WORD_PIXEL_SIZE
        self._FOVEA_RECT_WIDTH = cons.config["concrete_configs"]["foveal_size"][0]
        self._FOVEA_RECT_HEIGHT = cons.config["concrete_configs"]["foveal_size"][1]
        self._PARAFOVEA_RECT_WIDTH = cons.config["concrete_configs"]["parafoveal_size"][0]
        self._PARAFOVEA_RECT_HEIGHT = cons.config["concrete_configs"]["parafoveal_size"][1]
        self._TRAINING_FOVEA_AND_PARAFOVEA_RECT_WIDTH = cons.config["concrete_configs"]["training_foveal_and_parafoveal_size"][0]
        self._TRAINING_FOVEA_AND_PARAFOVEA_RECT_WIDTH = cons.config["concrete_configs"]["training_foveal_and_parafoveal_size"][1]

        # Read image environments -- determines which word to be identified
        image_envs_filename = self._config["resources"]["img_env_dir"]
        # --------------------------------------------------------------------------
        # Determine the dataset's mode -- using the training or testing dataset
        self.dataset_mode = None
        if self._config["rl"]["mode"] == cons.TRAIN or self._config["rl"]["mode"] == cons.CONTINUAL_TRAIN:
            self.dataset_mode = cons.TRAIN
        elif self._config["rl"]["mode"] == cons.SIMULATE:
            # self.dataset_mode = cons.TRAIN if self._config["rl"]["deploy"]["use_training_dataset"] is True else cons.TEST
            self.dataset_mode = cons.SIMULATE
        else:
            self.dataset_mode = cons.TEST
            if self._config["rl"]["test"]["use_training_dataset"] is True:
                self.dataset_mode = cons.TRAIN
        # --------------------------------------------------------------------------
        image_envs_dir = os.path.join(root_dir, "data", "gen_envs", image_envs_filename, self.dataset_mode)
        # --------------------------------------------------------------------------
        self._image_envs, self._image_filenames = aux.read_images(image_envs_dir)
        self._image_filenames_indexes = [int(name.split("_")[1].split(".")[0]) for name in self._image_filenames]
        self._num_image_envs = len(self._image_envs)
        self._image_filename = None
        # Output the loading information
        print(f"{cons.LV_THREE_DASHES}The image environments are loaded from {image_envs_dir}, the number of images is {self._num_image_envs}")

        # Pull the lexicon from the lexicon file -- determines the agent's internal word knowledge
        # To get the lexicon from the dataset: https://github.com/sapbmw/The-Oxford-3000
        # lexicon_file_dir = os.path.join(root_dir, 'assets', lexicon_name, 'ltm_corpus.txt')
        lexicon_dir = os.path.join(root_dir, "data", "assets", self._config["resources"]["lexicon_filename"]+'.txt')
        self._lexicon = sorted(list(set(aux.load_txt(lexicon_dir))))
        # Filter out words with non-alphabetic characters and convert to lowercase
        self._lexicon = [word.lower() for word in self._lexicon if word.isalpha()]
        # Get a dictionary of the lexicon encoded in a 16-dim vector -- slow, 230 fps
        self._encoded_lexicon = {}
        for word in self._lexicon:
            self._encoded_lexicon[word] = self._encode_letters_in_word(word)
        # Annoy index for quick querying the nearest word in the word-image environment
        # self._annoy_index = self._build_annoy_index(encoded_lexicon=self._encoded_lexicon)
        self._word_inferencer = WordInferencer(encoded_lexicon=self._encoded_lexicon)
        # Output the loading information
        print(f"{cons.LV_THREE_DASHES}The lexicon is loaded from {lexicon_dir}")

        # Read the JSON file (metadata)
        with open(os.path.join(image_envs_dir, cons.MD_FILE_NAME), "r") as file:
            self._metadata = json.load(file)

        # self._num_words_per_img = self._metadata[cons.md["config"]][cons.md["num words"]]
        self._num_words_on_img = None
        self._norm_num_words_per_img = None

        # Do a metadata check here
        num_imgs = self._metadata[cons.md["config"]][cons.md["num images"]]
        assert num_imgs == self._num_image_envs, f"The number of images in the metadata is {num_imgs}, " \

        # Get image-level metadata
        self._images_metadata = self._metadata[cons.md["images"]]
        self._image_metadata = None
        self._words_on_image = None
        self._words_lens_on_image = None

        # Define state related variables
        self._image_env = None     # The word image environment does the agent tries to identify
        self._image_env_pixels = None  # The pixel values of the image environment
        self._image_env_index = None
        self._target_word = None      # The word that the agent tries to identify among many words in the image, randomly sampled from the image
        self._target_word_metadata = None  # The metadata of the target word
        self._next_word_metadata = None  # The metadata of the next word
        self._target_word_len = None      # The length of the targeted word
        self._target_word_center_x = None  # The x coordinate of the target word's center
        self._target_word_center_y = None  # The y coordinate of the target word's center
        self._target_word_left_x = None  # The x coordinate of the target word's left
        self._target_word_sigma = None  # The sigma factor of the target word's center
        self._target_word_center_noisy_x = None  # The noisy x coordinate of the target word's center
        self._target_word_center_noisy_y = None  # The noisy y coordinate of the target word's center
        self._encoded_target_word = None  # The encoded target word that the agent tries to identify
        self._target_word_idx_on_image = None      # The index of the target word in the image
        self._norm_target_word_idx_on_image = None  # The normalized index of the target word in the image
        self._encoded_words = None       # The encoded words that the agent tries to identify
        # (several 16-dim vectors, dependent on the number of words on the image, range from -1 to 1)
        self._encoded_words_counters = None  # The counter for the encoded word
        self._encoded_words_counters_offsets = None  # The counter for the encoded word, which will set empty positions -1 to provide more information to the agent: which letters are seen (1), which are guessed (0), which are no letters (-1)
        self._norm_target_word_len = None      # The normalized length of the targeted word
        self._word_x_init = None          # The initial x coordinate of the words on the image -- hints in the observation to speed up training
        self._word_y_init = None          # The initial y coordinate of the words on the image

        # Define the perception noise
        self._letter_perception_sigma_range = [0, 1]  # An empirical range that sample out the letter sigma factors
        self._init_letter_perception_sigma = None
        self._letter_perception_sigma = None  # An empirical value that controls the initial perception noise -- initial letter sigma
        self._flag_fixate_on_target_word = None  # A flag that indicates whether the agent is on the target word: it could give the agent more continuous information
        self._flag_target_word_internal_letters_are_updated = None  # A flag that indicates whether the internal letters are updated

        self._inferred_words = None      # The inferred words (all words on the image) by the agent
        self._encoded_inferred_words = None  # The inferred encoded words (all words on the image) by the agent

        self._encoded_internal_words = None  # The encoded internal words (all words on the image) that is presented in the observation

        self._vect_dist = None          # Define the nearest distance of the word in the encoded vector space

        self._rho_spatial = 0.09        # The empirical factor set as 0.09, Reference: An Adaptive Model of Gaze-based Selection

        # Action / proprioception information
        self._norm_fix_x = None              # The x coordinate of the fixation position
        self._norm_fix_y = None              # The y coordinate of the fixation position
        self._fix_x = None              # The x coordinate of the fixation position
        self._fix_y = None              # The y coordinate of the fixation position

        self._fovea_seen_letters = None       # The letters that the agent has seen in the fovea
        self._parafovea_seen_letters = None   # The letters that the agent has seen in the parafovea
        self._is_seen_letters_acrs_img = None  # A flag that the agent sees letters

        # The index of the word in the image that the agent thinks is the target word
        self._agent_word_idx_on_image = None
        self._norm_agent_word_idx_on_image = None

        # Define the task completion flag
        self._task_completion = None

        # Define the training -- episode and rl related variables
        self._steps = None
        self.ep_len = 10        # The length of the episode, smaller, faster to converge
        self._terminate = cons.NA
        self._truncated = cons.NA

        # TODO <IMPORTANT> 1. add other variables, such as user profile, time awareness, .. later
        #  2. Use a different shape fovea vision, not a circle, but slanted; and apply a different size parafovea
        #  vision, think how to train that

        # Define the action space
        # Two normalized actions from -1 to 1, representing the x and y coordinates of the gaze;
        # The third action is the termination action
        self.action_space = Box(low=-1, high=1, shape=(3,))

        # Define the observation space
        self._original_img_env_width = self._metadata[cons.md["config"]][cons.md["img size"]][0]
        self._original_img_env_height = self._metadata[cons.md["config"]][cons.md["img size"]][1]
        self._foveal_width = cons.config['concrete_configs']['foveal_size'][0]
        self._foveal_height = cons.config['concrete_configs']['foveal_size'][1]
        self._peripheral_width = cons.config['concrete_configs']['peripheral_size'][0]
        self._peripheral_height = cons.config['concrete_configs']['peripheral_size'][1]
        self._training_foveal_and_parafoveal_width = cons.config['concrete_configs']['training_foveal_and_parafoveal_size'][0]
        self._training_foveal_and_parafoveal_height = cons.config['concrete_configs']['training_foveal_and_parafoveal_size'][1]
        self._num_stateful_info_obs = 1 + 5 + 3 * self.MAX_WORD_LEN    # Remove cons.MAX_NUM_WORDS
        self.observation_space = Dict({
            # "vision": Box(low=-1, high=1, shape=(2, self._foveal_height, self._foveal_width)),
            "vision": Box(low=-1, high=1, shape=(2, self._training_foveal_and_parafoveal_height, self._training_foveal_and_parafoveal_width)),
            "stateful information": Box(low=-1, high=1, shape=(self._num_stateful_info_obs,))
        })

        # Define the logger
        self.fixations_logger = None
        self.logger = None

        # Initialize the pseudo offline ocr model
        _word_size_in_px = self._metadata[cons.md["config"]][cons.md["word size"]]
        self._font = ImageFont.truetype(cons.FONT_PATH, _word_size_in_px)
        self._ocr_model = None

        # Get the number of episodes in testing
        self._num_test_episodes = self._config['rl']['test']['num_episodes']

    def reset(self, seed=None, inputs=None):

        self._steps = 0

        # Initialize the uncertainty levels -- sigma
        self._init_letter_perception_sigma = 0.5

        if inputs is None:
            # Randomly sample an image environment
            self._image_env_index = np.random.choice(self._num_image_envs, 1)[0]
        else:
            self._image_env_index = inputs['image_index']
        # Get the allocated image environment
        image_env_index_in_images = self._image_filenames_indexes.index(self._image_env_index)

        self._image_env = self._image_envs[image_env_index_in_images]
        # Get the corresponding image pixel values
        self._image_env_pixels = np.array(self._image_env)
        # self._image_env_pixels = self._add_noise_to_image(pixels=self._image_env_pixels,
        #                                                   scale=self._image_pixel_noise_scale)  # Add noise to the image to increase policy robustness when necessary

        # Get the corresponding image metadata
        self._image_metadata = self._images_metadata[self._image_env_index]

        # Reset the number of words on the image
        self._num_words_on_img = self._image_metadata[cons.md['num words']]
        self._norm_num_words_per_img = aux.normalise(self._num_words_on_img, 0, cons.MAX_NUM_WORDS, cons.ZERO, cons.ONE)

        # Initialize the pseudo offline ocr model with the image metadata
        self._ocr_model = ocr.PseudoOfflineOCRModel(self._font, self._image_metadata, debug=False)

        # Get words info in this image
        self._words_on_image = self._image_metadata[cons.md["selected words"]]
        self._words_lens_on_image = [len(word) for word in self._words_on_image]

        # Set the encoded word counter offsets, all 0, but exceeding the word length parts are set -1
        self._encoded_words_counters_offsets = [[cons.ZERO] * self.MAX_WORD_LEN for _ in range(len(self._words_on_image))]
        for _idx, _word_len in enumerate(self._words_lens_on_image):
            for _i in range(_word_len, self.MAX_WORD_LEN):
                self._encoded_words_counters_offsets[_idx][_i] = cons.NEGATIVE_ONE

        # Get the target word -- randomly sample one
        if inputs is None:
            self._target_word = np.random.choice(self._words_on_image, 1)[0]
            # Get the target word's index - randomly sample one
            self._target_word_idx_on_image = self._words_on_image.index(self._target_word)
        else:
            self._target_word_idx_on_image = np.clip(inputs['target_word_idx'], 0, self._num_words_on_img - 1)
            self._target_word = self._words_on_image[self._target_word_idx_on_image]

        # Clean the target word is in the lexicon -- lower-case and alphabetic (has no special characters)
        # Might be moved later when we generalize our models
        self._target_word = self._clean_word(word=self._target_word)

        # Get the target word's center
        self._target_word_metadata = self._image_metadata[cons.md["words metadata"]][self._target_word_idx_on_image]
        self._next_word_metadata = self._image_metadata[cons.md["words metadata"]][self._target_word_idx_on_image + 1] if self._target_word_idx_on_image < self._num_words_on_img - 1 else cons.NO_NEXT_WORD
        top_left_x = int(self._target_word_metadata[cons.md["position"]][cons.md["x"]])  # x_init is the first word's info
        top_left_y = int(self._target_word_metadata[cons.md["position"]][cons.md["y"]])
        word_width = int(self._target_word_metadata[cons.md["position"]][cons.md["word_width"]])
        word_height = int(self._target_word_metadata[cons.md["position"]][cons.md["word_height"]])
        self._target_word_center_x = top_left_x + word_width / 2
        self._target_word_left_x = top_left_x
        self._target_word_center_y = top_left_y + word_height / 2

        self._target_word_len = len(self._target_word)
        self._norm_target_word_idx_on_image = aux.normalise(self._target_word_idx_on_image, 0, cons.MAX_NUM_WORDS - 1, cons.ZERO, cons.ONE)

        # Encode the sampled word from a string list to a MAX_WORD_LEN-dim vector from 0 to 1
        self._encoded_target_word = self._encoded_lexicon[self._target_word]

        # Encode all the words in the image for the internal word representation
        self._encoded_words = [self._encoded_lexicon[self._clean_word(word=word)] for word in self._words_on_image]

        # Initialize the encoded words with all zeros
        self._encoded_words_counters = [[0] * self.MAX_WORD_LEN for _ in range(len(self._words_on_image))]
        # TODO get something like a placeholder for the non-words parts

        # Initialize the sampled word's observation vector -- internal representation
        self._encoded_internal_words = [[cons.NEGATIVE_ONE] * self.MAX_WORD_LEN for _ in range(len(self._words_on_image))]
        # TODO OR can we keep this, while appending the non-words parts?

        # Initialize the seen letters
        self._fovea_seen_letters = ['' for _ in range(len(self._words_on_image))]
        self._parafovea_seen_letters = ['' for _ in range(len(self._words_on_image))]
        self._is_seen_letters_acrs_img = [False for _ in range(len(self._words_on_image))]

        encoded_inferred_words = []
        inferred_words = []
        # Initialize the inferred words
        for _idx in range(self._num_words_on_img):
            nearest_id, _encoded_inferred_word, self._vect_dist, _inferred_words \
                = self._word_inferencer.infer_word(
                encoded_internal_word=self._encoded_internal_words[_idx],
                # word_len=self._words_lens_on_image[_idx],
            )
            encoded_inferred_words.append(_encoded_inferred_word)
            inferred_words.append(_inferred_words)
        self._encoded_inferred_words = encoded_inferred_words
        self._inferred_words = inferred_words

        if inputs is None:
            self._norm_fix_x = np.random.uniform(-1, 1)
            self._norm_fix_y = np.random.uniform(-1, 1)
        else:
            if inputs['norm_fix_x'] is None or inputs['norm_fix_y'] is None:
                self._norm_fix_x = np.random.uniform(-1, 1)
                self._norm_fix_y = np.random.uniform(-1, 1)
            else:
                self._norm_fix_x = inputs['norm_fix_x']
                self._norm_fix_y = inputs['norm_fix_y']
        self._fix_x, self._fix_y = self._get_fix_xy()
        self._see()

        # Create the noisy version of the target word's center x and y
        self._target_word_center_noisy_x, self._target_word_center_noisy_y = self._get_noisy_target_word_center()

        # Initialize the terminate action related flag
        self._terminate = False
        self._truncated = False

        # Initialize the task completion flag
        self._task_completion = False

        self._flag_target_word_internal_letters_are_updated = False

        # Initialize the logger
        self.fixations_logger = []

        # Log the first fixation
        self._get_logs(
            reward=self._compute_reward(done=False),
            done=False,
        )

        return self._get_obs(), {}

    def step(self, action):

        # Update the steps
        self._steps += 1

        # Get actions
        self._norm_fix_x, self._norm_fix_y = action[0:2]    # The first two actions

        self._terminate = True if action[2] > cons.ZERO else False  # The third action of determining the termination

        # Reset the flag at the beginning of each step
        self._is_seen_letters_acrs_img = False

        if self._terminate is False:
            # Convert the normalized fixation position to the pixel-wise fixation position
            self._fix_x, self._fix_y = self._get_fix_xy()
            # Update the seen letters if there are any
            self._see()

        self._task_completion = self._get_task_completion()

        done = self._is_done()

        reward = self._compute_reward(done)

        self.logger = {
            'episode': '',
            'step': self._steps,
            'agent': 'OculomotorController',
            'image id': self._image_env_index,
            'words': self._words_on_image,
            'num words': self._num_words_on_img,
            'target word': self._target_word,
            'target word len': self._target_word_len,
            'target word id': self._target_word_idx_on_image,
            'target word visuospatial info': self._get_visuospatial_info(word_idx=self._target_word_idx_on_image),
            'next word visuospatial info': self._get_visuospatial_info(word_idx=self._target_word_idx_on_image + 1) if self._target_word_idx_on_image + 1 < self._num_words_on_img else cons.NO_NEXT_WORD,
            'encoded target word': self._encoded_target_word,
            'norm_fix_x': self._norm_fix_x,
            'norm_fix_y': self._norm_fix_y,
            'fix_x': self._fix_x,
            'fix_y': self._fix_y,
            'foveal seen letters': self._fovea_seen_letters[self._target_word_idx_on_image],
            'parafoveal seen letters': self._parafovea_seen_letters[self._target_word_idx_on_image + 1] if self._target_word_idx_on_image + 1 < self._num_words_on_img else cons.NO_NEXT_WORD,
            'encoded internal word': self._encoded_internal_words[self._target_word_idx_on_image],
            'word counter': self._encoded_words_counters[self._target_word_idx_on_image],
            'inferred word': self._inferred_words[self._target_word_idx_on_image],
            'encoded inferred word': self._encoded_inferred_words[self._target_word_idx_on_image],
            'done': done,
            'done action': self._terminate,
            'reward': reward,
        }

        truncated = self._truncated

        info = {}

        # Log the fixation
        self._get_logs(
            reward=reward,
            done=done,
        )

        return self._get_obs(), reward, done, truncated, info

    def render(self, mode='human'):
        pass

    def _get_fix_xy(self):
        fix_x = aux.normalise(self._norm_fix_x, -1, 1, 0, self._original_img_env_width)
        fix_y = aux.normalise(self._norm_fix_y, -1, 1, 0, self._original_img_env_height)
        return fix_x, fix_y

    def _get_eccentricity(self):
        return math.sqrt((self._target_word_center_x - self._fix_x) ** 2 + (self._target_word_center_y - self._fix_y) ** 2)

    def _get_noisy_target_word_center(self):
        self._target_word_sigma = self._rho_spatial * self._get_eccentricity()
        return self._target_word_center_x + np.random.normal(0, self._target_word_sigma), self._target_word_center_y + np.random.normal(0, self._target_word_sigma)

    def _get_obs(self):
        """Get the observations"""
        """Get the visual perception observation input"""
        # Get the 2D vector
        # rgb_pixels = np.array(self._image_env)  # [H, W, C]
        original_stimuli_rgb_pixels = self._image_env_pixels # [H, W, C]

        # Get the foveal view and peripheral view
        foveal_view_rgb_pixels = self._get_foveal_view(
            original_stimuli_rgb_pixels=original_stimuli_rgb_pixels.copy(),
            foveal_view_width=self._foveal_width,
            foveal_view_height=self._foveal_height,
            fix_x=self._fix_x,
            fix_y=self._fix_y,
        )

        # Downsample the foveal view for training
        downsampled_foveal_view_rgb_pixels = self._downsample_image(
            foveal_view_rgb_pixels,
            self._training_foveal_and_parafoveal_width,
            self._training_foveal_and_parafoveal_height,
        )

        # Get the grayscale image with the fovea mask applied
        original_attention_annotation_gray_pixels = self._annotate_foveal_circle(
            rgb_pixels=original_stimuli_rgb_pixels.copy(),
            fixation_height=self._fix_y,
            fixation_width=self._fix_x,
            foveal_radius=self._FOVEA_RADIUS,
        )

        peripheral_view_rgb_pixels = self._get_peripheral_view(
            original_stimuli_rgb_pixels=original_attention_annotation_gray_pixels,
            peripheral_view_width=self._training_foveal_and_parafoveal_width,     # self._peripheral_width,
            peripheral_view_height=self._training_foveal_and_parafoveal_height,     # self._peripheral_height,
        )

        # Normalize the color values from 0 to 255 to -1 to 1
        # norm_rgb_pixels = aux.normalise(original_stimuli_rgb_pixels, 0, 255, -1, 1)
        norm_foveal_view_rgb_pixels = aux.normalise(foveal_view_rgb_pixels, 0, 255, -1, 1)
        norm_downsampled_foveal_view_rgb_pixels = aux.normalise(downsampled_foveal_view_rgb_pixels, 0, 255, -1, 1)
        norm_peripheral_view_rgb_pixels = aux.normalise(peripheral_view_rgb_pixels, 0, 255, -1, 1)

        # norm_foveal_view_gray_pixels = (norm_foveal_view_rgb_pixels[:, :, 0:1] * cons.RED_TO_GRAY +
        #                                 norm_foveal_view_rgb_pixels[:, :, 1:2] * cons.GREEN_TO_GRAY +
        #                                 norm_foveal_view_rgb_pixels[:, :, 2:3] * cons.BLUE_TO_GRAY)
        norm_downsampled_foveal_view_gray_pixels = (norm_downsampled_foveal_view_rgb_pixels[:, :, 0:1] * cons.RED_TO_GRAY +
                                        norm_downsampled_foveal_view_rgb_pixels[:, :, 1:2] * cons.GREEN_TO_GRAY +
                                        norm_downsampled_foveal_view_rgb_pixels[:, :, 2:3] * cons.BLUE_TO_GRAY)
        norm_peripheral_view_gray_pixels = (norm_peripheral_view_rgb_pixels[:, :, 0:1] * cons.RED_TO_GRAY +
                                            norm_peripheral_view_rgb_pixels[:, :, 1:2] * cons.GREEN_TO_GRAY +
                                            norm_peripheral_view_rgb_pixels[:, :, 2:3] * cons.BLUE_TO_GRAY)

        # Transpose the two pixels to [C, H, W]
        norm_downsampled_foveal_view_gray_pixels = np.transpose(norm_downsampled_foveal_view_gray_pixels, (2, 0, 1))
        norm_peripheral_view_gray_pixels = np.transpose(norm_peripheral_view_gray_pixels, (2, 0, 1))

        foveal_vision = norm_downsampled_foveal_view_gray_pixels
        peripheral_vision = norm_peripheral_view_gray_pixels

        # Stack them as two channels
        assert foveal_vision.shape == peripheral_vision.shape, "The foveal and peripheral vision should have the same shape"
        # Else, maybe resize the peripheral vision to the foveal vision
        vision = np.concatenate([norm_downsampled_foveal_view_gray_pixels, norm_peripheral_view_gray_pixels], axis=0)

        """Get the stateful information observation input"""
        # The stateful information
        norm_remaining_ep_len = (self.ep_len - self._steps) / self.ep_len * 2 - 1

        norm_terminate = 1 if self._terminate else -1

        # norm_target_word_x = aux.normalise(self._target_word_center_x, 0, self._original_img_env_width, -1, 1)
        norm_target_word_left_x = aux.normalise(self._target_word_left_x, 0, self._original_img_env_width, -1, 1)
        norm_target_word_center_y = aux.normalise(self._target_word_center_y, 0, self._original_img_env_height, -1, 1)

        # self._target_word_center_noisy_x, self._target_word_center_noisy_y = self._get_noisy_target_word_center()
        # norm_noisy_target_word_x = aux.normalise(self._target_word_center_noisy_x, 0, self._original_img_env_width, -1, 1)
        # norm_noisy_target_word_y = aux.normalise(self._target_word_center_noisy_y, 0, self._original_img_env_height, -1, 1)

        # Agent's proprioception information -- the fixation position
        norm_fix_x = self._norm_fix_x
        norm_fix_y = self._norm_fix_y

        norm_num_words_per_img = self._norm_target_word_idx_on_image

        norm_target_word_index = aux.normalise(self._target_word_idx_on_image, 0, cons.MAX_NUM_WORDS - 1, -1, 1)

        # Encoded words
        norm_internal_word = self._encoded_internal_words[self._target_word_idx_on_image]
        norm_inferred_word = self._encoded_inferred_words[self._target_word_idx_on_image]
        norm_word_counter = self._encoded_words_counters[self._target_word_idx_on_image]

        # Integrate the observations
        stateful_information = np.concatenate([
            [
                norm_remaining_ep_len,
                norm_terminate,
                norm_target_word_left_x, norm_target_word_center_y,
                # norm_fix_x, norm_fix_y,
                norm_num_words_per_img,
                norm_target_word_index,
            ],
            # self._encoded_target_word,
            norm_internal_word,
            norm_inferred_word,
            norm_word_counter,
        ])

        # Check the dimension of the observation
        if vision.shape != self.observation_space["vision"].shape:
            raise ValueError(f"Vision observation has an invalid shape: {vision.shape}, "
                             f"expected: {self.observation_space['vision'].shape}")
        if stateful_information.shape != self.observation_space["stateful information"].shape:
            raise ValueError(f"Stateful information observation has an invalid shape: {stateful_information.shape}, "
                             f"expected: {self.observation_space['stateful information'].shape}")

        return {"vision": vision, "stateful information": stateful_information}

    def _get_encoded_words_counters(self):
        return np.array(self._encoded_words_counters).flatten() + np.array(self._encoded_words_counters_offsets).flatten()

    def _encode_letters_in_word(self, word):

        """Encode the word letters into a 16-dim vector from 0 to 1 """
        encoded_word_letters = [-1] * self.MAX_WORD_LEN

        # Always convert the word to lowercase -- assumption: uppercase and lowercase are the same when being processed internally
        word = word.lower()

        # Encode every letter in the word from 26 to 0 to 1, if not enough to 16, then pad with -1
        for idx, letter in enumerate(word):
            encoded_word_letters[idx] = aux.normalise(ord(letter) - ord('a'), 0, 25, 0, 1)  # Lowercase
            # encoded_word_letters[idx] = auxiliaries.normalise(ord(letter), 0, 127, 0, 1)  # Full ASCII table

        return encoded_word_letters

    def _update_perception_uncertainty(self, encoded_word_counter, current_word_len, fixation_start_index, fixation_end_index):

        """Update the letter sigma based on the number of letters that have been observed."""
        is_updated = False
        # Collect the number of directly fixated letters before updating
        observed_letters_before = sum(1 for item in encoded_word_counter if item == 1)

        # Update the counter of directly fixated letters
        encoded_word_counter[fixation_start_index:fixation_end_index] = (
                [cons.ONE] * (fixation_end_index - fixation_start_index))

        # Number of explicitly observed letters
        observed_letters = sum(1 for item in encoded_word_counter if item == 1)

        proportion = observed_letters / current_word_len

        self._letter_perception_sigma = self._init_letter_perception_sigma * (1 - proportion)

        # Check if new internal letters have been observed
        if observed_letters_before < observed_letters:
            is_updated = True

        return is_updated

    def _update_internal_word(
            self,
            current_word_len,
            encoded_word,
            encoded_internal_word,
            encoded_word_counter,
            perception_uncertainty
    ):

        """
        Update the internal word representation (in the observation) with the fixation index.
        Get the blurring effect of the unseen letters - provide a rough guessing of the word.
        """

        # Apply the noisy masking first
        encoded_internal_word[:current_word_len] = self._add_noise_to_letters(encoded_word[:current_word_len], perception_uncertainty)

        # Get all the indexes of counted letters that are one, then update the internal word representation
        for idx, value in enumerate(encoded_word_counter):
            if value == cons.ONE:
                # Copy the corresponding letter from self._encoded_word to self._encoded_internal_word
                encoded_internal_word[idx] = encoded_word[idx]

        return encoded_internal_word

    def _get_task_completion(self):
        """Get the task completion status"""
        if self._target_word == self._inferred_words[self._target_word_idx_on_image]:
            return True
        else:
            return False

    def _compute_reward(self, done):

        # time_cost = -0.05        # -0.1 # Constant time penalty

        # Apply a position/distance-based reward
        step_reward = self._reward_shaping(
            center_x=self._target_word_center_x,    # OR maybe change this to a noisier version -- some place within the word
            center_y=self._target_word_center_y,
            fix_x=self._fix_x,
            fix_y=self._fix_y,
        )

        # Determine the bonus
        if done is True:
            if self._target_word == self._inferred_words[self._target_word_idx_on_image]:
                bonus = 1
            else:
                bonus = -1
                # TODO <IMPORTANT> To tune the agent's emergency level, use a weight factor here to control the final penalty,
                #  if urgent, then set it to a small value, if not, then pursue the correctness.
                #  It could control the agent's behavior to be more aggressive or conservative. regarding when to terminate the task.
        else:
            bonus = 0

        return np.clip(step_reward + bonus, -1, 1)

    def _reward_shaping(self, center_x, center_y, fix_x, fix_y, k=0.08, w=0.05):
        """Apply a position/distance-based reward shaping"""
        distance = np.sqrt((center_x - fix_x) ** 2 + (center_y - fix_y) ** 2)
        if self._flag_fixate_on_target_word is False:
            return w * (np.exp(-k * distance) - 1)
        else:
            return -0.01
        # return w * (np.exp(-k * distance) - 1)

    def _is_done(self):

        # If the agent wants to terminate the trial
        if self._terminate is True:
            return True

        # Time out
        if self._steps >= self.ep_len:
            self._truncated = True
            return True
        else:
            return False

    def _get_start_end_indexes(self, word, segment):

        start_index = word.find(segment)

        if start_index != -1:
            # The segment exists within the word
            # Calculate the end index
            end_index = start_index + len(segment)
        else:
            # The segment was not found in the word
            raise ValueError(f"Segment '{segment}' not found in '{word}', "
                             f"\nthe issue image is {self._image_filename}."
                             f"\nThe current fixation is {self._fix_x}, {self._fix_y}")

        return start_index, end_index

    @staticmethod
    def _add_noise_to_letters(values, sigma):
        """ Add Gaussian noise to a list of values (letters)."""
        # Add Gaussian noise centered at 0 with standard deviation sigma
        noises = np.random.normal(0, sigma, len(values))
        noisy_values = values + noises

        # Ensure the noisy values are clipped to the range [0, 1] if your encoding is in this range
        return np.clip(noisy_values, 0, 1)

    @staticmethod
    def _annotate_foveal_circle(rgb_pixels, fixation_height, fixation_width, foveal_radius):
        # Copy the original image to not alter the original pixels outside the circle
        manipulated_image = np.copy(rgb_pixels)

        # Image dimensions
        height, width = rgb_pixels.shape[:2]

        # Define the coordinates for the circle's center
        center_y, center_x = fixation_height, fixation_width

        # Create an array representing the indices of the image
        y_indices, x_indices = np.ogrid[:height, :width]

        # Calculate the distance of each pixel from the center
        distance_from_center = np.sqrt((y_indices - center_y) ** 2 + (x_indices - center_x) ** 2)

        # Create a mask for the pixels that are near the boundary of the circle
        # Define a thickness for the circle line
        circle_thickness = 1  # This will create a circle line that is 2 pixels thick
        lower_bound = foveal_radius - circle_thickness
        upper_bound = foveal_radius + circle_thickness
        circle_mask = (distance_from_center >= lower_bound) & (distance_from_center <= upper_bound)

        # Apply the circle mask to the image with a distinctive intensity value
        # Here we'll set the color to red for visibility
        manipulated_image[circle_mask] = [255, 0, 0]

        return manipulated_image

    @staticmethod
    def _get_foveal_view(
            original_stimuli_rgb_pixels,
            foveal_view_width,
            foveal_view_height,
            fix_x,
            fix_y,
    ):
        """
        Get the foveal view of the image with padding if necessary to ensure a fixed size of 40x40 pixels.
        """

        # Calculate foveal view coordinates
        x_left_up_foveal = max(int(fix_x - foveal_view_width / 2), 0)
        y_left_up_foveal = max(int(fix_y - foveal_view_height / 2), 0)

        x_right_down_foveal = min(int(fix_x + foveal_view_width / 2), original_stimuli_rgb_pixels.shape[1] - 1)
        y_right_down_foveal = min(int(fix_y + foveal_view_height / 2), original_stimuli_rgb_pixels.shape[0] - 1)

        # Ensure the coordinates are within valid ranges
        x_left_up_foveal = min(x_left_up_foveal, original_stimuli_rgb_pixels.shape[1] - 1)
        y_left_up_foveal = min(y_left_up_foveal, original_stimuli_rgb_pixels.shape[0] - 1)
        x_right_down_foveal = max(x_right_down_foveal, 0)
        y_right_down_foveal = max(y_right_down_foveal, 0)

        # Crop the foveal view
        foveal_view_rgb_pixels = original_stimuli_rgb_pixels[y_left_up_foveal:y_right_down_foveal,
                                 x_left_up_foveal:x_right_down_foveal]

        # Check the dimensions and pad if necessary
        if foveal_view_rgb_pixels.shape[0] != foveal_view_height or foveal_view_rgb_pixels.shape[
            1] != foveal_view_width:
            # Padding to ensure fixed size
            padded_foveal_view = np.full((foveal_view_height, foveal_view_width, 3), fill_value=255,
                                         dtype=np.uint8)  # Create a white background
            h, w, _ = foveal_view_rgb_pixels.shape

            y_offset = (foveal_view_height - h) // 2
            x_offset = (foveal_view_width - w) // 2

            # Ensure offsets are within bounds
            y_offset = max(y_offset, 0)
            x_offset = max(x_offset, 0)

            padded_foveal_view[y_offset:y_offset + h, x_offset:x_offset + w] = foveal_view_rgb_pixels
            foveal_view_rgb_pixels = padded_foveal_view

        assert foveal_view_rgb_pixels.shape[0] == foveal_view_height and foveal_view_rgb_pixels.shape[
            1] == foveal_view_width

        # # Visualise the image   --  # TODO comment later
        # plt.imshow(foveal_view_rgb_pixels)
        # plt.show()

        return foveal_view_rgb_pixels

    @staticmethod
    def _get_peripheral_view(
            original_stimuli_rgb_pixels,
            peripheral_view_width,
            peripheral_view_height,
    ):
        """
        Get the peripheral view of the image.
        """

        # Resize the original image to the peripheral view size
        peripheral_view_rgb_pixels = np.array(Image.fromarray(original_stimuli_rgb_pixels).resize(
            (peripheral_view_width, peripheral_view_height)))

        # # Visualise the image   --  TODO comment later
        # plt.imshow(peripheral_view_rgb_pixels)
        # plt.show()

        return peripheral_view_rgb_pixels

    @staticmethod
    def _downsample_image(
            original_stimuli_rgb_pixels,
            downsample_image_width,
            downsample_image_height,
    ):
        """
        Get the downsampled image of the original image.
        """

        # Resize the original image to the peripheral view size
        downsampled_image_rgb_pixels = np.array(Image.fromarray(original_stimuli_rgb_pixels).resize(
            (downsample_image_width, downsample_image_height)))

        # # Visualise the image   --  TODO comment later
        # plt.imshow(peripheral_view_rgb_pixels)
        # plt.show()

        return downsampled_image_rgb_pixels

    def _see(self):
        """Update the seen letters if there are any"""
        # Get the fixation position's corresponding letters from the "OCR" image
        foveal_covered_letters = self._ocr_model.test(
            image=self._image_env,  # Comment this because it is not used if the draw_image is False
            fixation_center=(self._fix_x, self._fix_y),
            foveal_size=(self._FOVEA_RECT_WIDTH, self._FOVEA_RECT_HEIGHT),
            target_word_idx=self._target_word_idx_on_image,
            draw_image=False,
        )
        parafoveal_covered_letters = self._ocr_model.test(
            image=self._image_env,  # Comment this because it is not used if the draw_image is False
            fixation_center=(self._fix_x, self._fix_y),
            foveal_size=(self._PARAFOVEA_RECT_WIDTH, self._PARAFOVEA_RECT_HEIGHT),
            target_word_idx=self._target_word_idx_on_image,
            draw_image=False,
        )

        # Update the word's internal representation and infer the word only when sees something
        self._fovea_seen_letters = [''.join(foveal_covered_letters[i]) for i in range(len(self._words_on_image))]
        self._parafovea_seen_letters = [''.join(parafoveal_covered_letters[i]) for i in range(len(self._words_on_image))]

        self._flag_fixate_on_target_word = False
        self._flag_target_word_internal_letters_are_updated = False

        # Traverse each word in the image
        for _idx, _seen_letters in enumerate(self._fovea_seen_letters):
            if _seen_letters != cons.NA:
                if _idx == self._target_word_idx_on_image:
                    self._flag_fixate_on_target_word = True
                    # self._num_steps_on_target_word += 1
                self._is_seen_letters_acrs_img = True
                # Get the fixation's start and end index
                start_idx, end_idx = self._get_start_end_indexes(
                    word=self._words_on_image[_idx],
                    segment=_seen_letters,
                )

                # Update the perception uncertainty
                _flag_internal_letters_are_updated = self._update_perception_uncertainty(
                    encoded_word_counter=self._encoded_words_counters[_idx],
                    current_word_len=self._words_lens_on_image[_idx],
                    fixation_start_index=start_idx,
                    fixation_end_index=end_idx,
                )
                if _idx == self._target_word_idx_on_image and _flag_internal_letters_are_updated is True:
                    self._flag_target_word_internal_letters_are_updated = True

                # Update the internal word representation -- the encoded internal word
                self._encoded_internal_words[_idx] = self._update_internal_word(
                    current_word_len=self._words_lens_on_image[_idx],
                    encoded_word=self._encoded_words[_idx],
                    encoded_internal_word=self._encoded_internal_words[_idx],
                    encoded_word_counter=self._encoded_words_counters[_idx],
                    perception_uncertainty=self._letter_perception_sigma,
                )

                # Infer the word from the internal word representation
                nearest_id, self._encoded_inferred_words[_idx], self._vect_dist, self._inferred_words[_idx] \
                    = self._word_inferencer.infer_word(
                    encoded_internal_word=self._encoded_internal_words[_idx],
                    # word_len=self._words_lens_on_image[_idx],
                )

            else:
                pass

    def _get_visuospatial_info(
            self,
            word_idx: int = None,
    ):
        """
        Get the target word's visuospatial information from the metadata.
        :return: None
        """
        word_metadata = self._image_metadata[cons.md["words metadata"]][word_idx]
        return {
            cons.md["line index"]: int(word_metadata[cons.md["visuospatial info"]][cons.md["line index"]]),
            cons.md["lines number"]: int(word_metadata[cons.md["visuospatial info"]][cons.md["lines number"]]),
            cons.md["word index in line"]: int(word_metadata[cons.md["visuospatial info"]][cons.md["word index in line"]]),
            cons.md["words number in line"]: int(word_metadata[cons.md["visuospatial info"]][cons.md["words number in line"]]),
        }

    @staticmethod
    def _clean_word(word):
        """
        Clean the word by removing all non-alphabetic characters and converting to lowercase.
        :param word: The word to clean.
        :return: The cleaned word.
        """
        return re.sub(r'[^a-zA-Z\s]', '', word).strip().lower()

    def _get_logs(self, reward, done):
        """Log the fixations"""
        self.fixations_logger.append((
            self._steps,
            self._image_env_index,
            self._words_on_image,
            self._num_words_on_img,
            self._target_word,
            self._encoded_target_word,
            self._target_word_len,
            self._target_word_idx_on_image,
            self._norm_fix_x,
            self._norm_fix_y,
            self._fix_x,
            self._fix_y,
            self._letter_perception_sigma if self._is_seen_letters_acrs_img else "Do not see letters",
            self._fovea_seen_letters[self._target_word_idx_on_image],
            self._parafovea_seen_letters[self._target_word_idx_on_image+1] if self._target_word_idx_on_image+1 < self._num_words_on_img else cons.NO_NEXT_WORD,
            self._encoded_internal_words[self._target_word_idx_on_image],
            self._encoded_words_counters[self._target_word_idx_on_image],
            self._inferred_words[self._target_word_idx_on_image],
            self._encoded_inferred_words[self._target_word_idx_on_image],
            self._vect_dist,
            self._flag_fixate_on_target_word,
            self._flag_target_word_internal_letters_are_updated,
            reward,
            self._terminate,  # The terminate action
            done,  # The done (terminate) flag
        ))


class WordInferencer:

    def __init__(self, encoded_lexicon, n_trees=10):
        self.MAX_WORD_LEN = cons.MAX_WORD_LEN
        self._encoded_lexicon = encoded_lexicon
        self._length_based_lexicons = None
        # self._annoy_indices = self._build_annoy_indices(encoded_lexicon, n_trees)
        self._annoy_index = self._build_annoy_index(encoded_lexicon, n_trees)

    def _build_annoy_index(self, encoded_lexicon, n_trees=10):
        # Initialize Annoy index with 16 dimensions and Euclidean metric
        t = AnnoyIndex(self.MAX_WORD_LEN, 'euclidean')

        # Add items to the index
        for i, (word, vec) in enumerate(encoded_lexicon.items()):
            t.add_item(i, vec)

        # Build the index with n_trees
        t.build(n_trees)

        return t

    def infer_word(self, encoded_internal_word):

        """Infer the word from the internal word representation -- find the nearest word in the encoded lexicon space
        to the internal word representation with ANNOY."""

        # Use a faster way -- annoy index to get the nearest word in the lexicon
        nearest_ids, distances = self._annoy_index.get_nns_by_vector(encoded_internal_word, 1,
                                                                     include_distances=True)
        nearest_id = nearest_ids[0]
        distance = distances[0]
        inferred_word = list(self._encoded_lexicon.keys())[nearest_id]
        nearest_vector = self._encoded_lexicon[inferred_word]

        return nearest_id, nearest_vector, distance, inferred_word

    # def _build_annoy_indices(self, encoded_lexicon, n_trees):
    #     """
    #     Implement a length-based pre-selection step to increase the search efficiency.
    #     """
    #     indices = {}    # Dictionary to hold Annoy indices based on word lengths
    #     self._length_based_lexicons = {}  # Dictionary to hold words by their lengths
    #
    #     for word, vec in encoded_lexicon.items():
    #         word_len = len(word)
    #         if word_len not in indices:
    #             indices[word_len] = AnnoyIndex(self.MAX_WORD_LEN, 'euclidean')
    #             self._length_based_lexicons[word_len] = {}
    #
    #         indices[word_len].add_item(indices[word_len].get_n_items(), vec)
    #         self._length_based_lexicons[word_len][word] = vec
    #
    #     # for i, (word, vec) in enumerate(encoded_lexicon.items()):
    #     #     word_len = len(word)
    #     #     if word_len not in indices:
    #     #         indices[word_len] = AnnoyIndex(self.MAX_WORD_LEN, 'euclidean')
    #     #         self._length_based_lexicons[word_len] = {}
    #     #
    #     #     indices[word_len].add_item(i, vec)
    #     #     self._length_based_lexicons[word_len][word] = vec
    #
    #     for index in indices.values():
    #         index.build(n_trees)
    #
    #     return indices
    #
    # def infer_word(self, encoded_internal_word, word_len):
    #     if word_len not in self._annoy_indices:
    #         raise ValueError(f"Word length {word_len} is not in the pre-built Annoy indices.")
    #         # return None, None, None, None
    #
    #     annoy_index = self._annoy_indices[word_len]
    #     nearest_ids, distances = annoy_index.get_nns_by_vector(
    #         encoded_internal_word,
    #         1,
    #         include_distances=True,
    #     )
    #     nearest_id = nearest_ids[0]
    #     distance = distances[0]
    #     # inferred_word = list(self._encoded_lexicon.keys())[nearest_id]
    #     # nearest_vector = self._encoded_lexicon[inferred_word]
    #     inferred_word = list(self._length_based_lexicons[word_len].keys())[nearest_id]
    #     nearest_vector = self._length_based_lexicons[word_len][inferred_word]
    #
    #     return nearest_id, nearest_vector, distance, inferred_word


# Define the test case
class TestWordInferencer:

    def setup(self):
        # Set up a simple encoded lexicon for testing
        self.encoded_lexicon = {
            'hello': [0.4, 0.1, 0.7, 0.7, 0.6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            'world': [0.2, 0.7, 0.5, 0.6, 0.3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            'test':  [0.1, 0.5, 0.6, 0.7, -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            'python':[0.6, 0.4, 0.7, 0.2, 0.5, 0.3,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        }
        self.word_inferencer = WordInferencer(self.encoded_lexicon, n_trees=10)

    def test_infer_word(self):
        # Encode a word that is similar to 'hello'
        encoded_internal_word = [0.1, 0.5, 0.6, 0.7, -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        word_len = 5
        # nearest_id, nearest_vector, distance, inferred_word = (
        #     self.word_inferencer.infer_word(encoded_internal_word, word_len))
        (nearest_id, nearest_vector,
         distance, inferred_word) = self.word_inferencer.infer_word(encoded_internal_word)

        # assert inferred_word == 'hello'
        # assert abs(distance - 0.0) < 1e-1  # Allowing a small tolerance


if __name__ == '__main__':
    test = TestWordInferencer()
    test.setup()
    test.test_infer_word()