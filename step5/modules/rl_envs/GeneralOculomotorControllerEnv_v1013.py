import random
import shutil

import numpy as np
import yaml
import json
import os
import re
import pickle
import math
import cv2
import matplotlib.pyplot as plt

from gymnasium import Env
from gymnasium.spaces import Box, Dict
from annoy import AnnoyIndex
from PIL import Image, ImageDraw, ImageFont

# from memory_profiler import profile
from multiprocessing import shared_memory

import pandas as pd

from step5.utils import auxiliaries as aux
from step5.utils import pseudo_offline_ocr_model as ocr
from step3.gens import image_generator as gens
from step5.utils import constants as cons


class GeneralOculomotorControllerEnv(Env):

    def __init__(
            self,
            shared_metadata_name,
            metadata_size,
            shared_lexicon_name,
            lexicon_size,
            ):

        """
        Created on 13 Oct 2024.
        This is the environment for the oculomotor controller agent.
        It is responsible for seeing letters of a word.
        I plan to move word identification to the working memory, so the agent does not need to do word inference here.
        Only perceive the word (without needing to map which word it is) and fixate on the target word.
        On the other hand, we will introduce a new variable to control the agent's reading urgency -->
            careful reading or fast reading.
        Author: Bai Yunpeng

        This version v1013 is for running under shared memory on multi-processes.
        """

        # Get the current root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Get the mode from the config yaml file
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        print(
            f"{cons.LV_TWO_DASHES}General Oculomotor Controller Environment V1013 -- Deploying the environment in the {self._config['rl']['mode']} mode.")

        # Define constants
        self.MAX_WORD_LEN = cons.MAX_WORD_LEN
        self.MIN_WORD_LEN = cons.MIN_WORD_LEN

        # Connect to the shared memory blocks   # TODO print, we need to make them as dictionaries later, TODO: and also handle the close function
        # Stimuli metadata
        # Access shared memory for metadata
        metadata_shm = shared_memory.SharedMemory(name=shared_metadata_name)
        metadata_bytes = bytes(metadata_shm.buf[:metadata_size])
        self._metadata = pickle.loads(metadata_bytes)
        self._metadata_shm = metadata_shm  # Keep a reference
        # Lexicon
        # Access shared memory for encoded lexicon
        lexicon_shm = shared_memory.SharedMemory(name=shared_lexicon_name)
        lexicon_bytes = bytes(lexicon_shm.buf[:lexicon_size])
        self._encoded_lexicon = pickle.loads(lexicon_bytes)
        self._lexicon_shm = lexicon_shm  # Keep a reference

        # Get the configurations of the image / visual stimulus
        self._STIMULUS_WIDTH = self._metadata["config"]["img size"][0]
        self._STIMULUS_HEIGHT = self._metadata["config"]["img size"][1]
        # self._WORD_PIXEL_SIZE = cons.config["concrete_configs"]["word_size"]
        self._WORD_PIXEL_SIZE = self._metadata["config"]["word size"]
        # self._FOVEA_RADIUS = cons.FOVEA_FACTOR * self._WORD_PIXEL_SIZE
        # self._FOVEA_RECT_WIDTH = cons.config["concrete_configs"]["foveal_size"][0]
        self._FOVEAL_RECT_WIDTH = self._metadata["config"]["foveal size"][0]
        # self._FOVEA_RECT_HEIGHT = cons.config["concrete_configs"]["foveal_size"][1]
        self._FOVEAL_RECT_HEIGHT = self._metadata["config"]["foveal size"][1]
        # self._PARAFOVEA_RECT_WIDTH = cons.config["concrete_configs"]["parafoveal_size"][0]
        self._PARAFOVEA_RECT_WIDTH = self._metadata["config"]["parafoveal size"][0]
        # self._PARAFOVEA_RECT_HEIGHT = cons.config["concrete_configs"]["parafoveal_size"][1]
        self._PARAFOVEA_RECT_HEIGHT = self._metadata["config"]["parafoveal size"][1]
        self._PERIPHERAL_RECT_WIDTH = self._metadata["config"]["peripheral size"][0]
        self._PERIPHERAL_RECT_HEIGHT = self._metadata["config"]["peripheral size"][1]
        # self._TRAINING_FOVEA_AND_PARAFOVEA_RECT_WIDTH = cons.config["concrete_configs"]["training_foveal_and_peripheral_size"][0]
        # self._TRAINING_FOVEA_AND_PARAFOVEA_RECT_WIDTH = cons.config["concrete_configs"]["training_foveal_and_peripheral_size"][1]
        self._TRAINING_RECT_WIDTH = self._metadata["config"]["training foveal and peripheral size"][0]
        self._TRAINING_RECT_HEIGHT = self._metadata["config"]["training foveal and peripheral size"][1]

        # Get the number of images
        self._num_image_envs = self._metadata[cons.md["config"]][cons.md["num images"]]
        self._image_filename = None

        # self._num_words_per_img = self._metadata[cons.md["config"]][cons.md["num words"]]
        self._num_words_on_img = None
        self._norm_num_words_per_img = None

        # Get image-level metadata
        self._images_metadata = self._metadata[cons.md["images"]]
        self._image_metadata = None
        self._words_on_image = None
        self._words_lens_on_image = None

        # Define state related variables
        self._image_env = None  # The word image environment does the agent tries to identify
        self._normalised_image_env_foveal_patch_pixels = None  # The pixel values of the image environment
        self._relative_bbox_foveal_patch = None  # The relative bounding box of the foveal patch
        self._normalised_masked_downsampled_peripheral_view_pixels = None  # The downsampled and masked peripheral view of the target word
        self._image_env_index = None
        self.target_word = None  # The word that the agent tries to identify among many words in the image, randomly sampled from the image
        self._target_word_metadata = None  # The metadata of the target word
        self._next_word_metadata = None  # The metadata of the next word
        self._target_word_len = None  # The length of the targeted word
        self._target_word_center_x = None  # The x coordinate of the target word's center
        self._target_word_center_y = None  # The y coordinate of the target word's center
        self._target_word_left_x = None  # The x coordinate of the target word's left
        self._target_word_sigma = None  # The sigma factor of the target word's center
        self._target_word_center_noisy_x = None  # The noisy x coordinate of the target word's center
        self._target_word_center_noisy_y = None  # The noisy y coordinate of the target word's center
        self._target_word_bbox = None  # The bounding box of the target word
        self._encoded_target_word = None  # The encoded target word that the agent tries to identify
        self._target_word_idx_on_image = None  # The index of the target word in the image
        self._norm_target_word_idx_on_image = None  # The normalized index of the target word in the image
        self._encoded_words = None  # The encoded words that the agent tries to identify
        # (several 16-dim vectors, dependent on the number of words on the image, range from -1 to 1)
        self._encoded_words_counters = None  # The counter for the encoded word
        self._encoded_words_counters_offsets = None  # The counter for the encoded word, which will set empty positions -1 to provide more information to the agent: which letters are seen (1), which are guessed (0), which are no letters (-1)
        self._norm_target_word_len = None  # The normalized length of the targeted word
        self._word_x_init = None  # The initial x coordinate of the words on the image -- hints in the observation to speed up training
        self._word_y_init = None  # The initial y coordinate of the words on the image

        # Define the perception noise
        self._letter_perception_sigma_range = [0, 1]  # An empirical range that sample out the letter sigma factors
        self._init_letter_perception_sigma = None
        self._letter_perception_sigma = None  # An empirical value that controls the initial perception noise -- initial letter sigma
        self._flag_fixate_on_target_word = None  # A flag that indicates whether the agent is on the target word: it could give the agent more continuous information
        self._flag_target_word_new_letters_updated = None  # A flag that indicates whether the internal letters are updated
        self._flag_fixate_inside_bbox = None  # A flag that indicates whether the agent is inside the target word's bounding box

        self._encoded_internal_words = None  # The encoded internal words (all words on the image) that is presented in the observation

        self._vect_dist = None  # Define the nearest distance of the word in the encoded vector space

        # Action / proprioception information
        # The current step
        self.norm_fix_x = None  # The x coordinate of the fixation position
        self.norm_fix_y = None  # The y coordinate of the fixation position
        self._fix_x = None  # The x coordinate of the fixation position
        self._fix_y = None  # The y coordinate of the fixation position
        # The previous step
        self._norm_previous_fix_x = None    # The previous x coordinate of the fixation position
        self._norm_previous_fix_y = None    # The previous y coordinate of the fixation position
        self._previous_fix_x = None    # The previous x coordinate of the fixation position
        self._previous_fix_y = None    # The previous y coordinate of the fixation position
        # Set the saccadic noise factor, reasons: 1) more realisitic, replicate human behaviors with higher fidelity; 2) learn a policy that is more robust; 3) the simulation/agent could better capture human stochasticity
        self._rho_spatial = 0.09  # The empirical factor set as 0.09, Reference: An Adaptive Model of Gaze-based Selection

        self.foveal_seen_letters = None  # The letters that the agent has seen in the fovea
        self.parafovea_seen_letters = None  # The letters that the agent has seen in the parafovea
        self._is_seen_letters_acrs_img = None  # A flag that the agent sees letters

        # Define the task completion value
        self._activation_level = None   # This is abstracted as a single value of probability, derived from the completeness (the progress the word has been processed)
        self._pre_activation_level = None

        # Define the recognizing flag: True for successful recognition, False for unsuccessful recognition
        self._recognize_flag = None

        # Weighting factors for rewards
        self._w_time_penalty_lower_bound = 1
        self._w_time_penalty_upper_bound = 20
        self._w_time_penalty = None
        self._w_time_pressure = None        # Time pressure weighting factor

        # Define the parafoveal preview letters
        self.parafoveal_preview_letters = None

        # Define the training -- episode and rl related variables
        self._steps = None
        self.ep_len = 50        # TODO maybe later when testing/simulating, try the shorter episodes
        # The length of the episode, smaller, faster to converge -- depends on the env size, if small env, yes.
        self._terminate = cons.NA
        self._truncated = cons.NA
        self.is_terminate_step = cons.NA

        # Define the action space
        # Two normalized actions from -1 to 1, representing the x and y coordinates of the gaze;
        # The third action is the termination action
        self.action_space = Box(low=-1, high=1, shape=(3,))

        # Define the observation space
        self._original_img_env_width = self._STIMULUS_WIDTH
        self._original_img_env_height = self._STIMULUS_HEIGHT
        self._foveal_width = self._FOVEAL_RECT_WIDTH
        self._foveal_height = self._FOVEAL_RECT_HEIGHT
        self._peripheral_width = self._PERIPHERAL_RECT_WIDTH
        self._peripheral_height = self._PERIPHERAL_RECT_HEIGHT
        self._training_foveal_and_peripheral_width = self._TRAINING_RECT_WIDTH
        self._training_foveal_and_peripheral_height = self._TRAINING_RECT_HEIGHT
        # self._num_stateful_info_obs = 11 + 2 * self.MAX_WORD_LEN
        self._num_stateful_info_obs = 9 + 2 * self.MAX_WORD_LEN
        self.observation_space = Dict({
            "vision": Box(low=-1, high=1, shape=(1, 2*self._training_foveal_and_peripheral_height, self._training_foveal_and_peripheral_width)),
            "stateful information": Box(low=-1, high=1, shape=(self._num_stateful_info_obs,))
        })

        # Define the logger
        self.fixations_logger = None
        self.logger = None

        # Initialize the pseudo offline ocr model
        _word_size_in_px = self._metadata[cons.md["config"]][cons.md["word size"]]
        self._font = ImageFont.truetype(cons.FONT_PATH, _word_size_in_px)
        self._ocr_model = None

    # @profile
    def reset(self, seed=None, inputs=None, ep_idx=None):

        self._steps = 0

        # Initialize the uncertainty levels -- sigma
        self._init_letter_perception_sigma = 0.5

        if inputs is None:
            # Randomly sample an image environment
            self._image_env_index = np.random.choice(self._num_image_envs, 1)[0]
        else:
            self._image_env_index = inputs['image_index']

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
            self.target_word = np.random.choice(self._words_on_image, 1)[0]
            # Get the target word's index - randomly sample one
            self._target_word_idx_on_image = self._words_on_image.index(self.target_word)
        else:
            self._target_word_idx_on_image = np.clip(inputs['target_word_index'], 0, self._num_words_on_img - 1)
            self.target_word = self._words_on_image[self._target_word_idx_on_image]

        # Get the target word's bounding box
        self._target_word_bbox = self._image_metadata[cons.md["words metadata"]][self._target_word_idx_on_image][cons.md["word_bbox"]]

        # Get the env pixels from the metadata
        self._normalised_image_env_foveal_patch_pixels = np.array(self._image_metadata[cons.md["words metadata"]][self._target_word_idx_on_image][cons.md["normalised_foveal_patch"]])  # TODO change this to only around the target word, do not need others. So there will be more images, but only
        self._relative_bbox_foveal_patch = self._image_metadata[cons.md["words metadata"]][self._target_word_idx_on_image][cons.md["relative_bbox_foveal_patch"]]

        # Get the target word's downsampled and masked peripheral view from the metadata
        self._normalised_masked_downsampled_peripheral_view_pixels = np.array(self._image_metadata[cons.md["words metadata"]][self._target_word_idx_on_image][cons.md["normalised_masked_downsampled_peripheral_view"]])[np.newaxis, :, :]

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

        self._target_word_len = len(self.target_word)
        self._norm_target_word_idx_on_image = aux.normalise(self._target_word_idx_on_image, 0, cons.MAX_NUM_WORDS - 1, cons.ZERO, cons.ONE)

        # Encode the sampled word from a string list to a MAX_WORD_LEN-dim vector from 0 to 1
        self._encoded_target_word = self._encoded_lexicon[self.target_word]

        # Encode all the words in the image for the internal word representation
        self._encoded_words = [self._encoded_lexicon[word] for word in self._words_on_image]

        # Initialize the encoded words with all zeros
        self._encoded_words_counters = [[0] * self.MAX_WORD_LEN for _ in range(len(self._words_on_image))]
        # Initialize with negative ones
        # Update with the offsets
        self._encoded_words_counters[self._target_word_idx_on_image] = np.array(
            self._encoded_words_counters[self._target_word_idx_on_image]) + np.array(
            self._encoded_words_counters_offsets[self._target_word_idx_on_image])

        # Initialize the sampled word's observation vector -- internal representation
        self._encoded_internal_words = [[cons.NEGATIVE_ONE] * self.MAX_WORD_LEN for _ in range(len(self._words_on_image))]

        # Initialize the seen letters
        self.foveal_seen_letters = ['' for _ in range(len(self._words_on_image))]
        self.parafovea_seen_letters = ['' for _ in range(len(self._words_on_image))]
        self._is_seen_letters_acrs_img = [False for _ in range(len(self._words_on_image))]

        if inputs is None:
            self.norm_fix_x = np.random.uniform(-1, 1)
            self.norm_fix_y = np.random.uniform(-1, 1)
        else:
            if inputs['norm_fix_x'] is None or inputs['norm_fix_y'] is None:
                self.norm_fix_x = np.random.uniform(-1, 1)
                self.norm_fix_y = np.random.uniform(-1, 1)
            else:
                self.norm_fix_x = inputs['norm_fix_x']
                self.norm_fix_y = inputs['norm_fix_y']
        # Initialize the previous fixation position
        self._norm_previous_fix_x = self.norm_fix_x
        self._norm_previous_fix_y = self.norm_fix_y
        # Initialize the fixation positions
        self._fix_x, self._fix_y = self._get_fix_xy()

        # Initialize the weighting factor for the penalty reward -- But actually this is not working right now
        if inputs is None:
            # Randomly sample a penalty reward weighting factor as an integer   # TODO remove this later
            self._w_time_penalty = np.random.randint(self._w_time_penalty_lower_bound, self._w_time_penalty_upper_bound + 1)
            # Ramdomly sample a time pressure weighting factor as an integer -- randomly sample from 0 to 1
            self._w_time_pressure = np.random.uniform(0, 1)
        else:
            # Get the assigned penalty reward weighting factor      # TODO remove this later
            self._w_time_penalty = inputs['time_awareness_weight']
            # Get the assigned time pressure weighting factor
            self._w_time_pressure = inputs['time_pressure_weight']      # TODO check the simulation later

        # Initialize the flag for recognizing the target word
        self._recognize_flag = False

        # Initialize the terminate action related flag
        self._terminate = False
        self._truncated = False
        self.is_terminate_step = False

        # Initialize the task completion flag
        self._activation_level: float = 0.0
        self._pre_activation_level: float = 0.0

        self._flag_fixate_on_target_word = False
        self._flag_target_word_new_letters_updated = False
        self._flag_fixate_inside_bbox = False

        # Reset the parafoveal preview letters
        self.parafoveal_preview_letters = ''

        # Initialize the logger
        self.fixations_logger = []

        # Log the first fixation
        self._get_logs_for_test(
            reward=self._compute_reward(done=False),
            done=False,
        )

        return self._get_obs(), {}

    # @profile
    def step(self, action):

        # Update the steps
        self._steps += 1

        # Get actions
        self.norm_fix_x, self.norm_fix_y = action[0:2]  # The first two actions
        self._terminate = True if action[2] > cons.ZERO else False  # The third action of determining the termination --> Reserve it because we need to model skim reading as well

        # Reset the flag at the beginning of each step
        self._is_seen_letters_acrs_img = False
        self._flag_fixate_inside_bbox = False
        self._flag_fixate_on_target_word = False
        self._flag_target_word_new_letters_updated = False

        if self._terminate is False:
            # Convert the normalized fixation position to the pixel-wise fixation position
            self._fix_x, self._fix_y = self._get_fix_xy()

            # Update the seen letters if there are any
            self._sample_letters_from_pixels()
        else:
            # Update the recognition flag
            self._recognize_flag = self._recognize(p=self._activation_level)

        # Update the task completion flag
        self._pre_activation_level = self._activation_level
        self._activation_level = self._compute_completeness(
            encoded_word_counter=self._encoded_words_counters[self._target_word_idx_on_image],
            current_word_len=self._target_word_len
        )

        # Check if the episode is done
        done = self._is_done()

        # Compute the reward
        reward = self._compute_reward(done=done)

        # Check if the episode is truncated
        truncated = self._truncated

        info = {}

        # Log the fixation for testing
        self._get_logs_for_test(reward=reward, done=done)

        return self._get_obs(), reward, done, truncated, info

    def render(self, mode='human'):
        pass

    @staticmethod
    def _recognize(p: float):
        """Recognize the target word based on the given activation level"""
        return random.choices([True, False], weights=[p, 1 - p], k=1)[0]

    def _get_fix_xy(self):
        """Get the fixation position in pixel coordinates, considering saccadic noise."""
        # Get the target fixation position
        target_fix_x = aux.normalise(self.norm_fix_x, -1, 1, 0, self._original_img_env_width)
        target_fix_y = aux.normalise(self.norm_fix_y, -1, 1, 0, self._original_img_env_height)

        # Get the previous fixation position
        previous_fix_x = aux.normalise(self._norm_previous_fix_x, -1, 1, 0, self._original_img_env_width)
        previous_fix_y = aux.normalise(self._norm_previous_fix_y, -1, 1, 0, self._original_img_env_height)

        # Calculate the saccade vector components and amplitude
        delta_x = target_fix_x - previous_fix_x
        delta_y = target_fix_y - previous_fix_y
        saccade_amplitude = np.sqrt(delta_x ** 2 + delta_y ** 2)

        # Determine the standard deviation of the saccadic noise
        std_noise = self._rho_spatial * saccade_amplitude

        # Generate independent Gaussian noise for x and y
        noise_x = np.random.normal(0, std_noise)
        noise_y = np.random.normal(0, std_noise)

        # Add the noise to the target fixation position
        fix_x = target_fix_x + noise_x
        fix_y = target_fix_y + noise_y

        # Update the previous normalized fixation position
        self._norm_previous_fix_x = self.norm_fix_x
        self._norm_previous_fix_y = self.norm_fix_y

        return fix_x, fix_y


    def _get_eccentricity(self):
        return math.sqrt((self._target_word_center_x - self._fix_x) ** 2 + (self._target_word_center_y - self._fix_y) ** 2)

    # @profile
    def _get_obs(self):
        """Get the observations"""
        """Get the visual perception observation input"""

        # Check whether the current fixation is on the target word's bounding box
        if self._target_word_bbox[0] <= self._fix_x <= self._target_word_bbox[2] and self._target_word_bbox[1] <= self._fix_y <= self._target_word_bbox[3]:
            # Get the foveal view
            normalised_downsampled_foveal_view_gray_pixels = self._get_and_downsample_foveal_view(
                normalised_foveal_patch_pixels=self._normalised_image_env_foveal_patch_pixels,
                relative_bbox_to_foveal_patch_left_up_corner=self._relative_bbox_foveal_patch,
                global_bbox=self._target_word_bbox,
                global_fix_x=self._fix_x,
                global_fix_y=self._fix_y,
                foveal_view_width=self._foveal_width,
                foveal_view_height=self._foveal_height,
                downsample_image_width=self._training_foveal_and_peripheral_width,
                downsample_image_height=self._training_foveal_and_peripheral_height,
            )
            # Update the flag that the agent is inside the bbox
            self._flag_fixate_inside_bbox = True
        else:
            # If the fixation is not on the target word's bounding box, then the foveal view is all white --  all are ones
            normalised_downsampled_foveal_view_gray_pixels = np.ones((self._training_foveal_and_peripheral_height, self._training_foveal_and_peripheral_width))

        # Transpose the two pixels to [C, H, W]
        norm_downsampled_foveal_view_gray_pixels = normalised_downsampled_foveal_view_gray_pixels[np.newaxis, :, :]
        norm_peripheral_view_gray_pixels = self._normalised_masked_downsampled_peripheral_view_pixels

        # Stack them along the height dimension
        vision = np.concatenate([norm_downsampled_foveal_view_gray_pixels, norm_peripheral_view_gray_pixels], axis=1)

        """Get the stateful information observation input"""
        # The stateful information
        norm_remaining_ep_len = (self.ep_len - self._steps) / self.ep_len * 2 - 1

        norm_terminate = 1 if self._terminate else -1

        norm_recognize_flag = 1 if self._recognize_flag else -1

        norm_inside_bbox_flag = 1 if self._flag_fixate_inside_bbox else -1

        norm_fix_x = self.norm_fix_x
        norm_fix_y = self.norm_fix_y

        norm_activation_level = self._activation_level

        # norm_penalty_weight = aux.normalise(self._w_time_penalty, self._w_time_penalty_lower_bound, self._w_time_penalty_upper_bound, -1, 1)        # TODO remove this later
        norm_time_pressure_weight = self._w_time_pressure

        norm_update_new_letters = 1 if self._flag_target_word_new_letters_updated else -1

        # Encoded words
        norm_internal_word = self._encoded_internal_words[self._target_word_idx_on_image]
        norm_word_counter = self._encoded_words_counters[self._target_word_idx_on_image]

        # Integrate the observations
        stateful_information = np.concatenate([
            [
                norm_remaining_ep_len,
                norm_terminate,
                norm_recognize_flag,
                norm_inside_bbox_flag,
                norm_fix_x, norm_fix_y,
                # norm_penalty_weight,
                norm_time_pressure_weight,
                norm_activation_level,
                norm_update_new_letters,
            ],
            norm_internal_word,
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

    def _update_target_word_counters(
            self,
            start_index: int,
            end_index: int,
    ):
        """Update the encoded words counters"""
        # Update with the new fixations
        self._encoded_words_counters[self._target_word_idx_on_image][start_index:end_index] = [cons.ONE] * (end_index - start_index)

    def _update_perception_uncertainty(
            self,
            encoded_word_counter,
            current_word_len,
            fixation_start_index,
            fixation_end_index
    ):
        """Update the letter sigma based on the number of letters that have been observed."""
        is_new_letters_updated = False
        # Collect the number of directly fixated letters before updating
        observed_letters_before = sum(1 for item in encoded_word_counter if item == 1)

        # Update the counter of directly fixated letters
        encoded_word_counter[fixation_start_index:fixation_end_index] = ([cons.ONE] * (fixation_end_index - fixation_start_index))

        # Number of explicitly observed letters
        observed_letters = sum(1 for item in encoded_word_counter if item == 1)

        proportion = self._compute_completeness(encoded_word_counter, current_word_len)

        self._letter_perception_sigma = self._init_letter_perception_sigma * (1 - proportion)

        # Check if new internal letters have been observed
        if observed_letters_before < observed_letters:
            is_new_letters_updated = True

        return is_new_letters_updated

    @staticmethod
    def _compute_completeness(encoded_word_counter, current_word_len):
        return np.clip(sum(1 for item in encoded_word_counter if item == 1) / current_word_len, 0, 1)

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

        We reserve this because we want to model how parafoveal view process adjacent letters.
        """

        # Apply the noisy masking first
        encoded_internal_word[:current_word_len] = self._add_noise_to_letters(encoded_word[:current_word_len], perception_uncertainty)

        # Get all the indexes of counted letters that are one, then update the internal word representation
        for idx, value in enumerate(encoded_word_counter):
            if value == cons.ONE:
                # Copy the corresponding letter from self._encoded_word to self._encoded_internal_word
                encoded_internal_word[idx] = encoded_word[idx]

        return encoded_internal_word

    def _compute_reward(
            self,
            done: bool = False,
    ):
        time_pressure_weight = aux.normalise(self._w_time_pressure, 0, 1, 1, 10)
        constant_coeff = 100
        time_penalty = time_pressure_weight * 0.02 * constant_coeff * self._reward_shaping(    # TODO decrease the reward penalties with larger and more complex environment
            center_x=self._target_word_center_x,
            center_y=self._target_word_center_y,
            fix_x=self._fix_x,
            fix_y=self._fix_y,
        )

        # Get the final step reward: Bonus
        if done is True:
            if self._recognize_flag is True:
                Bonus = 10
            else:
                Bonus = -10
        else:
            Bonus = 0

        return time_penalty + Bonus

    def _reward_shaping(self, center_x, center_y, fix_x, fix_y, k=0.005, w=0.1):    # TODO the reward shaping does not help much when the ambiguious parts increase; need to adapt the reward and observation
        """Apply a position/distance-based reward shaping"""
        distance = np.sqrt((center_x - fix_x) ** 2 + (center_y - fix_y) ** 2)
        if self._flag_fixate_on_target_word is False:
            return w * (np.exp(-k * distance) - 1)
        else:
            return w * -0.01

    def _is_done(self):

        # If the agent wants to terminate the trial
        if self._terminate is True:
            # Set the flag that this is the terminate step
            self.is_terminate_step = True
            return True

        # Time out
        if self._steps >= self.ep_len:
            self._truncated = True
            # Set the flag that this is the terminate step
            self.is_terminate_step = True
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
    def _get_and_downsample_foveal_view(
            normalised_foveal_patch_pixels,
            relative_bbox_to_foveal_patch_left_up_corner,
            global_bbox,
            global_fix_x,
            global_fix_y,
            foveal_view_width,
            foveal_view_height,
            downsample_image_width,
            downsample_image_height,
    ):
        """
        Get the foveal view of the image with padding if necessary to ensure a fixed size of 40x40 pixels.
        """

        # Get the relative fixation position relative to the bbox's left top corner = fix_x, fix_y - bbox left up corner
        relative_fix_x_to_bbox = global_fix_x - global_bbox[0]
        relative_fix_y_to_bbox = global_fix_y - global_bbox[1]

        # We have relative bbox = bbox left up corner - foveal patch left up corner

        # Get the relative fixation position to foveal path left up corner = relative fix to bbox + bbox relative position to foveal patch left up corner
        relative_fix_x_to_bbox = relative_fix_x_to_bbox + relative_bbox_to_foveal_patch_left_up_corner[0]
        relative_fix_y_to_bbox = relative_fix_y_to_bbox + relative_bbox_to_foveal_patch_left_up_corner[1]

        # Get the relative position of the foveal view using foveal width and height, and fixation position
        x_left_up_foveal = max(int(relative_fix_x_to_bbox - foveal_view_width / 2), 0)
        y_left_up_foveal = max(int(relative_fix_y_to_bbox - foveal_view_height / 2), 0)
        x_right_down_foveal = min(int(relative_fix_x_to_bbox + foveal_view_width / 2), normalised_foveal_patch_pixels.shape[1])
        y_right_down_foveal = min(int(relative_fix_y_to_bbox + foveal_view_height / 2), normalised_foveal_patch_pixels.shape[0])

        # Crop the foveal view without padding if within bounds
        foveal_view_gray_pixels = normalised_foveal_patch_pixels[y_left_up_foveal:y_right_down_foveal, x_left_up_foveal:x_right_down_foveal]

        assert foveal_view_gray_pixels.shape[0] == foveal_view_height and foveal_view_gray_pixels.shape[1] == foveal_view_width, f"The foveal view shape is not correct: {foveal_view_gray_pixels.shape}"

        # Downsample the foveal view using OpenCV for faster performance
        downsampled_foveal_view_gray_pixels = cv2.resize(foveal_view_gray_pixels,
                                                         (downsample_image_width, downsample_image_height),
                                                         interpolation=cv2.INTER_LINEAR)

        return downsampled_foveal_view_gray_pixels

    def _sample_letters_from_pixels(self):
        """Update the seen letters if there are any"""
        # Get the fixation position's corresponding letters from the "OCR" image
        foveal_covered_letters, parafoveal_covered_letters = self._ocr_model.test(
            image=self._image_env,
            fixation_center=(self._fix_x, self._fix_y),
            foveal_size=(self._FOVEAL_RECT_WIDTH, self._FOVEAL_RECT_HEIGHT),
            parafoveal_size=(self._PARAFOVEA_RECT_WIDTH, self._PARAFOVEA_RECT_HEIGHT),
            target_word_idx=self._target_word_idx_on_image,
            draw_image=False,
        )

        self._flag_fixate_on_target_word = False
        self._flag_target_word_new_letters_updated = False

        # Get the range of indices to check -- since this is the range of the target word, the end index should be one more
        range_bound_start_check_idx = max(0, self._target_word_idx_on_image - cons.WORDS_INTEREST_RANGE)
        range_bound_end_check_idx = min(len(self._words_on_image), self._target_word_idx_on_image + cons.WORDS_INTEREST_RANGE + 1)

        # Update only the words within the specified range
        self.foveal_seen_letters = [cons.NA] * len(self._words_on_image)
        self.parafovea_seen_letters = [cons.NA] * len(self._words_on_image)

        # Update the parafoveal preview letters for the next word
        self.parafoveal_preview_letters = self.parafovea_seen_letters[self._target_word_idx_on_image + 1] if self._target_word_idx_on_image + 1 < self._num_words_on_img else cons.NO_NEXT_WORD,

        # Update the internal representation only for the relevant indices
        for i in range(range_bound_start_check_idx, range_bound_end_check_idx):
            self.foveal_seen_letters[i] = ''.join(foveal_covered_letters[i])
            self.parafovea_seen_letters[i] = ''.join(parafoveal_covered_letters[i])

        # Traverse only the words around the target word
        for _idx in range(range_bound_start_check_idx, range_bound_end_check_idx):
            _seen_letters = self.foveal_seen_letters[_idx]
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
                _flag_new_letters_are_updated = self._update_perception_uncertainty(
                    encoded_word_counter=self._encoded_words_counters[_idx].copy(),
                    current_word_len=self._words_lens_on_image[_idx],
                    fixation_start_index=start_idx,
                    fixation_end_index=end_idx,
                )   # TO note that the _flag_new_letters_are_updated does not work, it is only a placeholder now here.
                if _idx == self._target_word_idx_on_image and _flag_new_letters_are_updated is True:
                    self._flag_target_word_new_letters_updated = True

                # Update the encoded words counters -- Only update using start_index and end_index for the target word
                if _idx == self._target_word_idx_on_image:
                    self._update_target_word_counters(start_index=start_idx, end_index=end_idx)

                # Update the internal word representation -- the encoded internal word
                self._encoded_internal_words[_idx] = self._update_internal_word(
                    current_word_len=self._words_lens_on_image[_idx],
                    encoded_word=self._encoded_words[_idx],
                    encoded_internal_word=self._encoded_internal_words[_idx].copy(),
                    encoded_word_counter=self._encoded_words_counters[_idx].copy(),
                    perception_uncertainty=self._letter_perception_sigma,
                )

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
            cons.md["word index in line"]: int(
                word_metadata[cons.md["visuospatial info"]][cons.md["word index in line"]]),
            cons.md["words number in line"]: int(
                word_metadata[cons.md["visuospatial info"]][cons.md["words number in line"]]),
        }

    def _get_logs_for_test(self, reward, done):
        """Log the fixations"""
        self.fixations_logger.append((
            self._steps,
            self._image_env_index,
            self._words_on_image,
            self._num_words_on_img,
            self.target_word,
            self._encoded_target_word,
            self._target_word_len,
            self._target_word_idx_on_image,
            self.norm_fix_x,
            self.norm_fix_y,
            self._fix_x,
            self._fix_y,
            self._letter_perception_sigma if self._is_seen_letters_acrs_img else "Do not see letters",
            self.foveal_seen_letters[self._target_word_idx_on_image],
            self.parafovea_seen_letters[self._target_word_idx_on_image + 1] if self._target_word_idx_on_image + 1 < self._num_words_on_img else cons.NO_NEXT_WORD,
            self._encoded_internal_words[self._target_word_idx_on_image],
            self._encoded_words_counters[self._target_word_idx_on_image],
            "",
            self._recognize_flag,
            self._activation_level,
            self._flag_fixate_on_target_word,
            self._flag_target_word_new_letters_updated,
            reward,
            self._terminate,  # The terminate action
            done,  # The done (terminate) flag
        ))

    # def get_logs(self, reward, done):
    def get_logs(self):
        """Log the fixations for simulation. This should always be consistent with the logger for testing."""
        self.logger = {
            'episode': '',
            'step': self._steps,
            'agent': 'OculomotorController',
            'image id': self._image_env_index,
            'words': self._words_on_image,
            'num words': self._num_words_on_img,
            'target word': self.target_word,
            'target word len': self._target_word_len,
            'target word id': self._target_word_idx_on_image,
            'target word visuospatial info': self._get_visuospatial_info(word_idx=self._target_word_idx_on_image),
            'next word visuospatial info': self._get_visuospatial_info(word_idx=self._target_word_idx_on_image + 1) if self._target_word_idx_on_image + 1 < self._num_words_on_img else cons.NO_NEXT_WORD,
            'encoded target word': self._encoded_target_word,
            'norm_fix_x': self.norm_fix_x,
            'norm_fix_y': self.norm_fix_y,
            'fix_x': self._fix_x,
            'fix_y': self._fix_y,
            'foveal seen letters': self.foveal_seen_letters[self._target_word_idx_on_image],
            'parafoveal seen letters': self.parafovea_seen_letters[self._target_word_idx_on_image + 1] if self._target_word_idx_on_image + 1 < self._num_words_on_img else cons.NO_NEXT_WORD,
            'encoded internal word': self._encoded_internal_words[self._target_word_idx_on_image],
            'word counter': self._encoded_words_counters[self._target_word_idx_on_image],
            'recognize flag': self._recognize_flag,
            'completeness': self._activation_level,
            # 'done': done,
            'done action': self._terminate,
            # 'reward': reward,
        }
    
    def close(self):
        # Close shared memory references
        self._metadata_shm.close()
        self._lexicon_shm.close()

if __name__ == '__main__':
    pass
