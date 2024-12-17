import math
import os
import yaml
import random

import numpy as np

from gymnasium import Env
from gymnasium.spaces import Box, Discrete, Dict

from collections import OrderedDict

from step5.utils import auxiliaries as aux
from step5.utils import constants as const


class SentenceLevelControllerEnv(Env):

    def __init__(self):
        """
        Create on 25 September 2024.
        This is the environment for the RL-based intermediate level -- sentence-level control agent: it controls word skippings in a given sentence
        Based on the current reading progress, time constraints/pressure, and following words' appraisals/activation levels,
            the agent needs to decide whether to skip the next word or not.
        When training, all appraisal level data are artificial/imitated.

        TODO: maybe we could add word-level regression here as well?

        TODO tunable parameters:
            - the time constraint weights for 30s, 60s, 90s

        Author: Bai Yunpeng
        """

        # Get the current root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Get the mode from the config yaml file
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"{const.LV_TWO_DASHES}RL Word Skip Controller Environment -- Deploying the environment in the {self._config['rl']['mode']} mode.")

        # Define the stateful information
        # Sentence level -- only one sentence
        pass
        # Word level
        self.num_words_in_sentence = None
        self.num_words_read_in_sentence = None
        self._num_remaining_words = None
        # Predictability states
        self._predictability_states = None
        # Contain both the appraisal levels and the decisions -- Note that only when the agent is fixating on a word,
        #   this word's appraisal level will be one and the next word's appraisal will be sampled.

        # Define the state-related variables
        self.fixated_word_index = None
        # Time constraints
        self._time_constraint_level_key = None
        self._norm_time_constraint_level = None
        # Time constraint related constants -- TODO make these tunable parameters later if needed
        self.THIRTY_SECONDS_WEIGHT = 0.4
        self.SIXTY_SECONDS_WEIGHT = 0.6
        self.NINETY_SECONDS_WEIGHT = 0.8
        # Counters
        self.num_words_skipped_in_sentence = None
        self.num_saccades_on_word_level = None

        # Define the RL env spaces
        self._steps = None
        self.ep_len = None  # The length of the episode is a variable in this case
        self.max_ep_len = self.NINETY_SECONDS_WEIGHT * const.MAX_NUM_WORDS_PER_SENTENCE
        self._terminate = None
        self._truncated = None

        # Define the action space
        self.READ_ACTION = 0
        self.SKIP_ACTION = 1
        self.UNDEFINED_ACTION = -1
        self.action_space = Discrete(2)     # 0: read; 1: skip the next word (one word only)

        # Define the observation space
        self._num_stateful_info_obs = 5
        self.observation_space = Box(low=0, high=1, shape=(self._num_stateful_info_obs,))

    def reset(self, seed=None, inputs: dict = None):
        """
        Reset the environment to the initial state.
        """

        self._steps = 0

        self.fixated_word_index = -1

        self.num_words_skipped_in_sentence = 0         # The number of words skipped -- the counter of the agent's skipping behaviors
        self.num_saccades_on_word_level = 0   # The number of saccades across words

        # Initialize the number of words in a sentence
        self._init_num_words_in_sentence(inputs=inputs)

        # Initialize the time constraint -- randomly pick one, and the episode length will be picked as well.
        self._init_time_constraint(
            inputs=inputs,
            time_constraint_weights={"30S": self.THIRTY_SECONDS_WEIGHT, "60S": self.SIXTY_SECONDS_WEIGHT, "90S": self.NINETY_SECONDS_WEIGHT,},
        )

        # Initialize the appraisal levels
        self._init_appraisal_states()

        return self._get_obs(), {}

    def step(self, action, external_llm_predictability: float = None):
        """
        Execute one step in the environment.
        """

        done = False
        reward = 0.0

        # Update the steps
        self._steps += 1

        # Take the action
        if action == 0:    # Read the next word
            if self.have_words_to_read():
                self._read_next_word(predictability=external_llm_predictability)
                reward = self._compute_continue_reading_reward()
            else:
                done = True
                reward = self._compute_terminate_reading_reward()
        elif action == 1:  # Skip the next word
            if self.have_words_to_skip():
                self._skip_next_word(predictability=external_llm_predictability)
                reward = self._compute_skip_word_reward()
            else:
                done = True
                reward = self._compute_terminate_reading_reward()
        else:
            # Invalid action, do nothing
            print(f"Invalid action: {action}, should be 0 or 1.")
            reward = 0.0

        # Check if the episode should be terminated -- if the sentence has been finished
        if self._check_terminate():
            done = True
            reward = self._compute_terminate_reading_reward()

        # Check if the episode should be truncated -- if the time constraint has been reached
        if self._check_truncate():
            done = True
            reward = self._compute_terminate_reading_reward()

        return self._get_obs(), reward, done, self._truncated, {}

    def render(self, mode='human'):
        """
        Render the environment.
        """
        pass

    def _get_obs(self):
        """
        Get the current observation.
        TODO note: the agent does not know the number of words, only the reading progress
        """

        # Episode length awareness
        # The normalised episode length
        norm_ep_len = aux.normalise(self.ep_len, 0, self.max_ep_len, -1, 1)
        # Remaining steps
        norm_ep_remaining_steps = aux.normalise(self.ep_len - self._steps, 0, self.ep_len, -1, 1)

        # Time constraint awareness -- pre-calculated to save computational resources
        norm_time_constraint_level = self._norm_time_constraint_level

        # Reading progress awareness -- the number of words read
        norm_reading_progress = aux.normalise(self.num_words_read_in_sentence, 0, self.num_words_in_sentence, -1, 1)

        # Appraisal states awareness -- the appraisal levels of the following words
        norm_next_word_predictability = self._predictability_states[self.fixated_word_index + 1][0] if self.fixated_word_index < self.num_words_in_sentence - 1 else -1

        stateful_info_obs = np.array([norm_ep_len, norm_ep_remaining_steps, norm_time_constraint_level, norm_reading_progress, norm_next_word_predictability])
        return stateful_info_obs

    def _init_num_words_in_sentence(self, inputs: dict = None):
        """
        Initialize the number of words in a sentence.
        """
        if inputs is not None:
            self.num_words_in_sentence = inputs["num_words_in_sentence"]
        else:
            self.num_words_in_sentence = random.randint(const.MIN_NUM_WORDS_PER_SENTENCE, const.MAX_NUM_WORDS_PER_SENTENCE)

        self.num_words_read_in_sentence = 0
        self._num_remaining_words = self.num_words_in_sentence - self.num_words_read_in_sentence

    def _init_time_constraint(self, inputs: dict = None, time_constraint_weights: dict = None):
        """
        Initialize the time constraint.
        TODO the time constraint weights could be a tunable parameter in the future.
        """

        assert self.num_words_in_sentence is not None, "The number of words in a sentence should be initialized first."

        if inputs is not None:
            self._time_constraint_level_key = inputs["time_constraint_level"]
        else:
            # Sample the total time constraint in seconds from the predefined dictionary: 30s, 60s, 90s,
            random_key = random.choice(list(const.TIME_CONSTRAINT_LEVELS.keys()))
            random_value = const.TIME_CONSTRAINT_LEVELS[random_key]
            # Get the corresponding level label
            self._time_constraint_level_key = random_key
        # Get the episode length, but it should be related to the number of words in the sentence
        if self._time_constraint_level_key == "30S":
            self._norm_time_constraint_level = -1
            self.ep_len = math.ceil(time_constraint_weights["30S"] * self.num_words_in_sentence)
        elif self._time_constraint_level_key == "60S":
            self._norm_time_constraint_level = 0
            self.ep_len = math.ceil(time_constraint_weights["60S"] * self.num_words_in_sentence)
        elif self._time_constraint_level_key == "90S":
            self._norm_time_constraint_level = 1
            self.ep_len = math.ceil(time_constraint_weights["90S"] * self.num_words_in_sentence)
        else:
            # raise ValueError(f"Invalid time constraint level keys: {self._time_constraint_level_key}, should be 30S, 60S, or 90S.")
            print(f"Invalid time constraint level keys: {self._time_constraint_level_key}, prefer to be 30S, 60S, or 90S. Will use 60S as default.")
            self._norm_time_constraint_level = 0
            self.ep_len = math.ceil(time_constraint_weights["60S"] * self.num_words_in_sentence)

    def _init_appraisal_states(self):
        """
        Initialize the appraisal states.
        """
        assert self.num_words_in_sentence is not None, "The number of words in a sentence should be initialized first."

        # Initialize the appraisal states
        self._predictability_states = OrderedDict()
        for i in range(self.num_words_in_sentence):
            # Sample the appraisal level
            predictability = 0.0   # The appraisal levels of the words are always 0.0, because not read or previewed
            decision = self.UNDEFINED_ACTION
            self._predictability_states[i] = (predictability, decision)

    def have_words_to_read(self):
        """
        Check if there are more words to read.
        """
        return self.num_words_read_in_sentence < self.num_words_in_sentence

    def _read_next_word(self, predictability: float = None):
        """
        Read the next word.
        """
        # Update the stateful information
        self.fixated_word_index += 1
        self.num_words_read_in_sentence += 1
        self._num_remaining_words = self.num_words_in_sentence - self.num_words_read_in_sentence
        self.num_saccades_on_word_level += 1

        # Update the appraisal states
        self._predictability_states[self.fixated_word_index] = (1.0, self.READ_ACTION)

        # Get the next word's predictability
        self._get_next_word_predictability(external_llm_predictability=predictability)

    def have_words_to_skip(self):
        """
        Check if there are more words to skip.
        """
        return self.num_words_read_in_sentence < (self.num_words_in_sentence - 1)

    def _skip_next_word(self, predictability: float = None):
        """
        Skip the next word.
        """
        # Update the stateful information
        self.fixated_word_index += 2
        self.num_words_read_in_sentence += 2
        self._num_remaining_words = self.num_words_in_sentence - self.num_words_read_in_sentence
        self.num_saccades_on_word_level += 1
        self.num_words_skipped_in_sentence += 1

        # Update the predictability states
        last_word_predictability_level = self._predictability_states[self.fixated_word_index-1][0]
        self._predictability_states[self.fixated_word_index-1] = (last_word_predictability_level, self.SKIP_ACTION)
        self._predictability_states[self.fixated_word_index] = (1.0, self.READ_ACTION)

        # Get the next word's predictability
        self._get_next_word_predictability(external_llm_predictability=predictability)
    
    def _get_next_word_predictability(self, external_llm_predictability: float = None):
        """
        Get the predictability of the next word using the large language model when simulating OR random sample when training.
        """
        if self.fixated_word_index < self.num_words_in_sentence - 1:
            if external_llm_predictability is None:
                next_word_predictability = random.uniform(0.2, 0.9)
            else:
                next_word_predictability = external_llm_predictability
            self._predictability_states[self.fixated_word_index + 1] = (next_word_predictability, self.UNDEFINED_ACTION)

    @staticmethod
    def _compute_continue_reading_reward():
        """
        Compute the reward for continuing reading.
        """
        return 0.1

    @staticmethod
    def _compute_skip_word_reward():
        """
        Compute the reward for skipping a word.
        """
        return 0.1

    def _compute_terminate_reading_reward(self):
        """
        Compute the reward for terminating the reading process.
        """
        # Assuming self._appraisal_states[idx][0] contains the appraisal value for each word
        predictability_values = [self._predictability_states[idx][0] for idx in range(self.num_words_in_sentence)]

        # Sample a Bernoulli random variable for each appraisal value
        # np.random.binomial(1, p) returns 1 with probability p and 0 with probability (1 - p)
        samples = [np.random.binomial(1, p_i) for p_i in predictability_values]

        # Compute the aggregated appraisal levels
        # For each sample, add +1 if correct (sample == 1), else -1 if incorrect (sample == 0)
        aggregated_predictability_values = sum([1 if sample == 1 else -1 for sample in samples])

        termination_reward = 10 * aggregated_predictability_values

        return termination_reward

    def _check_terminate(self):
        """
        Check if the episode should be terminated.
        """
        if self.num_words_read_in_sentence >= self.num_words_in_sentence:
            self._terminate = True
        else:
            self._terminate = False
        return self._terminate

    def _check_truncate(self):
        """
        Check if the episode should be truncated.
        """
        if self._steps >= self.ep_len:
            self._truncated = True
        else:
            self._truncated = False
        return self._truncated

    def get_logs(self):
        """
        Log the information.
        """
        step_log = {
            'steps': self._steps,
            'fixated_word_index': self.fixated_word_index,
            'time_constraint_level': self._time_constraint_level_key,
            'norm_time_constraint_level': self._norm_time_constraint_level,
            'ep_len': self.ep_len,
            'appraisal_states': self._predictability_states,
            'num_words_skipped': self.num_words_skipped_in_sentence,
            'num_word_level_saccades': self.num_saccades_on_word_level,
            'word_skipping_percentage': (self.num_words_skipped_in_sentence / self.num_words_read_in_sentence) * 100,
        }

        print(step_log)

        print(f"The number of words in the sentence: {self.num_words_in_sentence}, "
              f"the allocated time constraint: {self._time_constraint_level_key}, "
              f"the time constraint level: {self._norm_time_constraint_level}, "
              f"the episode length: {self.ep_len}.")
        print(f"The number of words read: {self.num_words_read_in_sentence}, "
              f"the number of words skipped: {self.num_words_skipped_in_sentence}, "
              f"the number of word-level saccades: {self.num_saccades_on_word_level}."
              f"The word skipping rate is: {self.num_words_skipped_in_sentence / self.num_words_read_in_sentence}.")
        print(f"The appraisal levels of the words: {self._predictability_states}.\n"
              f"The aggregated appraisal levels: {np.sum([self._predictability_states[idx][0] for idx in range(self.num_words_in_sentence)])}.")
        # print(f"Appraisal values: {appraisal_values}, samples: {samples}, aggregated appraisal levels: {aggregated_appraisal_levels}.\n")

        return step_log
