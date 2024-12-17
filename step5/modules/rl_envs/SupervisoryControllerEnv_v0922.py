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


class SupervisoryControllerEnv(Env):

    def __init__(self):

        """
        Create on 22 September 2024.
        This is the environment for the RL-based supervisory controller agent.
        Based on the current reading states and appraisal states, the agent needs to decide whether to regress or not.
        If regress, then determine which position to regress to.

        TODO tunable parameters: Appraisal levels and how to update appraisal levels

        Author: Bai Yunpeng
        """

        # Get the current root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Get the mode from the config yaml file
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
        print(
            f"{const.LV_TWO_DASHES}RL Supervisory Controller Environment -- Deploying the environment in the {self._config['rl']['mode']} mode.")

        # Define the stateful information
        # Sentence level
        self._total_num_sentences = None
        self._num_read_sentences = None
        self._num_remaining_sentences = None
        # Word level
        self._total_num_words = None
        self._num_read_words = None
        self._num_remaining_words = None
        self._sentences_lengths = None
        self._min_sentence_length = None
        self._max_sentence_length = None  
        # Time related
        self._total_time_in_seconds = None
        self.time_spent_in_seconds = None
        self.time_left_in_seconds = None
        # Appraisal states
        self._appraisal_states = None
        # Reading strategies
        self._reading_strategy_for_this_sentence = None
        # Revisit history
        self.revisited_sentences_list_log = None

        # Statistics
        self.num_revisit_sentences = None
        self.num_total_sentences_read = None
        self.num_revisit_saccades_on_word_level = None
        self._num_word_level_saccades = None

        # STM related
        self._stm = None

        # Inputted values
        self._inputted_read_sentence_appraisal_level = None

        # Tunable appraisal level weights
        self._individual_sentence_appraisal_level = None
        self.MAX_APPRAISAL_LEVEL = 1.0
        self.MIN_APPRAISAL_LEVEL = 0.0
        self._individual_sentence_appraisal_levels_list_log = None

        # Tunable reward weight
        self._reward_weight = None
        self.MAX_REWARD_WEIGHT = 1.0
        self.MIN_REWARD_WEIGHT = 0.0

        # Define the training parameters
        self._steps = None
        self.ep_len = 20
        self._terminate = None
        self._truncated = None

        # Define the action related variables (states)
        self._max_sentences = const.MAX_NUM_SENTENCES
        self.reading_sentence_index = None

        # Define the action space -- More efficient to use Discrete for now
        self.action_space = Discrete(self._max_sentences + 1)

        # Define the observation space
        # self._num_stateful_info_obs = 7 + const.STM_CAPACITY + 2 * self._max_sentences
        self._num_stateful_info_obs = 8 + self._max_sentences # TODO decrease the number of stateful information for POMDP formalism later
        self.observation_space = Box(low=-1, high=1, shape=(self._num_stateful_info_obs,))

    def reset(self, seed: int=None, inputs: dict=None):
        """
        Reset the environment to the initial state.
        """

        np.random.seed(42)  # Set the random seed for reproducibility

        self._steps = 0

        self.reading_sentence_index = 0   # Since we are determining the reading before the actual read action, so it is set to 0

        self.num_revisit_sentences = 0    # Number of regressions
        self.num_total_sentences_read = 0       # Number of sentence level saccades
        self.num_revisit_saccades_on_word_level = 0        # Number of regressions at word level
        self._num_word_level_saccades = 0           # Number of saccades at word level
        self.revisited_sentences_list_log = []
        self._individual_sentence_appraisal_levels_list_log = []

        # Initialize the sentence state -- Randomly sample the initial sentence states if not in the simulation mode, i.e., when no inputs are provided
        self._init_sentence_states(inputs=inputs)

        # Randomly sample the initial word states
        self._init_word_states(inputs=inputs)

        # Randomly sample the initial time states
        self._init_time_states(inputs=inputs)

        # Randomly sample the initial appraisal states according to the above states
        self._init_appraisal_states()

        # Randomly sample the reading strategies
        self._get_reading_strategy_for_the_sentence_to_be_read()

        # Initialize the STM
        self._init_stm()

        # Initialize the reward weight
        self._init_reward_weight()

        return self._get_obs(), {}

    def step(self, action, inputted_read_sentence_appraisal_level: float = None):
        """
        Take a step in the environment.
        """
        # Part 1: Determine the sentence to read
        reward = self.step_part1(action)

        # Part 2: After reading the sentence
        observation, reward, done, truncated, info = self.step_part2(reward, inputted_read_sentence_appraisal_level)

        return observation, reward, done, truncated, info
    
    def step_part1(self, action):
        """
        Determine the sentence to read. First part of the step function.
        """
        # Process the action
        self._steps += 1

        # Randomly sample the reading strategies -- TODO: only use it in the training mode
        self._get_reading_strategy_for_the_sentence_to_be_read()

        # Initialize reward
        reward = 0.0

        # Take the action
        if action == 0:  # Continue reading
            if self.reading_sentence_index < (self._total_num_sentences - 1):
                # Have not reached the end of the sentences
                self.reading_sentence_index += 1
                reward = self._compute_continue_reward()
            else:
                # Reached the end of the sentences -- stay put, do nothing
                reward = 0.0
        elif 1 <= action <= self._total_num_sentences:  # Revisit to a certain sentence
            # Check whether the action is valid -- only the sentence that has been read can be regressed to
            if action > self._num_read_sentences:
                # Invalid action: Regress to a non-existent or unread sentence -- stay put, do nothing
                reward = 0.0
            else:
                # Regress to the specified sentence
                self.reading_sentence_index = action - 1  # Convert to 0-based index
                # Update the reward
                reward = self._compute_regress_reward()
        else:
            # Invalid action: Regress to a non-existent sentence
            reward = 0.0

        return reward
    
    def step_part2(self, reward, inputted_read_sentence_appraisal_level):
        """
        After reading a sentence. Second part of the step function.
        """
        # Initialize done to False
        done = False

        # Update the inputted appraisal level
        self._inputted_read_sentence_appraisal_level = inputted_read_sentence_appraisal_level

        # Update the STM with the just-read sentence
        self._update_stm_with_just_read_sentence(sentence_index=self.reading_sentence_index)

        # Update the time states after reading a sentence
        self._update_time_states(sentence_index=self.reading_sentence_index)

        # Check if the reading task is over -- whether the time is up
        if self.time_left_in_seconds <= 0:
            done = True
            self._terminate = True
            reward = self._compute_time_up_reward()

        # Check if the episode is truncated
        truncated = False

        observation = self._get_obs()
        info = {}

        return observation, reward, done, truncated, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        """
        pass

    def _get_obs(self):
        """
        Get the current observation.

        States aren't observed:
            1. Number of sentences and number of words. The agent has to infer from the encoded (normalized) reading progress.
        """
        # Episode length awareness -- remaining steps in the episode, encoded as a continuous value ranges from -1 to 1
        norm_ep_remain_steps = aux.normalise(self.ep_len - self._steps, 0, self.ep_len, -1, 1)

        # Time awareness -- time condition we are in, encoded as -1, 0, 1
        if self._total_time_in_seconds == const.TIME_CONSTRAINT_LEVELS['30S']:
            norm_time_condition = -1
        elif self._total_time_in_seconds == const.TIME_CONSTRAINT_LEVELS['60S']:
            norm_time_condition = 0
        elif self._total_time_in_seconds == const.TIME_CONSTRAINT_LEVELS['90S']:
            norm_time_condition = 1
        else:
            raise ValueError(f"Invalid time constraint: {self._total_time_in_seconds}. Must be one of the following: {const.TIME_CONSTRAINT_LEVELS.values()}")

        # Encode the reading strategy for a specific sentence -- TODO remove this later
        if self._reading_strategy_for_this_sentence == const.READ_STRATEGIES['skim']:
            norm_reading_strategy = -1
        elif self._reading_strategy_for_this_sentence == const.READ_STRATEGIES['normal']:
            norm_reading_strategy = 0
        elif self._reading_strategy_for_this_sentence == const.READ_STRATEGIES['careful']:
            norm_reading_strategy = 1
        else:
            raise ValueError(f"Invalid reading strategies: {self._reading_strategy_for_this_sentence}. Must be one of the following: {const.READ_STRATEGIES.keys()}")

        # Remaining time awareness -- time remaining in seconds, encoded as a continuous value ranges from -1 to 1
        norm_remaining_time = aux.normalise(self.time_left_in_seconds, 0, self._total_time_in_seconds, -1, 1)

        # Sentence awareness -- the current sentence index we are reading, encoded as a continuous value ranges from -1 to 1
        norm_current_sentence_index = aux.normalise(self.reading_sentence_index, 0, self._max_sentences - 1, -1, 1)

        # Reading progress awareness -- the reading progress (sentence level), encoded as a continuous value ranges from -1 to 1
        norm_reading_progress = aux.normalise(self._num_read_sentences, 0, self._total_num_sentences, -1, 1)

        # Reward weights
        norm_reward_weight = aux.normalise(self._reward_weight, self.MIN_REWARD_WEIGHT, self.MAX_REWARD_WEIGHT, -1, 1)

        # Appraisal awareness -- the appraisal level of all sentences, encoded as a continuous value ranges from -1 to 1
        # Get the current appraisal level weight for one sentence
        norm_appraisal_level_weight = aux.normalise(self._individual_sentence_appraisal_level, self.MIN_APPRAISAL_LEVEL, self.MAX_APPRAISAL_LEVEL, -1, 1)
        # Outside 0 to total_num_sentences - 1, set as -1
        norm_appraisal_levels = np.array([self._appraisal_states[idx]['appraisal_level'] for idx in range(self._total_num_sentences)])
        # Top up to the max_sentences
        norm_appraisal_levels = np.pad(norm_appraisal_levels, (0, self._max_sentences - self._total_num_sentences), 'constant', constant_values=const.NEGATIVE_ONE)

        # The visit history -- temporally -- which sentences are in the stm now -- size: stm_capacity, default all -1, then replace with the stm indexes
        norm_stm = self._get_stm_observation()

        # Visit history awareness -- the visit count of all sentences, encoded as a continuous value ranges from -1 to 1
        visits_count = np.array([self._appraisal_states[idx]['visits_count'] for idx in range(self._total_num_sentences)])
        # Normalise them to [0, 1]
        norm_visits_count = aux.normalise(visits_count, 0, np.max(self.ep_len), 0, 1)
        # Top up to the max_sentences
        norm_visits_count = np.pad(norm_visits_count, (0, self._max_sentences - self._total_num_sentences), 'constant', constant_values=const.NEGATIVE_ONE)

        obs = np.array([norm_ep_remain_steps, norm_time_condition, norm_reading_strategy, norm_remaining_time, norm_current_sentence_index, norm_appraisal_level_weight, norm_reward_weight,
                        norm_reading_progress, 
                        # *norm_stm, *norm_visits_count, 
                        *norm_appraisal_levels])

        return obs

    def _get_stm_observation(self):
        # Get the list of sentence indexes from the STM
        stm_sentence_indexes = list(self._stm.keys())
        # Ensure the list does not exceed stm_capacity
        stm_sentence_indexes = stm_sentence_indexes[-const.STM_CAPACITY:]
        # Calculate the number of empty slots
        num_empty_slots = const.STM_CAPACITY - len(stm_sentence_indexes)
        # Create the STM observation with empty slots at the beginning
        stm_observation = [const.NEGATIVE_ONE] * num_empty_slots + stm_sentence_indexes
        # Convert to NumPy array
        return np.array(stm_observation)

    def _init_sentence_states(self, inputs: dict=None):
        """
        Initial sentence states: randomly sample the number of sentences if the inputs are none.
        """
        if inputs is not None:
            # Set the number of sentences according to the inputs
            self._total_num_sentences = inputs['total_num_sentences_in_stimulus']
        else:
            self._total_num_sentences = random.randint(const.MIN_NUM_SENTENCES, const.MAX_NUM_SENTENCES)
        
        self._num_read_sentences = 0
        self._num_remaining_sentences = self._total_num_sentences - self._num_read_sentences

    def _advance_reading_progress(self):
        """
        Update the sentence states.
        """
        # Update the number of read sentences
        self._num_read_sentences += 1
        # Update the number of remaining sentences
        self._num_remaining_sentences = self._total_num_sentences - self._num_read_sentences

    def _init_word_states(self, inputs: dict=None):
        """
        Sample the initial word states if inputs is None, else initializes the inputs.
        """
        if inputs is not None:
            self._sentences_lengths = inputs['num_words_per_sentences']
            self._total_num_words = sum(self._sentences_lengths)
            self._min_sentence_length = min(self._sentences_lengths)
            self._max_sentence_length = max(self._sentences_lengths)
        else:
            # Set the min and max number of words per sentence
            min_words_per_sentence = const.MIN_NUM_WORDS_PER_SENTENCE  # Minimum words in a sentence
            max_words_per_sentence = const.MAX_NUM_WORDS_PER_SENTENCE  # Maximum words in a sentence
            self._min_sentence_length = min_words_per_sentence
            self._max_sentence_length = max_words_per_sentence

            # Set the target total number of words (between 160 and 180)
            min_text_length = 165
            max_text_length = 180
            total_words_target = random.randint(min_text_length, max_text_length)

            N = self._total_num_sentences

            # Calculate the minimum and maximum possible total words
            min_total_words = N * min_words_per_sentence
            max_total_words = N * max_words_per_sentence

            # Adjust total_words_target if it's outside the possible range
            if total_words_target < min_total_words:
                total_words_target = min_total_words
            elif total_words_target > max_total_words:
                total_words_target = max_total_words

            # Initialize words per sentence with the minimum number of words
            num_words_per_sentence = [min_words_per_sentence] * N

            # Compute capacities (additional words that can be added to each sentence)
            capacities = [max_words_per_sentence - min_words_per_sentence] * N

            # Compute remaining words to distribute
            remaining_words = total_words_target - sum(num_words_per_sentence)

            # Distribute the remaining words randomly among sentences
            while remaining_words > 0:
                i = random.randint(0, N - 1)
                if capacities[i] > 0:
                    num_words_per_sentence[i] += 1
                    capacities[i] -= 1
                    remaining_words -= 1

            # Assign the word counts to the environment's state
            self._sentences_lengths = num_words_per_sentence
            self._total_num_words = sum(num_words_per_sentence)

        self._num_read_words = 0
        self._num_remaining_words = self._total_num_words - self._num_read_words

    def _init_time_states(self, inputs: dict=None):
        """
        Sample the initial time states. If inputs is None, sample the total time constraint in seconds from the predefined dictionary.
        """
        # Sample the total time constraint in seconds from the predefined dictionary: 30s, 60s, 90s,
        if inputs is not None:
            self._total_time_in_seconds = inputs['total_time_in_seconds']
        else:
            random_key = random.choice(list(const.TIME_CONSTRAINT_LEVELS.keys()))
            random_value = const.TIME_CONSTRAINT_LEVELS[random_key]
            self._total_time_in_seconds = random_value
        
        self.time_spent_in_seconds = 0
        self.time_left_in_seconds = self._total_time_in_seconds - self.time_spent_in_seconds

    def _update_time_states(self, sentence_index):
        """
        Update the time states.
        """
        # Update the time elapsed
        self.time_spent_in_seconds += self._sentences_lengths[sentence_index] * (1 / const.AVERAGE_READ_WORDS_PER_SECOND)
        # Update the time remaining
        self.time_left_in_seconds = np.clip(self._total_time_in_seconds - self.time_spent_in_seconds, 0, self._total_time_in_seconds)

    def _init_appraisal_states(self):
        """
        Sample the initial appraisal states.
        The number of appraisal states must match the number of sentences.
        """
        assert self._total_num_sentences > 0, "Initialize the number of sentences first."

        # Ensure the appraisal states match the number of sentences initialized
        self._appraisal_states = []

        # Example of how to define initial appraisal levels for each sentence
        initial_appraisal_level = const.ZERO  # Fixed for now, can be dynamic later

        for idx in range(self._total_num_sentences):
            appraisal_state = {
                'sentence_index': idx,
                'appraisal_level': initial_appraisal_level,  # This could be sampled dynamically if needed
                'visits_count': const.ZERO,
                'forgotten': False,
                'visit_step': const.NA,
            }
            self._appraisal_states.append(appraisal_state)
        
        # Initialize the appraisal level weight
        self._individual_sentence_appraisal_level = random.uniform(self.MIN_APPRAISAL_LEVEL, self.MAX_APPRAISAL_LEVEL)
        # Log the appraisal level weights
        self._individual_sentence_appraisal_levels_list_log.append(self._individual_sentence_appraisal_level)

    def _calculate_appraisal_level(self, sentence_index, operation):
        """
        Calculate the appraisal level for the given sentence index.
        """
        original_appraisal_level = self._appraisal_states[sentence_index]['appraisal_level']
        original_visits_count = self._appraisal_states[sentence_index]['visits_count']
        # original_forgotten_flag = self._appraisal_states[sentence_index]['forgotten']
        original_visit_step = self._appraisal_states[sentence_index]['visit_step']

        if operation == const.SENTENCE_STATES_IN_MEMORY['revisited']:
            # The sentence was revisited
            _new_level = original_appraisal_level + self._get_appraisal_level(inputted_individual_sentence_appraisal_level=self._inputted_read_sentence_appraisal_level)
            # Use this simple adding mechanism to update the appraisal level, change to something more complex later if needed
            new_level = np.clip(_new_level, 0.0, 1.0)
            new_visits_count = original_visits_count + 1
            new_forgotten_flag = False
            new_visit_step = self._steps
            # Update the number of regression counts at sentence level
            self.num_revisit_sentences += 1
            # Update the number of sentence level saccades
            self.num_total_sentences_read += 1
            # Update the number of regression/revisits at word level
            self.num_revisit_saccades_on_word_level += self._sentences_lengths[sentence_index]
            # Update the number of word level saccades
            self._num_word_level_saccades += self._sentences_lengths[sentence_index]
        elif operation == const.SENTENCE_STATES_IN_MEMORY['forgotten']:
            # Check if the sentence's appraisal level is below the threshold
            if original_appraisal_level < const.MEMORY_RETAIN_APPRAISAL_LEVEL_THRESHOLD:
                # The sentence was forgotten -- TODO Apply the simple forgetting mechanism now, change to something more complex later
                new_level = self._compute_appraisal_decrease_due_to_forget(original_appraisal_level)
                new_forgotten_flag = True
            else:
                # The sentence was not forgotten
                new_level = original_appraisal_level
                new_forgotten_flag = False
            new_visits_count = original_visits_count
            new_visit_step = original_visit_step
        elif operation == const.SENTENCE_STATES_IN_MEMORY['first_visit']:
            # The sentence was visited for the first time
            new_level = self._get_appraisal_level(inputted_individual_sentence_appraisal_level=self._inputted_read_sentence_appraisal_level)
            new_visits_count = const.ONE
            new_forgotten_flag = False
            new_visit_step = self._steps
            # Update the number of sentence level saccades
            self.num_total_sentences_read += 1
            # Update the number of word level saccades
            self._num_word_level_saccades += self._sentences_lengths[sentence_index]
        else:
            raise ValueError(f"Invalid operation: {operation}. Must be one of the following: {const.SENTENCE_STATES_IN_MEMORY.keys()}")

        # Update the appraisal states
        self._update_appraisal_state(sentence_index=sentence_index, new_level=new_level, new_visits_count=new_visits_count,
                                     new_forgotten_flag=new_forgotten_flag, new_visit_step=new_visit_step)

    @staticmethod
    def _compute_appraisal_decrease_due_to_forget(appraisal_level):     # TODO here exists some potential parameter tuning places
        """
        Compute the appraisal decrease due to forgotten.
        """
        numerator = math.exp(appraisal_level) - 1
        denominator = np.e - 1
        return numerator / denominator
    
    def _get_appraisal_level(self, inputted_individual_sentence_appraisal_level: float=None,):
        """
        Get the appraisal level based on reading strategy and sentence length.
        
        :param inputs: Dictionary containing the inputs for the appraisal level calculation. Consider the following keys:
            - information_sampling_rate: The information sampling rate for the sentence. Controlled by the SLC.
            - readability: The readability of the sentence. Calculated based on sentence length and complexity in the simulator.
            - sentence_reading_strategy: The reading strategy for the sentence.
            
        :return: Adjusted appraisal level between 0 and 1.
        """
        if inputted_individual_sentence_appraisal_level is not None:
            self._individual_sentence_appraisal_level = inputted_individual_sentence_appraisal_level
        else:
            self._individual_sentence_appraisal_level = random.uniform(self.MIN_APPRAISAL_LEVEL, self.MAX_APPRAISAL_LEVEL)
        self._individual_sentence_appraisal_levels_list_log.append(self._individual_sentence_appraisal_level)
        return self._individual_sentence_appraisal_level

    def _update_appraisal_state(self, sentence_index, new_level, new_visits_count, new_forgotten_flag, new_visit_step):
        self._appraisal_states[sentence_index]['appraisal_level'] = new_level
        self._appraisal_states[sentence_index]['visits_count'] = new_visits_count
        self._appraisal_states[sentence_index]['forgotten'] = new_forgotten_flag
        self._appraisal_states[sentence_index]['visit_step'] = new_visit_step

    def _get_reading_strategy_for_the_sentence_to_be_read(self, useless_inputs: float=None):
        """
        Sample the reading strategies. If the inputs are none, randomly sample the reading strategies from the predefined dictionary.
        """
        # if llm_actions_dict is not None:
        #     self._reading_strategy_for_this_sentence = llm_actions_dict['reading_strategy_for_this_sentence']
        # else:
        #     # Sample the reading strategies from the predefined dictionary
        #     random_key = random.choice(list(const.READ_STRATEGIES.keys()))
        #     self._reading_strategy_for_this_sentence = const.READ_STRATEGIES[random_key]
        if useless_inputs is not None:        # TODO these are not usefyl now, remove them later.
            self._reading_strategy_for_this_sentence = const.READ_STRATEGIES['normal']
            # self._reading_strategy_for_this_sentence = const.READ_STRATEGIES['skim']
        else:
            # Randomly sample the reading strategies if the inputs are none
            random_key = random.choice(list(const.READ_STRATEGIES.keys()))
            self._reading_strategy_for_this_sentence = const.READ_STRATEGIES[random_key]

    def _init_stm(self):
        """
        Initialize the STM.
        """
        self._stm = OrderedDict()
    
    def _init_reward_weight(self):
        """
        Initialize the reward weight.
        """
        # self._reward_weight = random.uniform(self.MIN_REWARD_WEIGHT, self.MAX_REWARD_WEIGHT)
        self._reward_weight = 0.5       # I empirically set it to 0.5 for now, see my analysis here: https://docs.google.com/presentation/d/1iqBCSFAXCpKWc2O41FQpZEW2AhDx9PsbZLY3jd1gpuA/edit#slide=id.g30bbbc2a2be_0_15

    def _update_stm_with_just_read_sentence(self, sentence_index):
        """
        Update the STM.
        """
        if sentence_index in self._stm:
            # Reinforce directly
            self._calculate_appraisal_level(
                sentence_index=sentence_index,
                # reading_strategy=self._reading_strategy_for_this_sentence,
                operation=const.SENTENCE_STATES_IN_MEMORY['revisited']
            )
            # Move to the end of the STM
            self._stm.move_to_end(sentence_index)
        else:
            # Remove the oldest sentence if the STM is full
            if len(self._stm) >= const.STM_CAPACITY:
                self._remove_oldest_sentence()
            # Initialize appraisal level if not already set
            if self._appraisal_states[sentence_index]['visits_count'] == const.ZERO:  # First visit
                self._calculate_appraisal_level(
                    sentence_index=sentence_index,
                    # reading_strategy=self._reading_strategy_for_this_sentence,
                    operation=const.SENTENCE_STATES_IN_MEMORY['first_visit']
                )
                # Update the number of read sentences
                self._advance_reading_progress()
            elif self._appraisal_states[sentence_index]['visits_count'] > const.ZERO:
                self._calculate_appraisal_level(
                    sentence_index=sentence_index,
                    # reading_strategy=self._reading_strategy_for_this_sentence,
                    operation=const.SENTENCE_STATES_IN_MEMORY['revisited']
                )
            # Add the sentence to STM (it will be added to the end)
            self._stm[sentence_index] = self._appraisal_states[sentence_index]['appraisal_level']

    def _remove_oldest_sentence(self):
        """
        Forget the oldest sentence in the STM.
        """
        oldest_sentence, _ = self._stm.popitem(last=False)
        appraisal_level = self._appraisal_states[oldest_sentence]['appraisal_level']
        if appraisal_level >= const.MEMORY_RETAIN_APPRAISAL_LEVEL_THRESHOLD:
            # Do not decrease appraisal, but it's removed from STM, do nothing to the appraisal level
            pass
        else:
            # Start decreasing appraisal over time
            self._calculate_appraisal_level(
                sentence_index=oldest_sentence, 
                # reading_strategy=self._reading_strategy_for_this_sentence,
                operation=const.SENTENCE_STATES_IN_MEMORY['forgotten']
            )

    def _compute_continue_reward(self):
        """
        Compute the reward for continuing to read.
        """
        return 0.1

    def _compute_regress_reward(self):
        """
        Compute the reward for regressing.
        """
        return 0.1

    def _compute_time_up_reward(self):
        """
        Compute the reward for time up.
        The reward should be composed of two parts: the overall reading progress and the appraisal levels
        """
        # TODO maybe change to Bernoulli distribution later
        # aggregated_appraisal_levels = 10 * np.sum([self._appraisal_states[idx]['appraisal_level'] for idx in range(self._total_num_sentences)])
        # return aggregated_appraisal_levels
        
        # Get appraisal levels
        appraisal_levels = [self._appraisal_states[idx]['appraisal_level'] for idx in range(self._total_num_sentences)]

        # Sample a Bernoulli random variable for each appraisal value
        # np.random.binomial(1, p) returns 1 with probability p and 0 with probability (1 - p)
        samples = [np.random.binomial(1, appraisal_level) for appraisal_level in appraisal_levels]

        # Weighted parameter tuning -- the reward is the sum of the Bernoulli samples
        weighted_aggregated_appraisal_levels = sum([1 if sample == 1 else -1 * self._reward_weight for sample in samples])

        termination_reward = 1 * weighted_aggregated_appraisal_levels

        return termination_reward

    def _get_reading_progress_at_word_level(self) -> int:
        """
        Get the reading progress at the word level.
        """
        # Check the last sentence read by the appraisal levels -- the last index in the appraisal state that is not zero
        last_sentence_read_index = -1
        for idx in range(self._total_num_sentences):
            if self._appraisal_states[idx]['appraisal_level'] != 0:
                last_sentence_read_index = idx
        return last_sentence_read_index

    def _get_total_num_words_read(self, last_sentence_read_index) -> int:
        """
        Get the total number of words read.
        """
        return int(np.sum([self._sentences_lengths[idx] for idx in range(last_sentence_read_index + 1)]))

    def get_logs(self):      # TODO make this concise and useful later
        """
        Log the information.
        """
        step_log = {
            'steps': self._steps,
            'current_sentence_index': self.reading_sentence_index,
            'total_num_sentences': self._total_num_sentences,
            'num_read_sentences': self._num_read_sentences,
            'num_remaining_sentences': self._num_remaining_sentences,
            'total_time_in_seconds': self._total_time_in_seconds,
            'time_elapsed_in_seconds': self.time_spent_in_seconds,
            'time_remaining_in_seconds': self.time_left_in_seconds,
            'reading_strategy_for_this_sentence': self._reading_strategy_for_this_sentence,
            'appraisal_states': self._appraisal_states,
            'stm': self._stm,
            'individual_sentence_appraisal_level_weight': self._individual_sentence_appraisal_level,
            'average_individual_sentence_appraisal_level_weight': np.mean(self._individual_sentence_appraisal_levels_list_log),
            'reward_weight': self._reward_weight,
            'num_regression': self.num_revisit_sentences,
            'num_sentence_level_saccades': self.num_total_sentences_read,
            'regression_rate_sentence_level': self.num_revisit_sentences / self.num_total_sentences_read,
            'revisit_percentage_word_level_using_saccades': (self.num_revisit_saccades_on_word_level / self._num_word_level_saccades) * 100,        # TODO this should be calculated by the SLC agent later
            'revisit_percentage_word_level_using_reading_progress': (self.num_revisit_saccades_on_word_level / self._get_total_num_words_read(self._get_reading_progress_at_word_level())) * 100,
        }

        # TODO debug delete later   
        print(f"The revisited log is: {self.revisited_sentences_list_log}\n")

        return step_log


if __name__ == '__main__':
    sc = SupervisoryControllerEnv()
    sc.reset()
