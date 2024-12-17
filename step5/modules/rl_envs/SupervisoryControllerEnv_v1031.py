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
        Create on 31 Octorber 2024.
        This is the environment for the RL-based supervisory controller agent.
        Based on the current reading states and appraisal states, the agent needs to decide whether to regress or not.
        If regress, then determine which position to regress to.

        Simplify the action logic: each action corresponds to a sentence. Difficult part -- how to set up observations effectively and set up the valid thresholds.

        Author: Bai Yunpeng

        This version aims to make the env only has one tunable parameter, that is the appraisal level (weight).
        """

        # Get the current root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Get the mode from the config yaml file
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
        print(
            f"=====================================\n"
            f"RL Supervisory Controller Environment Version 1031 -- Deploying the environment in the {self._config['rl']['mode']} mode. \n"
            f"=====================================\n"
            )

        # Define the stateful information
        # Sentence level
        self._total_num_sentences = None
        self._num_unique_sentences_read_OR_farthest_sentence_reached = None
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
        self.appraisal_states = None
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
        self._pre_sampled_appraisal_levels_acorss_sentences = None
        self.PRE_SAMPLED_APPRAISAL_LEVELS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        # Tunable reward weight
        self._reward_weight = None
        self._failure_penalty_weight = None
        self._exploit_weight = None
        self._explore_weight = None
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
        self._FLAG_new_info_sampled_OR_old_info_reinforced = None    # True of False, whether a sentence is read or revisited in a valid way

        # Define the action space -- More efficient to use Discrete for now
        # self.action_space = Discrete(self._max_sentences + 1)
        self.action_space = Discrete(2)     # 0 for continue, and 1 for revisit

        # Define the observation space
        self._num_stateful_info_obs = 8 + self._max_sentences 
        self.observation_space = Box(low=-1, high=1, shape=(self._num_stateful_info_obs,))

    def reset(self, seed: int=None, inputs: dict=None):
        """
        Reset the environment to the initial state.
        """

        np.random.seed(42)  # Set the random seed for reproducibility

        self._steps = 0

        self.reading_sentence_index = -1   # Since we are determining the reading before the actual read action, so it is set to 0

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

        # Initialize the STM
        self._init_stm()

        # Initialize the reward weight
        self._init_reward_weights()

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

        # Initialize reward
        reward = 0.0

        # Reset the validity of the reading action
        self._FLAG_new_info_sampled_OR_old_info_reinforced = False

        # Action masking -- in the first step, the agent always chooses to read the first sentence
        if self._steps == 1:
            action = 0

        # Take the action
        if action == 0:  # Continue reading
            if self._num_unique_sentences_read_OR_farthest_sentence_reached < self._total_num_sentences:       # Have not reached the end of the sentences
                # self.reading_sentence_index += 1
                self._FLAG_new_info_sampled_OR_old_info_reinforced = True
                self.reading_sentence_index = self._num_unique_sentences_read_OR_farthest_sentence_reached
                reward = self._compute_continue_read_new_sentence_reward()

                # TODO debug delete later
                print(f"////////////////////////////////////////////////////////////////////////")
                print(f"The current step is: {self._steps}, the action is: {action}, reward is {reward}")
                print(f"Valid Continue Reading -- Reading sentence index: {self.reading_sentence_index}")
                print(f"////////////////////////////////////////////////////////////////////////")
            else:
                # Reset the reading sentence index to the none-exist index
                # self.reading_sentence_index = const.NEGATIVE_ONE      # The reading index should stay put
                # Invalid action: Continue reading when all sentences have been read
                self._FLAG_new_info_sampled_OR_old_info_reinforced = False
                # Reached the end of the sentences -- stay put, do nothing
                reward = self._compute_penalty_for_invalid_actions_or_do_nothing()       # Penalize for doing nothing       # TODO some bugs here, when invalid actions of doing nothing the appraisal should not be reinforced.

                # TODO debug delete later
                print(f"////////////////////////////////////////////////////////////////////////")
                print(f"The current step is: {self._steps}, the action is: {action}, reward is {reward}")
                print(f"Invalid Continue Reading -- Stay put on the reading sentence index: {self.reading_sentence_index}")
                print(f"////////////////////////////////////////////////////////////////////////")

        # elif 1 <= action <= self._total_num_sentences:  # Revisit to a certain sentence     # TODO maybe only determine revisit or not, use appraisal levels to sample which sentence to revisit. --> found not very helpful, see version 1030
        elif action == 1:  # Revisit to the previous sentence
            # Check whether the action is valid -- only the sentence that has been read can be regressed to
            if self._num_unique_sentences_read_OR_farthest_sentence_reached > 0:
                # Regress to the specified sentence
                self.reading_sentence_index = self._sample_revisit_sentence_by_appraisal_levels()

                if self.appraisal_states[self.reading_sentence_index]['appraisal_level'] < 1:       # If there is space for reinforcement, 1 is the saturation point
                    # The reivist action is reasonable and reinforcing
                    self._FLAG_new_info_sampled_OR_old_info_reinforced = True
                else:
                    # The action is not reasonable
                    self._FLAG_new_info_sampled_OR_old_info_reinforced = False

                # Update the reward
                reward = self._compute_revisit_to_reinforce_old_sentence_reward()   # Set the flag inside the reward function

                # TODO debug delete later
                print(f"////////////////////////////////////////////////////////////////////////")
                print(f"The current step is: {self._steps}, the action is: {action}, reward is {reward}")
                print(f"Valid Revisit -- Revisited reading sentence index: {self.reading_sentence_index}")
                print(f"////////////////////////////////////////////////////////////////////////")
            else:
                # # Reset the reading sentence index to the none-exist index
                # self.reading_sentence_index = const.NEGATIVE_ONE      The reading index should stay put
                # A not reasonable action
                self._FLAG_new_info_sampled_OR_old_info_reinforced = False
                # Invalid action: Regress to a non-existent or unread sentence -- stay put, do nothing
                reward = self._compute_penalty_for_invalid_actions_or_do_nothing()

                # TODO debug delete later
                print(f"////////////////////////////////////////////////////////////////////////")
                print(f"The current step is: {self._steps}, the action is: {action}, reward is {reward}")
                print(f"Invalid Revisit -- Stay put on the reading sentence index: {self.reading_sentence_index}")
                print(f"////////////////////////////////////////////////////////////////////////")

        else:
            # # Reset the reading sentence index to the none-exist index
            # self.reading_sentence_index = const.NEGATIVE_ONE      # The readinng index should stay put
            # A not reasonable action
            self._FLAG_new_info_sampled_OR_old_info_reinforced = False
            # Invalid action: Regress to a non-existent sentence
            reward = self._compute_penalty_for_invalid_actions_or_do_nothing()   # Penalize non-existent regressions

            # TODO debug delete later
            print(f"////////////////////////////////////////////////////////////////////////")
            print(f"The current step is: {self._steps}, the action is: {action}, reward is {reward}")
            print(f"Valid Revisit -- Revisited reading sentence index: {self.reading_sentence_index}")
            print(f"////////////////////////////////////////////////////////////////////////")

        return reward
    
    def step_part2(self, reward, inputted_read_sentence_appraisal_level=None, 
                   inputted_num_words_explicitly_fixated_in_sentence=None):
        """
        After reading a sentence. Second part of the step function.
        """
        # Initialize done to False
        done = False

        # Update the inputted appraisal level
        self._inputted_read_sentence_appraisal_level = inputted_read_sentence_appraisal_level

        # # Update the STM and time states only if a sentence was read
        # if self._FLAG_new_info_sampled_OR_old_info_reinforced:
        #     # Update the STM with the just-read sentence
        #     self._update_stm_with_just_read_sentence(sentence_index=self.reading_sentence_index)

        #     # Update the time states after reading a sentence
        #     self._update_time_states(sentence_index=self.reading_sentence_index, 
        #                             inputted_num_words_explicitly_fixated_in_sentence=inputted_num_words_explicitly_fixated_in_sentence)   
        # else:
        #     # Update the time states after reading a sentence -- The time is still consumed, but the STM is not updated
        #     self._update_time_states(sentence_index=self.reading_sentence_index, 
        #                             inputted_num_words_explicitly_fixated_in_sentence=inputted_num_words_explicitly_fixated_in_sentence)
        
        # Update the STM with the just-read sentence
        self._update_stm_with_just_read_sentence(sentence_index=self.reading_sentence_index)

        # Update the time states after reading a sentence
        self._update_time_states(sentence_index=self.reading_sentence_index, 
                                inputted_num_words_explicitly_fixated_in_sentence=inputted_num_words_explicitly_fixated_in_sentence)

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

    # def _sample_revisit_sentence_by_appraisal_levels(self):     
    #     """
    #     Sample the sentence to revisit based on the appraisal levels.
    #     """
    #     # Sample the sentence to revisit based on the appraisal levels from the read sentences, using argmin appraisal levels
    #     # Get the appraisal levels of the read sentences
    #     appraisal_levels = [self.appraisal_states[idx]['appraisal_level'] for idx in range(self._num_unique_sentences_read_OR_farthest_sentence_reached)]
    #     # Get the sentence index with the minimum appraisal level
    #     sentence_index = np.argmin(appraisal_levels)
    #     assert sentence_index < self._num_unique_sentences_read_OR_farthest_sentence_reached, f"Invalid sentence index '{sentence_index}' to revisit, should be smaller than: {self._num_unique_sentences_read_OR_farthest_sentence_reached}."
    #     return sentence_index
    
    def _sample_revisit_sentence_by_appraisal_levels(self):     
        """
        Sample the sentence to revisit based on the appraisal levels.
        """
        # Get the appraisal levels of the read sentences
        appraisal_levels = [self.appraisal_states[idx]['appraisal_level'] 
                            for idx in range(self._num_unique_sentences_read_OR_farthest_sentence_reached)]
        # Find the minimum appraisal level
        min_appraisal_level = min(appraisal_levels)
        # Find all indices with the minimum appraisal level
        min_indices = [idx for idx, level in enumerate(appraisal_levels) if level == min_appraisal_level]
        # Randomly select one of the indices with the minimum appraisal level
        sentence_index = np.random.choice(min_indices)
        assert sentence_index < self._num_unique_sentences_read_OR_farthest_sentence_reached, (
            f"Invalid sentence index '{sentence_index}' to revisit, should be smaller than: "
            f"{self._num_unique_sentences_read_OR_farthest_sentence_reached}."
        )
        return sentence_index

    def _get_obs(self):     # TODO get an observation telling the agent how many sentences are left revisitable
        """
        Get the current observation.

        States aren't observed:
            1. Number of sentences and number of words. The agent has to infer from the encoded (normalized) reading progress.      

        """
        # Episode length awareness -- remaining steps in the episode, encoded as a continuous value ranges from -1 to 1
        norm_ep_remain_steps = aux.normalise(self.ep_len - self._steps, 0, self.ep_len, -1, 1)

        # Time awareness -- time condition we are in, encoded as -1, 0, 1
        if self._total_time_in_seconds == const.TIME_CONSTRAINT_LEVELS['30S']:
            norm_time_condition = aux.normalise(const.TIME_CONSTRAINT_LEVELS['30S'], 0, const.TIME_CONSTRAINT_LEVELS['90S'], -1, 1)
        elif self._total_time_in_seconds == const.TIME_CONSTRAINT_LEVELS['60S']:
            norm_time_condition = aux.normalise(const.TIME_CONSTRAINT_LEVELS['60S'], 0, const.TIME_CONSTRAINT_LEVELS['90S'], -1, 1)
        elif self._total_time_in_seconds == const.TIME_CONSTRAINT_LEVELS['90S']:
            norm_time_condition = aux.normalise(const.TIME_CONSTRAINT_LEVELS['90S'], 0, const.TIME_CONSTRAINT_LEVELS['90S'], -1, 1)
        else:
            raise ValueError(f"Invalid time constraint: {self._total_time_in_seconds}. Must be one of the following: {const.TIME_CONSTRAINT_LEVELS.values()}")

        # Remaining time awareness -- time remaining in seconds, encoded as a continuous value ranges from -1 to 1
        norm_remaining_time = aux.normalise(self.time_left_in_seconds, 0, self._total_time_in_seconds, -1, 1)

        # Sentence awareness -- the current sentence index we are reading, encoded as a continuous value ranges from -1 to 1
        norm_current_sentence_index = aux.normalise(self.reading_sentence_index, const.NEGATIVE_ONE, self._max_sentences - 1, -1, 1)

        # Action validity for the current state
        norm_action_validity = 1 if self._FLAG_new_info_sampled_OR_old_info_reinforced else -1

        # Reading progress awareness -- the reading progress (sentence level), encoded as a continuous value ranges from -1 to 1
        norm_reading_progress = aux.normalise(self._num_unique_sentences_read_OR_farthest_sentence_reached, 0, self._total_num_sentences, -1, 1)

        # Remaining sentences awareness -- the remaining sentences, encoded as a continuous value ranges from -1 to 1
        norm_remaining_sentences = aux.normalise(self._num_remaining_sentences, 0, self._total_num_sentences, -1, 1)

        # Appraisal awareness -- the appraisal level of all sentences, encoded as a continuous value ranges from -1 to 1
        # Get the current appraisal level weight for one sentence
        reading_sentence_appraisal_level = self.appraisal_states[self.reading_sentence_index]['appraisal_level']
        norm_individual_sentence_appraisal_level = aux.normalise(reading_sentence_appraisal_level, self.MIN_APPRAISAL_LEVEL, self.MAX_APPRAISAL_LEVEL, -1, 1)  # TODO some bugs here
        # Outside 0 to total_num_sentences - 1, set as -1
        norm_appraisal_levels = np.array([self.appraisal_states[idx]['appraisal_level'] for idx in range(self._total_num_sentences)])
        # Top up to the max_sentences
        norm_appraisal_levels = np.pad(norm_appraisal_levels, (0, self._max_sentences - self._total_num_sentences), 'constant', constant_values=const.NEGATIVE_ONE)

        # TODO think why the agent needs to revisit the sentences, and how are these information delivered to the agent.

        obs = np.array([norm_ep_remain_steps, norm_time_condition, norm_remaining_time, norm_current_sentence_index, norm_action_validity, norm_individual_sentence_appraisal_level, 
                        norm_remaining_sentences, norm_reading_progress, *norm_appraisal_levels])
        
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
        
        self._num_unique_sentences_read_OR_farthest_sentence_reached = 0
        self._num_remaining_sentences = self._total_num_sentences - self._num_unique_sentences_read_OR_farthest_sentence_reached

        self._FLAG_new_info_sampled_OR_old_info_reinforced = False

    def _advance_reading_progress(self):
        """
        Update the sentence states.
        """
        # Update the number of read sentences
        self._num_unique_sentences_read_OR_farthest_sentence_reached += 1
        # Update the number of remaining sentences
        self._num_remaining_sentences = self._total_num_sentences - self._num_unique_sentences_read_OR_farthest_sentence_reached

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
            min_text_length = const.MIN_TEXT_LENGTH
            max_text_length = const.MAX_TEXT_LENGTH
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

    def _update_time_states(self, sentence_index, inputted_num_words_explicitly_fixated_in_sentence=None):        
        """
        Update the time states.
        """
        if inputted_num_words_explicitly_fixated_in_sentence is not None:
            self.time_spent_in_seconds += inputted_num_words_explicitly_fixated_in_sentence * (1 / const.AVERAGE_READ_WORDS_PER_SECOND)
        else:     
            # I think do not need to randomize this, because: 
                #   1. in the obs, sentence length and reading time does not have tangled effect. Only the reading progress is mentioned in the obs. It is still sentence-level info.
                #   2. it is randomized already in the init method.
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
        self.appraisal_states = []

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
            self.appraisal_states.append(appraisal_state)
        
        # Initialize the pre-sampled appraisal levels
        self._pre_sampled_appraisal_levels_acorss_sentences = []
        for idx in range(self._total_num_sentences):
            self._pre_sampled_appraisal_levels_acorss_sentences.append(random.choice(self.PRE_SAMPLED_APPRAISAL_LEVELS))

        # Initialize here in case reset function errors
        # Initialize the appraisal level weight
        # self._individual_sentence_appraisal_level = random.uniform(self.MIN_APPRAISAL_LEVEL, self.MAX_APPRAISAL_LEVEL)
        self._individual_sentence_appraisal_level = self._pre_sampled_appraisal_levels_acorss_sentences[0]
        # Log the appraisal level weights
        self._individual_sentence_appraisal_levels_list_log.append(self._individual_sentence_appraisal_level)

    def _calculate_appraisal_level(self, sentence_index, operation):
        """
        Calculate the appraisal level for the given sentence index.
        """
        original_appraisal_level = self.appraisal_states[sentence_index]['appraisal_level']
        original_visits_count = self.appraisal_states[sentence_index]['visits_count']
        # original_forgotten_flag = self._appraisal_states[sentence_index]['forgotten']
        original_visit_step = self.appraisal_states[sentence_index]['visit_step']

        if operation == const.SENTENCE_STATES_IN_MEMORY['revisited']:
            # The sentence was revisited
            # _new_level = original_appraisal_level + self._get_appraisal_level(inputted_individual_sentence_appraisal_level=self._inputted_read_sentence_appraisal_level)
            _new_level = 1.0        # TODO try the simple one here to encourage revisits
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
            new_level = self._compute_appraisal_decrease_due_to_forget(original_appraisal_level)
            new_forgotten_flag = True
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

    @staticmethod       #@potential parameter tuning
    def _compute_appraisal_decrease_due_to_forget(appraisal_level):    
        """
        Compute the appraisal decrease due to forgotten.
        """
        # Decrease the appraisal level by 0.5
        return np.clip(appraisal_level - 0.5, 0.0, 1.0)
    
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
            # self._individual_sentence_appraisal_level = random.uniform(self.MIN_APPRAISAL_LEVEL, self.MAX_APPRAISAL_LEVEL)  # TODO change this to use an existing appraisal level candidates
            self._individual_sentence_appraisal_level = self._pre_sampled_appraisal_levels_acorss_sentences[self.reading_sentence_index]
        self._individual_sentence_appraisal_levels_list_log.append(self._individual_sentence_appraisal_level)
        return self._individual_sentence_appraisal_level

    def _update_appraisal_state(self, sentence_index, new_level, new_visits_count, new_forgotten_flag, new_visit_step):
        self.appraisal_states[sentence_index]['appraisal_level'] = new_level
        self.appraisal_states[sentence_index]['visits_count'] = new_visits_count
        self.appraisal_states[sentence_index]['forgotten'] = new_forgotten_flag
        self.appraisal_states[sentence_index]['visit_step'] = new_visit_step

    def _init_stm(self):
        """
        Initialize the STM.
        """
        self._stm = OrderedDict()
    
    def _init_reward_weights(self):     
        """
        Initialize the reward weight.
        """
        self._failure_penalty_weight = 0.5       
        self._explore_weight = 0.25
        self._exploit_weight = 1 - self._explore_weight

    def _update_stm_with_just_read_sentence(self, sentence_index):
        """
        Update the STM.
        """

        if sentence_index in self._stm:
            # Reinforce directly
            self._calculate_appraisal_level(
                sentence_index=sentence_index,
                operation=const.SENTENCE_STATES_IN_MEMORY['revisited']
            )
            # Move to the end of the STM
            self._stm.move_to_end(sentence_index)
        else:
            # Remove the oldest sentence if the STM is full
            if len(self._stm) >= const.STM_CAPACITY:
                self._remove_oldest_sentence()
            # Initialize appraisal level if not already set
            if self.appraisal_states[sentence_index]['visits_count'] == const.ZERO:  # First visit
                self._calculate_appraisal_level(
                    sentence_index=sentence_index,
                    operation=const.SENTENCE_STATES_IN_MEMORY['first_visit']
                )
                # Update the number of read sentences
                self._advance_reading_progress()
            elif self.appraisal_states[sentence_index]['visits_count'] > const.ZERO:
                self._calculate_appraisal_level(
                    sentence_index=sentence_index,
                    operation=const.SENTENCE_STATES_IN_MEMORY['revisited']
                )
            # Add the sentence to STM (it will be added to the end)
            self._stm[sentence_index] = self.appraisal_states[sentence_index]['appraisal_level']

    def _remove_oldest_sentence(self):
        """
        Forget the oldest sentence in the STM.
        """
        oldest_sentence, _ = self._stm.popitem(last=False)
        appraisal_level = self.appraisal_states[oldest_sentence]['appraisal_level']
        if appraisal_level >= const.MEMORY_RETAIN_APPRAISAL_LEVEL_THRESHOLD:
            # Do not decrease appraisal, but it's removed from STM, do nothing to the appraisal level
            pass
        else:
            # Start decreasing appraisal over time
            self._calculate_appraisal_level(
                sentence_index=oldest_sentence, 
                operation=const.SENTENCE_STATES_IN_MEMORY['forgotten']
            )

    def _compute_continue_read_new_sentence_reward(self):       # TODO a lot of bugs here, tune it
        """
        Compute the reward for continuing to read.
        """
        return 0.1      # TODO reward could associated with the magnitude of the got appaisal level

    def _compute_revisit_to_reinforce_old_sentence_reward(self):
        """
        Compute the reward for regressing.
        """
        if self.appraisal_states[self.reading_sentence_index]['appraisal_level'] < 1:       # If there is space for reinforcement, 1 is the saturation point
            return 0.1      # TODO reward could associated with the magnitude of the reinforcement
        else:
            # Invalid action: Regress to a fully reinforced sentence
            return self._compute_penalty_for_invalid_actions_or_do_nothing()
    
    def _compute_penalty_for_invalid_actions_or_do_nothing(self):
        """
        Compute the penalty for invalid actions or doing nothing.
        """
        return -0.1 * self._steps       # A time-based penalty

    def _compute_time_up_reward(self):
        """
        Compute the reward for time up.
        The reward should be composed of two parts: the overall reading progress and the appraisal levels
        """

        # Get appraisal levels for read sentences
        read_appraisal_levels = [self.appraisal_states[idx]['appraisal_level'] for idx in range(self._num_unique_sentences_read_OR_farthest_sentence_reached)]

        # num_remaining_sentences = self._num_remaining_sentences
        # read_sentences_rewards = sum(read_appraisal_levels)

        # # TODO debug delete later
        # print(f"The read appraisal levels are: {read_appraisal_levels}\n")
        
        # Step 1: Compute reward based on Bernoulli sampling for read sentences
        read_samples = [np.random.binomial(1, appraisal_level) for appraisal_level in read_appraisal_levels]
        penalty_weights = [1-appraisal_level for appraisal_level in read_appraisal_levels]
        
        # Positive reward for successful recall, penalty for failure
        # read_reward = sum([1 if sample == 1 else -1 * self._failure_penalty_weight for sample in read_samples])      # TODO maybe I need a parameter here as well
        read_reward = sum([1 if sample == 1 else -1 * penalty_weights[idx] for idx, sample in enumerate(read_samples)])

        # # Normalize the read reward
        # norm_read_reward = self._exploit_weight * read_reward / self._total_num_sentences
        
        # Step 2: Compute penalty for unread sentences
        num_unread_sentences = self._total_num_sentences - self._num_unique_sentences_read_OR_farthest_sentence_reached
        # norm_unread_penalty = -1 * self._explore_weight * num_unread_sentences / self._total_num_sentences
        unread_sentences_penalty = -0.5 * num_unread_sentences
        
        # Step 3: Combine rewards and penalties
        termination_reward = 1 * (read_reward + unread_sentences_penalty)
        
        return termination_reward

    def _get_reading_progress_at_word_level(self) -> int:
        """
        Get the reading progress at the word level.
        """
        # Check the last sentence read by the appraisal levels -- the last index in the appraisal state that is not zero
        last_sentence_read_index = -1
        for idx in range(self._total_num_sentences):
            if self.appraisal_states[idx]['appraisal_level'] != 0:
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
            'num_read_sentences': self._num_unique_sentences_read_OR_farthest_sentence_reached,
            'num_remaining_sentences': self._num_remaining_sentences,
            'total_time_in_seconds': self._total_time_in_seconds,
            'time_elapsed_in_seconds': self.time_spent_in_seconds,
            'time_remaining_in_seconds': self.time_left_in_seconds,
            'appraisal_states': self.appraisal_states,
            'stm': self._stm,
            'individual_sentence_appraisal_level_weight': self._individual_sentence_appraisal_level,
            'average_individual_sentence_appraisal_level_weight': np.mean(self._individual_sentence_appraisal_levels_list_log),
            # 'reward_weight': self._reward_weight,
            'failure_penalty_weight': self._failure_penalty_weight,
            'exploit_weight': self._exploit_weight,
            'explore_weight': self._explore_weight,
            'num_regression': self.num_revisit_sentences,
            'num_sentence_level_saccades': self.num_total_sentences_read,
            'regression_rate_sentence_level': self.num_revisit_sentences / self.num_total_sentences_read,
            'revisit_percentage_word_level_using_saccades': (self.num_revisit_saccades_on_word_level / self._num_word_level_saccades) * 100,        # TODO this should be calculated by the SLC agent later
            'revisit_percentage_word_level_using_reading_progress': (self.num_revisit_saccades_on_word_level / self._get_total_num_words_read(self._get_reading_progress_at_word_level())) * 100 if self._get_total_num_words_read(self._get_reading_progress_at_word_level()) != 0 else 0,
        }

        # # TODO debug delete later   
        # print(f"The revisited log is: {self.revisited_sentences_list_log}\n")

        return step_log


if __name__ == '__main__':
    sc = SupervisoryControllerEnv()
    sc.reset()
