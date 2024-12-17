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
        Create on 14 October 2024.
        This is the environment for the RL-based intermediate level -- sentence-level control agent: it controls word skippings in a given sentence
        Based on the current reading progress, time constraints/pressure, and following words' appraisals/activation levels,
            the agent needs to decide whether to skip the next word or not.
        When training, all appraisal level data are artificial/imitated.

        TODO: maybe we could add word-level regression here as well?

        Author: Bai Yunpeng

        This version is originated from v0925, it should be able to use the different length of episodes time awareness but still finish reading the sentence.
        """

        # Get the current root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Get the mode from the config yaml file
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
        print(
            f"=====================================\n"
            f"RL Sentence Level Controller version 1014 Environment -- Deploying the environment in the {self._config['rl']['mode']} mode. \n"
            f"=====================================\n"
            )

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
        # The placeholder for the average appraisal level of the words in the sentence, acuiqred after the sentence is finished
        self.avg_words_appraisals = None

        # Define the state-related variables
        self.fixated_word_index = None
        self.sampled_indexes_in_sentence = None
        # Time constraints
        self._time_constraint_level_key = None
        self._norm_time_constraint_level = None
        # Counters
        self.num_words_skipped_in_sentence = None
        self.num_saccades_on_word_level = None
        self._time_over_steps = None

        # Parameterized weights
        self.MAX_TIME_CONSTRAINT_WEIGHT = 1.0
        self.MIN_TIME_CONSTRAINT_WEIGHT = 0.0
        self._time_constraint_weight = None
        self.MAX_OVER_TIME_REWARD_PENALTY_WEIGHT = 1.0
        self.MIN_OVER_TIME_REWARD_PENALTY_WEIGHT = 0.0
        self._over_time_reward_penalty_weight = None
        self.EMPIRICAL_OVERTIME_PENALTY_WEIGHT = 0.5

        # Define the RL env spaces
        self._steps = None
        self.granted_time_constraints_in_steps = None  # The length of the episode is a variable in this case
        self.ep_len = 1 * const.MAX_NUM_WORDS_PER_SENTENCE
        self._terminate = None
        self._truncated = None

        # Define the action space
        self.READ_ACTION = 0
        self.SKIP_ACTION = 1
        self.UNDEFINED_ACTION = -1
        self.action_space = Discrete(2)     # 0: read; 1: skip the next word (one word only)

        # Define the observation space
        self._num_stateful_info_obs = 7
        self.observation_space = Box(low=0, high=1, shape=(self._num_stateful_info_obs,))

    def reset(self, seed=42, inputs: dict = None):
        """
        Reset the environment to the initial state.
        """

        super().reset(seed=seed)

        self._steps = 0

        self._truncated = False

        self.fixated_word_index = -1
        self.sampled_indexes_in_sentence = []

        self.num_words_skipped_in_sentence = 0         # The number of words skipped -- the counter of the agent's skipping behaviors
        self.num_saccades_on_word_level = 0   # The number of saccades across words
        self._time_over_steps = 0   

        # Initialize the number of words in a sentence
        self._init_num_words_in_sentence(inputs=inputs)

        # Initialize the time constraint -- randomly pick one, and the episode length will be picked as well.
        self._init_time_constraint(inputs=inputs)

        # Initialize the appraisal levels
        self._init_appraisal_states()

        return self._get_obs(), {}

    def step(self, action, external_llm_next_word_predictability: float = None):
        """
        Take a step in the environment
        """

        # Part 1: Determine the word in the sentence to read
        reward, done = self.step_part1(action)

        # Part 2: After reading the given target word, update the predictability of the following word from the currently fixated word
        observation, reward, done, truncated, info = self.step_part2(reward, done, external_llm_next_word_predictability)

        return observation, reward, done, truncated, info
    
    def step_part1(self, action):

        done = False
        reward = 0.0

        # Update the steps
        self._steps += 1

        # Take the action
        if action == self.READ_ACTION:    # Read the next word
            if self.have_words_to_read():
                # self._read_next_word(predictability=external_llm_following_word_predictability)
                self._read_next_word()
                reward += self._compute_continue_reading_reward()
            else:
                done = True
                reward += self._compute_terminate_reading_reward()
        elif action == self.SKIP_ACTION:  # Skip the next word
            if self.have_words_to_skip():
                # done = self._skip_next_word(predictability=external_llm_following_word_predictability)
                done = self._skip_next_word()
                if done == False:
                    reward += self._compute_skip_word_reward()
                else:
                    reward += self._compute_terminate_reading_reward()
            else:
                done = True
                reward += self._compute_terminate_reading_reward()
        else:
            # Invalid action, do nothing
            print(f"Invalid action: {action}, should be 0 or 1.\n")
            reward += 0.0
        
        return reward, done
    
    def step_part2(self, reward, done, external_llm_next_word_predictability: float = None):

        time_penalty = 0.0

        # Get the next word's predictability if there is a next word (determined outside, here is just a value / placeholder, no worries)
        self._get_next_word_predictability(external_llm_next_word_predictability=external_llm_next_word_predictability)

        # Check if the episode should be terminated -- if the sentence has been finished
        if self._check_terminate():
            done = True
            reward += self._compute_terminate_reading_reward()
            # Log the sampled indexes in the sentence
            self.sampled_indexes_in_sentence = self._get_sampled_indexes_in_sentence()
        
        # Apply time penalty if steps exceed ep_len
        if self._steps > self.granted_time_constraints_in_steps:
            time_penalty = self._compute_time_penalty()
            reward += time_penalty
            self._time_over_steps += 1  # Increment the over-time step counter

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
        norm_time_constraint_in_steps = aux.normalise(self.granted_time_constraints_in_steps, 0, self.ep_len, -1, 1)
        # Remaining steps
        norm_remaining_time_constraints_steps = aux.normalise(self.granted_time_constraints_in_steps - self._steps, 0, self.granted_time_constraints_in_steps, -1, 1)

        # Time constraint awareness -- pre-calculated to save computational resources
        # norm_time_constraint_level = self._norm_time_constraint_level

        # Time constraint weight awareness
        norm_time_tolerance_weight = aux.normalise(1 - self._time_constraint_weight, self.MIN_TIME_CONSTRAINT_WEIGHT, self.MAX_TIME_CONSTRAINT_WEIGHT, -1, 1)

        # Over time reward penalty weight awareness
        norm_over_time_reward_penalty_weight = aux.normalise(self._over_time_reward_penalty_weight, self.MIN_OVER_TIME_REWARD_PENALTY_WEIGHT, self.MAX_OVER_TIME_REWARD_PENALTY_WEIGHT, -1, 1)

        # Reading progress awareness -- the number of words read
        norm_reading_progress = aux.normalise(self.num_words_read_in_sentence, 0, self.num_words_in_sentence, -1, 1)

        # Appraisal states awareness -- the appraisal levels of the following words
        norm_next_word_predictability = self._predictability_states[self.fixated_word_index + 1][0] if self.fixated_word_index < self.num_words_in_sentence - 1 else -1

        # Time over step awareness
        norm_time_over_steps = aux.normalise(self._time_over_steps, 0, self.ep_len, -1, 1)      

        stateful_info_obs = np.array([norm_time_constraint_in_steps, norm_remaining_time_constraints_steps, norm_time_tolerance_weight, norm_over_time_reward_penalty_weight,
                                      norm_reading_progress, norm_next_word_predictability, norm_time_over_steps])
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
    
    #@potential_parameter_inference_place
    # REference: https://docs.google.com/presentation/d/1iqBCSFAXCpKWc2O41FQpZEW2AhDx9PsbZLY3jd1gpuA/edit#slide=id.g30ae2ee59b1_0_54
    def _init_time_constraint(self, inputs: dict = None):
        """
        Initialize the time constraint.
        """

        assert self.num_words_in_sentence is not None, "The number of words in a sentence should be initialized first."

        if inputs is not None:  # Has external inputs   # TODO IMPORTANT DO NOT DELETE parameterize these later
            self._time_constraint_level_key = inputs["time_constraint_level"]
            self._over_time_reward_penalty_weight = self.EMPIRICAL_OVERTIME_PENALTY_WEIGHT     # Emiprically set as 0.5; could try others if needed
            # Get the corresponding level label
            if self._time_constraint_level_key == "30S":
                self._norm_time_constraint_level = -1
                self._time_constraint_weight = 0.5      # TODO make these tunable parameters later IMPORTANT: they need to be tuned later 
            elif self._time_constraint_level_key == "60S":
                self._norm_time_constraint_level = 0
                self._time_constraint_weight = 0.4
            elif self._time_constraint_level_key == "90S":
                self._norm_time_constraint_level = 1
                self._time_constraint_weight = 0.35
        else:   # Not specify, randomize everything
            # Sample the total time constraint in seconds from the predefined dictionary: 30s, 60s, 90s,
            random_key = random.choice(list(const.TIME_CONSTRAINT_LEVELS.keys()))
            random_value = const.TIME_CONSTRAINT_LEVELS[random_key]
            # Get the corresponding level label
            self._time_constraint_level_key = random_key
            # Randomize the time constraints in steps
            self._time_constraint_weight = np.random.uniform(self.MIN_TIME_CONSTRAINT_WEIGHT, self.MAX_TIME_CONSTRAINT_WEIGHT)
            # Randomize the over-time reward penalty weight
            # self._over_time_reward_penalty_weight = np.random.uniform(self.MIN_OVER_TIME_REWARD_PENALTY_WEIGHT, self.MAX_OVER_TIME_REWARD_PENALTY_WEIGHT)     
            self._over_time_reward_penalty_weight = 0.0       # TODO use it when testing
        
        # Get the episode length -- time constraints, but it should be related to the number of words in the sentence
        self.granted_time_constraints_in_steps = math.ceil((1 - self._time_constraint_weight) * self.num_words_in_sentence)

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
        
        # Initialize the next word's predictability
        self.avg_words_appraisals = 0.0

    def have_words_to_read(self):
        """
        Check if there are more words to read.
        """
        return self.fixated_word_index + 1 < self.num_words_in_sentence

    # def _read_next_word(self, predictability: float = None):
    def _read_next_word(self):
        """
        Read the next word.
        """
        # Update the stateful information
        self.fixated_word_index += 1
        self.num_words_read_in_sentence += 1
        self._num_remaining_words = self.num_words_in_sentence - self.num_words_read_in_sentence
        self.num_saccades_on_word_level += 1

        # Update the appraisal states if within bounds
        if self.fixated_word_index < self.num_words_in_sentence:
            self._predictability_states[self.fixated_word_index] = (1.0, self.READ_ACTION)      # This word is read explicitly (by fixation)

        # # Get the next word's predictability if there is a next word
        # self._get_next_word_predictability(external_llm_predictability=predictability)

    def have_words_to_skip(self):
        """
        Check if there are more words to skip.
        """
        return self.fixated_word_index + 2 <= self.num_words_in_sentence

    # def _skip_next_word(self, predictability: float = None):
    def _skip_next_word(self):
        """
        Skip the next word.
        """
        done = False

        words_remaining = self.num_words_in_sentence - self.fixated_word_index - 1

        if words_remaining >= 2:
            # Can skip one word and read the next
            skipped_word_index = self.fixated_word_index + 1
            self.fixated_word_index += 2
            read_word_index = self.fixated_word_index
            self.num_words_read_in_sentence += 2
        elif words_remaining == 1:
            # Skip the last word and finish
            skipped_word_index = self.fixated_word_index + 1
            read_word_index = None
            self.num_words_read_in_sentence += 1
            done = True
        else:
            # No words left to skip or read
            return

        self._num_remaining_words = self.num_words_in_sentence - self.num_words_read_in_sentence
        self.num_saccades_on_word_level += 1
        self.num_words_skipped_in_sentence += 1

        # Update the predictability states
        if skipped_word_index < self.num_words_in_sentence:
            last_word_predictability_level = self._predictability_states[skipped_word_index][0]
            self._predictability_states[skipped_word_index] = (last_word_predictability_level, self.SKIP_ACTION)    # Documenting that this word is skipped -- (current word index, (predictability, the current word is skip or not))
        if read_word_index is not None and read_word_index < self.num_words_in_sentence:
            self._predictability_states[read_word_index] = (1.0, self.READ_ACTION)      # This word was explicitly read, so its appraisal level is set to 1.0

        # # Get the next word's predictability
        # self._get_next_word_predictability(external_llm_predictability=predictability)

        return done
    
    def _get_next_word_predictability(self, external_llm_next_word_predictability: float = None):
        """
        Get the predictability of the next word (regarding the current fixated word) using the large language model when simulating OR random sample when training.
        It is used for next step's decision making.

        Or it could be called the "next next word" from the agent's perspective before actually reading anything.
        """
        if self.fixated_word_index + 1 < self.num_words_in_sentence:
            if external_llm_next_word_predictability is None:
                next_word_predictability = random.uniform(0.2, 0.9)
            else:
                next_word_predictability = external_llm_next_word_predictability
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
    
    def _compute_time_penalty(self):
        """
        Compute the penalty for exceeding the allocated time.
        """
        # Example: Negative reward proportional to the number of extra steps
        # penalty_per_step = -2 - self._over_time_reward_penalty_weight * 1
        penalty_per_step = -2 - self.EMPIRICAL_OVERTIME_PENALTY_WEIGHT * 1     # Empirically set as 0.5; could try others if needed
        return penalty_per_step

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

        termination_reward = 1 * aggregated_predictability_values

        return termination_reward

    def _check_terminate(self):
        """
        Check if the episode should be terminated.
        """
        if self.num_words_read_in_sentence >= self.num_words_in_sentence:
            self.avg_words_appraisals = np.mean([self._predictability_states[idx][0] for idx in range(self.num_words_in_sentence)])
            self._terminate = True
        else:
            self._terminate = False
        return self._terminate

    def _check_truncate(self):
        """
        Check if the episode should be truncated.
        """
        if self._steps >= self.granted_time_constraints_in_steps:
            self._truncated = True
        else:
            self._truncated = False
        return self._truncated

    def _get_sampled_indexes_in_sentence(self):
        """
        Get the sampled indexes in the sentence.
        """
        return [idx for idx in range(self.num_words_in_sentence) if self._predictability_states[idx][1] == self.READ_ACTION]

    def get_logs(self):
        """
        Log the information.
        """
        step_log = {
            'steps': self._steps,
            'fixated_word_index': self.fixated_word_index,
            'sampled_indexes_in_sentence': self.sampled_indexes_in_sentence,
            'time_constraint_level': self._time_constraint_level_key,
            'time_constraint_weight': self._time_constraint_weight,
            'over_time_reward_penalty_weight': self._over_time_reward_penalty_weight,
            'norm_time_constraint_level': self._norm_time_constraint_level,
            'ep_len': self.granted_time_constraints_in_steps,
            'appraisal_states': self._predictability_states,
            'num_words_skipped': self.num_words_skipped_in_sentence,
            'num_word_level_saccades': self.num_saccades_on_word_level,
            'avg_words_appraisals': np.mean([self._predictability_states[idx][0] for idx in range(self.num_words_in_sentence)]),
            'word_skipping_percentage': (self.num_words_skipped_in_sentence / self.num_words_read_in_sentence) * 100,
        }

        return step_log
