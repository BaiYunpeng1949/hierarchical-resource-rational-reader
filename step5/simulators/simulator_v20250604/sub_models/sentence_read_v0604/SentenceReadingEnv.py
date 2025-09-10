import math
import os
import yaml
import random
import torch
import numpy as np
import json

from scipy import stats
from typing import Optional, Sequence

from gymnasium import Env
from gymnasium.spaces import Box, Discrete, Dict

from .TimeConditionManager import TimeConditionManager
from .SentencesManager import SentencesManager
from .TransitionFunction import TransitionFunction
from .RewardFunction import RewardFunction
from . import Constants
from . import Utilities


# # Tunable parameters -- Units: words per second
# THIRTY_SECONDS_EXPECTED_READING_SPEED = Constants.READING_SPEED / 2
# SIXTY_SECONDS_EXPECTED_READING_SPEED = Constants.READING_SPEED
# NINETY_SECONDS_EXPECTED_READING_SPEED = Constants.READING_SPEED * 1.5

# # Dataset
# DATASET = "Ours"      # NOTE: I recommend using this dataset for testing
# # DATASET = "ZuCo1.0"       # NOTE: I recommend using this dataset for training 


class SentenceReadingUnderTimePressureEnv(Env):
    def __init__(self):
        """
        Create on 19 March 2025.
        This is the environment for the RL-based intermediate level -- sentence-level control agent: 
            it controls, word skippings, word revisits, and when to stop reading in a given sentence
        Updated in July, 2025.
            I tried to amortisely train the model, learn an agent that could adapt to different parameters. But did not go well.
        Updated on 25 July, 2025.
            I tried to use the Bayesian optimization to tune the parameters. Only one parameter to train at a time.
        Updated on 29 July, 2025.
            I tried to tune the agent's perception of the time pressure directly, set it as a tunable parameter.
            Change the structure to v1014. Grant some time, if exceeds, terminate.
        
        Features: predict the word skipping and word revisiting decisions, and when to stop reading.
        Cognitive constraints: (limited time and cognitive resources) 
            Foveal vision (so need multiple fixations), 
            STM (so need to revisit sometimes, provide limited contextual predictability),
            attention resource (so need to skip the unnecessary words),
            Time pressure (so need to finish reading before time runs out)
        """
        # Load configuration
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
        self._mode = self._config["rl"]["mode"]

        if self._mode == "simulate":
            DATASET = "Ours"
            assert DATASET == "Ours", f"Invalid dataset: {DATASET}, should be 'Ours' when running the simulator!"
        elif self._mode == "train" or self._mode == "continual_train" or self._mode == "debug":
            DATASET = "ZuCo1.0"
            assert DATASET == "ZuCo1.0", f"Invalid dataset: {DATASET}, should be 'ZuCo1.0' when training the model!"
        print(f"\n===================================================================================================================")
        print(f"Sentence Reading Under Time Pressure Environment V0604 -- Deploying in {self._mode} mode with the dataset {DATASET}")
        print(f"===================================================================================================================\n")

        # Initialize components
        self.time_condition_manager = TimeConditionManager()
        self.sentences_manager = SentencesManager(dataset=DATASET)
        self.transition_function = TransitionFunction()
        self.reward_function = RewardFunction()

        # Text information (for the time pressure approximation)
        self._text_word_count = None

        # State tracking, mainly the sentence-level information
        self._sentence_info = None
        self._sentence_len = None
        self.current_word_index = None
        self._previous_word_index = None
        self._actual_reading_word_index = None
        self._word_beliefs = None  # Store beliefs for each word
        self._read_words = set()  # Track which words have been read

        # Reading behavior tracking
        self._skipped_words_indexes = None      
        self._regressed_words_indexes = None
        self.reading_sequence = None
        self.local_actual_fixation_sequence_in_sentence = None      # Local means within the sentence
        self.global_actual_fixation_sequence_in_text = None         # Global means outside of the sentence, but within the text

        # Comprehension score tracking
        self._ongoing_sentence_comprehension_score = None
        self._valid_words_beliefs = None

        # Time tracking
        self._time_condition = None
        self._time_condition_value = None
        self._baseline_time_needed_to_read_text = None
        self.elapsed_time = None
        self._sentence_wise_remaining_time_in_seconds = None
        self._min_sentence_wise_remaining_time_in_seconds = None
        self._time_pressure_scalar_for_the_sentence = None
        self._sentence_wise_expected_time_pressure_in_seconds = None
        self._sampled_n_words_reading_time_for_text_ndarray = None
        self._sampled_m_words_reading_time_for_sentence_ndarray = None

        # Environment parameters
        self._steps = None
        self.ep_len = 2 * Constants.MAX_SENTENCE_LENGTH
        self._granted_step_budget = None
        self._terminate = None
        self._truncated = None
        self._episode_id = None

        # Action space
        self._REGRESS_ACTION = 0
        self._READ_ACTION = 1
        self._SKIP_ACTION = 2
        self._STOP_ACTION = 3    
        self.action_space = Discrete(4)     
        
        # Observation space - simplified to scalar signals
        self._num_stateful_obs = 6 + 3      # +2 for the time condition and remaining granted steps and whether overtime or not
        self.observation_space = Box(low=0, high=1, shape=(self._num_stateful_obs,))
        self._noisy_obs_sigma = Constants.NOISY_OBS_SIGMA

        # Free parameters
        self._w_regression_cost = None          # NOTE: the most important parameter to tune for natural reading
        self._w_skipping_cost = None            # NOTE: maybe the most important parameter to tune for reading under time pressure
        self._w_comprehension_vs_time_pressure = None    # NOTE: another parameter might be needed
        self.MIN_W_COMPREHENSION_VS_TIME_PRESSURE = 0.0
        self.MAX_W_COMPREHENSION_VS_TIME_PRESSURE = 0.5
        self._w_skip_degradation_factor = None
        self.MIN_W_SKIP_DEGRADATION_FACTOR = 0.25
        self.MAX_W_SKIP_DEGRADATION_FACTOR = 1.00
        self._w_step_wise_comprehension_gain = None
        self.MIN_W_STEP_WISE_COMPREHENSION_GAIN = 0.0
        self.MAX_W_STEP_WISE_COMPREHENSION_GAIN = 1.0
        
        self._w_time_perception = None

        # Log variables
        self._log_individual_step_action = None
        self._log_elapsed_time_list_for_each_index = None
        self._log_remaining_time_list_for_each_index = None
        self._log_num_skips = None
        self._log_num_regressions = None
        self._log_terminate_reward = None
        self._log_terminate_reward_logs = None
        
    def reset(self, seed=42, inputs: dict=None):          
        """
        Reset environment and initialize states.

        inputs: inputted configurations, including assigned sentence index and episode id
        """
        super().reset(seed=seed)

        self._steps = 0
        self._terminate = False
        self._truncated = False
        # self._episode_id = episode_id
        self._episode_id = inputs["episode_id"] if inputs is not None else 0

        # Get new sentence
        self._sentence_info, self._text_word_count = self.sentences_manager.reset(inputs=inputs)

        # Get the sentence information
        self._sentence_len = len(self._sentence_info['words'])

        # TODO 1. organize the documentation, see which parameters are functioning; 
        # 2.when training, use the sampled elapsed time (I could sampled from our simulation data); 
        # 3. link the real time elapse when simulating 

        # Initialize word beliefs from pre-computed data
        self._word_beliefs = [-1] * self._sentence_len
        self._read_words = []
        
        # Reset reading state
        self.current_word_index = -1
        self._previous_word_index = None
        self._actual_reading_word_index = None
        
        # Reset tracking
        self._skipped_words_indexes = []    # only check the first-pass skipped words
        self._regressed_words_indexes = []
        self.reading_sequence = []
        self.local_actual_fixation_sequence_in_sentence = []
        self.global_actual_fixation_sequence_in_text = []

        # Reset the comprehension score tracking
        self._ongoing_sentence_comprehension_score = 0.0
        self._valid_words_beliefs = []

        # Reset the time related variables
        self._time_condition, self._time_condition_value = self.time_condition_manager.reset(inputs=inputs)
        # Sample words reading time for this text, sampling pool was approximated from the actual simulation data. Do the random sampling for a more robust policy learning.
        self._sampled_n_words_reading_time_for_text_ndarray = sample_n_individual_word_elapsed_duration_from_our_simulation_data(
            dist_name="gamma", 
            n=self._text_word_count,
            params=(1.827785449759574, 25.65868356976778, 154.48236013843115),  # (a, loc, scale)
        )
        # Approximate the time pressure for each sentence # Get the text information, because need this to approximate the time pressure for each sentence
        # self._baseline_time_needed_to_read_text = self._text_word_count * Constants.READING_SPEED
        self._baseline_time_needed_to_read_text = np.sum(self._sampled_n_words_reading_time_for_text_ndarray)      
        # Get the time pressure for each sentence
        self._time_pressure_scalar_for_the_sentence = self._time_condition_value / self._baseline_time_needed_to_read_text    # belongs to [0, infinity]
        
        # NOTE: our tunable parameter working
        self._w_time_perception = 0.35          # Now I assume it is a tunable parameter   # TODO try a new function that tear bigger gap between 30s and 60s conditions --> NOTE: try 0.5 later, where applies more time pressure perception
        granted_step_budget_factor = self.calc_time_pressure_to_factor(x=self._time_pressure_scalar_for_the_sentence, w=self._w_time_perception)
        # Granted step budget
        self._granted_step_budget = np.ceil(granted_step_budget_factor * self._sentence_len)      # This value is definitely smaller than the sentence lenght.

        # reading_speed = Constants.READING_SPEED     # TODO change this
        # # TODO we need to do a sentence-words sampling here. 1. collect data from simulation; 2. see what distribution it is; 3. see what ranges it cover; 
        # # 4. can I use a simple stuff to cover it. Or, can I use a simple average value to stand-for it. I still use this average value. But dyanmical values are better because they are more dynamic. The learned policy could be more robust.
        # self._sentence_wise_expected_time_pressure_in_seconds = reading_speed * self._sentence_len * self._time_pressure_scalar_for_the_sentence    
        # # TODO: re-sample this using our sampled values. TODO: resample a random index

        # NOTE: the individual word's reading time should be obtained from the lower-level agent. But for training efficiency, 
        #   I directly sample them from the distribution observed from simulation data. And sample randomly (first sent_len) values in the sampled text reading time
        #   as sampled sentence reading time.
        #   I will keep using this with simulation mode as well, bc I don't think these individual words' reading time are dominant factors.
        self._sampled_m_words_reading_time_for_sentence_ndarray = self._sampled_n_words_reading_time_for_text_ndarray[0:self._sentence_len]
        sampled_expected_sentence_reading_time = np.sum(self._sampled_m_words_reading_time_for_sentence_ndarray)
        self._sentence_wise_expected_time_pressure_in_seconds = sampled_expected_sentence_reading_time * self._time_pressure_scalar_for_the_sentence

        # # TODO debug delete later
        # print(f"\nthe total time needed: {self._baseline_time_needed_to_read_text}, the self._time_pressure_scalar_for_the_sentence is: {self._time_pressure_scalar_for_the_sentence}")
        # print(f"The granted_step_budget_factor is: {granted_step_budget_factor}, the self._granted_step_budget is: {self._granted_step_budget}, the sentence len is: {self._sentence_len}")

        self.elapsed_time = 0
        self._sentence_wise_remaining_time_in_seconds = self._sentence_wise_expected_time_pressure_in_seconds - self.elapsed_time
        self._min_sentence_wise_remaining_time_in_seconds = - self._sentence_wise_expected_time_pressure_in_seconds   
        # NOTE: empirically set a negative value, for tracking the exceeded time

        # Initialize a random regression cost
        # self._w_regression_cost = random.uniform(0, 1)   # NOTE: uncomment when training!!!!
        self._w_regression_cost = 1.0    # NOTE: uncomment when testing!!!! --> For the reading under time constraint, no need to change, keep it as constant in both training and testing.
        
        # NOTE: The two tunable parameters, try, if identified, get it into the Bayesian optimization later
        # TODO: not useful parameters, maybe delete
        self._w_skip_degradation_factor = 1.0
        self._w_comprehension_vs_time_pressure = 0.5
        self._w_step_wise_comprehension_gain = 0.5      # Tunable step-wise parameter

        # Initialize the log variables
        self._log_elapsed_time_list_for_each_index = []
        self._log_remaining_time_list_for_each_index = []
        self._log_num_skips = 0
        self._log_num_regressions = 0
        self._log_terminate_reward = 0

        return self._get_obs(), {}
    
    def step(self, action):
        """Take action and update states"""
        self._steps += 1
        reward = 0

        self._log_individual_step_action = action
        old_word_beliefs = self._word_beliefs.copy()

        if action == self._REGRESS_ACTION:
            self.current_word_index, action_validity = (
                self.transition_function.update_state_regress(
                    self.current_word_index,
                    self._sentence_len
                )    # The current word now is the word before
            )
            self._actual_reading_word_index = self.current_word_index
            if action_validity:
                self._regressed_words_indexes.append(self.current_word_index)
                self.reading_sequence.append(self.current_word_index)
                self.local_actual_fixation_sequence_in_sentence.append(self.current_word_index)
                # NOTE: plan 3 -- reinforce both the revisited word and the difficult word: source: https://docs.google.com/presentation/d/1JYPKUz5k5Ncp_WJHWshXA4j_h5Nnnd_D2RlHfh9Taoo/edit?slide=id.g349565993ea_0_0#slide=id.g349565993ea_0_0
                self._word_beliefs[self.current_word_index] = 1.0
                self._word_beliefs[self.current_word_index+1] = 1.0    # Simple reinforcment, directly set to 1.0

                # Lower the cost of regression: jump to the last and AUTOMATICALLY jump back. Objective: see whether the agent would try regressions more often
                self.current_word_index += 1     # Jump back to the last word read before
                self._previous_word_index = self.current_word_index - 1
                # Because both words are reinforced, so we read the word after the regressed word as well.
                self.reading_sequence.append(self.current_word_index)
                self.local_actual_fixation_sequence_in_sentence.append(self.current_word_index)

                # Update the time related variables
                self._update_time_related_variables()      

                self._log_num_regressions += 1

            reward = self.reward_function.compute_regress_reward(w_regression_cost=self._w_regression_cost) + self.reward_function.compute_step_wise_comprehension_gain(
                w_comprehension_gain=self._w_step_wise_comprehension_gain, 
                old_beliefs=old_word_beliefs, 
                new_beliefs=self._word_beliefs.copy()
                )
        
        elif action == self._SKIP_ACTION:
            self.current_word_index, action_validity = (
                self.transition_function.update_state_skip_next_word(
                    self.current_word_index,
                    self._sentence_len
                )
            )    # The current word now is the word after the word being skipped.
            self._actual_reading_word_index = self.current_word_index
            if action_validity:
                # Use pre-computed ranked integration probability as belief
                skipped_word_index = self.current_word_index - 1
                self._previous_word_index = skipped_word_index
                # self._word_beliefs[skipped_word_index] = self._sentence_info['words_predictabilities_for_running_model'][skipped_word_index]
                
                # If the skipped word has been read before, use the original integration values
                if skipped_word_index in self._read_words:
                    pass
                else:
                    # If the skipped word has not been read before, use the pre-processed integration values
                    perfect_skipped_word_integration_prob = self._sentence_info['predicted_words_ranked_integration_probabilities_for_running_model'][skipped_word_index]
                    noisy_skipped_word_integration_prob = self._get_noisy_skipped_word_integration_prob(perfect_skipped_word_integration_prob)
                    self._word_beliefs[skipped_word_index] = noisy_skipped_word_integration_prob
                
                # If the skip destination word (the word after the skipped word) has been read before, use the original integration values
                if self.current_word_index in self._read_words:
                    pass
                else:
                    self._word_beliefs[self.current_word_index] = self._sentence_info['words_ranked_word_integration_probabilities_for_running_model'][self.current_word_index]

                # Check if the skipped word is the first-pass skipped word
                if skipped_word_index not in self.reading_sequence:
                    self._skipped_words_indexes.append(skipped_word_index)

                self.reading_sequence.append(skipped_word_index)
                self.reading_sequence.append(self.current_word_index)

                self.local_actual_fixation_sequence_in_sentence.append(self._actual_reading_word_index)

                # Update the time related variables
                self._update_time_related_variables()

                self._log_num_skips += 1

            reward = self.reward_function.compute_skip_reward() + self.reward_function.compute_step_wise_comprehension_gain(
                w_comprehension_gain=self._w_step_wise_comprehension_gain, 
                old_beliefs=old_word_beliefs, 
                new_beliefs=self._word_beliefs.copy()
                )
        
        elif action == self._READ_ACTION:
            self.current_word_index, action_validity = (
                self.transition_function.update_state_read_next_word(
                    self.current_word_index,
                    self._sentence_len
                )
            )    # The current word now is the next word being read.
            self._actual_reading_word_index = self.current_word_index
            if action_validity:
                # Sample from prediction candidates with highest probabilit
                self.reading_sequence.append(self.current_word_index)
                self.local_actual_fixation_sequence_in_sentence.append(self._actual_reading_word_index)

                self._previous_word_index = self.current_word_index - 1

                # If the read word has been read before, use the original integration values
                if self.current_word_index in self._read_words:
                    pass
                else:
                    self._word_beliefs[self.current_word_index] = self._sentence_info['words_ranked_word_integration_probabilities_for_running_model'][self.current_word_index]
                    
                    # Update the time related variables
                    self._update_time_related_variables()

            reward = self.reward_function.compute_read_reward() + self.reward_function.compute_step_wise_comprehension_gain(
                w_comprehension_gain=self._w_step_wise_comprehension_gain, 
                old_beliefs=old_word_beliefs, 
                new_beliefs=self._word_beliefs.copy()
                )

        elif action == self._STOP_ACTION:
            self._terminate = True
        
        #######################################################################################

        # Check whether exceeds the allocated step numbers
        if self._steps >= self._granted_step_budget:
            # Apply small penalties
            reward += self.reward_function.compute_step_wise_overtime_penalty()
        
        # Check termination
        if self._steps >= self.ep_len:
            self._terminate = True
            self._truncated = True

        if self._terminate: 
            info = self.get_episode_log()
            
            # Compute final comprehension reward
            valid_words_beliefs = [b for b in self._word_beliefs if b != -1]
            reward, logs = self.reward_function.compute_terminate_reward(
                sentence_len=self._sentence_len,
                num_words_read=len(valid_words_beliefs),
                words_beliefs=valid_words_beliefs,
                remaining_time=self._sentence_wise_remaining_time_in_seconds,
                expected_sentence_reading_time=self._sentence_wise_expected_time_pressure_in_seconds,
                w_comprehension_vs_time_pressure=self._w_comprehension_vs_time_pressure
            )
            self._log_terminate_reward = reward
            self._log_terminate_reward_logs = logs
        else:
            info = {}
        
        return self._get_obs(), reward, self._terminate, self._truncated, info
    
    @staticmethod
    def normalise(x, x_min, x_max, a, b):
    # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a
    
    @staticmethod
    def calc_time_pressure_to_factor(x, w):
        """
        Use a linear or non-linear function to describe the perceived time pressure on each sentence, from allocated time
        """
        offset = 1.0
        return 1.0 - w / (x + offset)
    
    def _get_obs(self):
        """Get observation with simplified scalar signals"""
        # Get current position (normalized)
        current_position = self.normalise(self.current_word_index, 0, self._sentence_len - 1, 0, 1)
        
        valid_words_beliefs = [b for b in self._word_beliefs if b != -1]
        self._valid_words_beliefs = valid_words_beliefs

        # Get remaining words (normalized)
        remaining_words = self.normalise(self._sentence_len - len(valid_words_beliefs), 0, self._sentence_len, 0, 1)
        
        # Get the previous word's belief
        norm_previous_word_belief = np.clip(self._word_beliefs[self._previous_word_index], 0, 1) if self._previous_word_index is not None and 0 <= self._previous_word_index < self._sentence_len else 1
        norm_current_word_belief = np.clip(self._word_beliefs[self.current_word_index], 0, 1) if self.current_word_index is not None and 0 <= self.current_word_index < self._sentence_len else 1
        norm_next_word_predictability = np.clip(self._sentence_info['words_predictabilities_for_running_model'][self.current_word_index + 1], 0, 1) if self.current_word_index + 1 is not None and 0 <= self.current_word_index + 1 < self._sentence_len else 1

        # Apply noisy observation for a more robust model -- compensation for the limited data
        observed_previous_word_belief = norm_previous_word_belief
        observed_current_word_belief = norm_current_word_belief
        observed_next_word_predictability = norm_next_word_predictability
        # Get the on-going comprehension scalar
        # on_going_comprehension_scalar = np.clip(math.prod(valid_words_beliefs), 0, 1)
        self._ongoing_sentence_comprehension_score = self._compute_ongoing_sentence_comprehension_score(valid_words_beliefs)
        
        # Weights
        # norm_w_regression_cost = self.normalise(self._w_regression_cost, 0, 1, 0, 1)
        # # norm_w_skipping_cost = self.normalise(self._w_skipping_cost, self.MIN_W_SKIPPING_COST, self.MAX_W_SKIPPING_COST, 0, 1)
        # # norm_noisy_skipped_word_integration_prob_sigma = self.normalise(self._noisy_skipped_word_integration_prob_sigma, self.MIN_NOISY_SKIPPED_WORD_INTEGRATION_PROB_SIGMA, self.MAX_NOISY_SKIPPED_WORD_INTEGRATION_PROB_SIGMA, 0, 1)
        # # norm_w_skip_degradation_factor = self.normalise(self._w_skip_degradation_factor, self.MIN_W_SKIP_DEGRADATION_FACTOR, self.MAX_W_SKIP_DEGRADATION_FACTOR, 0, 1)
        # norm_w_comprehension_vs_time_pressure = self.normalise(self._w_comprehension_vs_time_pressure, self.MIN_W_COMPREHENSION_VS_TIME_PRESSURE, self.MAX_W_COMPREHENSION_VS_TIME_PRESSURE, 0, 1)

        # Time related variables
        if self._time_condition == "30s":
            norm_time_condition = -1
        elif self._time_condition == "60s":
            norm_time_condition = 0
        elif self._time_condition == "90s":
            norm_time_condition = 1
        else:
            raise ValueError(f"Invalid time condition: {self._time_condition}")
        
        # min_remaining_time = self._min_sentence_wise_remaining_time_in_seconds
        # max_remaining_time = self._sentence_wise_expected_time_pressure_in_seconds
        # clipped_remaining_time = np.clip(self._sentence_wise_remaining_time_in_seconds, min_remaining_time, max_remaining_time)
        # norm_remaining_time = self.normalise(clipped_remaining_time, min_remaining_time, max_remaining_time, 0, 1)   
        # In the simulation mode, when the time is running out, or reading out of the sentence, the remaining time is negative. I clip it to 0.

        # Variables related to whether overtime or not in terms of the granted step budget
        is_overtime = 1 if self._steps >= self._granted_step_budget else 0
        clipped_remaining_granted_step = np.clip(self._granted_step_budget-self._steps, 0, self._granted_step_budget)
        norm_remaining_granted_step_budget = self.normalise(clipped_remaining_granted_step, 0, self._granted_step_budget, 0, 1)

        stateful_obs = np.array([
            current_position,
            remaining_words,
            observed_previous_word_belief,
            observed_current_word_belief,
            observed_next_word_predictability,
            self._ongoing_sentence_comprehension_score,
            # norm_w_regression_cost,
            # norm_w_skip_degradation_factor,
            # norm_w_comprehension_vs_time_pressure,
            norm_time_condition,
            # norm_remaining_time
            norm_remaining_granted_step_budget,
            is_overtime
        ])

        assert stateful_obs.shape == (self._num_stateful_obs,)

        return stateful_obs
    
    ############################### Helper functions ###############################

    def _get_global_actual_fixation_sequence_in_text(self):
        """Get the global actual fixation sequence in text"""
        # Get the current sentence info
        sentence_id = self._sentence_info['sentence_id']
        
        # Load metadata to get sentence start indices
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        metadata_path = os.path.join(root_dir, "sentence_read_v0604", "assets", "metadata_sentence_indeces.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Find the stimulus and sentence that contains our sentence content
        current_sentence_content = self._sentence_info['sentence_content']
        current_stimulus = None
        current_sentence_info = None
        
        for stimulus in metadata:
            for sentence in stimulus['sentences']:
                if sentence['sentence'] == current_sentence_content:
                    current_stimulus = stimulus
                    current_sentence_info = sentence
                    break
            if current_stimulus:
                break
                
        if not current_stimulus or not current_sentence_info:
            raise ValueError(f"Sentence content not found in metadata")
            
        # Get the start index of the current sentence in the global text
        sentence_start_idx = current_sentence_info['start_idx']
        
        # Convert local fixation indices to global indices
        global_fixation_sequence = [idx + sentence_start_idx for idx in self.local_actual_fixation_sequence_in_sentence]
        
        return global_fixation_sequence

    def _update_time_related_variables(self):
        """Update the time related variables"""
        self.elapsed_time, self._sentence_wise_remaining_time_in_seconds = self.transition_function.update_state_time(
            elapsed_time=self.elapsed_time,
            expected_sentence_reading_time=self._sentence_wise_expected_time_pressure_in_seconds,
            # word_reading_time=self._sentence_info['individual_word_reading_time']   # NOTE Now it is the priority! NOT Priority: I will not apply the real words' reading time here for now. Implement when needed later, or too much nuances
            word_reading_time=self._sampled_m_words_reading_time_for_sentence_ndarray[self._actual_reading_word_index]  # Version 0910  # NOTE a slightly more realistic value, but still not at the max fidelity.
        )
        
    def _get_noisy_skipped_word_integration_prob(self, perfect_skipped_word_integration_prob):
        """Get the noisy skipped word integration probability"""
        degraded_skipped_word_integration_prob = perfect_skipped_word_integration_prob * self._w_skip_degradation_factor
        return degraded_skipped_word_integration_prob
    
    def _compute_ongoing_sentence_comprehension_score(self, valid_words_beliefs):
        """Compute the ongoing sentence comprehension score"""
        if len(valid_words_beliefs) > 0:
            # Apply the softmin function to calculate the sentence-appraisals, such to stress the importance of the accurate word understandings, i.e., higher appraisals
            ongoing_sentence_comprehension_score = Utilities.calc_dynamic_sentence_comprehension_score(valid_words_beliefs, mode="mean")
        else:
            ongoing_sentence_comprehension_score = 0.0
        
        # Clip the ongoing sentence comprehension score to be in the range [0, 1]
        ongoing_sentence_comprehension_score = np.clip(ongoing_sentence_comprehension_score, 0, 1)

        return ongoing_sentence_comprehension_score
    
    def _get_actually_sampled_words_in_sentence(self) -> list:
        """
        Get the words actually sampled from the sentence
        Includes: 1. actually read or regressed words; and 2. skipped words, represented by their predictions
        """
        sampled_words = []

        for word_index in self.reading_sequence.copy():
            if word_index in self._skipped_words_indexes.copy():    # If this was an skipped word
                # NOTE: predicted words start from the 2nd in the sentence, i.e., the first word in the sentence does not have a prediction.
                skipped_word_index = word_index - 1
                if skipped_word_index >= 0:
                    predicted_word = self._sentence_info['predicted_words_for_running_model'][skipped_word_index]
                    sampled_words.append(predicted_word)
                else:
                    sampled_words.append('')
            # elif word_index in self.local_actual_fixation_sequence_in_sentence.copy():   
            else:           # If this is a normally fixated word
                sampled_words.append(self._sentence_info['words'][word_index])
        
        return sampled_words

    
    def get_individual_step_log(self) -> dict:
        """Get individual step log"""
        if self._log_individual_step_action == self._REGRESS_ACTION:
            action_information = "regress"
        elif self._log_individual_step_action == self._SKIP_ACTION:
            action_information = "skip"
        elif self._log_individual_step_action == self._READ_ACTION:
            action_information = "read"
        elif self._log_individual_step_action == self._STOP_ACTION:
            action_information = "stop"
        else:
            raise ValueError(f"Invalid action: {self._log_individual_step_action}")
        
        individual_step_log = {
            "step": self._steps,
            "action": action_information,
            "current_word_index": self.current_word_index,
            "actual_reading_word_index": self.reading_sequence[-1],
            "elapsed_time": self.elapsed_time,      # TODO check here, not right.
            "remaining_time": self._sentence_wise_remaining_time_in_seconds,
            "valid_words_beliefs": self._valid_words_beliefs.copy(),
            "ongoing_sentence_comprehension_score": self._ongoing_sentence_comprehension_score,
            "word_recognition_summary": {},
            "word_recognition_logs": [],
        }
        return individual_step_log
    
    def get_summarised_sentence_reading_logs(self) -> dict:
        """Get summarised sentence reading logs"""
        summarised_sentence_reading_logs = {
            "sentence_id": self._sentence_info['sentence_id'],
            "num_words_in_sentence": self._sentence_len, 
            "sampled_words_in_sentence": self._get_actually_sampled_words_in_sentence(),
            "original_words_intergration_values (beliefs)": self._sentence_info['words_ranked_word_integration_probabilities_for_running_model'].copy(),
            "num_steps_or_fixations": self._steps,
            "num_regressions": self._log_num_regressions,
            "num_skips": self._log_num_skips,
            "num_stops": 1,
            "reading_sequence": self.reading_sequence.copy(),
            "local_actual_fixation_sequence_in_sentence": self.local_actual_fixation_sequence_in_sentence.copy(),
            "global_actual_fixation_sequence_in_text": self._get_global_actual_fixation_sequence_in_text(),
            "skipped_words_indexes": self._skipped_words_indexes.copy(),
            "regressed_words_indexes": self._regressed_words_indexes.copy(),
            "sentence_wise_expected_reading_time_in_seconds": self._sentence_wise_expected_time_pressure_in_seconds,
            "w_regression_cost": self._w_regression_cost,
            "w_skip_degradation_factor": self._w_skip_degradation_factor,
            "w_comprehension_vs_time_pressure": self._w_comprehension_vs_time_pressure,
            "elapsed_time": self.elapsed_time,
            "remaining_time": self._sentence_wise_remaining_time_in_seconds,
            "sentence_terminate_reward": self._log_terminate_reward,
            "sentence_terminate_reward_logs": self._log_terminate_reward_logs,
        }
        return summarised_sentence_reading_logs
    
    def get_episode_log(self) -> dict:
        """Get logs for the episode"""
        words_data_list = []
        for word_idx in range(self._sentence_len):
            word_data = {
                "word": self._sentence_info['words'][word_idx],
                "word_clean": self._sentence_info['word_cleans'][word_idx],
                "word_id": self._sentence_info['word_ids'][word_idx],
                "length": self._sentence_info['word_lengths_for_analysis'][word_idx],
                "frequency_per_million": self._sentence_info['word_frequencies_per_million_for_analysis'][word_idx],
                "log_frequency": self._sentence_info['word_log_frequencies_per_million_for_analysis'][word_idx],
                "difficulty": self._sentence_info['word_difficulties_for_analysis'][word_idx],
                "predictability": self._sentence_info['word_predictabilities_for_analysis'][word_idx],
                "logit_predictability": self._sentence_info['word_logit_predictabilities_for_analysis'][word_idx],
                "belief_in_next_word_predictability": self._sentence_info['words_predictabilities_for_running_model'][word_idx],
                "is_first_pass_skip": word_idx in self._skipped_words_indexes,
                "is_regression_target": word_idx in self._regressed_words_indexes,
                "FFD": [],
                "GD": [],
                "TRT": [],
                "nFixations": [],
            }
            words_data_list.append(word_data)
        
        episode_log = {
            "episode_id": self._episode_id,
            'sentence_id': self._sentence_info['sentence_id'],
            'participant_id': self._sentence_info['participant_id'],
            'sentence_content': self._sentence_info['sentence_content'],
            'sentence_len': self._sentence_len,
            "words": words_data_list,
            "time_condition": self._time_condition,
            "time_condition_value": self._time_condition_value,
            "elapsed_time": self.elapsed_time,
            "remaining_time": self._sentence_wise_remaining_time_in_seconds,
            "sentence_wise_expected_reading_time_in_seconds": self._sentence_wise_expected_time_pressure_in_seconds,
            "w_regression_cost": self._w_regression_cost,
            "w_skip_degradation_factor": self._w_skip_degradation_factor,
            "num_steps": self._steps,
            "num_regressions": self._log_num_regressions,
            "num_skips": self._log_num_skips,
            "sentence_wise_regression_rate": self._log_num_regressions / self._sentence_len,
            "sentence_wise_skip_rate": self._log_num_skips / self._sentence_len,
            "sentence_wise_reading_speed": None if self.elapsed_time == 0 else self._sentence_len / self.elapsed_time * 60,
        }

        return episode_log
    

def sample_n_individual_word_elapsed_duration_from_our_simulation_data(dist_name: str, params: Sequence[float], n: int, random_state: Optional[int] = None):
    """
    Draw n samples from a SciPy distribution given its name and fitted params.
    params must be the exact tuple returned by dist.fit(...): (shape(s)?, loc, scale)
    """
    if random_state is not None:
        np.random.seed(random_state)
    dist = getattr(stats, dist_name)
    return dist.rvs(*params, size=n)


if __name__ == "__main__":
    pass