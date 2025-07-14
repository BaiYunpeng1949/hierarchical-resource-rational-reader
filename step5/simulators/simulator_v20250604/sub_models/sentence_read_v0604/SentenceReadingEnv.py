import math
import os
import yaml
import random
import torch
import numpy as np
import json

from gymnasium import Env
from gymnasium.spaces import Box, Discrete, Dict

from .TimeConditionManager import TimeConditionManager
from .SentencesManager import SentencesManager
from .TransitionFunction import TransitionFunction
from .RewardFunction import RewardFunction
from . import Constants


# Tunable parameters
THIRTY_SECONDS_EXPECTED_READING_SPEED = Constants.READING_SPEED / 2
SIXTY_SECONDS_EXPECTED_READING_SPEED = Constants.READING_SPEED
NINETY_SECONDS_EXPECTED_READING_SPEED = Constants.READING_SPEED * 1.5

# Dataset
DATASET = "Ours"      # NOTE: I recommend using this dataset for testing
# DATASET = "ZuCo1.0"       # NOTE: I recommend using this dataset for training 


class SentenceReadingUnderTimePressureEnv(Env):
    def __init__(self):
        """
        Create on 19 March 2025.
        This is the environment for the RL-based intermediate level -- sentence-level control agent: 
            it controls, word skippings, word revisits, and when to stop reading in a given sentence
        
        Features: predict the word skipping and word revisiting decisions, and when to stop reading.
        Cognitive constraints: (limited time and cognitive resources) 
            Foveal vision (so need multiple fixations), 
            STM (so need to revisit sometimes, provide limited contextual predictability),
            attention resource (so need to skip the unnecessary words),
            Time pressure (so need to finish reading before time runs out)
        
        TODO: 1. change the POMDP to the time-pressured version
        2. process the stimuli data to be used
        3. train the model
        4. tune the reward weights or change the reward function
        5. plot results, especially the regression rate and the skipping rate
        6. reiterate from 3 until results are satisfactory
        7. if all works fine, then try to test on the dataset of our stimulus. (maybe just train based on our dataset, for safety)

        NOTE: training method 1 and 2
        For the training method 1: use three discrete time conditions: 30s, 60s, 90s, and manufacture the pressure;
           Advantage: more controllable. A simplified version that treats the time pressure applied uniformly to every sentence.
           Disadvantage: not realistic.
        For the training method 2: use the real time left for a specific sentence, maybe range from 0s to 90s; then see whether the agent could learn to read adaptively, 
           i.e., the agent would read slower, have more regressions and less skips when time is sufficient; and when time is running out, the agent would read faster.
           Advantage: more realisitic.
           Challenge: when human are reading, the time pressure might occur when reading the very first sentence, because human have a global estimation of the time. 
           The pressure is not evenly applied to every sentence.
           NOTE: how to solve that -- use some mathematical equation to regulate this perception of time. TODO: go check with GPT.
        NOTE: apply the first method first, see results for quick results production.
        

        """
        # Load configuration
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
        self._mode = self._config["rl"]["mode"]

        if self._mode == "simulate":
            assert DATASET == "Ours", f"Invalid dataset: {DATASET}, should be 'Ours' when running the simulator!"
        elif self._mode == "train" or self._mode == "continual_train" or self._mode == "debug":
            assert DATASET == "ZuCo1.0", f"Invalid dataset: {DATASET}, should be 'ZuCo1.0' when training the model!"
        
        print(f"Sentence Reading Under Time Pressure Environment V0604 -- Deploying in {self._mode} mode with the dataset {DATASET}")

        # Initialize components
        self.time_condition_manager = TimeConditionManager()
        self.sentences_manager = SentencesManager(dataset=DATASET)     # TODO set different modes to sample sentences
        self.transition_function = TransitionFunction()
        self.reward_function = RewardFunction()

        # State tracking
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
        self.local_actual_fixation_sequence_in_sentence = None
        self.global_actual_fixation_sequence_in_text = None

        # Time tracking
        self._time_condition = None
        self._time_condition_value = None
        self.elapsed_time = None
        self._remaining_time = None
        self._sentence_wise_expected_reading_time_in_seconds = None

        # Environment parameters
        self._steps = None
        self.ep_len = 2 * Constants.MAX_SENTENCE_LENGTH
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
        self._num_stateful_obs = 6 + 2 + 2      # +2 for the regression cost and the weighted skipped word integration probability sigma, +2 for the time condition and remaining time
        self.observation_space = Box(low=0, high=1, shape=(self._num_stateful_obs,))
        self._noisy_obs_sigma = Constants.NOISY_OBS_SIGMA

        # Free parameters
        self._w_regression_cost = None          # NOTE: the most important parameter to tune for natural reading
        self._w_skipping_cost = None            # NOTE: maybe the most important parameter to tune for reading under time pressure
        self.MIN_W_SKIPPING_COST = 1.0
        self.MAX_W_SKIPPING_COST = 10.0
        self._w_comprehension_vs_reading_time = None    # NOTE: another parameter might be needed
        # self._noisy_skipped_word_integration_prob_sigma = None  # NOTE: the sigma for the noisy skipped word integration probability, the tunable parameter
        # self.MIN_NOISY_SKIPPED_WORD_INTEGRATION_PROB_SIGMA = 0.0
        # self.MAX_NOISY_SKIPPED_WORD_INTEGRATION_PROB_SIGMA = 0.05

        # Log variables
        self._log_individual_step_action = None
        self._log_elapsed_time_list_for_each_index = None
        self._log_remaining_time_list_for_each_index = None
        self._log_num_skips = None
        self._log_num_regressions = None
        
    def reset(self, seed=42, inputs: dict=None):          # TODO do the resets later, get inputs to forcefully assign a sentence to read
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
        self._sentence_info = self.sentences_manager.reset(inputs=inputs)
        self._sentence_len = len(self._sentence_info['words'])

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

        # Reset the time related variables
        self._time_condition, self._time_condition_value = self.time_condition_manager.reset(inputs=inputs)
        if self._time_condition == "30s":
            reading_speed = THIRTY_SECONDS_EXPECTED_READING_SPEED
        elif self._time_condition == "60s":
            reading_speed = SIXTY_SECONDS_EXPECTED_READING_SPEED
        elif self._time_condition == "90s":
            reading_speed = NINETY_SECONDS_EXPECTED_READING_SPEED
        else:
            raise ValueError(f"Invalid time condition: {self._time_condition}")
        self._sentence_wise_expected_reading_time_in_seconds = reading_speed * self._sentence_len
        self.elapsed_time = 0
        self._remaining_time = self._sentence_wise_expected_reading_time_in_seconds - self.elapsed_time

        # Initialize a random regression cost
        # self._w_regression_cost = random.uniform(0, 1)   # NOTE: uncomment when training!!!!
        self._w_regression_cost = 1.0    # NOTE: uncomment when testing!!!!

        self._w_comprehension_vs_reading_time = 1.0    # Start from a deterministic value

        # Initialize the skipping cost
        if self._mode == "train" or self._mode == "continual_train" or self._mode == "debug":
            self._w_skipping_cost = random.randint(self.MIN_W_SKIPPING_COST, self.MAX_W_SKIPPING_COST)
            # print(f"The sampled skipping cost is {self._w_skipping_cost} now -----------------------------------------")
        elif self._mode == "simulate":
            self._w_skipping_cost = 1.0
            print(f"NOTE: set the skipping cost to a fixed value when running the simulator! Now the skipping cost is {self._w_skipping_cost}")

        # # Initialize the noisy skipped word integration probability sigma
        # if self._mode == "train" or self._mode == "continual_train" or self._mode == "debug":
        #     self._noisy_skipped_word_integration_prob_sigma = random.randint(self.MIN_NOISY_SKIPPED_WORD_INTEGRATION_PROB_SIGMA * 100, self.MAX_NOISY_SKIPPED_WORD_INTEGRATION_PROB_SIGMA * 100) / 100
        #     # print(f"The sampled noisy skipped word integration probability sigma is {self._noisy_skipped_word_integration_prob_sigma} now")
        # elif self._mode == "simulate":
        #     self._noisy_skipped_word_integration_prob_sigma = 0.05
        #     print(f"NOTE: set the noisy skipped word integration probability sigma to a fixed valuewhen running the simulator!")

        # Initialize the log variables
        self._log_elapsed_time_list_for_each_index = []
        self._log_remaining_time_list_for_each_index = []
        self._log_num_skips = 0
        self._log_num_regressions = 0

        return self._get_obs(), {}
    
    def step(self, action):
        """Take action and update states"""
        self._steps += 1
        reward = 0

        self._log_individual_step_action = action

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

                # Update the time related variables
                self._update_time_related_variables()      

                self._log_num_regressions += 1

            reward = self.reward_function.compute_regress_reward(w_regression_cost=self._w_regression_cost)
        
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
                    # noisy_skipped_word_integration_prob = np.clip(perfect_skipped_word_integration_prob + np.random.normal(0, self._noisy_skipped_word_integration_prob_sigma), 0, 1)
                    self._word_beliefs[skipped_word_index] = perfect_skipped_word_integration_prob
                
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

            reward = self.reward_function.compute_skip_reward(w_skipping_cost=self._w_skipping_cost)
        
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

            reward = self.reward_function.compute_read_reward()

        elif action == self._STOP_ACTION:
            self._terminate = True
            # Compute final comprehension reward
            
            valid_words_beliefs = [b for b in self._word_beliefs if b != -1]

            reward = self.reward_function.compute_terminate_reward(
                sentence_len=self._sentence_len,
                num_words_read=len(valid_words_beliefs),
                words_beliefs=valid_words_beliefs,
                remaining_time=self._remaining_time,
                expected_sentence_reading_time=self._sentence_wise_expected_reading_time_in_seconds,
                w_comprehension_vs_reading_time=self._w_comprehension_vs_reading_time
            )
        
        # NOTE: different from the text-level reading, I do not cut down the reading time for the sentence. 
        #       Because I hold the assumption that sentences could always be finished.
        
        # Check termination
        if self._steps >= self.ep_len:
            self._terminate = True
            self._truncated = True

        if self._terminate: 
            info = self.get_episode_log()
        else:
            info = {}
        
        return self._get_obs(), reward, self._terminate, self._truncated, info
    
    @staticmethod
    def normalise(x, x_min, x_max, a, b):
    # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a
    
    def _get_obs(self):
        """Get observation with simplified scalar signals"""
        # Get current position (normalized)
        current_position = self.normalise(self.current_word_index, 0, self._sentence_len - 1, 0, 1)
        
        valid_words_beliefs = [b for b in self._word_beliefs if b != -1]

        # Get remaining words (normalized)
        remaining_words = self.normalise(self._sentence_len - len(valid_words_beliefs), 0, self._sentence_len, 0, 1)
        
        # Get the previous word's belief
        norm_previous_word_belief = np.clip(self._word_beliefs[self._previous_word_index], 0, 1) if self._previous_word_index is not None and 0 <= self._previous_word_index < self._sentence_len else 1
        norm_current_word_belief = np.clip(self._word_beliefs[self.current_word_index], 0, 1) if self.current_word_index is not None and 0 <= self.current_word_index < self._sentence_len else 1
        norm_next_word_predictability = np.clip(self._sentence_info['words_predictabilities_for_running_model'][self.current_word_index + 1], 0, 1) if self.current_word_index + 1 is not None and 0 <= self.current_word_index + 1 < self._sentence_len else 1
        

        # Apply noisy observation for a more robust model -- compensation for the limited data
        # NOTE: the noisy observation is applied to the normalized values. 
        # NOTE: if it is too small, no effect to model's stochasticity, if too large, the agent finds it hard to learn reasonable policy (no regression or skipping)
        observed_previous_word_belief = np.clip(norm_previous_word_belief + np.random.normal(0, self._noisy_obs_sigma), 0, 1)
        observed_current_word_belief = np.clip(norm_current_word_belief + np.random.normal(0, self._noisy_obs_sigma), 0, 1) 
        observed_next_word_predictability = np.clip(norm_next_word_predictability + np.random.normal(0, self._noisy_obs_sigma), 0, 1)
        # Get the on-going comprehension scalar
        # on_going_comprehension_scalar = np.clip(math.prod(valid_words_beliefs), 0, 1)
        on_going_comprehension_log_scalar = 0.0
        if len(valid_words_beliefs) > 0:
            overall_comprehension_log = 0.0
            for b in valid_words_beliefs:
                overall_comprehension_log += math.log(max(b, 1e-9))
            # geometric mean
            on_going_comprehension_log_scalar = math.exp(overall_comprehension_log / len(valid_words_beliefs))
        else:
            on_going_comprehension_log_scalar = 0.0
        
        on_going_comprehension_log_scalar = np.clip(on_going_comprehension_log_scalar, 0, 1)
        
        # Weights
        norm_w_regression_cost = self.normalise(self._w_regression_cost, 0, 1, 0, 1)
        norm_w_skipping_cost = self.normalise(self._w_skipping_cost, self.MIN_W_SKIPPING_COST, self.MAX_W_SKIPPING_COST, 0, 1)
        # norm_noisy_skipped_word_integration_prob_sigma = self.normalise(self._noisy_skipped_word_integration_prob_sigma, self.MIN_NOISY_SKIPPED_WORD_INTEGRATION_PROB_SIGMA, self.MAX_NOISY_SKIPPED_WORD_INTEGRATION_PROB_SIGMA, 0, 1)

        # Time related variables
        if self._time_condition == "30s":
            norm_time_condition = -1
        elif self._time_condition == "60s":
            norm_time_condition = 0
        elif self._time_condition == "90s":
            norm_time_condition = 1
        else:
            raise ValueError(f"Invalid time condition: {self._time_condition}")
        norm_remaining_time = self.normalise(self._remaining_time, 0, self._sentence_wise_expected_reading_time_in_seconds, 0, 1)

        stateful_obs = np.array([
            current_position,
            remaining_words,
            observed_previous_word_belief,
            observed_current_word_belief,
            observed_next_word_predictability,
            on_going_comprehension_log_scalar,
            norm_w_regression_cost,
            norm_w_skipping_cost,
            norm_time_condition,
            norm_remaining_time
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
        self.elapsed_time, self._remaining_time = self.transition_function.update_state_time(
            elapsed_time=self.elapsed_time,
            expected_sentence_reading_time=self._sentence_wise_expected_reading_time_in_seconds,
            word_reading_time=self._sentence_info['individual_word_reading_time']
        )
        
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
            "elapsed_time": self.elapsed_time,
            "remaining_time": self._remaining_time,
        }
        return individual_step_log
    
    def get_summarised_sentence_reading_logs(self) -> dict:
        """Get summarised sentence reading logs"""
        summarised_sentence_reading_logs = {
            "sentence_id": self._sentence_info['sentence_id'],
            "num_words_in_sentence": self._sentence_len, 
            "num_steps_or_fixations": self._steps,
            "num_regressions": self._log_num_regressions,
            "num_skips": self._log_num_skips,
            "num_stops": 1,
            "reading_sequence": self.reading_sequence.copy(),
            "local_actual_fixation_sequence_in_sentence": self.local_actual_fixation_sequence_in_sentence.copy(),
            "global_actual_fixation_sequence_in_text": self._get_global_actual_fixation_sequence_in_text(),
            "skipped_words_indexes": self._skipped_words_indexes.copy(),
            "regressed_words_indexes": self._regressed_words_indexes.copy(),
            "sentence_wise_expected_reading_time_in_seconds": self._sentence_wise_expected_reading_time_in_seconds,
            "w_regression_cost": self._w_regression_cost,
            "w_comprehension_vs_reading_time": self._w_comprehension_vs_reading_time,
            "elapsed_time": self.elapsed_time,
            "remaining_time": self._remaining_time,
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
            "remaining_time": self._remaining_time,
            "sentence_wise_expected_reading_time_in_seconds": self._sentence_wise_expected_reading_time_in_seconds,
            "w_regression_cost": self._w_regression_cost,
            "w_comprehension_vs_reading_time": self._w_comprehension_vs_reading_time,
            "num_steps": self._steps,
            "num_regressions": self._log_num_regressions,
            "num_skips": self._log_num_skips,
            "sentence_wise_regression_rate": self._log_num_regressions / self._sentence_len,
            "sentence_wise_skip_rate": self._log_num_skips / self._sentence_len,
            "sentence_wise_reading_speed": None if self.elapsed_time == 0 else self._sentence_len / self.elapsed_time * 60,
        }

        return episode_log


if __name__ == "__main__":
    pass