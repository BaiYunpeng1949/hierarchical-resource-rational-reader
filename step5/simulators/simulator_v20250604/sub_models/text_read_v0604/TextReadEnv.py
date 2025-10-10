import math
import os
import yaml
import random
import torch
import numpy as np
import logging

from gymnasium import Env
from gymnasium.spaces import Box, Discrete, Dict

from .TextManager import TextManager
from .TimeConditionManager import TimeConditionManager
from .TransitionFunction import TransitionFunction
from .RewardFunction import RewardFunction
from . import Constants
from .Utilities import calc_dynamic_text_comprehension_score


logger = logging.getLogger(__name__)
if not logger.handlers:               # avoid duplicate handlers when workers fork
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()  # or FileHandler(...)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
    )
    logger.addHandler(handler)


# DATA_SOURCE = "real_stimuli"
DATA_SOURCE = "generated_stimuli"    # NOTE: please set this when training the model


class TextReadingUnderTimePressureEnv(Env):
    def __init__(self):
        """
        Create on 04 June 2025.
            Updated on 09 July 2025.
        
        This is the environment for the RL-based high level -- text-level control agent: 
            it controls externally which sentence to read (proceed or regress to a previous one).

            Each episode reads one text (a fixed number of sentences)
        
        Cognitive constraints: (limited time and cognitive resources) 
            STM (so need to revisit sometimes, provide limited contextual predictability),
            attention resource (so need to skip the unnecessary words),
            LTM
            Time pressure (so need to finish reading before time runs out)
        
        Version: 1
            A simple version for sentence reading but under time pressure.
        
        Future work:
        1. Graph-based gist
        2. Fluid schema
        3. Reading flow interruption costs (apply from the memory perspective)

        Training Objective:
            4th June 2025:
                - More sentence regressions when time is sufficient
        """
        # Load configuration
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
        self._mode = self._config["rl"]["mode"]
        self.num_episodes = self._config["rl"]["test"]["num_episodes"]

        # Make sure the DATA_SOURCE is set correctly when training the model
        if self._mode == "train" or self._mode == "continual_train" or self._mode == "debug":
            assert DATA_SOURCE == "generated_stimuli", f"Invalid DATA_SOURCE: {DATA_SOURCE}, should be 'generated_stimuli' when training the model!"
        elif self._mode == "simulate":
            assert DATA_SOURCE == "real_stimuli", f"Invalid DATA_SOURCE: {DATA_SOURCE}, should be 'real_stimuli' when testing the model!"
        
        print(f"Text Reading (Under Time Pressure) Environment V0604 -- Deploying in {self._mode} mode")

        # Initialize components
        self.text_manager = TextManager(data_source=DATA_SOURCE)
        self.time_condition_manager = TimeConditionManager()
        self.transition_function = TransitionFunction()
        self.reward_function = RewardFunction()

        # Variables
        self._sampled_text = None  
        self._num_sentences = None
        self._num_sentences_read = None
        self._num_remaining_sentence = None
        self._sentence_appraisal_scores_distribution = None
        self._regress_sentence_index = None
        # Internal states
        self._already_read_sentences_appraisal_scores_distribution = None
        # External states
        self.current_sentence_index = None     # Actually the reading progress, because the revisited sentence index is not trakced here
        self.actual_reading_sentence_index = None  # Tracking the revisited sentence index

        # Time conditions
        self._time_condition = None     # Select from 30s, 60s, 90s
        self._time_condition_value = None     # Value of the time condition, in seconds
        self._elapsed_time = None     # Timer for ticking the time, in seconds
        self._remaining_time = None     # Remaining time, in seconds

        # Environment parameters
        self._steps = None
        self.ep_len = 3 * Constants.MAX_NUM_SENTENCES           # Enough steps for the agent to stop reading actively
        self._terminate = None
        self._truncated = None

        self._individual_step_log = None
        self._step_wise_logs = None

        # Action space
        self.action_space = Dict({
            "action_type": Discrete(3),               # 0: read-next, 1: stop, 2: regress
            "regress_target": Discrete(Constants.MAX_NUM_SENTENCES) # Only used when action_type == 2
        })


        # Observation space - simplified to scalar signals
        self._num_stateful_obs = Constants.MAX_NUM_SENTENCES + 3 + 2 + 1 + 1     
        # Distribution of the appraisal scores over the sentences, current sentence index, time awarenesss and remaining time, and the time awareness, and the free parameters (coverage factor)
        self.observation_space = Box(low=-1, high=1, shape=(self._num_stateful_obs,))

        #########################################################   NOTE: log related variables
        self.episode_id = 0        # Initialize here because need to accumulate across episodes
        self._log_number_regressions = None
        self._log_episodic_regression_rate = None
        self._log_actions = None

        ###################  Get data from the simulator or not  #######################
        self._get_data_from_other_agents = None

        ###################  Tunable parameter  #######################
        self._free_param_coverage_factor = None       # A value ranges from 0 to 1
        self.MAX_COVERAGE_FACTOR = 3
        self.MIN_COVERAGE_FACTOR = 0

    def reset(self, seed=None, options=None, inputs: dict=None, params: dict=None):                
        """Reset environment and initialize states"""
        super().reset(seed=seed)

        self.episode_id += 1

        self._steps = 0
        self._terminate = False
        self._truncated = False

        # Get new sentence
        self._sampled_text_metadata = self.text_manager.reset(inputs=inputs)
        stimulus_id = self._sampled_text_metadata["stimulus_id"]

        stimulus_source = self._sampled_text_metadata["stimulus_source"]
        self._num_sentences = self._sampled_text_metadata["num_sentences"]
        self._num_remaining_sentence = self._num_sentences
        self._sentence_appraisal_scores_distribution = self._sampled_text_metadata["sentence_appraisal_scores_distribution"]    # NOTE: these comprehensions could somehow be integrated with the sentence-level comprehension scores; 
        self._already_read_sentences_appraisal_scores_distribution = [-1] * self._num_sentences
        self.current_sentence_index = -1
        self._num_sentences_read = 0
        self._regress_sentence_index = -1    # -1 means no regress# NOTE: if the agent does not learn, include this into the observation space
        self.actual_reading_sentence_index = self.current_sentence_index         
        
        self._individual_step_log = {}
        self._step_wise_logs = []

        # Get the time condition
        self._time_condition, self._time_condition_value = self.time_condition_manager.reset(inputs=inputs)
        self._elapsed_time = 0
        self._remaining_time = self._time_condition_value - self._elapsed_time        

        # Get the running mode
        self._get_data_from_other_agents = inputs["get_data_from_agents"] if inputs is not None else False
        assert self._get_data_from_other_agents in [True, False], f"Invalid configuration on data source: {self._get_data_from_other_agents}, should be True or False"

        # Initialize log variables
        self._log_actions = {}
        self._log_number_regressions = 0
        self._log_episodic_regression_rate_over_num_read_sentences = 0
        self._log_episodic_regression_rate_over_steps = 0

        # Initialize the tunable parameters
        # Randomly sample a value from the range [1, 2], but discrete with a step of 0.1
        if self._mode == "train" or self._mode == "continual_train" or self._mode == "debug":
            self._free_param_coverage_factor = random.randint(self.MIN_COVERAGE_FACTOR * 10, self.MAX_COVERAGE_FACTOR * 10) / 10
            # print(f"sampling coverage factor: {self._free_param_coverage_factor}")
        else:
            if params is None:
                self._free_param_coverage_factor = 3.0
                print(f"    Text Read: fixed _free_param_coverage_factor as {self._free_param_coverage_factor}")
            else:
                self._free_param_coverage_factor = params['coverage_factor']
        # NOTE: now range from [1, 2], all over-weighting / encouraging the agent to read more sentences, 
        #   maybe need to range from [0, 2] later, also consider different coverage factor for different texts
        
        return self._get_obs(), {}
    
    def step(self, action, time_info: dict = None):
        """Take action and update states"""
        
        # Apply actions
        action_type = action["action_type"]
        regress_target = action["regress_target"]
        
        self._steps += 1
        reward = 0

        ####################################################### Execute actions #######################################################
        if action_type == 0:   # READ NEXT SENTENCE
            # Then update state with new sentence
            new_scores, action_validity = self.transition_function.update_state_read_next_sentence(
                current_sentence_index=self.current_sentence_index,
                sentence_appraisal_scores_distribution=self._sentence_appraisal_scores_distribution,
                num_sentences=self._num_sentences
            )
            
            if action_validity:
                # Only update the new sentence's score, preserve decayed scores for others
                self.current_sentence_index = self.current_sentence_index + 1
                self.actual_reading_sentence_index = self.current_sentence_index
                
                # TODO debug delete later
                if self.actual_reading_sentence_index >= self._num_sentences:
                    logger.warning(f"actual_reading_sentence_index={self.actual_reading_sentence_index} is out of range (len={self._num_sentences}) -- issue is caught here, in READING NEXT SENTENCE")
                    # self.actual_reading_sentence_index = self._num_sentences - 1

                self._num_sentences_read += 1
                self._num_remaining_sentence -= 1
                # Apply memory decay first
                self._already_read_sentences_appraisal_scores_distribution = self.transition_function.apply_time_independent_memory_decay(
                    self._already_read_sentences_appraisal_scores_distribution, 
                    self.current_sentence_index,
                    apply=False
                )
                # Update the sentences' appraisal scores
                self._already_read_sentences_appraisal_scores_distribution[self.current_sentence_index] = new_scores[self.current_sentence_index]
                # Update the elapsed time
                if self._get_data_from_other_agents:
                    self._elapsed_time = time_info["elapsed_time"]
                    self._remaining_time = time_info["remaining_time"]
                else:   
                    self._elapsed_time, self._remaining_time = self.transition_function.update_state_time(
                        elapsed_time=self._elapsed_time,
                        sentence_reading_time=self._sampled_text_metadata["sentence_reading_times"][self.actual_reading_sentence_index],
                        time_condition_value=self._time_condition_value
                    )
            else: 
                self._already_read_sentences_appraisal_scores_distribution = self.transition_function.apply_time_independent_memory_decay(
                    self._already_read_sentences_appraisal_scores_distribution, 
                    -1,
                    apply=False
                )
            # Get the reward    
            reward = self.reward_function.compute_read_next_sentence_reward()
        
        elif action_type == 1:   # STOP READING
            self._terminate = True
            self._truncated = False
        
        elif action_type == 2:   # REGRESS TO A PREVIOUS SENTENCE
            # Regress to a previously read sentence
            revised_sentence_index = min(regress_target, self.current_sentence_index)  # prevent forward regress -- but cause a severe issue: it will set the last sentence's appraisal score to 1.0
            
            # Do a validity check 
            if revised_sentence_index != -1:        # Valid regression action

                self.actual_reading_sentence_index = revised_sentence_index
            
                # Apply memory decay first
                self._already_read_sentences_appraisal_scores_distribution = self.transition_function.apply_time_independent_memory_decay(
                    self._already_read_sentences_appraisal_scores_distribution, 
                    revised_sentence_index,
                    apply=False
                )
            
                # Then update the regressed sentence's score
                self._already_read_sentences_appraisal_scores_distribution, action_validity = self.transition_function.update_state_regress_to_sentence(
                    revised_sentence_index=revised_sentence_index,
                    furtherest_read_sentence_index=self.current_sentence_index,
                    read_sentence_appraisal_scores_distribution=self._already_read_sentences_appraisal_scores_distribution
                )

                # Update the elapsed time
                if self._get_data_from_other_agents:
                    self._elapsed_time = time_info["elapsed_time"]     
                    self._remaining_time = time_info["remaining_time"]
                else:   
                    self._elapsed_time, self._remaining_time = self.transition_function.update_state_time(
                        elapsed_time=self._elapsed_time,
                        sentence_reading_time=self._sampled_text_metadata["sentence_reading_times"][self.actual_reading_sentence_index],
                        time_condition_value=self._time_condition_value
                    )
                
                self.current_sentence_index = self.current_sentence_index     # Just a placeholder here -- automatically jumps back to the latest sentence that read. NOTE: make this complex later    
                
                # Update the log variables
                self._log_number_regressions += 1
            
            else:    # Invalid regression action, do nothing
                pass

            # Get the reward
            reward = self.reward_function.compute_regress_to_sentence_reward()

        # Check termination by the episode length
        if self._steps >= self.ep_len:
            self._terminate = True
            self._truncated = True
        
        # Check termination by the time condition
        if self._remaining_time <= 0:
            self._terminate = True
            self._truncated = True
        
        # Log the actions
        self._log_actions = {
            # "read_or_regress_action_raw_value": read_or_regress_action,
            # "raw_regress_sentence_value_raw_value": raw_regress_sentence_value,
            # "continue_or_stop_action_raw_value": continue_or_stop_action,
            "read_or_regress_action": "read" if action_type == 0 else "regress" if action_type == 2 else "stop",
            "raw_regress_sentence_value": regress_target,     
            "continue_or_stop_action": "continue" if action_type != 1 else "stop",
        }

        if self._terminate: 
            reward = self.reward_function.compute_terminate_reward(self._num_sentences, self._num_sentences_read, self._already_read_sentences_appraisal_scores_distribution, self._free_param_coverage_factor)
            info = self.get_episode_log()
        else:
            info = {}
        
        return self._get_obs(), reward, self._terminate, self._truncated, info
    
    @staticmethod
    def normalise(x, x_min, x_max, a, b):
        """Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]"""
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a
    
    def _get_obs(self):
        """Get observation with simplified scalar signals"""

        # Remaining episode length awareness
        remaining_episode_length_awareness = self.normalise(self.ep_len - self._steps, 0, self.ep_len, 0, 1)

        # Normalised number of remaining sentences
        # norm_num_remaining_sentence = self.normalise(self._num_remaining_sentence, 0, self._num_sentences, 0, 1)
        norm_remaining_sentence = 1 if self._num_remaining_sentence > 0 else 0      # A noisier but more realistic signal denoting the reading progress

        # Get current sentence position (normalized)    NOTE: maybe add the revised sentence index ot the agent as an observation, not always the current sentence index
        norm_current_position = self.normalise(self.current_sentence_index, 0, Constants.MAX_NUM_SENTENCES - 1, 0, 1)
        
        # Get the valid sentences appraisal scores
        valid_sentences_appraisals = [a for a in self._already_read_sentences_appraisal_scores_distribution if a != -1]
        # Pad valid_sentences_appraisals to have fixed length
        padded_appraisals = [-1] * Constants.MAX_NUM_SENTENCES
        padded_appraisals[:len(valid_sentences_appraisals)] = valid_sentences_appraisals
        
        # Get the on-going comprehension scalar
        on_going_comprehension_log_scalar = 0.0
        if len(valid_sentences_appraisals) > 0:
            on_going_comprehension_log_scalar = max(0, calc_dynamic_text_comprehension_score(valid_sentences_appraisals, mode=Constants.COMPREHENSION_SCORE_MODE, tau=Constants.TAU))
        else:
            on_going_comprehension_log_scalar = 0.0
        
        on_going_comprehension_log_scalar = np.clip(on_going_comprehension_log_scalar, 0, 1)

        # Get the time condition awareness
        if self._time_condition == "30s":
            time_condition_awareness = -1
        elif self._time_condition == "60s":
            time_condition_awareness = 0
        elif self._time_condition == "90s":
            time_condition_awareness = 1
        else:
            raise ValueError(f"Invalid time condition: {self._time_condition}")

        # Get the coverage factor
        normalized_coverage_factor = self.normalise(self._free_param_coverage_factor, self.MIN_COVERAGE_FACTOR, self.MAX_COVERAGE_FACTOR, 0, 1)
        
        # Get the remaining time awareness
        norm_remaining_time = self.normalise(self._remaining_time, 0, self._time_condition_value, 0, 1)

        # Concatenate the observations
        stateful_obs = np.concatenate([padded_appraisals, [norm_current_position], [remaining_episode_length_awareness], [norm_remaining_sentence], 
                                      [on_going_comprehension_log_scalar], [time_condition_awareness], [norm_remaining_time], [normalized_coverage_factor]])

        assert stateful_obs.shape[0] == self._num_stateful_obs, f"expected {self._num_stateful_obs} but got {stateful_obs.shape[0]}"

        try:
            if self.actual_reading_sentence_index is None:
                num_words_in_sentence = None
            else:
                sl = self._sampled_text_metadata["sentence_lengths"]
                num_words_in_sentence = sl[self.actual_reading_sentence_index]
        except IndexError:
            logger.warning(
                "actual_reading_sentence_index=%s is out of range (len=%s) -- this is actually the text length, i.e., the number of sentences in the text",
                self.actual_reading_sentence_index,
                self._num_sentences,    # This is actually the text length
            )
            num_words_in_sentence = None

        ################## Update step-wise log here because some values are computed here ################## 
        self._individual_step_log = {
            "step": self._steps,
            "action_information": self._log_actions,
            "current_sentence_index": self.current_sentence_index,
            "num_words_in_sentence": num_words_in_sentence,
            "actual_reading_sentence_index": self.actual_reading_sentence_index,
            # "remaining_episode_length_awareness": remaining_episode_length_awareness,
            "already_read_sentences_appraisal_scores_distribution": self._already_read_sentences_appraisal_scores_distribution.copy(),
            "on_going_comprehension_log_scalar": on_going_comprehension_log_scalar,
            "remaining_time": self._remaining_time,
            "terminate": self._terminate,
            "sentence_reading_summary": {},
            "sentence_reading_logs": [],
        }

        self._step_wise_logs.append(self._individual_step_log)

        return stateful_obs
    
    ############################### Helper functions ###############################
    def get_individual_step_log(self) -> dict:
        """Get individual step log"""
        return self._individual_step_log.copy()
        
    def get_episode_log(self) -> dict:
        """Get logs for the episode"""

        # Update the log variables
        self._log_episodic_regression_rate_over_num_read_sentences = self._log_number_regressions / self._num_sentences_read if self._num_sentences_read > 0 else 0
        self._log_episodic_regression_rate_over_steps = self._log_number_regressions / self._steps if self._steps > 0 else 0

        episode_log = {
            "episode_id": self.episode_id,
            "total_episodes": self.num_episodes,
            "time_condition": self._time_condition,
            "time_condition_value": self._time_condition_value,
            "num_sentences": self._num_sentences,
            "init_sentence_appraisal_scores_distribution": self._sentence_appraisal_scores_distribution,
            "sentence_lengths": self._sampled_text_metadata["sentence_lengths"],
            "sentence_reading_times": self._sampled_text_metadata["sentence_reading_times"],
            "log_number_regressions": self._log_number_regressions,
            "log_episodic_regression_rate_over_num_read_sentences": self._log_episodic_regression_rate_over_num_read_sentences,
            "log_episodic_regression_rate_over_steps": self._log_episodic_regression_rate_over_steps,

            "step_wise_log": self._step_wise_logs,
        }

        return episode_log


if __name__ == "__main__":
    pass