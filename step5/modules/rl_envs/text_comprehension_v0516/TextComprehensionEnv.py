import math
import os
import yaml
import random
import torch
import numpy as np

from gymnasium import Env
from gymnasium.spaces import Box, Discrete, Dict

from .TextManager import TextManager
from .TransitionFunction import TransitionFunction
from .RewardFunction import RewardFunction
from . import Constants
from .Utilities import calc_dynamic_text_comprehension_score


class TextComprehensionEnv(Env):
    def __init__(self):
        """
        Create on 16 May 2025.
        This is the environment for the RL-based high level -- text-level control agent: 
            it controls externally which sentence to read (proceed or regress to a previous one).

            Each episode reads one text (a fixed number of sentences)
        
        Cognitive constraints: (limited time and cognitive resources) 
            STM (so need to revisit sometimes, provide limited contextual predictability),
            attention resource (so need to skip the unnecessary words),
            LTM
            Time pressure (so need to finish reading before time runs out)
        
        Version: 1
            A simple version for sentence reading, where only regress to the previous sentence is the action. Observations are directly the 
        
        Future work:
        1. Graph-based gist
        2. Fluid schema
        3. Reading flow interruption costs (apply from the memory perspective)

        """
        # Load configuration
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
        self._mode = self._config["rl"]["mode"]
        self.num_episodes = self._config["rl"]["test"]["num_episodes"]
        
        print(f"Text Comprehension Environment V0516 -- Deploying in {self._mode} mode")

        # Initialize components
        self.text_manager = TextManager()
        self.transition_function = TransitionFunction()
        self.reward_function = RewardFunction()

        # Variables
        self._sampled_text = None   # TODO to be extended later
        self._num_sentences = None
        self._num_sentences_read = None
        self._num_remaining_sentence = None
        self._sentence_appraisal_scores_distribution = None
        self._regress_sentence_index = None
        # Internal states
        self._already_read_sentences_appraisal_scores_distribution = None
        self._previous_already_read_sentences_appraisal_scores_distribution = None
        # External states
        self._current_sentence_index = None     # Actually the reading progress, because the revisited sentence index is not trakced here
        self._actual_reading_sentence_index = None  # Tracking the revisited sentence index
        # State flags
        self._flag_is_regress = None
        self._flag_is_sentence_all_finished = None

        # Environment parameters
        self._steps = None
        # self.ep_len = 2 * Constants.MAX_NUM_SENTENCES           # Enough steps for the agent to stop reading actively
        self.ep_len = None
        self._terminate = None
        self._truncated = None
        self._step_wise_log = None

        # Action space
        # 0 to MAX_NUM_SENTENCES, corresponding to valid sentence indexes for revisiting; max_num_sentences + 1 is the read next sentence action, max_num_sentences + 2 is the stop action
        # self._REGRESS_PREVIOUS_SENTENCE_ACTION = 0
        # self._READ_NEXT_SENTENCE_ACTION = 1
        # self._STOP_ACTION = 2
        # self.action_space = Discrete(3)      
        self._regress_proceed_division = 0.5
        self._stop_division = 0.5
        # self.action_space = Box(low=0, high=1, shape=(3,))      # First action decides keeps reading or not; second acction decides where to regress to; third action decides whether to stop
        # NOTE change from the continuous action space to the discrete action space
        self.action_space = Dict({
            "action_type": Discrete(3),               # 0: read-next, 1: stop, 2: regress
            "regress_target": Discrete(Constants.MAX_NUM_SENTENCES) # Only used when action_type == 2
        })
        
        # Observation space - simplified to scalar signals
        self._num_stateful_obs = Constants.MAX_NUM_SENTENCES + 3 + 1     # Distribution of the appraisal scores over the sentences, current sentence index, and the time awareness 
        self.observation_space = Box(low=-1, high=1, shape=(self._num_stateful_obs,))

        #########################################################
        self.episode_id = 0        # Initialize here because need to accumulate across episodes
        
    # def reset(self, seed=42):
    def reset(self, *, seed=None, options=None):
        """Reset environment and initialize states"""
        super().reset(seed=seed)

        self.episode_id += 1

        self._steps = 0
        self._terminate = False
        self._truncated = False

        # Get new sentence
        self._sampled_text_metadata = self.text_manager.reset()
        text_id = self._sampled_text_metadata["text_id"]
        self._num_sentences = self._sampled_text_metadata["num_sentences"]
        self._num_remaining_sentence = self._num_sentences
        self._sentence_appraisal_scores_distribution = self._sampled_text_metadata["sentence_appraisal_scores_distribution"]
        self._already_read_sentences_appraisal_scores_distribution = [-1] * self._num_sentences
        self._previous_already_read_sentences_appraisal_scores_distribution = self._already_read_sentences_appraisal_scores_distribution.copy()
        self._current_sentence_index = -1
        self._num_sentences_read = 0
        self._regress_sentence_index = -1    # -1 means no regress# NOTE: if the agent does not learn, include this into the observation space

        # Reset state flags
        self._flag_is_regress = False
        self._flag_is_sentence_all_finished = False
        self._sentence_appraisal_before_regress = None

        # Initialize the episode length
        self.ep_len = self._num_sentences + 3     # NOTE: trial of limited episode length

        # Step-wise log
        self._step_wise_log = []
        
        return self._get_obs(), {}
    
    def step(self, action):
        """Take action and update states"""

        # Apply actions
        action_type = action["action_type"]
        regress_target = action["regress_target"]
        
        # Update variables
        self._steps += 1
        reward = 0

        # read_or_regress_action = action[0]
        # raw_regress_sentence_value = action[1]
        # continue_or_stop_action = action[2]

        # Store the previous state
        self._previous_already_read_sentences_appraisal_scores_distribution = self._already_read_sentences_appraisal_scores_distribution.copy()

        # Update the state flags
        self._flag_is_regress = False
        
        ############################################### Execute the action in the new state ###############################################
        if action_type == 0:  # READ NEXT SENTENCE
            new_scores, action_validity = self.transition_function.update_state_read_next_sentence(
                current_sentence_index=self._current_sentence_index,
                sentence_appraisal_scores_distribution=self._sentence_appraisal_scores_distribution,
                num_sentences=self._num_sentences
            )

            if action_validity:
                self._current_sentence_index += 1
                self._actual_reading_sentence_index = self._current_sentence_index
                self._num_sentences_read += 1
                self._num_remaining_sentence -= 1
                # Apply memory decay
                self._already_read_sentences_appraisal_scores_distribution = self.transition_function.apply_time_independent_memory_decay(
                    self._already_read_sentences_appraisal_scores_distribution,
                    self._current_sentence_index,
                    apply=False
                )
                self._already_read_sentences_appraisal_scores_distribution[self._current_sentence_index] = new_scores[self._current_sentence_index]
            else:
                self._already_read_sentences_appraisal_scores_distribution = self.transition_function.apply_time_independent_memory_decay(
                    self._already_read_sentences_appraisal_scores_distribution,
                    -1,
                    apply=False
                )
            # Get the reward
            reward = self.reward_function.compute_read_next_sentence_reward()
        
        elif action_type == 1:  # STOP READING
            self._terminate = True
            self._truncated = False
        
        elif action_type == 2:  # REGRESS TO A SENTENCE
            revised_sentence_index = min(regress_target, self._current_sentence_index)  # prevent forward regress
            self._actual_reading_sentence_index = revised_sentence_index

            self._already_read_sentences_appraisal_scores_distribution = self.transition_function.apply_time_independent_memory_decay(
                self._already_read_sentences_appraisal_scores_distribution,
                revised_sentence_index,
                apply=False
            )

            self._already_read_sentences_appraisal_scores_distribution, action_validity = self.transition_function.update_state_regress_to_sentence(
                revised_sentence_index=revised_sentence_index,
                furtherest_read_sentence_index=self._current_sentence_index,
                read_sentence_appraisal_scores_distribution=self._already_read_sentences_appraisal_scores_distribution
            )

            self._current_sentence_index = self._current_sentence_index     # Just a placeholder here -- automatically jumps back to the latest sentence that read. NOTE: make this complex later    

            self._flag_is_regress = True
            reward = self.reward_function.compute_regress_to_sentence_reward()

        else:
            raise ValueError(f"Invalid action_type: {action_type}")
            
              
        # # Execute the action in the new state
        # if continue_or_stop_action <= self._stop_division:
        #     if read_or_regress_action > self._regress_proceed_division:
        #         # Then update state with new sentence
        #         new_scores, action_validity = self.transition_function.update_state_read_next_sentence(
        #             current_sentence_index=self._current_sentence_index,
        #             sentence_appraisal_scores_distribution=self._sentence_appraisal_scores_distribution,
        #             num_sentences=self._num_sentences
        #         )
                
        #         if action_validity:
        #             # Only update the new sentence's score, preserve decayed scores for others
        #             self._current_sentence_index = self._current_sentence_index + 1
        #             self._actual_reading_sentence_index = self._current_sentence_index
        #             self._num_sentences_read += 1
        #             self._num_remaining_sentence -= 1
        #             # Apply memory decay first
        #             self._already_read_sentences_appraisal_scores_distribution = self.transition_function.apply_time_independent_memory_decay(
        #                 self._already_read_sentences_appraisal_scores_distribution, 
        #                 self._current_sentence_index,
        #                 apply=False
        #             )

        #             self._already_read_sentences_appraisal_scores_distribution[self._current_sentence_index] = new_scores[self._current_sentence_index]
        #         else: 
        #             self._already_read_sentences_appraisal_scores_distribution = self.transition_function.apply_time_independent_memory_decay(
        #                 self._already_read_sentences_appraisal_scores_distribution, 
        #                 -1,
        #                 apply=False
        #             )
        #         # Get the reward    
        #         reward = self.reward_function.compute_read_next_sentence_reward()
        #     else:
        #         # Regress to a previously read sentence
        #         revised_sentence_index = self._get_regress_sentence_index(raw_regress_sentence_value)
        #         self._actual_reading_sentence_index = revised_sentence_index
                
        #         # Apply memory decay first
        #         self._already_read_sentences_appraisal_scores_distribution = self.transition_function.apply_time_independent_memory_decay(
        #             self._already_read_sentences_appraisal_scores_distribution, 
        #             revised_sentence_index,
        #             apply=False
        #         )
                
        #         # Then update the regressed sentence's score
        #         self._already_read_sentences_appraisal_scores_distribution, action_validity = self.transition_function.update_state_regress_to_sentence(
        #             revised_sentence_index=revised_sentence_index,
        #             furtherest_read_sentence_index=self._current_sentence_index,
        #             read_sentence_appraisal_scores_distribution=self._already_read_sentences_appraisal_scores_distribution
        #         )
                
        #         self._current_sentence_index = self._current_sentence_index     # Just a placeholder here -- automatically jumps back to the latest sentence that read. NOTE: make this complex later    

        #         # Update state flags    
        #         self._flag_is_regress = True
                
        #         # Get the reward
        #         reward = self.reward_function.compute_regress_to_sentence_reward()
        # else:
        #     # Stop reading
        #     self._terminate = True
        #     self._truncated = False
        
        # Check if the sentence is all finished
        if self._num_remaining_sentence == 0:
            self._flag_is_sentence_all_finished = True

        # Check termination
        if self._steps >= self.ep_len:
            self._terminate = True
            self._truncated = True

        if self._terminate: 
            reward = self.reward_function.compute_terminate_reward(self._num_sentences, self._num_sentences_read, self._already_read_sentences_appraisal_scores_distribution)
            info = self.get_episode_log()
        else:
            info = {}
        
        return self._get_obs(), reward, self._terminate, self._truncated, info
    
    @staticmethod
    def normalise(x, x_min, x_max, a, b):
    # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a
    
    def _get_regress_sentence_index(self, raw_regress_sentence_value):
        regress_sentence_index = int(self.normalise(raw_regress_sentence_value, 0, 1, 0, self._num_sentences - 1))
        return regress_sentence_index
    
    def _get_obs(self):
        """Get observation with simplified scalar signals"""

        # Remaining episode length awareness
        remaining_episode_length_awareness = self.normalise(self.ep_len - self._steps, 0, self.ep_len, 0, 1)

        # Normalised number of remaining sentences
        # norm_num_remaining_sentence = self.normalise(self._num_remaining_sentence, 0, self._num_sentences, 0, 1)
        norm_remaining_sentence = 1 if self._num_remaining_sentence > 0 else 0      # A noisier but more realistic signal denoting the reading progress

        # Get current sentence position (normalized)    NOTE: maybe add the revised sentence index ot the agent as an observation, not always the current sentence index
        norm_current_position = self.normalise(self._current_sentence_index, 0, Constants.MAX_NUM_SENTENCES - 1, 0, 1)
        
        # Get the valid sentences appraisal scores
        valid_sentences_appraisals = [a for a in self._already_read_sentences_appraisal_scores_distribution if a != -1]
        # Pad valid_sentences_appraisals to have fixed length
        padded_appraisals = [-1] * Constants.MAX_NUM_SENTENCES
        padded_appraisals[:len(valid_sentences_appraisals)] = valid_sentences_appraisals
        
        # Get the on-going comprehension scalar
        on_going_comprehension_log_scalar = 0.0
        if len(valid_sentences_appraisals) > 0:
            # overall_comprehension_log = 0.0
            # for b in valid_sentences_appraisals:
            #     overall_comprehension_log += math.log(max(b, 1e-9))
            # # geometric mean
            # on_going_comprehension_log_scalar = math.exp(overall_comprehension_log / len(valid_sentences_appraisals))
            on_going_comprehension_log_scalar = max(0, calc_dynamic_text_comprehension_score(valid_sentences_appraisals, mode=Constants.COMPREHENSION_SCORE_MODE, tau=Constants.TAU))
        else:
            on_going_comprehension_log_scalar = 0.0
        
        on_going_comprehension_log_scalar = np.clip(on_going_comprehension_log_scalar, 0, 1)

        stateful_obs = np.concatenate([padded_appraisals, [norm_current_position], [remaining_episode_length_awareness], [norm_remaining_sentence], [on_going_comprehension_log_scalar]])

        assert stateful_obs.shape[0] == self._num_stateful_obs, f"expected {self._num_stateful_obs} but got {stateful_obs.shape[0]}"
        
        ################## Update step-wise log here because some values are computed here ##################
        self._step_wise_log.append({
            "step": self._steps,
            "current_sentence_index": self._current_sentence_index,
            "actual_reading_sentence_index": self._actual_reading_sentence_index,
            "sentence_appraisal_before_regress": self._previous_already_read_sentences_appraisal_scores_distribution[self._actual_reading_sentence_index] if self._flag_is_regress else None,
            "is_regress": self._flag_is_regress,
            "is_sentence_all_finished": self._flag_is_sentence_all_finished,
            "remaining_episode_length_awareness": remaining_episode_length_awareness,
            "already_read_sentences_appraisal_scores_distribution": self._already_read_sentences_appraisal_scores_distribution.copy(),
            "on_going_comprehension_log_scalar": on_going_comprehension_log_scalar,
            "terminate": self._terminate,
        })

        return stateful_obs
        
    def get_episode_log(self) -> dict:
        """Get logs for the episode"""
        episode_log = {
            "episode_id": self.episode_id,
            "total_episodes": self.num_episodes,
            "num_sentences": self._num_sentences,
            "init_sentence_appraisal_scores_distribution": self._sentence_appraisal_scores_distribution,
            "step_wise_log": self._step_wise_log,
        }

        return episode_log


if __name__ == "__main__":
    pass