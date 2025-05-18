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
        # Internal states
        self._already_read_sentences_appraisal_scores_distribution = None
        # External states
        self._current_sentence_index = None

        # Environment parameters
        self._steps = None
        self.ep_len = 1.5 * Constants.MAX_NUM_SENTENCES
        self._terminate = None
        self._truncated = None

        # Action space
        # 0 to MAX_NUM_SENTENCES, corresponding to valid sentence indexes for revisiting; max_num_sentences + 1 is the read next sentence action, max_num_sentences + 2 is the stop action
        self._REGRESS_PREVIOUS_SENTENCE_ACTION = 0
        self._READ_NEXT_SENTENCE_ACTION = 1
        self._STOP_ACTION = 2
        self.action_space = Discrete(3)      
        
        # Observation space - simplified to scalar signals
        self._num_stateful_obs = Constants.MAX_NUM_SENTENCES + 3 + 1     # Distribution of the appraisal scores over the sentences, current sentence index, and the time awareness 
        self.observation_space = Box(low=-1, high=1, shape=(self._num_stateful_obs,))

        
    def reset(self, seed=42):
        """Reset environment and initialize states"""
        super().reset(seed=seed)

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
        self._current_sentence_index = -1
        self._num_sentences_read = 0

        # # TODO debug delete later
        # print(f"Text ID sampled: {text_id}")
        
        return self._get_obs(), {}
    
    def step(self, action):
        """Take action and update states"""
        self._steps += 1
        reward = 0

        # # TODO debug delete later
        # print(f"Agent's action is: {action}")

        # Read the next sentence
        if action == self._READ_NEXT_SENTENCE_ACTION:
            self._already_read_sentences_appraisal_scores_distribution, action_validity = self.transition_function.update_state_read_next_sentence(
                current_sentence_index=self._current_sentence_index,
                sentence_appraisal_scores_distribution=self._sentence_appraisal_scores_distribution,
                num_sentences=self._num_sentences
            )
            if action_validity:
                self._current_sentence_index = self._current_sentence_index + 1
                self._num_sentences_read += 1
                self._num_remaining_sentence -= 1
            reward = self.reward_function.compute_read_next_sentence_reward()
        
        # Regress to a previously read sentence
        if action == self._REGRESS_PREVIOUS_SENTENCE_ACTION:
            revised_sentence_index = self.transition_function.optimize_select_sentence_to_regress_to(
                current_sentence_index=self._current_sentence_index,
                read_sentence_appraisal_scores_distribution=self._already_read_sentences_appraisal_scores_distribution
            )
            self._already_read_sentences_appraisal_scores_distribution, action_validity = self.transition_function.update_state_regress_to_sentence(
                revised_sentence_index=revised_sentence_index,
                furtherest_read_sentence_index=self._current_sentence_index,
                read_sentence_appraisal_scores_distribution=self._already_read_sentences_appraisal_scores_distribution
            )
            self._current_sentence_index = self._current_sentence_index     # Just a placeholder here       
            reward = self.reward_function.compute_regress_to_sentence_reward()

        # Stop reading
        if action == self._STOP_ACTION:
            self._terminate = True
            self._truncated = False
            reward = self.reward_function.compute_terminate_reward(self._num_sentences, self._num_sentences_read, self._sentence_appraisal_scores_distribution)

        # Check termination
        if self._steps >= self.ep_len:
            self._terminate = True
            self._truncated = True

        if self._terminate: 
            reward = self.reward_function.compute_terminate_reward(self._num_sentences, self._num_sentences_read, self._sentence_appraisal_scores_distribution)
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

        # Remaining episode length awareness
        remaining_episode_length_awareness = self.normalise(self.ep_len - self._steps, 0, self.ep_len, 0, 1)

        # Normalised number of remaining sentences
        norm_num_remaining_sentence = self.normalise(self._num_remaining_sentence, 0, self._num_sentences, 0, 1)

        # Get current sentence position (normalized)
        norm_current_position = self.normalise(self._current_sentence_index, 0, Constants.MAX_NUM_SENTENCES - 1, 0, 1)
        
        # Get the valid sentences appraisal scores
        valid_sentences_appraisals = [a for a in self._already_read_sentences_appraisal_scores_distribution if a != -1]
        # Pad valid_sentences_appraisals to have fixed length
        padded_appraisals = [-1] * Constants.MAX_NUM_SENTENCES
        padded_appraisals[:len(valid_sentences_appraisals)] = valid_sentences_appraisals
        
        # Get the on-going comprehension scalar
        on_going_comprehension_log_scalar = 0.0
        if len(valid_sentences_appraisals) > 0:
            overall_comprehension_log = 0.0
            for b in valid_sentences_appraisals:
                overall_comprehension_log += math.log(max(b, 1e-9))
            # geometric mean
            on_going_comprehension_log_scalar = math.exp(overall_comprehension_log / len(valid_sentences_appraisals))
        else:
            on_going_comprehension_log_scalar = 0.0
        
        on_going_comprehension_log_scalar = np.clip(on_going_comprehension_log_scalar, 0, 1)

        stateful_obs = np.concatenate([padded_appraisals, [norm_current_position], [remaining_episode_length_awareness], [norm_num_remaining_sentence], [on_going_comprehension_log_scalar]])

        assert stateful_obs.shape[0] == self._num_stateful_obs, f"expected {self._num_stateful_obs} but got {stateful_obs.shape[0]}"

        # # TODO debug delete later
        # print(f"Stateful observation is: {stateful_obs}")

        return stateful_obs
        
    def get_episode_log(self) -> dict:
        """Get logs for the episode"""
        sentence_data_list = []
        for sentence_id in range(self._num_sentences):
            sentence_data = {
                # "sentence_content": self._sampled_text.sentences[sentence_id],
                # "sentence_len": self._sampled_text.sentences[sentence_id].num_words,
            }
            sentence_data_list.append(sentence_data)
        
        episode_log = {
            # "episode_id": self._episode_id,
            # 'sentence_id': self._sentence_info['sentence_id'],
            # 'participant_id': self._sentence_info['participant_id'],
            # 'sentence_content': self._sentence_info['sentence_content'],
            # 'sentence_len': self._sentence_len,
            # "words": words_data_list
        }

        return episode_log


if __name__ == "__main__":
    pass