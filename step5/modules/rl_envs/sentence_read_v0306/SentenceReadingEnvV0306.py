import math
import os
import yaml
import random

import numpy as np

from gymnasium import Env
from gymnasium.spaces import Box, Discrete, Dict

from modules.rl_envs.sentence_read_v0306.SentencesManager import SentencesManager
from modules.rl_envs.sentence_read_v0306.TransitionFunction import TransitionFunction
from modules.rl_envs.sentence_read_v0306.RewardFunction import RewardFunction
from modules.rl_envs.sentence_read_v0306 import Constants


class SentenceReadingEnv(Env):

    def __init__(self):
        """
        Create on 6 March 2025.
        This is the environment for the RL-based intermediate level -- sentence-level control agent: 
            it controls word skippings and word revisits in a given sentence
        
        Features: predict the word skipping and word revisiting decisions, and when to stop reading.
        Cognitive constraints: (limited time and cognitive resources) 
            Foveal vision (so need multiple fixations), 
            STM (so need to revisit sometimes, provide limited contextual predictability),
            attention resource (so need to skip the unnecessary words),
            Time pressure (so need to finish reading before time runs out) -- this is an additional trait for sec 2.4/2.5
        Optimality: maximizing overall comprehension.
        State (addtional stuff in resource rationality): continuous monitoring of certainty. 
            (How confident am I that I already got that word? Or that I understood the prior sentence?) 
        Noisy information input (observation).

        NOTE: my understanding of word skipping and revisiting
            1. Word Skipping: “I Already Know Enough.”
                High Confidence / Low Utility of Further Fixation
                Rate of Information Gain Falls Below Threshold
                Hence a Single “Skip or Not” Decision
            2. Revisiting (Regressions): “I Don't Understand Enough Yet.”
                Detecting a Comprehension or Integration Problem
                Cost-Benefit Trade-Off
                Adaptive Correction (coherence)
        
        Task: each episode read one sentence. 
        Objective: read it as soon as possible, with the highest overall comprehension.
        
        Author: Bai Yunpeng

        NOTE: this version is originated from v1014 in the old repo. I rewrite it for the Nature Human Behavior paper.
        NOTE: the dataset to compare with are mainly from ZuCo 1.0 Task 2 Natural Reading, where P=12, number of sentences=300
        NOTE: this env is for explaining word features' effect on the word skipping and regressions. 
            For e.g., word length, frequency, and predictability's effect on word skipping; 
                and word difficultiy's effect on regressions; and my custom effect:  
            As for the when to stop and specific sentence-level word skipping and regression replications (with specific values), 
                will be added in the 4th section and maybe 6th section of simulation results (maybe such a section is not needed for a Nature Human Behavior paper).
        
        Formalism: both word skipping and regressions are a result of the resource-rationality. 
            These are results of the optimazation of the trade-off between the information gain and costs of eye movements.
        
        NOTE: empirical data evidence:
            Source: ZuCo 1.0 Task 2 Natural Reading, processed by me, aggregated metrics could be found: /home/baiy4/ScanDL/scripts/data/zuco/bai_processed_task2_NR_ET_aggregated_metrics
            Word skipping, average skip distance: 1.93 words; max skip distance mean: 3.51 words;
            Word regression, average regression distance: 1.59 words; max regression distance mean: 1.96 words;
            NOTE: But there are a lot of outliers with huge jumps. So for the simplicity, we are mainly simulating one word regression and one word skipping first.
        """

        # Get the current root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        # Get the mode from the config yaml file
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        self._mode = self._config["rl"]["mode"]
        
        print(f"Sentence Reading Environment V0306 -- Deploying the environment in the {self._mode} mode.")

        #########################################################################################################

        self.sentences_manager = SentencesManager()
        self.transition_function = TransitionFunction()
        self.reward_function = RewardFunction()

        #########################################################################################################
        
        # Define states
        # Words in the sentence
        self._sentence_len = None
        self._num_read_words_in_sentence = None     # Including the words that have been skipped
        self._num_left_words_in_sentence = None
        # Preset predictabilities of the words in the sentence, it helps the agent to make the decision of word skipping and regressions, 
        #   this does not change according to the actions, but providing evidence for forming the belief / appraisal
        self._words_predictabilities_in_sentence = None     # The overall predictability of the words (all in one metric)
        self._words_contextual_predictabilities_in_sentence = None     # The contextual predictability of the words (contextual predictability)
        self._words_frequencies_in_sentence = None          # Frequency of the words
        self._words_difficulties_in_sentence = None         # Difficulty of the words
        self._words_lengths_in_sentence = None              # Length of the words
        # Words beliefs / appraisals, the agent actively update it itself through actions
        self._words_appraisals_in_sentence = None                # Appraisal of the sentence, how the reader feels about his understanding of the sentence

        # Other states
        self._current_word_index = None
        self._current_word_predictability = None
        self._next_word_predictability = None

        # Counter variables
        # NOTE: same definition of how we process the ZuCo 1.0 Task 2 Natural Reading data; 
        #   skipped words are words that are not fixed during the first-pass reading
        self._skipped_words_indexes = None      
        # regressed words are words that are regressed to the previous word (before the reading progress and current fixation)
        self._regressed_words_indexes = None
        # The reading sequence of the words
        self._reading_sequence_of_word_indexes = None

        #########################################################################################################

        # Define the RL env spaces
        self._steps = None
        self.ep_len = 2 * Constants.MAX_SENTENCE_LENGTH      # times 2 because we have word regressions
        self._terminate = None
        self._truncated = None

        # Define the action space
        self._REGRESS_ACTION = 0
        self._READ_ACTION = 1
        self._SKIP_ACTION = 2
        self._STOP_ACTION = 3    
        self.action_space = Discrete(4)     
        # 0: regression to the one previous word; 1: read the next word; 2: skip the next word to read the next next.
        #   NOTE: We do not specify the termination control here, it belongs to sec.2.5. 
        #   So the agent stops reading when it reaches the end of the sentence.

        # Define the observation space -- The overall appraisal of the sentence, current fixation (word index), next word predictability
        self._num_stateful_info_obs = Constants.MAX_SENTENCE_LENGTH + 1 + 1
        self.observation_space = Box(low=0, high=1, shape=(self._num_stateful_info_obs,))
        
    def reset(self, seed=42, inputs: dict = None):
        """
        Reset the environment to the initial state.
        """

        super().reset(seed=seed)

        self._steps = 0

        self._truncated = False
        self._terminate = False

        # Reset the sentences manager (external environment)
        self._sentence_len, self._words_predictabilities_in_sentence = self.sentences_manager.reset()

        # Reset the transition function
        self._words_appraisals_in_sentence = self.transition_function.reset(sentence_length=self._sentence_len)

        # Reset the action-related states for logs later
        self._current_word_index = -1
        self._next_word_predictability = self._words_predictabilities_in_sentence[0]    # Start with the first word's predictability

        # Reset the logs
        self._skipped_words_indexes = []
        self._regressed_words_indexes = []
        self._reading_sequence_of_word_indexes = []
        return self._get_obs(), {}
    
    def step(self, action):
        """
        Take a step in the environment
        """

        self._terminate = False
        self._truncated = False

        reward = 0

        # Update the steps
        self._steps += 1

        # Update the states
        if action == self._REGRESS_ACTION:
            
            self._words_appraisals_in_sentence, self._current_word_index, action_validity = self.transition_function.update_state_regress(
                appraisals=self._words_appraisals_in_sentence,
                current_word_index=self._current_word_index,
                
            )    
            reward += self.reward_function.compute_regress_reward()    
        
        elif action == self._READ_ACTION:
            
            self._words_appraisals_in_sentence, self._current_word_index, action_validity = self.transition_function.update_state_read_next_word(
                appraisals=self._words_appraisals_in_sentence,
                current_word_index=self._current_word_index,
                sentence_length=self._sentence_len,
            )    
            reward += self.reward_function.compute_read_reward()    
        
        elif action == self._SKIP_ACTION:
            
            self._words_appraisals_in_sentence, self._current_word_index, action_validity = self.transition_function.update_state_skip_next_word(
                appraisals=self._words_appraisals_in_sentence,
                current_word_index=self._current_word_index,
                sentence_length=self._sentence_len,
                skip_word_predictability=self._next_word_predictability,
            )    
            reward += self.reward_function.compute_skip_reward()  

        elif action == self._STOP_ACTION:
            self._terminate = True
            self._truncated = False
            # Get the reward later
        
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # Check termination and truncation
        if self._steps >= self.ep_len:     # Truncation case
            self._terminate = True
            self._truncated = True
        
        # Get the termination reward
        if self._terminate or self._truncated:
            reward += self.reward_function.compute_terminate_reward(sentence_appraisals=self._words_appraisals_in_sentence)
        
        # Update the logs
        self._update_logs()
        
        return self._get_obs(), reward, self._terminate, self._truncated, {}
    
    def _get_obs(self):
        """
        Get the observation
        """

        # Normalize the current word index, allowed to take values in [0, 1], because <EOS> is included.
        normalized_current_word_index = self._current_word_index / self._sentence_len   
        
        stateful_obs = np.concatenate([self._words_appraisals_in_sentence, [normalized_current_word_index], [self._next_word_predictability]])

        assert len(stateful_obs) == self._num_stateful_info_obs, f"expected {self._num_stateful_info_obs} but got {len(stateful_obs)}"

        return stateful_obs
    
    def _update_logs(self):
        """
        Update the logs
        """

        self._reading_sequence_of_word_indexes.append(self._current_word_index)

        return self._reading_sequence_of_word_indexes
    

if __name__ == "__main__":

    # Test here 
    env = SentenceReadingEnvV0306()
    env.reset()
    print(env._get_obs())
    print(env._get_logs())
        
        
        
