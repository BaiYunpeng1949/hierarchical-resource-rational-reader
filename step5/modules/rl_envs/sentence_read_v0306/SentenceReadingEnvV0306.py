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


class SentenceReadingEnvV0306(Env):

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
        """

        # Get the current root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        # Get the mode from the config yaml file
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        self._mode = self._config["rl"]["mode"]
        
        print(f"Sentence Reading Environment V0306 -- Deploying the environment in the {self._mode} mode.")

        #########################################################################################################

        self._sentences_manager = SentencesManager()
        self._transition_function = TransitionFunction()
        self._reward_function = RewardFunction()

        #########################################################################################################
        
        # Define states
        # Words in the sentence
        self._num_words_in_sentence = None
        self._num_read_words_in_sentence = None     # Including the words that have been skipped
        self._num_left_words_in_sentence = None
        # Belief-level states
        self._predictabilities = None               # Includes the parafoveal preview, frequency, and contextual predictability
        # Belief
        self._sentence_belief = None                # Appraisal of the sentence, how the reader feels about his understanding of the sentence

        # Counter variables
        pass  # TODO fill them in later

        #########################################################################################################

        # Define the RL env spaces
        self._steps = None
        self.ep_len = 1 * Constants.MAX_NUM_WORDS_PER_SENTENCE
        self._terminate = None
        self._truncated = None
