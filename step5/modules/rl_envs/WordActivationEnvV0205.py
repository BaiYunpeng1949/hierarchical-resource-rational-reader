import random

import numpy as np
import yaml
import json
import os
import re
import math
import cv2
import matplotlib.pyplot as plt

from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete, Tuple
from annoy import AnnoyIndex

# from memory_profiler import profile

import pandas as pd

from step5.utils import auxiliaries as aux
from step5.utils import pseudo_offline_ocr_model as ocr
from step5.utils import constants as cons


class WordActivationRLEnv(Env):
    """
    Oculomotor Controller RL Environment

    A toy application for a high-fidelity simulation of how human activate words when reading and recognizing.

    Model objectives:
        1. Bayesian Reader as a belief update. P(word|sampled letters) = P(word) * P(letters|word) / Sigma_w' P(w') * P(letters|w'). 
            It could naturally account for frequency (through P(word)), length and predictability's effects (through P(letters|word)).
        2. Activation issue -- a large lexicon: 1) Candidate set pruning -- either restrict to a top-k condidate list, or 2. Use an approximate match structure (e.g., trie, prefix tree, or approximate nearest-neighbor on some letter embedding) 2) On-the-Fly Updating
        3. Competition among multiple words -- Bayesian belief distribution could be a vector of probabilities of words that has been parallelly activated. 
            The highest posterior wins the recognition competition.
    Learning objectives (what do I want the model to achieve):
        1. Learn to fixate on the most informative letter (could be a sub-optimal strategy, just like human)
        2. Learn to stop sampling when the word is recognized as soon and accurate as possible
    
    NOTE: this function will be called n times, where n is the number of parallel environments. So better configure everything only once, 
        outside of this function for training efficiency.

    NOTE: Response time (RT) vs. the number of fixations
        Emergent ‘Time’ Effects. In most implementations, recognition time is modeled by how many “samples” (or cycles of evidence) 
        the system needs before a decision threshold is reached. High-frequency words typically cross threshold in fewer samples, 
        which is loosely analogous to “less time.”
    """

    def __init__(self):
        
        # Get the current root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Get the mode from the config yaml file
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        print(f"Oculomotor Controller Environment V0205 -- Deploying the environment in the {self._config['rl']['mode']} mode.")

        # Internal belief of the agent: belief over the top-k words, empirically set to a constant
        self._top_k = 5
        self._belief_distribution = self._reset_belief_distribution()      # The belief distribution has to be normalized, sumed up to 1

        # Define the environment configuration
        # Word lengths
        self.MAX_WORD_LEN = 10
        self.MIN_WORD_LEN = 1

        # Define the cognitive constraints
        self._foveal_size = 3       # one letter on the left, one in the middle, one on the right side  
        
        # Initialize the transition function
        self.transition_function = TransitionFunction(max_word_len=self.MAX_WORD_LEN, num_top_k_candidates=self._top_k)
        
        # Representations
        self._word_representation = self.transition_function.reset_word_representation()    # The word representation, it stores the sampled letters. Here is a vector of letters sampled from the ASCII space
        self._ground_truth_word_representation = None

        # Define the word that is recognized
        self._word_to_activate = None
        
        # Define the action space: 
        self.action_space = Discrete(self.MAX_WORD_LEN + 1)    # 0-9: fixate on the letter at the position, 10: stop the sampling and recognize the word

        # Define the observation space:
        self._num_stateful_obs = len(self._belief_distribution) + len(self._word_representation) # Belief distribution, word representation with sampled letters
        self.observation_space = Dict({
            "stateful_obs": Box(low=-1, high=1, shape=(self._num_stateful_obs,)),
            "action_obs": Discrete(self.MAX_WORD_LEN+1)     # The action of continue fixating or stop    
        })

        # Initialize the reward function
        self.reward_function = RewardFunction()
        
        # Define the training:
        self._ep_len = 20
        self._steps = None
        self._truncated = None

        # Define the logger:
        self._logger = None

        # Define the Bayesian Updater
        self.bayesian_inference = BayesianInference()

        # Define the training and testing data (temporary, in the formal training deploy separately)
        # TODO think how to structure the data, should we get a look-up table?
        # Q1, how do we get the possible words from the first place? Where should we grab the words from?
        # So the pipeline could be: the agent first samples some letters from any word position, then it would sample some certain words with such combination of letters; 
        #   these words are the init words; then as the agent samples other parts, both letters' position and new information will update this candidate list, 
        #   with an updating belief distribution.
        self.lex_manager = LexiconManager()
        self.lexicon_w_freq = self.lex_manager.lexicon_w_freq
        first2pairs = {k: self.lexicon_w_freq[k] for k in sorted(self.lexicon_w_freq.keys())[:2]}
        print(f"Lexicon with frequency samples: {first2pairs}")
        self.train_test_words_data = self.lex_manager.sample_train_test_words(num_words=20, mode="random")
        print(f"Train/Test words data: {self.train_test_words_data}")

    
    def reset(self, seed=None, inputs=None, ep_idx=None):
        """
        Reset the environment to the initial state
        """

        self._steps = 0
        self._truncated = False

        # Reset the belief distribution
        self._belief_distribution = self.transition_function.reset_belief_distribution()

        # Reset the word representation
        self._word_representation = self.transition_function.reset_word_representation()

        # Sample the word to be recognized
        self._word = self.lex_manager.get_word()
        self._word_to_activate = None

        # Initialize the ground truth representation -- the word to be recognize is encoded as:
        self._ground_truth_word_representation = self.transition_function.get_normalized_ground_truth_word_representation(target_word=self._word)

        # TODO debug later
        print(f"Reset the environment with the word: {self._word}")

        return self._get_obs(), {}

    def step(self, action):
        """
        Take an action and return the response
        """

        # Initialize variables
        done = False
        self._truncated = False
        info = {}
        reward = 0

        # Move to the next step
        self._steps += 1

        # Update states
        if action <= self.MAX_WORD_LEN - 1:     # Still fixating on the letters
            self._word_representation[action] = self._word[action]
            reward = self.reward_function.get_step_wise_effort_cost()

            self.transition_function.update_sampled_letters(s, a)

            self.transition_function.update_belief_distribution()

        else:   # Stop the sampling and recognize the word
            reward = self.reward_function.get_terminate_reward(
                word_to_recognize=self._word,
                word_to_activate=self._word_to_activate
            )

            self.transition_function.activate_a_word()
        
        return self._get_obs(), reward, done, self._truncated, info

    def render(self, mode='human'):
        pass
    
    def _get_obs(self):
        """
        Get the current observation
        """
        return self._belief_distribution
    
    def get_logs(self):
        pass        # TODO later


class RewardFunction():
    """
    Reward Function
    """

    def __init__(self, weight_effort_cost=1.0, weight_recognition_bonus=1.0):
        self._weight_effort_cost = weight_effort_cost
        self._weight_recognition_bonus = weight_recognition_bonus

    def get_step_wise_effort_cost(self):
        return -1 * self._weight_effort_cost

    def get_terminate_reward(self, word_to_recognize, word_to_activate):
        
        Bonus = self._weight_recognition_bonus * 10
        
        if word_to_recognize == word_to_activate:
            return Bonus
        else:
            return -1 * Bonus


class TransitionFunction():
    """
    Transition Function
    """

    def __init__(self, max_word_len, num_top_k_candidates):
        
        # Inherited configurations
        self.MAX_WORD_LEN = max_word_len
        self._top_k = num_top_k_candidates

        # The python ORD baselines, up and bottom
        self.MIN_ORD = 32
        self.MAX_ORD = 126

    def reset_belief_distribution(self):
        return np.ones(self._top_k) / self._top_k

    def reset_word_representation(self):
        return -1 * np.ones(self.MAX_WORD_LEN)
    
    def get_normalized_ground_truth_word_representation(self, target_word):
        gt_word_rep = [ord(c) for c in target_word]
        norm_gt_word_rep = [aux.normalise(w, self.MIN_ORD, self.MAX_ORD, -1, 1) for w in gt_word_rep] 
        return norm_gt_word_rep

    def update_sampled_letters(self):
        pass

    def update_state(self):
        pass

    def get_state(self):
        pass


class BayesianInference():
    """
    Bayesian Updater
    """

    def __init__(self):
        pass

    def reset(self):
        pass

    def update_belief(self):
        pass

    def get_belief(self):
        pass


class LexiconManager():
    """
    Lexicon Manager
    """

    def __init__(self):
        
        self.lexicon_300 = [
            # 1-letter words (2 total)
            "a", "i",

            # 2-letter words (10 total)
            "am", "an", "as", "at", "be", "by", "do", "go", "he", "if",

            # 3-letter words (20 total)
            "cat", "dog", "pig", "cow", "ant", "bee", "car", "bar", "win", "fun",
            "run", "day", "sky", "red", "mom", "dad", "sun", "jet", "cut", "fit",

            # 4-letter words (30 total)
            "tree", "milk", "over", "past", "club", "hard", "soft", "fast", "math", "love",
            "code", "ball", "corn", "film", "grow", "hero", "lady", "tell", "bear", "king",
            "jump", "silk", "talk", "walk", "rock", "rain", "turn", "lift", "mind", "weak",

            # 5-letter words (30 total)
            "smile", "angry", "apple", "baker", "cable", "chalk", "dance", "eager", "fancy", "giant",
            "honor", "igloo", "jelly", "knead", "latch", "mimic", "noble", "ocean", "piano", "queen",
            "reach", "sauce", "tiger", "urban", "vague", "waltz", "xylem", "yield", "zesty", "raven",

            # 6-letter words (40 total)
            "absorb", "absent", "bright", "canvas", "damage", "danger", "effort", "family",
            "garden", "harbor", "hammer", "junior", "kidnap", "launch", "legacy", "matter",
            "narrow", "orange", "police", "quiver", "review", "silver", "shadow", "thrive",
            "unrest", "vacuum", "wonder", "xenial", "yolked", "zephyr", "bounty", "caught",
            "desert", "excuse", "fridge", "gender", "hiatus", "injury", "ignite", "banana",

            # 7-letter words (40 total)
            "already", "ammonia", "another", "bargain", "caution", "central", "darkest", "denying",
            "examine", "fiction", "freedom", "genuine", "harvest", "impress", "jubilee", "kingdom",
            "mixture", "nuggets", "observe", "plunder", "quality", "realign", "rebirth", "scanner",
            "teacher", "unclear", "villain", "warrior", "xeroxed", "yellowy", "zippers", "academy",
            "brownie", "captain", "delight", "exploit", "flaming", "hostage", "justice", "captors",  # replaced an accidental repeat

            # 8-letter words (40 total)
            "absolute", "attitude", "backpack", "baseball", "building", "carefully", "chatroom", "creative",
            "dinosaur", "director", "doorstep", "equipment", "eyepiece", "firewall", "football", "forgotten",
            "handmade", "homework", "infinity", "keyboard", "lifestyle", "linoleum", "marathon", "movement",
            "overcome", "particle", "quizzical", "regional", "shoulder", "sketches", "terrible", "tutorial",
            "umbrella", "vertical", "weekdays", "whenever", "wireless", "woodland", "xenoliths", "yardstick",

            # 9-letter words (30 total)
            "aardvarks", "adjective", "aquariums", "blackboard", "blueprint", "buttercup", "childbirth", "commenter",
            "confident", "creations", "dangerous", "dedicated", "defensive", "direction", "dimension", "drizzling",
            "excellent", "factories", "fantastic", "favourite", "formation", "forgotten", "gravitons", "hairbrush",
            "hysterics", "important", "landscape", "machinery", "drainpipes", "testflight",  # added 2 to keep total correct

            # 10-letter words (58 total)
            "accelerate", "advisories", "alphabetic", "amazements", "apocalypse", "authorized", "background", "bestseller",
            "blacksmith", "bombardiers", "carbonation", "chronicling", "commercial", "confiscate", "controller", "courthouse",
            "deactivate", "departures", "enthusiasm", "excitations", "foundation", "glittering", "greenhouse", "influencer",
            "journalism", "leadership", "livelihood", "lumberjack", "mainstream", "marketplace", "miraculous", "narrations",
            "noticeably", "overloading", "peppermint", "personality", "philosophy", "principled", "protecting", "quintupling",
            "reassuring", "renovations", "retrieving", "separations", "spectacles", "spiralling", "summertime", "switchboard",
            "vocational", "typography", "understand", "unfathomed", "university", "vacationer", "wrongdoings", "yearningly",
            "watermelon", "woodcutting"
        ]

        random.seed(42)
        
        # Create a dictionary mapping each word to a random frequency in [1..100]
        self.lexicon_w_freq = {w: random.randint(1, 100) for w in self.lexicon_300}

        # Other initialization
        self.train_test_words_data = None
    
    def sample_train_test_words(self, num_words=20, mode="random"):

        # Pick 20 words for a train/test subset
        self.train_test_words_data = random.sample(self.lexicon_300, num_words)
        return self.train_test_words_data


    def get_top_k_words(self):
        pass

    def update_lexicon(self):
        pass

    def get_word(self):
        return random.choice(self.train_test_words_data)


if __name__ == "__main__":
    env = WordActivationRLEnv()
    for i in range(20):
        env.reset()
    # env.step(1)
    # env.render()