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

# from memory_profiler import profile

import pandas as pd

from step5.utils import auxiliaries as aux
from step5.utils import constants as cons


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

        # Create a probability distribution of frequency over the lexicon
        total_freq = sum(self.lexicon_w_freq.values())
        self.lexicon_w_freq_prob = {w: f / total_freq for w, f in self.lexicon_w_freq.items()}

        # Other initialization
        self.train_test_words_data = None
    
    def sample_train_test_words(self, num_words=20, mode="random"):
        """
        TODO: maybe remove this later when training is good.
        """

        # Pick 20 words for a train/test subset
        self.train_test_words_data = random.sample(self.lexicon_300, num_words)
        return self.train_test_words_data


    def get_top_k_words(self, sampled_letters_so_far, top_k):
        """
        Return exactly self._top_k words from the lexicon that contain 'partial_string'.
        If fewer than _top_k words match, pad the list with None.
        
        Example: 
            If partial_string = "ell", 
            it will match words like "hello", "jelly", "teller", etc.

        :param sampled_letters_so_far: a substring of letters that the agent has discovered (e.g. "ell")
        :return: list of length self._top_k, sorted by freq (descending),
                padded with None if not enough matches.
        """
        matched_words = []
        
        # 1. Find all words that contain 'partial_string'
        for w in self.lexicon_300:
            if sampled_letters_so_far in w:
                matched_words.append(w)
        
        # # 2. Sort matches by frequency (descending)
        # matched_words.sort(key=lambda w: self.lexicon_w_freq[w], reverse=True)
        
        # 3. Build (word, freq) tuples
        # results = [(w, self.lexicon_w_freq[w]) for w in matched_words]
        results = []
        for w in matched_words:
            freq = self.lexicon_w_freq_prob[w]
            likelihood = self.get_likelihood_by_sampled_letters_so_far(sampled_letters_so_far, w, mode="fractional")
            results.append((w, freq, likelihood))

        # 4. If not enough matches, pad with (None, 0)
        # if len(results) < self._top_k:
        #     results += [(None, 0)] * (self._top_k - len(results))
        prob_epsilons = 0.001
        num_missing = max(0, top_k - len(results))
        padding = [(f"non-word-{i+1}", prob_epsilons, prob_epsilons) for i in range(num_missing)]
        results.extend(padding)
        
        # 5. Return exactly top_k items
        return results[:top_k]

    def get_likelihood_by_sampled_letters_so_far(self, sampled_letters_so_far, word, mode="fractional"):
        """
        Fractional approach (Fractional “Coverage” Likelihood):
        likelihood(w) = (# of sampled letters that appear in w) / (length of sampled_letters_so_far)
        
        If none appear, we give a small non-zero probability like 0.01 to avoid strict zero-likelihood.

        NOTE on Likelihood Computation:     # TODO leave this as an issue, run the model first -- 09 Feb 2025; 
                    maybe DIY a reasonable/simple linear likelihood generator, as a function of numbers of letters sampled. -- 11 Feb 2025

        1. This "fractional coverage" likelihood is noisy because it only checks
           whether each sampled letter appears *anywhere* in the candidate word.
           - It does NOT account for the specific positions of letters.
           - It does NOT distinguish multiple occurrences (e.g., "l" appearing twice).
           - It ignores letter order, which can cause overestimation if letters match
             but in different positions.
        
        2. We also do not handle more nuanced constraints that often matter in reading:
           - Word length constraints beyond basic substring checks.
           - Syntactic or semantic context (e.g., predictability from neighboring words).
           - Morphological rules or syllabic structure.
        
        3. Despite these oversights, the intuition remains:
           - If a shorter word already fits the sampled letters, it is more likely
             (because there are fewer letters left to "explain").
           - More frequent words also tend to dominate the posterior.
        
        This simple approach can be sufficient for a proof of concept, but 
        for a more realistic reading model, you would likely incorporate 
        letter-position alignment, repetition handling, and context-based constraints.
        """

        count = 0
        for ch in sampled_letters_so_far:
            if ch in word:
                count += 1
        
        if len(sampled_letters_so_far) > 0:
            frac = count / len(sampled_letters_so_far)
            # Optionally avoid exact zero:
            return max(frac, 0.01)
        else:
            # If no letters are sampled, everything is equally likely
            return 1.0

    def get_word(self):
        return random.choice(self.train_test_words_data)


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
        Emergent 'Time' Effects. In most implementations, recognition time is modeled by how many “samples” (or cycles of evidence) 
        the system needs before a decision threshold is reached. High-frequency words typically cross threshold in fewer samples, 
        which is loosely analogous to “less time.”
    
    NOTE: Primary assumptions:
        1. All the words presented are known by the reader -- only words within the lexical memory are presented

    TODO: double check whether the predictability's effect is assuming only the likelihood probability related to the context, so there is a constant value.
        The predictability -- likelihood could be composed of two parts: static and dynamic. The original contextual predictability, 
        and the dynamic predictability that changes as the agent samples new letters. 
        Need to check whether the prior work focuses on the contextual predictability (static part) only.
        # TODO 0211: check this and come up with a likelihood function if needed.
    """
    # TODO 0211: retrain the model.

    def __init__(self):
        
        # Get the current root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Get the mode from the config yaml file
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        print(f"Word Activation (No Vision) Environment V0205 -- Deploying the environment in the {self._config['rl']['mode']} mode.")

        self._mode = self._config["rl"]["mode"]

        # Define constants -- configurations
        # Define word lengths
        self.MAX_WORD_LEN = 11
        self.MIN_WORD_LEN = 1
        # Define the top-k candidates when competing for recognition
        self._top_k = 3
        # Define the foveal vision size
        self._foveal_size = 3       # one letter on the left, one in the middle, one on the right side  

        # Initialize necessary classes
        # Initialize the transition function
        self.transition_function = TransitionFunction(max_word_len=self.MAX_WORD_LEN, num_top_k_candidates=self._top_k, foveal_size=self._foveal_size)

        # Internal belief of the agent: belief over the top-k words, empirically set to a constant
        self._normalized_belief_distribution = self.transition_function.reset_state_belief_distribution()      # The belief distribution has to be normalized, sumed up to 1
        self._normalized_belief_distribution_dict = None

        self._word = None           # The word to be recognized
        self._word_len = None       # The length of the word to be recognized
        self._word_freq_prob = None      # The frequency of the word to be recognized -- ranges from 0 to 1
        self._word_predictability_prob = None    # The predictability of the word to be recognized (actually the likelihood prob) -- ranges from 0 to 1
        self._word_dynamic_predictability_prob = None    # The dynamic predictability of the word to be recognized (actually the likelihood prob) -- ranges from 0 to 1, it changes as the agent samples new letters
        self._sampled_letters_so_far = None    # The letters that have been sampled

        # Representations
        self._word_representation = self.transition_function.reset_state_word_representation()    # The word representation, it stores the sampled letters. Here is a vector of letters sampled from the ASCII space
        self._normalized_ground_truth_word_representation = None

        self._sampled_letters_so_far_representation = None   # The letters that have been sampled

        # Define the word that is recognized
        self._word_to_activate = None

        # Define the action 
        self._action = None
        
        # Define the action space: 
        self.action_space = Discrete(self.MAX_WORD_LEN + 1)    # 0-9: fixate on the letter at the position, 10: stop the sampling and recognize the word

        # Define the observation space:
        self.STATEFUL_OBS = "stateful_obs"
        self.ACTION_OBS = "action_obs"
        self._num_stateful_obs = len(self._normalized_belief_distribution) + len(self._word_representation) + 1 + (self.MAX_WORD_LEN + 1 + 1) # Belief distribution, word representation with sampled letters, word length
        self.observation_space = Box(low=-1, high=1, shape=(self._num_stateful_obs,))

        # Initialize the reward function
        self.reward_function = RewardFunction()
        
        # Define the training:
        self.ep_len = 10
        self._steps = None
        self._truncated = None
        self._done = None

        # Define the logger:
        self.log_cumulative_version = None

        # Define the training and testing data (temporary, in the formal training deploy separately)
        self.lex_manager = LexiconManager()
        self.lexicon_w_freq_prob = self.lex_manager.lexicon_w_freq_prob
        # first2pairs = {k: self.lexicon_w_freq_prob[k] for k in sorted(self.lexicon_w_freq_prob.keys())[:2]}
        # print(f"Lexicon with frequency samples: {first2pairs}")
        self.train_test_words_data = self.lex_manager.sample_train_test_words(num_words=20, mode="random")
        # print(f"Train/Test words data: {self.train_test_words_data}")       # TODO comment later during training

    
    def reset(self, seed=None, inputs=None, ep_idx=None):
        """
        Reset the environment to the initial state
        """

        self._steps = 0
        self._truncated = False
        self._done = False
        self.log_cumulative_version = {}

        # Initialize the action
        self._action = -1

        # Reset the belief distribution
        NA = 'non-word'
        self._normalized_belief_distribution_dict = {NA: 0.333, NA + '-1': 0.333, NA + '-2': 0.333}
        self._normalized_belief_distribution = self.transition_function.reset_state_belief_distribution()

        # Reset the word representation
        self._word_representation = self.transition_function.reset_state_word_representation()

        # Reset the seen letters
        self._sampled_letters_so_far_representation = [-1] * self.MAX_WORD_LEN
        self._sampled_letters_so_far = ""

        # Sample the word to be recognized
        self._word = self.lex_manager.get_word()
        self._word_len = len(self._word)
        self._word_freq_prob = self.lex_manager.lexicon_w_freq_prob[self._word]
        self._word_likelihood_prob = self.lex_manager.get_likelihood_by_sampled_letters_so_far(
            sampled_letters_so_far=self._sampled_letters_so_far, word=self._word, mode="fractional"
            )
        self._word_to_activate = None

        # Initialize the ground truth representation -- the word to be recognize is encoded as:
        self._normalized_ground_truth_word_representation = self.transition_function.get_normalized_ground_truth_word_representation(target_word=self._word)
        # This is only used for identifying words and numerical computations

        return self._get_obs(), self._get_logs(is_initialization=True, mode=self._mode)

    def step(self, action):
        """
        Take an action and return the response
        """

        # Initialize variables
        self._done = False
        self._truncated = False
        # info = {}
        reward = 0

        self._action = action

        # Move to the next step
        self._steps += 1

        # Update states
        if action <= self.MAX_WORD_LEN - 1:     # Still fixating on the letters

            if action <= self._word_len - 1:    # The action is valid, sampling letters
                
                self._sampled_letters_so_far_representation, self._sampled_letters_so_far = self.transition_function.update_state_seen_letters(
                    action=action, norm_gt_word_rep=self._normalized_ground_truth_word_representation, 
                    seen_letters_representation=self._sampled_letters_so_far_representation, 
                    seen_letters=self._sampled_letters_so_far, word=self._word, word_len=self._word_len
                )

                self._normalized_belief_distribution_dict, self._normalized_belief_distribution, words_freqs_pred_top_k_dict = self.transition_function.update_state_belief_distribution_dict(
                    seen_letters=self._sampled_letters_so_far, lexicon_manager=self.lex_manager
                )

                reward = self.reward_function.get_step_wise_effort_cost(is_action_valid=True)
            
            else:   # The action is invalid, do nothing
                
                reward = self.reward_function.get_step_wise_effort_cost(is_action_valid=False)

        else:   # Stop the sampling and recognize the word

            self._word_to_activate = self.transition_function.activate_a_word(normalized_belief_distribution_dict=self._normalized_belief_distribution_dict, deterministic=True) 
            
            reward = self.reward_function.get_terminate_reward(
                word_to_recognize=self._word,
                word_to_activate=self._word_to_activate
            )

            self._done = True

            # # TODO comment later
            # print(f"Word to be recognized: {self._word}, the word to be activated: {self._word_to_activate}")
            # print(f"Reward: {reward}")

        if self._steps >= self.ep_len:     # Truncation case
            self._word_to_activate = self.transition_function.activate_a_word(normalized_belief_distribution_dict=self._normalized_belief_distribution_dict, deterministic=True) 
            
            reward = self.reward_function.get_terminate_reward(
                word_to_recognize=self._word,
                word_to_activate=self._word_to_activate
            )
            # TODO debug see the errors below in the terminal
            self._truncated = True
            self._done = True


        return self._get_obs(), reward, self._done, self._truncated, self._get_logs(is_initialization=False, mode=self._mode)

    def render(self, mode='human'):
        pass
    
    def _get_obs(self):   
        """
        Get the current observation
        """

        # Encode the discrete action into a one-hot vector
        action_obs = np.zeros(self.MAX_WORD_LEN + 1 + 1)        # three types of actions -1, fixations, stop
        action_obs[self._action + 1] = 1

        stateful_obs = np.concatenate([self._normalized_belief_distribution, self._word_representation, [self._word_len], action_obs])

        assert len(stateful_obs) == self._num_stateful_obs, f"expected {self._num_stateful_obs} but got {len(stateful_obs)}"

        return stateful_obs
    
    def _get_logs(self, is_initialization=False, mode="train"):
        """
        Obtain the logs
        """        
        if mode == "train":
            return {}
        elif mode == "debug" or mode == "test":
            if is_initialization:   # Return the initializations, mainly the 

                self.log_cumulative_version = {
                    "episode_idnex": "TBD",   # The episode index, to be filled
                    "word": self._word,
                    "word_len": self._word_len,     # Used for analyzing the length's effect
                    "word_frequency": self._word_freq_prob,     # Used for analyzing the frequency's effect
                    "word_representation": self._word_representation,   
                    "normalized_ground_truth_word_representation": self._normalized_ground_truth_word_representation,
                    "fixations": [],
                }

                return self.log_cumulative_version
            else:
                self.log_cumulative_version["fixations"].append({
                    "steps": self._steps,
                    "action": self._action,
                    "done": self._done,
                    "word_predictability": self.lex_manager.get_likelihood_by_sampled_letters_so_far(
                        sampled_letters_so_far=self._sampled_letters_so_far, word=self._word, mode="fractional"
                        ),    # The likelihood probability: P(sampled letters so far | word)
                    "sampled_letters_so_far": self._sampled_letters_so_far,
                    "sampled_letters_so_far_representation": self._sampled_letters_so_far_representation,
                    "word_to_activate": self._word_to_activate,
                    "normalized_belief_distribution": self._normalized_belief_distribution,
                    "normalized_belief_distribution_dict": self._normalized_belief_distribution_dict
                })
                return self.log_cumulative_version


class RewardFunction():
    """
    Reward Function
    """

    def __init__(self, weight_effort_cost=1.0, weight_recognition_bonus=1.0):
        self._weight_effort_cost = weight_effort_cost
        self._weight_recognition_bonus = weight_recognition_bonus

    def get_step_wise_effort_cost(self, is_action_valid):
        if is_action_valid:
            return -1 * self._weight_effort_cost
        else:
            return -1 * self._weight_effort_cost * 2

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

    def __init__(self, max_word_len, num_top_k_candidates, foveal_size):
        
        # Inherited configurations
        self.MAX_WORD_LEN = max_word_len
        self._top_k = num_top_k_candidates
        self._foveal_size = foveal_size

        # The python ORD baselines, up and bottom
        self.MIN_ORD = 32
        self.MAX_ORD = 126

    def reset_state_belief_distribution(self):
        return np.ones(self._top_k) / self._top_k

    def reset_state_word_representation(self):
        return -1 * np.ones(self.MAX_WORD_LEN)
    
    def get_normalized_ground_truth_word_representation(self, target_word):
        gt_word_rep = [ord(c) for c in target_word]
        norm_gt_word_rep = [aux.normalise(w, self.MIN_ORD, self.MAX_ORD, -1, 1) for w in gt_word_rep] 

        # Pad the representation to the max length, non letter positions encode as -1
        if len(norm_gt_word_rep) < self.MAX_WORD_LEN:
            norm_gt_word_rep += [-1] * (self.MAX_WORD_LEN - len(norm_gt_word_rep))
        
        # # TODO debug delete later
        # print(f"Ground truth word representation: {gt_word_rep}, the normalized version: {norm_gt_word_rep}")

        return norm_gt_word_rep

    def update_state_seen_letters(self, action, norm_gt_word_rep, seen_letters_representation, seen_letters, word, word_len):
        """
        Update the word representation by sampling letters
        """

        # Determine the letters to be sampled using the foveal vision
        half_fovea = self._foveal_size // 2     # An symmetric foveal vision

        # Calculate leftmost and rightmost positions
        left_index = max(0, action - half_fovea)
        right_index = min(word_len - 1, action + half_fovea)

        # Update the seen letters representation in the foveal vision
        for i in range(left_index, right_index+1):
            seen_letters_representation[i] = norm_gt_word_rep[i]
        
        # Update the seen letters -- get all letters that are not -1 in representation from word
        # seen_letters = "".join([word[i] for i in range(len(word)) if seen_letters_representation[i] != -1])
        seen_letters = "".join([word[i] for i in range(len(word)) if i < len(seen_letters_representation) and seen_letters_representation[i] != -1]) or "NON_WORDS"

        # # TODO comment later
        # print(f"The action value is: {action}, left index: {left_index}, right index: {right_index}; the target word is: {word}, the word length is: {word_len}")
        # print(f"Seen letters representation: {seen_letters_representation}, seen letters: {seen_letters}")

        return seen_letters_representation, seen_letters
    
    def update_state_belief_distribution_dict(self, seen_letters, lexicon_manager: LexiconManager):
        """
        Update the belief distribution
        p(w_i|sampled letters so far) = p(w_i) * p(sampled letters so far|w_i) / Sigma_w' p(w') * p(sampled letters|w')
        In the Bayesian Reader, the p(w) is constant through out the updating process. Only the likelihood prob is changing due to the new sampled letters.

        Though everytime the posterior was not used in the next step, it is still a Bayesian updating process. Just not a full inference.
            As the exact full inferene is often impractical.
        The all-in-one manner: All-in-One: Always take the original prior p(w) (based on frequency) and re-compute the likelihood based on all sampled letters so far.
        We don't necessarily have to feed the old posterior directly back in as the next prior if our likelihood function always accounts for all the sampled letters so far.
        """

        # Find the top five words, in terms of the closest letters, and the length (criteria); also get the corresponding frequency probability, and the likelihood probability
        words_freqs_likelihoods = lexicon_manager.get_top_k_words(sampled_letters_so_far=seen_letters, top_k=self._top_k)

        # Update the beliefs
        belief_distribution_dict = {wfl[0]: wfl[1] * wfl[2] for wfl in words_freqs_likelihoods}

        # Normalize the beliefs
        normalized_belief_distribution_dict = {k: v / sum(belief_distribution_dict.values()) for k, v in belief_distribution_dict.items()}

        normalized_belief_distribution = list(normalized_belief_distribution_dict.values())

        # # TODO comment out later
        # print(f"Words, frequencies, and likelihoods: {words_freqs_likelihoods}")
        # print(f"Belief distribution dict: {belief_distribution_dict}, belief distribution: {normalized_belief_distribution}")
        # print(f"Normalized belief distribution: {normalized_belief_distribution_dict}")

        return normalized_belief_distribution_dict, normalized_belief_distribution, words_freqs_likelihoods

    def activate_a_word(self, normalized_belief_distribution_dict, deterministic=True):
        """
        Activate a word from the belief distribution, choose the highest for simplicity
        """
        if deterministic:
            activated_word = max(normalized_belief_distribution_dict, key=normalized_belief_distribution_dict.get)
            return activated_word
        else:
            return np.random.choice(list(normalized_belief_distribution_dict.keys()), p=list(normalized_belief_distribution_dict.values()))

if __name__ == "__main__":
    env = WordActivationRLEnv()
    env.reset()
    env.step(1)
    env.step(10)

    # env = LexiconManager()
    # print(env.get_top_k_words("ell"))
    # print(env.get_top_k_words("tell"))