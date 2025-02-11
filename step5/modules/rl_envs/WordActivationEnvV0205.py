import random

import numpy as np
import yaml
import json
import os
import re
import math
import cv2
import matplotlib.pyplot as plt
import markovify
import fasttext

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

        # Get the random predictability of the word in the context
        self.lexicon_w_pseudo_pred_prob = {w: self.get_predictability(mode="random") for w in self.lexicon_300}

        # Other initialization
        self.train_test_words_data = None
    
    def sample_train_test_words(self, num_words=20, mode="random"):     # TODO solve this, make the train test data set the global dataset
        """
        TODO: maybe remove this later when training is good.
        """

        # # Pick 20 words for a train/test subset
        # self.train_test_words_data = random.sample(self.lexicon_300, num_words)
        # return self.train_test_words_data

        self.train_test_words_data = self.lexicon_300
        return self.lexicon_300


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
            pred = self.lexicon_w_pseudo_pred_prob[w]
            likelihood = self.get_likelihood_by_sampled_letters_so_far(sampled_letters_so_far, w, mode="fractional")
            results.append((w, freq, pred, likelihood))

        # 4. If not enough matches, pad with (None, epsilon values)
        prob_epsilons = 0.001
        num_missing = max(0, top_k - len(results))
        padding = [(f"non-word-{i+1}", prob_epsilons, prob_epsilons, prob_epsilons) for i in range(num_missing)]
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
    
    def get_predictability(self, mode="random"):
        """
        Get the predictability of the word in the context
        """
        
        if mode == "random":
            return random.uniform(0, 1)
        else:
            # Use language models to get the predictability
            return 0.97

    def get_word(self):
        return random.choice(self.train_test_words_data)


class WordGenerator:
    def __init__(self, corpus):
        """Train a simple Markov model on word corpus."""
        self.model = markovify.Text(corpus, state_size=2)  # Uses 2-character states
    
    def generate_similar_word(self, prefix):
        """Generate a similar word given a prefix."""
        word = prefix
        for _ in range(10):  # Max word length
            next_char = self.model.make_sentence(tries=10)  # Sample next letter
            if next_char:
                word += next_char[0]  # Take first letter from the sampled word
            if len(word) > 10 or next_char is None:  # Stop at max length
                break
        return word


class ApproximateWordGenerator:
    def __init__(self, alphabet="abcdefghijklmnopqrstuvwxyz"):
        self.alphabet = list(alphabet)
    
    def generate_similar_word(self, prefix):
        """Generate words similar to the given prefix with small modifications."""
        word = list(prefix)
        
        if len(word) < 10 and random.random() > 0.5:
            word.append(random.choice(self.alphabet))  # Add a random letter
        
        if len(word) > 2 and random.random() > 0.3:
            word[random.randint(0, len(word) - 1)] = random.choice(self.alphabet)  # Substitute
        
        if len(word) > 3 and random.random() > 0.3:
            word.pop(random.randint(0, len(word) - 1))  # Delete a letter

        return "".join(word)


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

    NOTE: word predictability's effect
    Reference: Word length and frequency effects on text reading are highly similar in 12 alphabetic
    Quotation: 
        "It is well established that words that are more predictable in their context are recognized faster and require less cognitive effort of processing."
        "There are two common ways of measuring a word's predictability, and neither is practical to use here. One approach uses human responses (cloze probability), 
        and the other relies on computational models estimating word expectancy based on large corpora."
    Assumption:
        1. Not just reading time / processing time, the number of fixations are also affected by the word predictability. 
            Words that are more predictable in their context have fewer number of fixations. 
    Implementation: we combine word predictability into the **prior**, i.e., p(word)=p(word_freq) * p(word_predictability), instead of merging into the likelihood.
        This is because Priors influence word recognition before letter processing begins [Bayesian Reader], while likelihoods are updated as letters are sampled.
        Predictability reduces lexical access time by increasing prior probability before letters are processed, Skipping rates increase before letter evidence is considered [EZReader].
    Constraints: The predictability and frequency would be the same, but they should be distinct in the real-world application [EZReader].
        1. Word frequency effects are more consistent (they affect all words equally).
        2. Predictability effects are stronger for high-context situations (e.g., sentence constraints matter).
        3. The effect of predictability is typically stronger on skipping rates, while word frequency mainly affects fixation duration. -- Do a task-specific control parameter?
    """
    # TODO 0211: retrain the model with a more reasonable word activation from the corpus. -- Next pirority: work on a larger corpus, and a more activation process
    # That can be sensitive to the freq and pred effects. -- 0211

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
        self.MAX_WORD_LEN = 15
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
        # print(f"Train/Test words data: {self.train_test_words_data}")    
    
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
        self._word_predictability_prob = self.lex_manager.lexicon_w_pseudo_pred_prob[self._word]    # Only for training and testing, for actual simulation, need to use actual LLMs
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

                self._normalized_belief_distribution_dict, self._normalized_belief_distribution, words_freqs_pred_likelihood_top_k_dict = self.transition_function.update_state_belief_distribution_dict(
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

            # print(f"Word to be recognized: {self._word}, the word to be activated: {self._word_to_activate}")
            # print(f"Reward: {reward}")

        if self._steps >= self.ep_len:     # Truncation case
            self._word_to_activate = self.transition_function.activate_a_word(normalized_belief_distribution_dict=self._normalized_belief_distribution_dict, deterministic=True) 
            
            reward = self.reward_function.get_terminate_reward(
                word_to_recognize=self._word,
                word_to_activate=self._word_to_activate
            )
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
                    "Word predictability": self._word_predictability_prob,    # Used for analyzing the predictability's effect
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
            return -1 * self._weight_effort_cost        # For faster training, could remove

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
        words_freqs_preds_likelihoods = lexicon_manager.get_top_k_words(sampled_letters_so_far=seen_letters, top_k=self._top_k)

        # Update the beliefs: prior * likelihood, where the prior is p(freq) * p(pred)
        belief_distribution_dict = {wfpl[0]: wfpl[1] * wfpl[2] * wfpl[3] for wfpl in words_freqs_preds_likelihoods}

        # Normalize the beliefs
        normalized_belief_distribution_dict = {k: v / sum(belief_distribution_dict.values()) for k, v in belief_distribution_dict.items()}

        normalized_belief_distribution = list(normalized_belief_distribution_dict.values())

        return normalized_belief_distribution_dict, normalized_belief_distribution, words_freqs_preds_likelihoods

    def activate_a_word(self, normalized_belief_distribution_dict, deterministic=True):
        """
        Activate a word from the belief distribution, choose the highest for simplicity

        NOTE: some problems:
        1. The Lexical Size Constraint & Unique Words Activating Too Early
            Some words are unique in the lexicon, so they get activated too quickly with very few fixations. 
            This doesn't reflect how humans read, because humans don't search for words—they map letters to possible words and handle ambiguity.
        2. Activation Shouldn't Be Just Relative; It Should Depend on Posterior Values
            Right now, the agent activates words based only on relative belief sizes (e.g., highest belief wins).
            But: Sometimes, a word shouldn't be activated, even if it has the highest belief, because its absolute probability is still low.
        3. Rethinking Word Activation: Recognizing vs. Searching
            The current model searches for words in a fixed lexicon, but humans don't do that -- they recognize and map letters dynamically. 
            Humans don't have a giant lexicon lookup table; they infer meaning from partial, noisy inputs.
        
        Solutions:
        1. Introduce Ambiguous Alternatives
        2. Bernoulli Sampling for Activation Probability
        3. Use a Vector Space / Embeddings for Fuzzy Matching

        How to Implement Solution 1: Ambiguous Alternatives
        Method 1: Character-Level Language Model (Fast, Efficient), e.g., markovify (https://github.com/jsvine/markovify)
        Method 2: Approximate String Matching / Edit Distance (Fast)
        Method 3: Use a Pretrained Word Embedding Model (Most Realistic)
        """
        if deterministic:   # The greedy approach
            # Simple, efficient, and deterministic; always picks the most likely word; 
            #   but the agent does not capture human uncertainty or competition between words. Does not account that in the real case, 
            #   people sometimes misrecognize words if two candidates have similar probabilties.
            activated_word = max(normalized_belief_distribution_dict, key=normalized_belief_distribution_dict.get)
            return activated_word
        else:               # The stochastic selection (softmax sampling)
            # return np.random.choice(list(normalized_belief_distribution_dict.keys()), p=list(normalized_belief_distribution_dict.values()))
            words, probs = zip(*normalized_belief_distribution_dict.items())  # Unpack words & their probabilities
            activated_word = random.choices(words, weights=probs, k=1)[0]
            return activated_word
        # NOTE: Other methods: confidence-based thresholding (stop when certain). Cons: not adaptive, the threshold needs to be tuned.
        # NOTE: Other methods: Bayesian Stopping Rule (Confidence + Uncertainty Reduction)
        # Codes:
        # def entropy(prob_dist):
        #     """Compute entropy of a probability distribution."""
        #     return -sum(p * np.log2(p) for p in prob_dist.values() if p > 0)

        # def recognize_using_entropy(belief_distribution, uncertainty_threshold=0.2):
        #     best_word, best_prob = max(belief_distribution.items(), key=lambda x: x[1])
        #     prob_entropy = entropy(belief_distribution)

        #     if prob_entropy < uncertainty_threshold:
        #         return best_word  # Stop sampling
        #     return None  # Keep sampling more letters

if __name__ == "__main__":
    # env = WordActivationRLEnv()
    # env.reset()
    # env.step(1)
    # env.step(10)

    # env = LexiconManager()
    # print(env.get_top_k_words("ell"))
    # print(env.get_top_k_words("tell"))

    # # Example usage:      NOTE: issues -- only output 'hel'
    # corpus = "hello help helmet hall held held hero heat heavy height hello jelly yellow mellow"
    # word_gen = WordGenerator(corpus)
    # for i in range(10):
    #     print(word_gen.generate_similar_word("hel"))  # Might generate "helton" or "helder"

    # # Example usage:        NOTE: issue -- only output words with the same length of 'hel'
    # word_gen = ApproximateWordGenerator()
    # for i in range(10):
    #     print(word_gen.generate_similar_word("hel"))  # Outputs: "helo", "hep", "hella"

    # Load the FastText model once (global variable)
    fasttext_model = fasttext.load_model("cc.en.300.bin")  # Load once

    def generate_similar_word_fasttext(prefix, k=5):
        """Find similar words in FastText word embeddings."""
        similar_words = fasttext_model.get_nearest_neighbors(prefix, k=k)
        return [word for _, word in similar_words]

    # Example usage:
    for i in range(10):
        print(generate_similar_word_fasttext("hel"))  # Outputs: ['hello', 'help', 'helium', 'helmet']