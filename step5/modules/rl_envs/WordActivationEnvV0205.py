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

    def __init__(self, zipf_param=1.5):
        
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

        # NOTE **very important** when evaluating the effects of word length, frequency, and predictability, 
        #   I must control the variable! Fix the other two and test the one.

        # Initialize the prior probabilities: word frequencies and contextual predictabilities
        # Initialize the word frequencies
        self.zipf_param = zipf_param

        # Other initialization
        self.train_test_words_data = None

        # Initialize the Approximate Word Generator to generate likely words
        self.approximate_word_generator = ApproximateWordGenerator()
    
    def reset(self):
        # Initialize the word frequencies
        self._initialize_fixed_frequencies()  # Precompute only for fixed lexicon

        # Initialize the word predictabilities
        self._initialize_fixed_predictabilities()  # Precompute only for fixed lexicon

        # Sample the training and testing words
        self.train_test_words_data = self.lexicon_300
    
    # def _initialize_fixed_frequencies(self):
    #     """Precompute Zipfian frequencies for words in the fixed lexicon."""
    #     # random.seed(42)
    #     ranks = np.arange(1, len(self.lexicon_300) + 1)
    #     zipf_probs = 1 / (ranks ** self.zipf_param)
    #     zipf_probs /= zipf_probs.sum()
    #     self.lexicon_w_pseudo_freq_prob = {word: prob for word, prob in zip(self.lexicon_300, zipf_probs)}

    def _initialize_fixed_frequencies(self):
        """Precompute Zipfian frequencies for words in the fixed lexicon (randomized order)."""
        random.seed()  # Ensure different shuffle each time

        shuffled_words = self.lexicon_300.copy()
        random.shuffle(shuffled_words)  # Shuffle words before assigning ranks

        ranks = np.arange(1, len(shuffled_words) + 1)  # Assign new ranks
        zipf_probs = 1 / (ranks ** self.zipf_param)
        zipf_probs /= zipf_probs.sum()  # Normalize to sum to 1

        self.lexicon_w_pseudo_freq_prob = {word: prob for word, prob in zip(shuffled_words, zipf_probs)}

    
    def _generate_dynamic_frequency(self, word):
        # """
        # Generate a frequency probability for a word NOT in the lexicon.
        # Instead of storing it, we dynamically compute it when needed.
        # """
        # word_hash = abs(hash(word)) % 1000  # Create a pseudo-rank from the word's hash
        # rank_approx = (word_hash % len(self.lexicon_300)) + 1  # Map to an approximate rank
        # freq = 1 / (rank_approx ** self.zipf_param)
        # return freq
        """
        Dynamically generate a frequency probability for a word NOT in the lexicon.
        Instead of computing from scratch, sample from existing values to match the dataset's scale.
        """
        sampled_existing_freq = random.choice(list(self.lexicon_w_pseudo_freq_prob.values()))  # Sample existing frequency
        noise_factor = np.random.uniform(0.8, 1.2)  # Add slight variation
        return max(np.clip(sampled_existing_freq * noise_factor, 0, 1), 0.0001)  # Avoid zero probability

    def get_word_frequency_probability(self, word):
        """Retrieve or dynamically generate the word's probability."""
        if word in self.lexicon_w_pseudo_freq_prob:
            return self.lexicon_w_pseudo_freq_prob[word]
        else:
            return self._generate_dynamic_frequency(word)

    def _initialize_fixed_predictabilities(self):
        """Precompute random predictabilities for words in the fixed lexicon."""
        # random.seed(42)
        self.lexicon_w_pseudo_pred_prob = {w: self.get_predictability(mode="random") for w in self.lexicon_300}
    
    def _generate_dynamic_predictability(self, word):
        # """
        # Generate a predictability probability for a word NOT in the lexicon.
        # Instead of storing it, we dynamically compute it when needed.
        # """
        # return self.get_predictability(mode="random")
        """
        Dynamically generate a predictability probability for a word NOT in the lexicon.
        Similar to frequency, we sample from the dataset's distribution.
        """
        sampled_existing_pred = random.choice(list(self.lexicon_w_pseudo_pred_prob.values()))  # Sample from existing
        noise_factor = np.random.uniform(0.9, 1.1)  # Slight variation
        return max(np.clip(sampled_existing_pred * noise_factor, 0, 1), 0.001)  # Avoid zero probability

    def get_word_predictability_probability(self, word):
        """Retrieve or dynamically generate the word's probability."""
        if word in self.lexicon_w_pseudo_pred_prob:
            return self.lexicon_w_pseudo_pred_prob[word]
        else:
            return self._generate_dynamic_predictability(word)
    
    def sample_train_test_words(self, num_words=20, mode="random"):
        """
        Sample a subset of words for training and testing
        """

        # Use the whole lexicon for training and testing
        self.train_test_words_data = self.lexicon_300
        return self.lexicon_300

    def get_top_k_words(self, sampled_letters_so_far_with_spaces, original_word, top_k):
        """
        Use the Approximate Word Generator to get the top-k words that contain the sampled letters so far.
        We generate similar words based on:
            - Rough word length
            - Sampled letters (must be contained in the words)
            - Sampled letters' rough positions within the word.
        
        NOTE: Worth mentioning this in the paper.

        :param sampled_letters_so_far_with_spaces: Sampled letters in space-separated format (e.g., "com hen")
        :param original_word: The actual target word
        :param top_k: Number of words to return
        :return: List of tuples (word, freq, pred, likelihood) where freq and pred are sampled
        """
        generated_words = self.approximate_word_generator.generate_similar_words(sampled_letters_so_far_with_spaces, original_word, top_k=top_k)

        results = []
        for w in generated_words:
            freq = self.get_word_frequency_probability(w)
            pred = self.get_word_predictability_probability(w)
            likelihood, correct_factor_alpha = self.get_likelihood_by_sampled_letters_so_far(sampled_letters_so_far_with_spaces, w, word_len=len(w))        
            results.append((w, freq, pred, likelihood, correct_factor_alpha))

        return results

    def get_likelihood_by_sampled_letters_so_far(self, sampled_letters_so_far, word, word_len, mode="positional"):
        """
        Computes P(w) using a dynamic weighting of prior vs. likelihood.

        Improvements:
            **Likelihood gradually dominates as more letters are sampled**.
            **Prior remains relevant initially, but fades with evidence**.
            **Maintains stability & prevents zero-probability errors**.

        New Formula:
            P(w) = Prior^(1-alpha) * Likelihood^alpha
            - α = (# sampled letters) / (word length)
            - Prior is **important at the start**, but **likelihood becomes dominant over time**.
        """

        # Convert sampled letters into list format (split by spaces for non-contiguous sampling)
        sampled_segments = sampled_letters_so_far.split()  # e.g., "omp hen" -> ["omp", "hen"]
        
        # Step 1: Count correctly placed and misplaced letters
        matched_count = 0
        misplaced_count = 0
        total_sampled = sum(len(segment) for segment in sampled_segments)  # Total sampled letters
        word_set = set(word)  # Quick lookup for misplaced letters

        decay_factor = 0.5  # Penalizes misplaced letters but still gives some probability

        for segment in sampled_segments:
            match_start = word.find(segment)  # Find position of segment in word
            if match_start != -1:
                matched_count += len(segment)  # Correctly placed letters
            else:
                misplaced_count += sum(1 for ch in segment if ch in word_set)  # Misplaced but present letters

        # Compute likelihood
        likelihood = (matched_count + decay_factor * misplaced_count) / word_len

        # Compute alpha: how much likelihood dominates (alpha increases with sampled letters)
        alpha = min(1.0, total_sampled / word_len)  # Ensure alpha is in [0,1]

        # # Compute final probability using prior-likelihood combination
        # combined_prob = (prior_prob ** (1 - alpha)) * (likelihood ** alpha)

        # Prevent strict zero probabilities
        # return max(combined_prob, 0.01)
        return likelihood, alpha

    
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


class ApproximateWordGenerator:
    def __init__(self, alphabet="abcdefghijklmnopqrstuvwxyz", variation_prob=0.3):
        """
        Initialize the word generator with an alphabet and variation probability.
        - `variation_prob`: Controls how often random modifications happen.

        NOTE: this function is only called when letters are sampled.
        """
        self.alphabet = list(alphabet)
        self.variation_prob = variation_prob  # Adjusts randomness level

    def generate_similar_words(self, sampled_letters, original_word, top_k=5):
        """
        Generate exactly `top_k` similar words.
        - Ensures sampled letters (contiguous or non-contiguous) remain in the word.
        - Keeps sampled letters in their approximate position.
        - Randomly modifies unobserved letters.
        - Always includes the original word.
        - Pads missing entries with synthetic words if fewer than `top_k` are generated.
        """
        word_length = len(original_word)  
        generated_words = {original_word}  # Use a set to avoid duplicates

        # 🔹 Identify the approximate positions of sampled letter chunks in the original word
        sampled_chunks = sampled_letters.split()  # Assumes spaces separate discontinuous parts
        chunk_positions = []

        # 🔹 Find where each chunk appears in the original word
        for chunk in sampled_chunks:
            pos = original_word.find(chunk)
            if pos == -1:
                pos = random.randint(0, max(1, word_length - len(chunk)))  # Approximate if not found
            chunk_positions.append((chunk, pos))

        while len(generated_words) < top_k:  # Ensure we get exactly `top_k` words
            word = list(original_word)  # Start with the original word
            new_word = [""] * word_length  # Empty template

            # 🔹 Insert each sampled chunk into its approximate position 🔹
            for chunk, pos in chunk_positions:
                pos = min(pos, word_length - len(chunk))  # Ensure position is within bounds
                new_word[pos:pos + len(chunk)] = list(chunk)

            # 🔹 Fill in the remaining positions randomly 🔹
            for i in range(word_length):
                if new_word[i] == "":  # If it's still empty, insert a letter
                    if random.random() > self.variation_prob:
                        new_word[i] = random.choice(self.alphabet)  # Substitute with a random letter
                    else:
                        new_word[i] = word[i] if i < len(word) else random.choice(self.alphabet)  # Keep some original letters
            
            # 🔹 Randomly insert/delete a letter to create slight length variation 🔹
            if len(new_word) > 3 and random.random() > 0.5:
                del new_word[random.randint(0, len(new_word) - 1)]

            if len(new_word) < 15 and random.random() > 0.5:
                new_word.insert(random.randint(0, len(new_word)), random.choice(self.alphabet))

            generated_words.add("".join(new_word))

        # Convert to list and ensure exactly `top_k` words
        generated_words = list(generated_words)
        
        # If we still don’t have enough words, generate dummy words
        while len(generated_words) < top_k:
            filler_word = "".join(random.choices(self.alphabet, k=word_length))  # Generate a random word
            if filler_word not in generated_words:
                generated_words.append(filler_word)

        return generated_words[:top_k]  # Ensure exactly `top_k` words


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
    
    NOTE: maybe the fixation sequence will not always go from left to right; because we do not have a phonological (sound) system. 
        And humans have prior knowledge / are trained to behave in that way.

    TODO: 1. Merge the freq and pred to a single prior parameter that is predifined (not predifined is also fine). 
        2. Change the obs's word representation to the sampled letters so far.
        3. The current papers are saying freq and pred's effect on processing the word, are these really factors that could affect the number of fixations?
        4. Maybe just randomize the words that is read, so get free from the lexicon size.
        5. TODO: do this, if the proior is useful, then express it before the first fixation sampling, when initializing the word, give it to the agent.
            Also need to re-design the word activation dynamics.

    Why Is the Agent Ignoring Prior Knowledge (Freq & Pred)?
        The prior (freq & pred) is multiplied, but the likelihood dominates too quickly.

        This means the agent learns reading (sampling letters) is the only reliable strategy.
        It does not learn to make educated guesses early based on prior knowledge.
        The penalty for excessive reading is too weak.

        The agent has no strong incentive to guess early.
        It keeps reading letters instead of using prior knowledge.
        Unclear if RL agent gets "reward shaping" for correct recognition using prior knowledge.

        If the agent is not explicitly rewarded for stopping early and activating the right word, it has no reason to do so.    

        NOTE: now the agent relys more on sampled letters since prior knowledge is less deterministic. 
        
    """

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
        self._top_k = 5
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
        self._sampled_letters_so_far_with_spaces = None    # The letters that have been sampled

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
    
    def reset(self, seed=None, inputs=None, ep_idx=None):
        """
        Reset the environment to the initial state
        """

        self._steps = 0
        self._truncated = False
        self._done = False
        self.log_cumulative_version = {}

        # Reset the lexicon
        self.lex_manager.reset()

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
        self._sampled_letters_so_far_with_spaces = ""

        # Sample the word to be recognized
        if inputs is not None:
            self._word = inputs["word"]
        else:
            self._word = self.lex_manager.get_word()
        self._word_len = len(self._word)
        self._word_freq_prob = self.lex_manager.get_word_frequency_probability(word=self._word)    # Only for training and testing, for actual simulation, need to use actual LLMs  NOTE: could set as controllable parameters
        self._word_predictability_prob = self.lex_manager.get_word_predictability_probability(word=self._word)    # Only for training and testing, for actual simulation, need to use actual LLMs   NOTE: could set as controllable parameters
        # self._word_likelihood_prob = self.lex_manager.get_likelihood_by_sampled_letters_so_far(
        #     sampled_letters_so_far=self._sampled_letters_so_far_with_spaces, word=self._word, mode="fractional"
        #     )
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
                
                self._sampled_letters_so_far_representation, self._sampled_letters_so_far_with_spaces = self.transition_function.update_state_sampled_letters_so_far(
                    action=action, norm_gt_word_rep=self._normalized_ground_truth_word_representation, 
                    seen_letters_representation=self._sampled_letters_so_far_representation, 
                    seen_letters=self._sampled_letters_so_far_with_spaces, word=self._word, word_len=self._word_len
                )

                assert self._sampled_letters_so_far_with_spaces != "NO_LETTER_SAMPLED", f"no letters sampled so far, the word is {self._word}, the action is {action}, the word length is {self._word_len}"
                
                self._normalized_belief_distribution_dict, self._normalized_belief_distribution, _ = self.transition_function.update_state_belief_distribution_dict(
                    sampled_letters_so_far_with_spaces=self._sampled_letters_so_far_with_spaces, word_to_be_recognized=self._word, lexicon_manager=self.lex_manager
                )

                reward = self.reward_function.get_step_wise_effort_cost(is_action_valid=True)
            
            else:   # The action is invalid, sampling nothing, doing nothing, wasting time
                
                reward = self.reward_function.get_step_wise_effort_cost(is_action_valid=False)

        else:   # Stop the sampling and recognize the word
            reward, self._done = self._terminate_step()

        if self._steps >= self.ep_len:     # Truncation case
            reward, self._done = self._terminate_step()
            self._truncated = True

        return self._get_obs(), reward, self._done, self._truncated, self._get_logs(is_initialization=False, mode=self._mode)

    def render(self, mode='human'):
        pass
    
    def _terminate_step(self):
        self._word_to_activate = self.transition_function.activate_a_word(
            normalized_belief_distribution_dict=self._normalized_belief_distribution_dict, deterministic=False
        ) 
            
        reward = self.reward_function.get_terminate_reward(
            word_to_recognize=self._word,
            word_to_activate=self._word_to_activate
        )

        done = True

        return reward, done

    def _get_obs(self):   
        """
        Get the current observation
        """

        # Encode the discrete action into a one-hot vector
        action_obs = np.zeros(self.MAX_WORD_LEN + 1 + 1)        # three types of actions -1, fixations, stop
        action_obs[self._action + 1] = 1

        stateful_obs = np.concatenate([self._normalized_belief_distribution, self._sampled_letters_so_far_representation, [self._word_len], action_obs])

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
                    "word_likelihood": self.lex_manager.get_likelihood_by_sampled_letters_so_far(
                        sampled_letters_so_far=self._sampled_letters_so_far_with_spaces, word=self._word, word_len=self._word_len
                        ),    # The likelihood probability: P(sampled letters so far | word)
                    "sampled_letters_so_far": self._sampled_letters_so_far_with_spaces,
                    "sampled_letters_so_far_representation": self._sampled_letters_so_far_representation.copy(),
                    "word_to_activate": self._word_to_activate,
                    "normalized_belief_distribution": self._normalized_belief_distribution.copy(),
                    "normalized_belief_distribution_dict": self._normalized_belief_distribution_dict.copy(),
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

    def update_state_sampled_letters_so_far(self, action, norm_gt_word_rep, seen_letters_representation, seen_letters, word, word_len):
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
        
        # # Update the seen letters -- get all letters that are not -1 in representation from word
        # # seen_letters = "".join([word[i] for i in range(len(word)) if seen_letters_representation[i] != -1])
        # seen_letters = "".join([word[i] for i in range(len(word)) if i < len(seen_letters_representation) and seen_letters_representation[i] != -1]) or "NON_WORDS"
        # # print(f"The action value is: {action}, left index: {left_index}, right index: {right_index}; the target word is: {word}, the word length is: {word_len}")
        # # print(f"Seen letters representation: {seen_letters_representation}, seen letters: {seen_letters}")

        # Handle both contiguous and non-contiguous letters
        seen_letters_list = []
        current_segment = []

        for i in range(word_len):
            if i < len(seen_letters_representation) and seen_letters_representation[i] != -1:   # Collect contiguous letters
                current_segment.append(word[i])  
            else:       # Collect non-contiguous letters
                if current_segment:  # If a segment exists, store it to the seen_letter_list of segments and start a new one
                    seen_letters_list.append("".join(current_segment))
                    current_segment = []
        
        # Add the last segment if it exists
        if current_segment:
            seen_letters_list.append("".join(current_segment))

        # Join segments with spaces
        seen_letters = " ".join(seen_letters_list) if seen_letters_list else cons.NO_LETTERS_SAMPLED

        return seen_letters_representation, seen_letters
    
    def update_state_belief_distribution_dict(self, sampled_letters_so_far_with_spaces, word_to_be_recognized, lexicon_manager: LexiconManager):
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
        words_freqs_preds_likelihoods_alphas = lexicon_manager.get_top_k_words(
            sampled_letters_so_far_with_spaces=sampled_letters_so_far_with_spaces, original_word=word_to_be_recognized, top_k=self._top_k
        )

        # def calc_belief(word_freq, word_pred, word_likelihood, correct_factor_alpha):
        #     prior = word_freq * word_pred
        #     likelihood = word_likelihood
        #     corrected_posterior = (prior ** (1 - correct_factor_alpha)) * (likelihood ** correct_factor_alpha)
        #     return corrected_posterior

        def calc_belief(word_freq, word_pred, word_likelihood, correct_factor_alpha):
            """
            Compute the final belief (posterior probability) with stronger likelihood dominance.
            
            - `correct_factor_alpha` dynamically increases as more letters are sampled.
            - The likelihood is **exponentiated more aggressively** to dominate priors faster.
            """

            word_freq = float(word_freq)  # Ensure numeric
            word_pred = float(word_pred)  # Ensure numeric
            word_likelihood = float(word_likelihood)  # Ensure numeric

            # Increase Likelihood Dominance as More Letters Are Sampled
            proportion_sampled = correct_factor_alpha  # % of word seen
            likelihood_strength = 1 + 10 * proportion_sampled  # Strengthen likelihood as more is sampled

            # Strengthen likelihood dominance over prior
            likelihood_corrected = (word_likelihood ** likelihood_strength) * 10  # Scaling to counteract prior imbalance

            # Apply Correcting Factor (Adjustable)
            corrected_posterior = (word_freq * word_pred) ** (1 - correct_factor_alpha) * (likelihood_corrected ** correct_factor_alpha)

            return corrected_posterior

        # Update the beliefs: prior * likelihood, where the prior is p(freq) * p(pred)
        belief_distribution_dict = {wfpl[0]: calc_belief(word_freq=wfpl[1], word_pred=wfpl[2], word_likelihood=wfpl[3], correct_factor_alpha=wfpl[4]) for wfpl in words_freqs_preds_likelihoods_alphas}
        # belief_distribution_dict_without_correction = {wfpl[0]: wfpl[1] * wfpl[2] * wfpl[3] for wfpl in words_freqs_preds_likelihoods_alphas}   

        # Normalize the beliefs -- we can do the softmax from this later
        normalized_belief_distribution_dict = {k: v / sum(belief_distribution_dict.values()) for k, v in belief_distribution_dict.items()}
        # normalized_belief_distribution_dict_without_correction = {k: v / sum(belief_distribution_dict_without_correction.values()) for k, v in belief_distribution_dict_without_correction.items()} 

        normalized_belief_distribution = list(normalized_belief_distribution_dict.values())

        return normalized_belief_distribution_dict, normalized_belief_distribution, words_freqs_preds_likelihoods_alphas

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
    env = WordActivationRLEnv()
    inputs = {"word": "watermelon"}

    for i in range(3):
        print(f"*********************************************************************")
        env.reset(inputs=inputs)
        env.step(1)
        env.step(5)
        env.step(9)
        env.step(8)
        env.step(3)
        env.step(15)

    # env = LexiconManager()
    # print(env.get_top_k_words("ell"))
    # print(env.get_top_k_words("tell"))

    # # Example usage:      NOTE: issues -- only output 'hel'
    # corpus = "hello help helmet hall held held hero heat heavy height hello jelly yellow mellow"
    # word_gen = MarkovWordGenerator(corpus)
    # print(f"The Markov model: {word_gen.model}")
    # for i in range(10):
    #     print(word_gen.generate_similar_word("hel"))  # Might generate "helton" or "helder"

    # # Example usage:     
    # word_gen = ApproximateWordGenerator()   

    # # Example: If sampled "com   hen" from "comprehensive"
    # original_word = "comprehensive"
    # sampled_letters = "com re ve"  # Two non-contiguous sampled parts
    # top_k = 5

    # print(f"Original word: {original_word}, Sampled letters: {sampled_letters}")
    # print(word_gen.generate_similar_words(sampled_letters, original_word, top_k=top_k))

    # # Load the FastText model once (global variable)
    # fasttext_model = fasttext.load_model("/home/baiy4/reader-agent-zuco/cc.en.300.bin")

    # def generate_similar_word_fasttext(prefix, k=5):
    #     """Find similar words in FastText word embeddings."""
    #     similar_words = fasttext_model.get_nearest_neighbors(prefix, k=k)
    #     return [word for _, word in similar_words]

    # # Example usage:
    # for i in range(10):
    #     print(generate_similar_word_fasttext("hel"))  # Outputs: ['hello', 'help', 'helium', 'helmet']

    # print(f"8888888888888888888888888888888888")

    # for i in range(10):     # NOTE: good, this works, but kind of slow, so maybe need to run everything beforehand offline. And also need to deal with special marks/symbols. 
    #     print(fasttext_model.get_nearest_neighbors("hel", k=5))