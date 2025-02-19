import random
import math
import numpy as np
from collections import Counter

from modules.rl_envs.word_activation_v0218.Utilities import ApproximateWordGenerator


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
        # TODO train with a random larger lexicon later, the words could be non-words, for generalizability

        # Initialize the prior probabilities: word frequencies and contextual predictabilities
        # Initialize the word frequencies
        self.zipf_param = zipf_param

        # Initialize the prior probability generator
        # NOTE: when testing freq and pred's effect, use a tunable parameter, say w_p, we could define prior = freq * w_p * pred; 
        #   Or we just report the freq's effect.
        self._prior_prob_generator = None
        self.weight_prior_pred_to_freq = 1.5   # This means prior = freq * pred, pred = freq * 1.5
        self.prior_dict = None                 # Store all the activated words' prior probabilities during one given episode

        # Other initialization
        self.train_test_words_data = None

        # Initialize the Approximate Word Generator to generate likely words
        self.approximate_word_generator = ApproximateWordGenerator()
    
    def reset(self):
        """
        Reset the lexicon manager

        This function is reset for every RL training episode
        """
        # # Initialize the word frequencies
        # self._initialize_fixed_frequencies()  # Precompute only for fixed lexicon

        # # Initialize the word predictabilities
        # self._initialize_fixed_predictabilities()  # Precompute only for fixed lexicon

        # Sample the training and testing words
        self.train_test_words_data = self.lexicon_300

        # Reset the prior dictionary
        self.prior_dict = {}
    
    @staticmethod
    def _generate_prior_probability():
        """
        """
        prior_prob = random.uniform(0, 1)
        return prior_prob

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
    
    def get_predictability(self, mode="random"):
        """
        Get the predictability of the word in the context
        """
        
        if mode == "random":
            return random.uniform(0, 1)
        else:
            # Use language models to get the predictability
            return 0.97
    
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

        assert original_word in self.prior_dict, f"the original word {original_word} is not in the prior dictionary"

        for w in generated_words:
            if w not in self.prior_dict:
                if w is not original_word:  # Randomize a prior probability for the original word (target word)
                    # # Method 1
                    # discount_factor = random.uniform(0.1, 0.5)  # Discount factor for non-target words   # TODO note this as a key assumption if used
                    # self.prior_dict[w] = self.prior_dict[original_word] * discount_factor
                    # Method 2
                    self.prior_dict[w] = self._generate_prior_probability()     # Randomize the prior probability   # NOTE: good good!
            prior = self.prior_dict[w]
            likelihood, correct_factor_alpha = self.get_likelihood_by_sampled_letters_so_far(
                sampled_letters_so_far=sampled_letters_so_far_with_spaces,
                candidate_word=w, original_word=original_word
            )        
            results.append((w, prior, likelihood, correct_factor_alpha))

        return results

    @staticmethod
    def levenshtein_distance(a: str, b: str) -> int:
        """
        Compute the Levenshtein distance between two strings a and b.
        This is the minimal number of single-character edits (insert, delete, substitute)
        needed to transform a into b.
        """
        n, m = len(a), len(b)
        dp = [[0]*(m+1) for _ in range(n+1)]

        # Initialize boundaries
        for i in range(n+1):
            dp[i][0] = i
        for j in range(m+1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,     # deletion
                    dp[i][j-1] + 1,     # insertion
                    dp[i-1][j-1] + cost # substitution
                )

        return dp[n][m]


    def get_likelihood_by_sampled_letters_so_far(
        self,
        sampled_letters_so_far: str,
        candidate_word: str,
        original_word: str,
        distance_scale: float = 0.5,
        misplaced_decay: float = 0.5
    ):
        """
        Computes a likelihood P(sampled_letters_so_far | candidate_word, original_word)
        by combining:
        1) Coverage of sampled letters in the candidate (duplicate-aware),
        2) Edit-distance penalty between candidate_word and original_word,
        3) alpha = #sampled_letters / len(original_word).

        Returns:
        (likelihood, alpha)

        -----------
        Explanation
        -----------

        1) We remove spaces from the sampled letters (e.g., "gr w" -> "grw") to count them properly.
        2) We compute a "coverage score" of how many sampled letters appear in candidate_word.
        - e.g., if we have "g", "r", "o" vs. candidate "grow", coverage ~ 3/3 = 1.0
        - partial credit for letters that appear but not in the correct positions can be added
            if you wish to track positions explicitly; for now we do a simpler approach.

        3) We measure how different candidate_word is from the original_word via Levenshtein distance.
        - dist_factor = exp(-distance_scale * distance).
            If distance is 0, factor=1, if distance is large, factor~0.

        4) Combine coverage score & dist_factor => coverage_score * dist_factor

        5) alpha = min(1, (#sampled_letters) / len(original_word))
        This indicates how strongly we trust the likelihood. If you eventually want to combine
        with a prior, do:
            combined_prob = (prior_prob)^(1-alpha) * (likelihood)^(alpha).

        6) Return (likelihood, alpha).

        -----------
        Parameters
        -----------
        sampled_letters_so_far : str
            The user-sampled letters, possibly containing spaces (e.g. "gr w").
        candidate_word : str
            The candidate word for which we want to compute the likelihood.
        original_word : str
            The actual (ground-truth) word, used to measure how far the candidate is from the real word.
        distance_scale : float
            Strength of the edit distance penalty. Larger = stronger penalty for bigger differences.
        misplaced_decay : float
            Partial credit factor for letters that exist in the candidate but exceed the count
            or are "extra" after coverage. Lower = bigger penalty for mismatch.

        -----------
        Returns
        -----------
        (likelihood: float, alpha: float)
        """

        # 1. Remove spaces in typed letters
        sample_str = sampled_letters_so_far.replace(" ", "")
        sample_len = len(sample_str)

        # 2. Count how many typed letters are in the candidate (accounting for duplicates)
        candidate_counts = Counter(candidate_word)
        sample_counts    = Counter(sample_str)

        matched_letters = 0
        misplaced_letters = 0

        for ch, s_count in sample_counts.items():
            c_count = candidate_counts.get(ch, 0)
            if s_count <= c_count:
                matched_letters += s_count
            else:
                matched_letters += c_count
                leftover = s_count - c_count
                misplaced_letters += leftover

        # --- IMPORTANT CHANGE HERE ---
        # coverage is based on candidate_word length, not just how many letters were typed
        candidate_len = len(candidate_word)
        if candidate_len > 0:
            coverage_score = (matched_letters + misplaced_decay * misplaced_letters) / candidate_len
        else:
            coverage_score = 0.0

        # 3. Edit distance penalty between candidate_word & original_word
        dist = self.levenshtein_distance(candidate_word, original_word)
        dist_factor = math.exp(-distance_scale * dist)

        # 4. Combine coverage with distance factor
        base_likelihood = coverage_score * dist_factor

        # 5. alpha = #typed / #original
        orig_len = len(original_word)
        if orig_len > 0:
            alpha = min(1.0, sample_len / orig_len)
        else:
            alpha = 1.0

        return base_likelihood, alpha

    def get_word(self):
        word_to_recognize = random.choice(self.train_test_words_data)
        # Reset the target word in the dictionary
        self.prior_dict[word_to_recognize] = self._generate_prior_probability()
        return word_to_recognize