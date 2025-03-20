import random
import math
import numpy as np
from collections import Counter

from modules.rl_envs.word_activation_v0218 import Constants
from modules.rl_envs.word_activation_v0218.Utilities import ApproximateWordGenerator


class LexiconManager():
    """
    Lexicon Manager
    """

    def __init__(self, zipf_param=1.5):
        
        # Initialize the type of prior
        self.prior_type = None

        # Initialize the prior probabilities: word frequencies and contextual predictabilities
        # Initialize the word frequencies
        self.raw_occurances, self.normalized_freqs = self.sample_pareto(
            n_words=Constants.NUM_WORDS_IN_LEXICON, alpha=Constants.ZIPF_PARAM_PARETO_ALPHA
        )

        # Initialize the prior probability generator
        # NOTE: when testing freq and pred's effect, use a tunable parameter, say w_p, we could define prior = freq * w_p * pred; 
        #   Or we just report the freq's effect.
        self._prior_prob_generator = None
        self.weight_prior_pred_to_freq = 1.5   # This means prior = freq * pred, pred = freq * 1.5
        self.prior_dict = None                 # Store all the activated words' prior probabilities during one given episode

        # # Other initialization
        # self.train_test_words_data = None

        # Initialize the Approximate Word Generator to generate likely words
        self.approximate_word_generator = ApproximateWordGenerator()
    
    def reset(self, prior_type=Constants.PRIOR_AS_PRED):
        """
        Reset the lexicon manager

        This function is reset for every RL training episode
        """
        # # Sample the training and testing words
        # self.train_test_words_data = self.lexicon_300

        # The type of prior probability
        self.prior_type = prior_type

        # Reset the prior dictionary
        self.prior_dict = {}
    
    def _generate_prior_probability(self):
        """
        Generate a random prior probability for a word. Could either be a frequency or predictability.
        Depends on the prior type input.
        """
        if self.prior_type == Constants.PRIOR_AS_FREQ:
            index_in_the_list = random.randint(0, len(self.normalized_freqs) - 1)
            raw_occurrance = self.raw_occurances[index_in_the_list]
            prior_prob = self.normalized_freqs[index_in_the_list]

        elif self.prior_type == Constants.PRIOR_AS_PRED:
            prior_prob = random.uniform(0, 1)
            raw_occurrance = 0
        else:
            raise ValueError(f"Invalid prior type: {self.prior_type}")
        return prior_prob, raw_occurrance

    def sample_pareto(self, n_words=1000000, alpha=1.0):
        """
        Samples frequencies from a Pareto (power-law) distribution:
        p(x) ~ (1/x^(alpha+1)) for x >= 1
        Then normalizes them to sum=1.
        The parameter 'alpha' controls how steep the tail is.
        """
        # 1) sample from Pareto, this is for computing the raw frequencies, serve as the denominator
        zipf_raw = np.random.pareto(alpha, size=n_words)

        # 2) shift so it starts ~ 1
        #    Pareto gives x >= 0, so we often do x+1 or something similar. 
        zipf_raw += Constants.XMIN

        self.zipf_raw_max = max(zipf_raw)
        self.zipf_raw_min = min(zipf_raw)

        # 3) normalize
        freqs = zipf_raw / zipf_raw.sum()

        # # 3) create a new list of raw words, but following a uniform distribution
        # # NOTE: this is for computing the prior probabilities, serve as the numerator. 
        # #       If not uniformly sampled, the training and testing would be unbalanced.
        # uniform_raw = np.random.uniform(self.zipf_raw_min, self.zipf_raw_max, size=n_words)
        
        # # 4) nomralize
        # freqs = uniform_raw / zipf_raw.sum()
        
        return zipf_raw, freqs
    
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
                    # TODO fix here, differentiate the type of prior probability
                    self.prior_dict[w], _ = self._generate_prior_probability()     # Randomize the prior probability   # NOTE: good good!
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
        misplaced_decay: float = 0.5,
        c: float = 0.618  # Decay constant for non-original words
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
        Updated Logic:  
        -----------  
        - If `candidate_word == original_word`, likelihood = progress (coverage).  
        - Otherwise, likelihood is scaled by (1 - c^(1 - progress)).  

        -----------  
        Parameters:  
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
            Partial credit factor for letters that exist in the candidate but exceed the count.  
        c : float  
            Decay constant for scaling non-original words' probabilities.  

        -----------  
        Returns:  
        -----------  
        (likelihood: float, alpha: float)  
        """

        # 1. Remove spaces in sampled letters
        sample_str = sampled_letters_so_far.replace(" ", "")
        sample_len = len(sample_str)

        # 2. Count how many sampled letters appear in the candidate
        candidate_counts = Counter(candidate_word)
        sample_counts = Counter(sample_str)

        matched_letters = 0
        misplaced_letters = 0

        for ch, s_count in sample_counts.items():
            c_count = candidate_counts.get(ch, 0)
            if s_count <= c_count:
                matched_letters += s_count
            else:
                matched_letters += c_count
                misplaced_letters += s_count - c_count

        # 3. Compute coverage score based on candidate word length
        candidate_len = len(candidate_word)
        if candidate_len > 0:
            coverage_score = (matched_letters + misplaced_decay * misplaced_letters) / candidate_len
        else:
            coverage_score = 0.0

        # 4. Compute progress for the original word
        orig_len = len(original_word)
        progress = min(1.0, sample_len / orig_len) if orig_len > 0 else 1.0

        # 5. If candidate is the original word, return progress as likelihood
        if candidate_word == original_word:
            likelihood = progress
        else:
            # # Method 1: calculate the distance between the candidate and the original word 
            # # Compute base likelihood using edit distance
            # dist = self.levenshtein_distance(candidate_word, original_word)
            # dist_factor = math.exp(-distance_scale * dist)
            # base_likelihood = coverage_score * dist_factor

            # # Apply (1 - c^(1 - progress))
            # likelihood = base_likelihood * (1 - c ** (1 - progress))

            # Method 2: directly apply progress factor onto the coverage -- A very simple one
            likelihood = coverage_score * (1 - progress)

        # 6. Compute alpha
        alpha = progress

        return max(Constants.EPSILON, likelihood), alpha  # Avoid zero likelihood

    def get_word(self):

        # word_to_recognize = random.choice(self.train_test_words_data)

        # Generate a random word
        # First, randomly choose the word length
        word_length = random.randint(Constants.MIN_WORD_LENGTH, Constants.MAX_WORD_LENGTH)
        # Then, generate a random word with the given length
        word_to_recognize = self.approximate_word_generator.generate_a_random_word(word_length=word_length)

        # Reset the target word in the dictionary
        self.prior_dict[word_to_recognize], raw_occurance_of_target_word = self._generate_prior_probability()
        return word_to_recognize, self.prior_dict[word_to_recognize], raw_occurance_of_target_word