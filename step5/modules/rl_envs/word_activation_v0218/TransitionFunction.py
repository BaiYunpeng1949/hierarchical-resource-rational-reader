import random
import numpy as np

from step5.utils import auxiliaries as aux
from step5.utils import constants as cons

from modules.rl_envs.word_activation_v0218 import Constants
from modules.rl_envs.word_activation_v0218.LexiconManager import LexiconManager


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

    def reset_state_normalized_belief_distribution(self):
        return np.ones(self._top_k) / self._top_k

    def reset_state_word_representation(self):
        return -1 * np.ones(self.MAX_WORD_LEN)
    
    def get_normalized_ground_truth_word_representation(self, target_word):
        gt_word_rep = [ord(c) for c in target_word]
        norm_gt_word_rep = [aux.normalise(w, self.MIN_ORD, self.MAX_ORD, -1, 1) for w in gt_word_rep] 

        # Pad the representation to the max length, non letter positions encode as -1
        if len(norm_gt_word_rep) < self.MAX_WORD_LEN:
            norm_gt_word_rep += [-1] * (self.MAX_WORD_LEN - len(norm_gt_word_rep))
        
        return norm_gt_word_rep

    def update_state_sampled_letters_so_far_include_non_contiguous_letters(
            self, action, norm_gt_word_rep, seen_letters_representation, seen_letters, word, word_len
        ):
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
    
    def update_state_normalized_belief_distribution_dict(
            self, 
            sampled_letters_so_far_with_spaces, 
            word_to_recognize, 
            parallelly_activated_words_beliefs_dict,
            lexicon_manager: LexiconManager
        ):
        """
        Update the belief distribution -- Following work: The Bayesian Reader
        p(w_i|sampled letters so far) = p(w_i) * p(sampled letters so far|w_i) / Sigma_w' p(w') * p(sampled letters|w'), where p(w_i) it the prior; 
            in the initial step, it is a joint probability of the frequency and predictability of the word; in the following steps, 
            it is the last step's posterior probability
        
        NOTE: assumption: the top k words are only activated once, so we do not need to dynamically calculating them every fixation. 
            Such simplification is reasonable because human are limited in the STM buffer, and the number of words they can activate at the same time.
            Technically speaking, this is beneficial in terms of computational efficiency.
        
        TODO: debug this, see whether them making sense, 
            if not, we can apply Bayesian Reader's multi-dim space and Gaussian-based likelihood calculation
        Not making sense in terms of the word length's effect
        """
        
        norm_belief_distribution_list = []

        words_prior_dict = {}

        # Do the Bayesian updating
        first_activated_word = next(iter(parallelly_activated_words_beliefs_dict))  # Get the first activated word
        if first_activated_word == Constants.NON_WORD:  # If this is the first valid fixation, i.e., sampling letter information, then use fixed freq and preds as priors
            words_priors_likelihoods_alphas = lexicon_manager.get_top_k_words(
                sampled_letters_so_far_with_spaces=sampled_letters_so_far_with_spaces, original_word=word_to_recognize, top_k=self._top_k
            )  
            words_init_priors_dict = {wpla[0]: wpla[1] for wpla in words_priors_likelihoods_alphas}
            words_norm_init_priors_dict = {k: v / sum(words_init_priors_dict.values()) for k, v in words_init_priors_dict.items()}
            words_likelihoods_dict = {wpla[0]: wpla[2] for wpla in words_priors_likelihoods_alphas}
            words_norm_belief_dict = self._bayesian_inference(
                words_norm_beliefs_dict=None, words_likelihood_dict=words_likelihoods_dict, words_norm_freq_dict=words_norm_init_priors_dict
            )
            words_prior_dict = words_norm_init_priors_dict
        else:   # If this is not the first valid fixation, i.e., sampling letter information, then use the last step's posterior as priors
            # Get the likelihoods of the activated words
            words_likelihoods_dict = {w: lexicon_manager.get_likelihood_by_sampled_letters_so_far(sampled_letters_so_far=sampled_letters_so_far_with_spaces, candidate_word=w, original_word=word_to_recognize)[0] for w in parallelly_activated_words_beliefs_dict}

            # Bayesian update
            words_norm_belief_dict = self._bayesian_inference(
                words_norm_beliefs_dict=parallelly_activated_words_beliefs_dict, words_likelihood_dict=words_likelihoods_dict, 
                words_norm_freq_dict=None
            )

            words_prior_dict = parallelly_activated_words_beliefs_dict.copy()
        
        # Get the norm belief distribution as a list
        norm_belief_distribution_list = [words_norm_belief_dict[w] for w in words_norm_belief_dict]

        # NOTE: last time the agent could not learn well maybe because the agent does not know the belief's corresponding words, which is big and small. 
        #   Maybe think about how to encode the dictionary into the agent's observation
        return words_prior_dict, words_norm_belief_dict, norm_belief_distribution_list, words_likelihoods_dict
        
    def _bayesian_inference(self, words_norm_beliefs_dict=None, words_likelihood_dict=None, words_norm_freq_dict=None):
        """
        Bayesian Inference

        return: the normalized posteriors as belief distribution
        """

        # Initialize the posterior dictionary
        words_posteriors_dict = {}

        # Calculate the bayesian inference
        if words_norm_freq_dict is not None:     # For the first fixation, we use the fixed freq and preds as priors
            for word in words_norm_freq_dict:
                words_posteriors_dict[word] = words_norm_freq_dict[word] * words_likelihood_dict[word]
        else:   # For the following fixations, we use the last step's posterior as priors
            assert words_norm_beliefs_dict is not None, "The last step's belief distribution should be provided for the Bayesian inference."
            for word in words_norm_beliefs_dict:
                words_posteriors_dict[word] = words_norm_beliefs_dict[word] * words_likelihood_dict[word] 
        
        # Normalize the posteriors
        words_norm_posteriors_dict = {k: v / sum(words_posteriors_dict.values()) for k, v in words_posteriors_dict.items()}

        return words_norm_posteriors_dict
    
    # def calculate_fixation_duration_in_ms_nonlinear(self, t0=250, lamda=1.0, entropy_diff=0.0):
    #     """
    #     Calculate the fixation duration in milliseconds with gamma-distributed noise
    #     NOTE Based on EMMA's assumption and Reichle et al., 1998
    #     """
    #     # Calculate mean duration
    #     # mean_duration = t0 * (1 + np.exp(-lamda * entropy_diff))
    #     entropy_change_magnitude = np.abs(entropy_diff)
    #     mean_duration = t0 * (1 + lamda * entropy_change_magnitude)
        
    #     # Set standard deviation to 1/3 of mean (following EMMA)
    #     std_dev = mean_duration / 3
        
    #     # Calculate gamma distribution parameters
    #     # For gamma dist: mean = k*theta, var = k*theta^2
    #     # where k is shape, theta is scale
    #     theta = (std_dev ** 2) / mean_duration  # scale
    #     k = mean_duration / theta               # shape
        
    #     # Sample from gamma distribution
    #     duration = np.random.gamma(k, theta)
        
    #     return duration

    # def calculate_saccade_duration_in_ms_emma(self, delta_visual_angle_in_degree=2.0):
    #     """
    #     Calculate the saccade duration in milliseconds (T_exec in EMMA)
    #     NOTE: the gaze duration should not include any forms of saccade lengths
    #     """
    #     return 50 + 20 + delta_visual_angle_in_degree * 2
    
    @staticmethod
    def calc_fixation_duration_ms(
        entropy_diff: float,
        t_processing_baseline: float = 200,
        kappa: float = 3.75,
        shape: float = 2.0,
        v_min: float = 200.0,
        v_max: float = 250.0
    ) -> float:
        """
        Sample a single fixation duration (ms) from a Gamma distribution.

        Parameters
        ----------
        entropy_diff : float
            Absolute change in entropy / surprisal produced by this fixation.
        t0 : float
            Baseline fixation (~150=170 ms in human readers).   # TODO we could make this something else, like prep time, encoding time, and processing time stuff; not using a wrapped gaze duration time
        kappa : float
            Additive time cost (ms) per bit of |Δentropy| or surprisal.
        shape : float
            Gamma shape parameter (1-3 keeps the heavy right tail).
        
        Two-piece model:
            visual/lexical  ~ Gamma
            + constant 40 ms non-labile motor stage

        Returns
        -------
        float
            Fixation duration in milliseconds (includes visual processing +
            saccade motor-programming time, but **not** the eye-in-flight time).
        """
        mean_vis = np.clip(t_processing_baseline + kappa * entropy_diff, v_min, v_max)
        scale = mean_vis / shape
        t_visual_lex = np.random.gamma(shape, scale)
        return t_visual_lex
    
    def calc_gaze_duration_ms(self, entropy_diffs, **fix_kwargs) -> float:
        """
        Sum first-pass fixations on a word.  *Do not* add intra-word saccades.

        Parameters
        ----------
        entropy_diffs : Iterable[float]
            Sequence of entropy/surprisal reductions for each fixation the agent
            makes before it leaves the word.
        **fix_kwargs
            Forwarded to `fixation_duration_ms` for easy tuning.

        Returns
        -------
        float
            Gaze duration in milliseconds.
        """

        if len(entropy_diffs) > 0:
            return sum(self.calc_fixation_duration_ms(d, **fix_kwargs) for d in entropy_diffs)
        else:
            return 0
    
    def calc_individual_saccade_duration_ms(self) -> float:
        """
        Calculate the individual saccade duration in milliseconds.
        Following EZReader's implementation, set it as a fixed value at 25 ms.
        """
        return 25
    
    def calc_total_saccade_duration_ms(self, entropy_diffs) -> float:
        """
        Calculate the total saccade duration in milliseconds.
        It includes the number of fixations' saccade durations, one of them is leaving the word.
        """
        if len(entropy_diffs) > 0:
            return sum(self.calc_individual_saccade_duration_ms() for _ in entropy_diffs)
        else:
            return 0

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
        if deterministic:   # The greedy approach -- NOTE: this is the one Bayesian Reader applies -- NOTE: but this might cause the agent does no trust belief and will execute to the last step; maybe we could feed-in the history of belief distribution, demonstrating that the agent is gaining evidence correctly (the target word's belief should be rising)
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