import numpy as np
import yaml
import os
import math

from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete, Tuple

from modules.rl_envs.word_activation_v0218 import Constants

from modules.rl_envs.word_activation_v0218.TransitionFunction import TransitionFunction
from modules.rl_envs.word_activation_v0218.RewardFunction import RewardFunction
from modules.rl_envs.word_activation_v0218.LexiconManager import LexiconManager


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
    
    NOTE: 
        1. according to SWIFT, both frequency and predictability are contributing to the word recognition and the predictability is happening 
            earlier than freqeuncy's effect.
        2. Both of them could serve as parts of prior, but not neither simply adding or multiplying together.
        3. In the paper "Length, frequency, and predictability effects of words on eye movements in reading" they use CELEX Frequency Norms to obtain
            their corpus's word frequency (as log frequency); and use human to lable words' predictability (as logit predictability). So we could map their 
            data and our pseudo priors ranges from 0 to 1 independently and accordingly. Because in the paper's study, they intentionally use words 
            that could independently analyze the words' freqeuncy and predictability's effects.
        
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
    
    TODO: check Competition Models (McClelland & Rumelhart, 1981)
    Some justifications given by gpt, need to verify:
        1.  Interactive Activation Model (McClelland & Rumelhart, 1981)
            Word recognition is parallel → multiple words get activated at once.
            Competing words are those that share the same prefix (because they match early letters).
        2.  Lexical Competition (Marslen-Wilson, 1987)
            As more letters are sampled, the likelihood of a single word grows.
            Competing words drop out of the race based on frequency + predictability.
    
    Why Should We Lower Competing Words' Priors?
        1.  Predictability suppresses competitors
            If "coffee" is highly expected (due to context), then "coffer" and "cofounder" should have much lower priors.
            This models real-world prediction-based reading where humans don't fixate equally on all words.
        2.  Better reinforcement learning signal (I first thought of this, then trying to find the cognitive perspective evidence/justifications)
        3.  Human Eye-Tracking Evidence (Rayner, 1998)
            Readers skip over predictable words faster.
            Competing words are only considered when predictability is low.
    
    NOTE: now the issue is, the word cannot easily revive from a very low init prior, even though short and could be easily fully traversed.
    """

    def __init__(self):
        
        # Get the current root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        # Get the mode from the config yaml file
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        print(f"Word Activation (No Vision) Environment V0218 -- Deploying the environment in the {self._config['rl']['mode']} mode.")

        self._mode = self._config["rl"]["mode"]

        # Define constants -- configurations
        # Define word lengths
        self.MAX_WORD_LEN = 15
        self.MIN_WORD_LEN = 1
        # Define the top-k candidates when competing for recognition
        self._top_k = Constants.WORKING_MEMORY_SIZE     # Set as five for the STM buffer's limitation
        # Define the foveal vision size
        self._foveal_size = Constants.FOVEAL_SIZE       # one letter on the left, one in the middle, one on the right side  

        # Initialize necessary classes
        # Initialize the transition function
        self.transition_function = TransitionFunction(max_word_len=self.MAX_WORD_LEN, num_top_k_candidates=self._top_k, foveal_size=self._foveal_size)

        # Internal belief of the agent: belief over the top-k words, empirically set to a constant
        self._normalized_belief_distribution_parallel_activation_with_k_words = self.transition_function.reset_state_normalized_belief_distribution()      # The belief distribution has to be normalized, sumed up to 1
        self._normalized_belief_distribution_dict_parallel_activation_with_k_words = None

        # Prior type: frequency and predictability
        #   Since our prior could represent either frequency or predictability, we need to 
        #   differentiate them because they have very differnet distributions, for e.g., predictability is uniform, 
        #   but frequency follows the Zipf's law
        self._prior_type = None     # 0 stands for frequency, 1 stands for predictability
        
        # Initialize the prior dictionary across candidate words
        self._prior_distribution_dict_parallel_activation_with_k_words = None
        # Initialize the likelihood dictionary across candidate words
        self._likelihood_dict_parallel_activation_with_k_words = None

        self._word = None           # The word to be recognized
        self._word_len = None       # The length of the word to be recognized
        self._word_freq_prob = None      # The frequency of the word to be recognized -- ranges from 0 to 1
        self._word_predictability_prob = None    # The predictability of the word to be recognized (actually the likelihood prob) -- ranges from 0 to 1
        self._word_dynamic_predictability_prob = None    # The dynamic predictability of the word to be recognized (actually the likelihood prob) -- ranges from 0 to 1, it changes as the agent samples new letters
        self._word_prior_prob = None     # The prior probability of the word to be recognized -- ranges from 0 to 1, which is a combination of the frequency and predictability
        self._sampled_letters_so_far_with_spaces = None    # The letters that have been sampled

        # Entropy
        self._previous_step_entropy = None
        self._current_step_entropy = None
        self._entropy_diff = None

        # Representations
        self._word_representation = self.transition_function.reset_state_word_representation()    # The word representation, it stores the sampled letters. Here is a vector of letters sampled from the ASCII space
        self._normalized_ground_truth_word_representation = None

        self._sampled_letters_so_far_representation = None   # The letters that have been sampled

        # Temporal variables
        self._current_fixation_duration = None      # Unit is milliseconds, the time spent for a single fixation
        self._individual_fixations_durations_list = None    # The list of individual fixation durations
        self._current_saccade_duration = None       # Unit is milliseconds, the time spent for a single saccade
        self._individual_saccades_durations_list = None     # The list of individual saccade durations
        self._gaze_duration = None      # Unit is milliseconds, the total time spent for a first-pass of a word, including both fixation durations and saccade durations

        # Fixtion to duration mapping non-linear equation parameters
        self._t_0 = Constants.DEFAULT_FIXATION_DURATION    # The default average fixation duration, unit is milliseconds
        self._lamda = Constants.GAZE_DURATION_LAMDA    # The decay rate, the larger the value, the faster the decay

        # Define the word that is recognized
        self._word_to_activate = None

        # Define the action 
        self._action = None
        
        # Define the action space: 
        self.action_space = Discrete(self.MAX_WORD_LEN + 1)    # 0-9: fixate on the letter at the position, 10: stop the sampling and recognize the word

        # Define the observation space:
        self.STATEFUL_OBS = "stateful_obs"
        self.ACTION_OBS = "action_obs"
        self._num_stateful_obs = len(self._normalized_belief_distribution_parallel_activation_with_k_words) + len(self._word_representation) + 1 + (self.MAX_WORD_LEN + 1 + 1) + 1 # Belief distribution, word representation with sampled letters, word length, prior type
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

        # Initialize the prior's type       # TODO: set this as a controllable parameter later
        self._prior_type = np.random.choice([Constants.PRIOR_AS_FREQ, Constants.PRIOR_AS_PRED])

        # Reset the lexicon
        self.lex_manager.reset(prior_type=self._prior_type)

        # Initialize the action
        self._action = -1

        # Reset the belief distribution
        non_word = Constants.NON_WORD
        self._normalized_belief_distribution_dict_parallel_activation_with_k_words = {non_word: 0.20, non_word + '-1': 0.20, non_word + '-2': 0.20, non_word + '-3': 0.20, non_word + '-4': 0.20}
        self._normalized_belief_distribution_parallel_activation_with_k_words = self.transition_function.reset_state_normalized_belief_distribution()

        # Reset the likelihodd distribution dictionary
        self._likelihood_dict_parallel_activation_with_k_words = {non_word: 0.20, non_word + '-1': 0.20, non_word + '-2': 0.20, non_word + '-3': 0.20, non_word + '-4': 0.20}

        # Reset the prior distribution dictionary
        self._prior_distribution_dict_parallel_activation_with_k_words = {non_word: 0.20, non_word + '-1': 0.20, non_word + '-2': 0.20, non_word + '-3': 0.20, non_word + '-4': 0.20}

        # Reset the entropy
        self._previous_step_entropy = self._calculate_entropy(probability_distribution=self._normalized_belief_distribution_parallel_activation_with_k_words)

        self._current_step_entropy = self._previous_step_entropy
        self._entropy_diff = 0

        # Reset the word representation
        self._word_representation = self.transition_function.reset_state_word_representation()

        # Temporal variables
        self._current_fixation_duration = 0
        self._individual_fixations_durations_list = []
        self._current_saccade_duration = 0
        self._individual_saccades_durations_list = []
        self._gaze_duration = 0

        # Reset the seen letters
        self._sampled_letters_so_far_representation = [-1] * self.MAX_WORD_LEN
        self._sampled_letters_so_far_with_spaces = ""

        # Sample the word to be recognized
        if inputs is not None:
            self._word = inputs["word"]
            self._word_prior_prob = self.lex_manager.prior_dict[self._word] 
            self._raw_occurance = inputs["raw_occurance"]   # TODO, if use, fix
        else:
            self._word, self._word_prior_prob, self._raw_occurance = self.lex_manager.get_word()

        self._word_len = len(self._word)
        # self._word_prior_prob = self.lex_manager.prior_dict[self._word] 

        # weight_prior_pred_to_freq = self.lex_manager.weight_prior_pred_to_freq    # NOTE we separetely analyze word frequency and predictability's effect
        # self._word_freq_prob = math.sqrt(self._word_prior_prob / weight_prior_pred_to_freq)    # Only for training and testing, for actual simulation, need to use actual LLMs  NOTE: could set as controllable parameters
        # self._word_predictability_prob = self._word_freq_prob * weight_prior_pred_to_freq    # Only for training and testing, for actual simulation, need to use actual LLMs   NOTE: could set as controllable parameters
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

            if action <= self._word_len - 1:    # The action is a valid fixation, sampling letters
                
                self._sampled_letters_so_far_representation, self._sampled_letters_so_far_with_spaces = self.transition_function.update_state_sampled_letters_so_far_include_non_contiguous_letters(
                    action=action, norm_gt_word_rep=self._normalized_ground_truth_word_representation, 
                    seen_letters_representation=self._sampled_letters_so_far_representation, 
                    seen_letters=self._sampled_letters_so_far_with_spaces, word=self._word, word_len=self._word_len
                )

                assert self._sampled_letters_so_far_with_spaces != "NO_LETTER_SAMPLED", f"no letters sampled so far, the word is {self._word}, the action is {action}, the word length is {self._word_len}"
                
                self._prior_distribution_dict_parallel_activation_with_k_words, self._normalized_belief_distribution_dict_parallel_activation_with_k_words, self._normalized_belief_distribution_parallel_activation_with_k_words, self._likelihood_dict_parallel_activation_with_k_words = self.transition_function.update_state_normalized_belief_distribution_dict(
                    sampled_letters_so_far_with_spaces=self._sampled_letters_so_far_with_spaces, word_to_recognize=self._word, 
                    parallelly_activated_words_beliefs_dict=self._normalized_belief_distribution_dict_parallel_activation_with_k_words,
                    lexicon_manager=self.lex_manager
                ) 

                # Calculate the entropy change
                # self._current_step_entropy = self._calculate_entropy(probability_distribution=self._normalized_belief_distribution_parallel_activation_with_k_words)
                # self._entropy_diff = self._previous_step_entropy - self._current_step_entropy
                # self._previous_step_entropy = self._current_step_entropy
                self._calculate_entropy_diff()

                # Get the fixation durations
                # self._current_fixation_duration = self.transition_function.calculate_fixation_duration_in_ms_nonlinear(
                #     lamda=self._lamda, t0=self._t_0, entropy_diff=self._entropy_diff
                # )
                # self._individual_fixations_durations_list.append(self._current_fixation_duration)
                # # Get the saccade durations
                # self._current_saccade_duration = self.transition_function.calculate_saccade_duration_in_ms_emma()
                # self._individual_saccades_durations_list.append(self._current_saccade_duration)
                # # Update the gaze duration
                # segment_duration = self._current_fixation_duration + self._current_saccade_duration
                # self._gaze_duration += segment_duration
                # # self._gaze_duration += self._current_fixation_duration
                self._calculate_gaze_duration()

                # Get the reward
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
            normalized_belief_distribution_dict=self._normalized_belief_distribution_dict_parallel_activation_with_k_words, deterministic=Constants.DETERMINISTIC_WORD_ACTIVATION
        )
            
        reward = self.reward_function.get_terminate_reward(
            word_to_recognize=self._word,
            word_to_activate=self._word_to_activate
        )       

        done = True

        return reward, done
    
    @staticmethod
    def _calculate_entropy(probability_distribution):
        """
        Calculate the entropy of a probability distribution
        """
        entropy = 0
        for prob in probability_distribution:
            if prob > 0:
                entropy -= prob * math.log(prob)
        return entropy

    def _get_obs(self):   
        """
        Get the current observation
        """

        # Encode the discrete action into a one-hot vector
        action_obs = np.zeros(self.MAX_WORD_LEN + 1 + 1)        # three types of actions -1, fixations, stop
        action_obs[self._action + 1] = 1

        stateful_obs = np.concatenate([self._normalized_belief_distribution_parallel_activation_with_k_words, self._sampled_letters_so_far_representation, [self._word_len], action_obs, [self._prior_type]])

        assert len(stateful_obs) == self._num_stateful_obs, f"expected {self._num_stateful_obs} but got {len(stateful_obs)}"

        return stateful_obs

    def _calculate_entropy_diff(self):
        """
        Calculate the entropy difference
        """
        self._current_step_entropy = self._calculate_entropy(probability_distribution=self._normalized_belief_distribution_parallel_activation_with_k_words)
        self._entropy_diff = self._previous_step_entropy - self._current_step_entropy
        self._previous_step_entropy = self._current_step_entropy


    def _calculate_gaze_duration(self):
        """
        Calculate the gaze duration
        """
        self._current_fixation_duration = self.transition_function.calculate_fixation_duration_in_ms_nonlinear(
                    lamda=self._lamda, t0=self._t_0, entropy_diff=self._entropy_diff
                )
        self._individual_fixations_durations_list.append(self._current_fixation_duration)
        # Get the saccade durations
        self._current_saccade_duration = self.transition_function.calculate_saccade_duration_in_ms_emma()
        self._individual_saccades_durations_list.append(self._current_saccade_duration)
        # Update the gaze duration
        segment_duration = self._current_fixation_duration + self._current_saccade_duration
        self._gaze_duration += segment_duration

    
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
                    "prior_type": self._prior_type,     # Used for analyzing the prior's effect
                    "word_prior_prob": self._word_prior_prob,     # Used for analyzing the prior's effect
                    "occurance": self._raw_occurance,     # Used for analyzing the frequency's effect
                    "word_frequency": self._word_freq_prob,     # Used for analyzing the frequency's effect
                    "word_predictability": self._word_predictability_prob,    # Used for analyzing the predictability's effect
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
                        sampled_letters_so_far=self._sampled_letters_so_far_with_spaces, candidate_word=self._word, original_word=self._word
                        ),    # The likelihood probability: P(sampled letters so far | word)
                    "sampled_letters_so_far": self._sampled_letters_so_far_with_spaces,
                    "sampled_letters_so_far_representation": self._sampled_letters_so_far_representation.copy(),
                    "word_to_activate": self._word_to_activate,
                    "prior_distribution_dict": self._prior_distribution_dict_parallel_activation_with_k_words.copy(),
                    "likelihood_distribution_dict": self._likelihood_dict_parallel_activation_with_k_words.copy(),
                    "normalized_belief_distribution_dict": self._normalized_belief_distribution_dict_parallel_activation_with_k_words.copy(),
                    "normalized_belief_distribution": self._normalized_belief_distribution_parallel_activation_with_k_words.copy(),
                    "current_step_entropy": self._current_step_entropy,
                    "entropy_diff": self._entropy_diff,
                    "current_fixation_duration": self._current_fixation_duration,
                    "individual_fixations_durations_list": self._individual_fixations_durations_list.copy(),
                    "gaze_duration": self._gaze_duration,
                    "accurate_recognition": self._word_to_activate == self._word if self._done else None
                })
                return self.log_cumulative_version
            

if __name__ == "__main__":

    lex_manager = LexiconManager()
    print(lex_manager.get_likelihood_by_sampled_letters_so_far(
        sampled_letters_so_far="gro", candidate_word="grow", original_word="grow"
    ))
    print(lex_manager.get_likelihood_by_sampled_letters_so_far("gro", "gro", "grow"))
    print(lex_manager.get_likelihood_by_sampled_letters_so_far("sil", "silk", "silk"))
    print(lex_manager.get_likelihood_by_sampled_letters_so_far("sil", "ssil", "silk"))
    print(lex_manager.get_likelihood_by_sampled_letters_so_far("si k", "silk", "silk"))