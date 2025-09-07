import numpy as np
import yaml
import os
import math

from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete, Tuple

# from modules.rl_envs.word_activation_v0218 import Constants
# from modules.rl_envs.word_activation_v0218.TransitionFunction import TransitionFunction
# from modules.rl_envs.word_activation_v0218.RewardFunction import RewardFunction
# from modules.rl_envs.word_activation_v0218.LexiconManager import LexiconManager

from . import Constants
from .TransitionFunction import TransitionFunction
from .RewardFunction import RewardFunction
from .LexiconManager import LexiconManager


class WordRecognitionEnv(Env):
    """
    Vectorized word recognition RL Environment

    Model capabilities: recognize any alphabetic language word, out put number of fixations, place of fixations, gaze duration, and saccade durations.
    """

    def __init__(self):
        
        # Get the current root directory
        # Load configuration
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
        self._mode = self._config["rl"]["mode"]

        assert self._mode in ['simulate', 'debug', 'test', 'train', 'continual_train'], f"Invalid mode: {self._mode} !!!!" 

        print(f"Word Activation (No Vision) Environment V0807 (orginated from V0218) -- Deploying the environment in the {self._config['rl']['mode']} mode.")

        # Define constants -- configurations
        # Define word lengths
        self.MAX_WORD_LEN = Constants.MAX_WORD_LEN
        self.MIN_WORD_LEN = Constants.MIN_WORD_LEN
        # Define the top-k candidates when competing for recognition
        self._top_k = Constants.WORKING_MEMORY_SIZE     # Set as five for the STM buffer's limitation
        # Define the foveal vision size
        self._foveal_size = Constants.FOVEAL_SIZE       # one letter on the left, one in the middle, one on the right side  

        # Initialize necessary classes
        # Initialize the transition function
        self.transition_function = TransitionFunction(max_word_len=self.MAX_WORD_LEN, num_top_k_candidates=self._top_k, foveal_size=self._foveal_size)
        # Initialize the reward function
        self.reward_function = RewardFunction()
        # Define the training and testing data (temporary, in the formal training deploy separately)
        self.lex_manager = LexiconManager(mode=self._mode)

        # Internal belief of the agent: belief over the top-k words, empirically set to a constant
        self._normalized_belief_distribution_parallel_activation_with_k_words = self.transition_function.reset_state_normalized_belief_distribution()      # The belief distribution has to be normalized, sumed up to 1
        self._normalized_belief_distribution_dict_parallel_activation_with_k_words = None

        # Prior type: frequency and predictability
        #   Since our prior could represent either frequency or predictability, we need to 
        #   differentiate them because they have very differnet distributions, for e.g., predictability is uniform, 
        #   but frequency follows the Zipf's law
        self._prior_type = None     # 0 stands for frequency, 1 stands for predictability  # NOTE: need to look into this, should I set freq or predictability as the prior? or a joint value?
        
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
        # Define the word that is recognized
        self._word_to_activate = None

        # Entropy (describes the information gain and uncertainty reduction)
        self._previous_step_entropy = None
        self._current_step_entropy = None
        self._entropy_diff = None
        self._entropy_diffs_list = None

        # Representations
        self._word_representation = self.transition_function.reset_state_word_representation()    # The word representation, it stores the sampled letters. Here is a vector of letters sampled from the ASCII space
        self._normalized_ground_truth_word_representation = None
        self._sampled_letters_so_far_representation = None   # The letters that have been sampled

        # Temporal variables
        self._gaze_duration_for_this_word = None            # Unit is milliseconds, the total time spent for a first-pass of a word, only including the fixation durations
        self._sum_saccade_duration_for_this_word = None     # Unit is milliseconds, the total time spent for all the saccades when reading the word
        self._total_time_cost_for_this_word = None          # Unit is milliseconds, the totall time spent for everything, inflated by overhead components (realized by an inflation percentage).

        # Define the action 
        self._action = None
        # Define the action space: 
        self.action_space = Discrete(self.MAX_WORD_LEN + 1)    # 0-14: fixate on the letter at the position, 15: stop the sampling and recognize the word

        # Define the observation space:
        self.STATEFUL_OBS = "stateful_obs"
        self.ACTION_OBS = "action_obs"
        self._num_stateful_obs = len(self._normalized_belief_distribution_parallel_activation_with_k_words) + len(self._word_representation) + 1 + (self.MAX_WORD_LEN + 1 + 1) + 1 # Belief distribution, word representation with sampled letters, word length, prior type
        self.observation_space = Box(low=-1, high=1, shape=(self._num_stateful_obs,))
        
        # Define the training:
        self.ep_len = 10
        self._steps = None
        self._truncated = None
        self._done = None

        # Tunable parameters
        self._kappa = None
        self._rho_inflation_percentage = None

        # Define the logger:
        self.log_cumulative_version = None
        self._log_valid_sampled_letters_indexes_list = None
    
    def reset(self, seed=None, inputs=None, ep_idx=None):
        """
        Reset the environment to the initial state
        """

        self._steps = 0
        self._truncated = False
        self._done = False
        self.log_cumulative_version = {}
        
        # TODO: here set for the special simulate mode
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
        self._entropy_diffs_list = []

        # Reset the word representation
        self._word_representation = self.transition_function.reset_state_word_representation()

        # Temporal variables
        self._gaze_duration_for_this_word = 0
        self._sum_saccade_duration_for_this_word = 0
        self._total_time_cost_for_this_word = 0

        # Reset the seen letters
        self._sampled_letters_so_far_representation = [-1] * self.MAX_WORD_LEN
        self._sampled_letters_so_far_with_spaces = ""

        # Sample the word to be recognized
        if self._mode == 'simulate':
        # if inputs is not None:
            self._word = inputs["word"]
            word_freq_prob = inputs["word_freq_prob"]
            word_pred_prob = inputs["word_pred_prob"]
            # self._word_prior_prob = self.lex_manager.prior_dict[self._word] 
            self._word_prior_prob = max(word_freq_prob, word_pred_prob)     # NOTE Justification: usually the pred prob is very small and neglectable, we use it only if it is dominating the freq
            self._raw_occurance = word_freq_prob * 1_000_000   # NOTE: this seem to be not used when inputting words for simulation, thus set as 0
            assert 0 <= self._word_prior_prob <= 1, (
                f"Invalid word prior prob, should be within [0, 1]. "
                f"Word freq prob is: {word_freq_prob} and word pred prob is: {word_pred_prob}."
            )
            # Update the prior dict in the lexicon manager, register the target word there
            self.lex_manager.update_prior_dict(word_to_recognize=self._word, prior_prob=self._word_prior_prob)
        else:
            self._word, self._word_prior_prob, self._raw_occurance = self.lex_manager.get_a_generated_word()

        self._word_len = len(self._word)

        self._word_to_activate = None

        # Initialize the ground truth representation -- the word to be recognize is encoded as:
        self._normalized_ground_truth_word_representation = self.transition_function.get_normalized_ground_truth_word_representation(target_word=self._word)
        # This is only used for identifying words and numerical computations

        # Reset the tunable parameter
        self._rho_inflation_percentage = 0.2    # TODO: put into the reset arguments later when tuning. Grid searcch should be enough.
        self._kappa = None  # TBD
        
        # Reset the log
        self._log_valid_sampled_letters_indexes_list = []

        return self._get_obs(), self._get_logs(is_initialization=True, mode=self._mode)

    def step(self, action):
        """
        Take an action and return the response
        """

        # Initialize variables
        self._done = False
        self._truncated = False
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
                self._calculate_entropy_diff()

                # Get the reward
                reward = self.reward_function.get_step_wise_effort_cost(is_action_valid=True)

                # Update the log
                self._log_valid_sampled_letters_indexes_list.append(action)
            
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
        """
        Terminate the episode

        Update states, word representation, belief distribution, and temporal variables
        """

        # Activate a word
        self._word_to_activate = self.transition_function.activate_a_word(
            normalized_belief_distribution_dict=self._normalized_belief_distribution_dict_parallel_activation_with_k_words, deterministic=Constants.DETERMINISTIC_WORD_ACTIVATION
        )
            
        reward = self.reward_function.get_terminate_reward(
            word_to_recognize=self._word,
            word_to_activate=self._word_to_activate
        )       

        done = True

        # Get the durations: gaze duration and sum saccade duration
        self._gaze_duration_for_this_word = self._get_gaze_duration_for_this_word()
        self._sum_saccade_duration_for_this_word = self._get_sum_saccade_duration_for_this_word()
        self._total_time_cost_for_this_word = self._get_total_time_cost_for_this_word()

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
    
    ############################################### Helper Functions ###############################################

    def get_current_letter_index(self):
        """
        Get the current letter index that is fixated
        """
        if self._action >= self.MAX_WORD_LEN:
            return (None, True)     # Terminate step: none of the fixation index, and True of termination
        else:
            if self._action <= self._word_len - 1:
                return (self._action, False)    # Valid fixation, False of termination
            else:
                return (-1, False)      # Invalid fixation, False of termination
    
    def get_elapsed_time_in_ms(self):
        """
        Get the elapsed time, including the gaze duration (sum of the first-fixations) and total saccade durations
        """
        if len(self._entropy_diffs_list) == 0:
            return 0        # No information sampling, just wasting steps
        else:
            # return self._get_gaze_duration_for_this_word() + self._get_sum_saccade_duration_for_this_word()
            return self._get_total_time_cost_for_this_word()

    def _calculate_entropy_diff(self):
        """
        Calculate the entropy difference
        """
        self._current_step_entropy = self._calculate_entropy(probability_distribution=self._normalized_belief_distribution_parallel_activation_with_k_words)
        self._entropy_diff = self._previous_step_entropy - self._current_step_entropy
        self._previous_step_entropy = self._current_step_entropy
        self._entropy_diffs_list.append(self._entropy_diff)

    def _get_gaze_duration_for_this_word(self):
        """
        Calculate the gaze duration
        """
        return self.transition_function.calc_gaze_duration_ms(entropy_diffs=self._entropy_diffs_list)
    
    def _get_inflated_gaze_duration_for_this_word(self):
        """
        Calculate the inflated gaze duration
        """
        return self.transition_function.calc_inflated_gaze_duration_ms(entropy_diffs=self._entropy_diffs_list, rho_inflation_percentage=self._rho_inflation_percentage)
    
    def _get_sum_saccade_duration_for_this_word(self):
        """
        Get the sum saccade duration for this word
        """
        return self.transition_function.calc_total_saccades_duration_ms(entropy_diffs=self._entropy_diffs_list)
    
    def _get_total_time_cost_for_this_word(self):
        """
        Get the sum of gaze duration, saccade durations, and non-fixation non-saccade overhead durations

        So it turns to be the sum of saccades durations + inflated gaze duration
        """
        return self._get_inflated_gaze_duration_for_this_word() + self._get_sum_saccade_duration_for_this_word()
    
    def get_recognized_word(self):
        """
        Get the recognized word str
        """
        return self._word_to_activate
    
    def get_valid_sampled_letters_indexes_list(self):
        """
        Get the valid sampled letters' indexes as a list
        """
        return self._log_valid_sampled_letters_indexes_list

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
                    "free_param_rho_inflation_percentage": self._rho_inflation_percentage,
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
                    "gaze_duration_for_this_word": self._gaze_duration_for_this_word,
                    "sum_saccade_duration_for_this_word": self._sum_saccade_duration_for_this_word,
                    "total_time_cost_for_this_word": self._total_time_cost_for_this_word,
                    "accurate_recognition": self._word_to_activate == self._word if self._done else None
                })
                return self.log_cumulative_version
    
    def get_individual_step_log(self):
        """
        Get individual step log for simulation documentation and data rolling.
        """
        if self._action >= self.MAX_WORD_LEN:
            action_information = "activate"
            current_letter_index = None
        else:
            if self._action <= self._word_len - 1:
                action_information = "sampling"
                current_index_letter = self._action
            else:
                action_information = "invalid sampling"
                current_index_letter = -1

        individual_step_log = {
            "step": self._steps,
            "action": action_information,
            "current_letter_index": self._action,
            "elapsed_time": 'NOTE: sampled in the simulator', # self.get_elapsed_time_in_ms() / 1_000,
            "recognized_word_str": self._word_to_activate,
        }
        return individual_step_log
        

if __name__ == "__main__":

    lex_manager = LexiconManager()
    print(lex_manager.get_likelihood_by_sampled_letters_so_far(
        sampled_letters_so_far="gro", candidate_word="grow", original_word="grow"
    ))
    print(lex_manager.get_likelihood_by_sampled_letters_so_far("gro", "gro", "grow"))
    print(lex_manager.get_likelihood_by_sampled_letters_so_far("sil", "silk", "silk"))
    print(lex_manager.get_likelihood_by_sampled_letters_so_far("sil", "ssil", "silk"))
    print(lex_manager.get_likelihood_by_sampled_letters_so_far("si k", "silk", "silk"))