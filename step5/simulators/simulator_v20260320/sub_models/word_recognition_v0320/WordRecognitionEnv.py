import numpy as np
import yaml
import os
import math

from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete, Tuple

from . import Constants
from .TransitionFunction import TransitionFunction
from .RewardFunction import RewardFunction
from .LexiconManager import LexiconManager


class WordRecognitionEnv(Env):
    """
    Oculomotor Controller RL Environment
    Coarse-region version:
    0 = beginning
    1 = mid_left
    2 = mid_right
    3 = ending
    4 = stop
    """

    REGION_BEGINNING = 0
    REGION_MID_LEFT = 1
    REGION_MID_RIGHT = 2
    REGION_ENDING = 3
    ACTION_STOP = 4

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
        self._raw_occurance = None
        self._sampled_letters_so_far_with_spaces = None    # The letters that have been sampled

        # Entropy
        self._previous_step_entropy = None
        self._current_step_entropy = None
        self._entropy_diff = None
        self._entropy_diffs_list = None

        # Representations
        self._word_representation = self.transition_function.reset_state_word_representation()    # The word representation, it stores the sampled letters. Here is a vector of letters sampled from the ASCII space
        self._normalized_ground_truth_word_representation = None

        self._sampled_letters_so_far_representation = None   # The letters that have been sampled

        # # Temporal variables
        # self._current_fixation_duration = None      # Unit is milliseconds, the time spent for a single fixation
        # self._individual_fixations_durations_list = None    # The list of individual fixation durations
        # self._current_saccade_duration = None       # Unit is milliseconds, the time spent for a single saccade
        # self._individual_saccades_durations_list = None     # The list of individual saccade durations
        # self._gaze_duration = None      # Unit is milliseconds, the total time spent for a first-pass of a word, including both fixation durations and saccade durations

        # # Fixtion to duration mapping non-linear equation parameters
        # self._t_0 = Constants.DEFAULT_FIXATION_DURATION    # The default average fixation duration, unit is milliseconds
        # self._lamda = Constants.GAZE_DURATION_LAMDA    # The decay rate, the larger the value, the faster the decay

        # Temporal variables (time-aware version)
        self.gaze_duration_for_this_word = None
        self.sum_saccade_duration_for_this_word = None
        self.total_elapsed_time_for_this_word = None

        # Define the word that is recognized
        self._word_to_activate = None

        # Whether to use the oculomotor noise
        self._apply_fixation_noise = True

        # Define the action 
        self._action = None
        # Fixation position variables
        self._intended_action = None
        self._executed_action = None
        self._target_action = None
        self._sampled_region_window  = None # For logging
        
        # # Define the action space:      NOTE: the original, disable
        # self.action_space = Discrete(self.MAX_WORD_LEN + 1)    # 0-9: fixate on the letter at the position, 10: stop the sampling and recognize the word

        # Action space: 0=beginning, 1=mid_left, 2=mid_right, 3=ending, 4=stop
        self.action_space = Discrete(5)

        # Define the observation space:
        self.STATEFUL_OBS = "stateful_obs"
        self.ACTION_OBS = "action_obs"
        self._num_action_obs = 6        # 1 init slot + 5 action slots
        # self._num_stateful_obs = len(self._normalized_belief_distribution_parallel_activation_with_k_words) + len(self._word_representation) + 1 + (self.MAX_WORD_LEN + 1 + 1) + 1 # Belief distribution, word representation with sampled letters, word length, prior type   NOTE: disable, the original
        self._num_stateful_obs = (
            len(self._normalized_belief_distribution_parallel_activation_with_k_words)
            + len(self._word_representation)
            + 1
            + self._num_action_obs
            + 1
        )
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

        ############################################################################################
        # Tunable Parameter
        ############################################################################################
        self._kappa = None
        self._rho_inflation_percentage = None
    
    def reset(self, seed=None, inputs=None, ep_idx=None, params=None):
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
        # Reset the fixation variables
        self._intended_action = -1
        self._executed_action = -1
        self._target_action = -1
        self._sampled_region_window  = None

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

        # # Temporal variables
        # self._current_fixation_duration = 0
        # self._individual_fixations_durations_list = []
        # self._current_saccade_duration = 0
        # self._individual_saccades_durations_list = []
        # self._gaze_duration = 0

        # Temporal variables
        self.gaze_duration_for_this_word = 0
        self.sum_saccade_duration_for_this_word = 0
        self.total_elapsed_time_for_this_word = 0

        # Reset the seen letters
        self._sampled_letters_so_far_representation = [-1] * self.MAX_WORD_LEN
        self._sampled_letters_so_far_with_spaces = ""

        # Sample the word to be recognized
        if inputs is not None:
            self._word = inputs["word"]
            self._word_prior_prob = self.lex_manager.prior_dict[self._word] 
            self._raw_occurance = inputs["raw_occurance"]   # TODO, if use, fix
        else:
            self._word, self._word_prior_prob, self._raw_occurance = self.lex_manager.get_a_generated_word()

        self._word_len = len(self._word)
        self._word_to_activate = None

        # Initialize the ground truth representation -- the word to be recognize is encoded as:
        self._normalized_ground_truth_word_representation = self.transition_function.get_normalized_ground_truth_word_representation(target_word=self._word)
        # This is only used for identifying words and numerical computations

        # print(f"Hello BYP, I am here")

        # # Check whether there are out-of-model tunable parameters
        # if params is not None:
        #     self.param_kappa = params['kappa']
        # else:
        #     self.param_kappa = 3.75     # An default value from the literature

        # Reset the tunable parameters for the time-aware version
        self._kappa = 3.50
        if params is None:
            self._rho_inflation_percentage = 0.2
        else:
            self._rho_inflation_percentage = params["rho_inflation_percentage"]

        return self._get_obs(), self._get_logs(is_initialization=True, mode=self._mode)

    def step(self, action):
        """
        Take an action and return the response.

        Action semantics:
            0 = beginning
            1 = mid_left
            2 = mid_right
            3 = ending
            4 = stop
        """
        self._done = False
        self._truncated = False
        reward = 0

        action = int(action)
        self._intended_action = action
        self._action = action

        self._steps += 1

        if action == self.ACTION_STOP:
            self._executed_action = action
            reward, self._done = self._terminate_step()
            # self._calc_gaze_duration()

        else:
            # Step 1: choose a target letter within the selected coarse region
            target_letter_idx = self._sample_target_action_from_region(action)
            self._target_action = target_letter_idx

            # Step 2: apply local oculomotor uncertainty around that target
            executed_letter_idx = self._sample_landed_action_from_target(target_letter_idx)

            # Final landed fixation location
            self._executed_action = executed_letter_idx

            self._sampled_letters_so_far_representation, self._sampled_letters_so_far_with_spaces = (
                self.transition_function.update_state_sampled_letters_so_far_include_non_contiguous_letters(
                    action=executed_letter_idx,
                    norm_gt_word_rep=self._normalized_ground_truth_word_representation,
                    seen_letters_representation=self._sampled_letters_so_far_representation,
                    seen_letters=self._sampled_letters_so_far_with_spaces,
                    word=self._word,
                    word_len=self._word_len,
                )
            )

            assert self._sampled_letters_so_far_with_spaces != "NO_LETTER_SAMPLED", (
                f"no letters sampled so far, the word is {self._word}, "
                f"the region action is {action}, the target letter is {target_letter_idx}, "
                f"the executed action is {executed_letter_idx}, "
                f"the word length is {self._word_len}"
            )

            (
                self._prior_distribution_dict_parallel_activation_with_k_words,
                self._normalized_belief_distribution_dict_parallel_activation_with_k_words,
                self._normalized_belief_distribution_parallel_activation_with_k_words,
                self._likelihood_dict_parallel_activation_with_k_words,
            ) = self.transition_function.update_state_normalized_belief_distribution_dict(
                sampled_letters_so_far_with_spaces=self._sampled_letters_so_far_with_spaces,
                word_to_recognize=self._word,
                parallelly_activated_words_beliefs_dict=self._normalized_belief_distribution_dict_parallel_activation_with_k_words,
                lexicon_manager=self.lex_manager,
            )

            self._calculate_entropy_diff()
            reward = self.reward_function.get_step_wise_effort_cost(is_action_valid=True)

        if self._steps >= self.ep_len:
            reward, self._done = self._terminate_step()
            self._truncated = True

        return (
            self._get_obs(),
            reward,
            self._done,
            self._truncated,
            self._get_logs(is_initialization=False, mode=self._mode),
        )

    def render(self, mode='human'):
        pass
    
    def _get_adaptive_region_window_size(self) -> int:
        """
        Adapt region window size to word length.

        Suggested mapping:
            word_len <= 6  -> 3
            word_len <= 9  -> 4
            word_len >= 10 -> 5
        """
        if self._word_len <= 6:
            return 3
        elif self._word_len <= 9:
            return 4
        else:
            return 5
    
    def _get_region_window(self, region_action: int):
        """
        Return an adaptive window of valid integer indices within the word.

        Regions:
            0 = beginning
            1 = mid_left
            2 = mid_right
            3 = ending

        Window size is adaptive:
            word_len <= 6  -> 3
            word_len <= 9  -> 4
            word_len >= 10 -> 5
        """
        window_size = self._get_adaptive_region_window_size()

        # If the word is shorter than the desired window, just return the whole word
        if self._word_len <= window_size:
            return list(range(self._word_len))

        max_start = self._word_len - window_size

        # Mid anchors based on proportional positions in the word
        region_starts = [
            0,                                   # beginning
            int(round(max_start * 0.25)),        # mid_left
            int(round(max_start * 0.60)),        # mid_right
            max_start                            # ending
        ]

        if region_action == self.REGION_BEGINNING:
            start = region_starts[0]
        elif region_action == self.REGION_MID_LEFT:
            start = region_starts[1]
        elif region_action == self.REGION_MID_RIGHT:
            start = region_starts[2]
        elif region_action == self.REGION_ENDING:
            start = region_starts[3]
        else:
            raise ValueError(f"Invalid region action: {region_action}")

        return list(range(start, start + window_size))
    

    def _sample_target_action_from_region(self, region_action: int) -> int:
        """
        Sample an intended target letter uniformly from the selected region.
        """
        window = self._get_region_window(region_action)
        self._sampled_region_window = window.copy()
        return int(np.random.choice(window))

    
    # def _terminate_step(self):
    #     self._word_to_activate = self.transition_function.activate_a_word(
    #         normalized_belief_distribution_dict=self._normalized_belief_distribution_dict_parallel_activation_with_k_words, 
    #         deterministic=Constants.DETERMINISTIC_WORD_ACTIVATION
    #     )
            
    #     reward = self.reward_function.get_terminate_reward(
    #         word_to_recognize=self._word,
    #         word_to_activate=self._word_to_activate
    #     )       

    #     done = True

    #     return reward, done

    def _terminate_step(self):
        """
        Terminate the episode.

        Update recognized word and time-related outputs.
        """
        self._word_to_activate = self.transition_function.activate_a_word(
            normalized_belief_distribution_dict=self._normalized_belief_distribution_dict_parallel_activation_with_k_words,
            deterministic=Constants.DETERMINISTIC_WORD_ACTIVATION
        )

        reward = self.reward_function.get_terminate_reward(
            word_to_recognize=self._word,
            word_to_activate=self._word_to_activate
        )

        done = True

        # Time-aware outputs
        self.gaze_duration_for_this_word, self.total_elapsed_time_for_this_word = self.get_gaze_and_elapsed_duration_in_ms()
        self.sum_saccade_duration_for_this_word = self._get_sum_saccade_duration_for_this_word()

        return reward, done
    
    def _sample_landed_action_from_target(self, target_action: int) -> int:
        """
        Given a sampled target letter, sample the actual landed fixation
        from an adaptive local noise window around target_action.

        Noise window size:
            word_len <= 6  -> 3 positions
            word_len <= 9  -> 4 positions
            word_len >= 10 -> 5 positions

        All positions are clipped to valid word boundaries.
        """
        noise_window_size = self._get_adaptive_region_window_size()

        if noise_window_size % 2 == 1:
            # odd-sized window, symmetric around target
            half = noise_window_size // 2
            offsets = list(range(-half, half + 1))
        else:
            # even-sized window, slightly right-biased
            left = noise_window_size // 2 - 1
            right = noise_window_size // 2
            offsets = list(range(-left, right + 1))

        candidates = [
            pos for pos in [target_action + offset for offset in offsets]
            if 0 <= pos <= self._word_len - 1
        ]

        return int(np.random.choice(candidates))

    # def _sample_landed_action_from_target(self, target_action: int) -> int:
    #     """
    #     Given a sampled target letter, sample the actual landed fixation
    #     from a fixed 3-letter window:
    #         [target-1, target, target+1],
    #     clipped to valid word boundaries.
    #     """
    #     candidates = [
    #         pos for pos in [
    #             target_action - 1,
    #             target_action,
    #             target_action + 1,
    #         ]
    #         if 0 <= pos <= self._word_len - 1
    #     ]

    #     return int(np.random.choice(candidates))
    
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
        Observation uses coarse-action one-hot.

        Slots:
            0 -> initialization / no previous action
            1 -> beginning
            2 -> mid_left
            3 -> mid_right
            4 -> ending
            5 -> stop
        """
        action_obs = np.zeros(self._num_action_obs)

        if self._action == -1:
            action_obs[0] = 1
        elif self._action == self.REGION_BEGINNING:
            action_obs[1] = 1
        elif self._action == self.REGION_MID_LEFT:
            action_obs[2] = 1
        elif self._action == self.REGION_MID_RIGHT:
            action_obs[3] = 1
        elif self._action == self.REGION_ENDING:
            action_obs[4] = 1
        elif self._action == self.ACTION_STOP:
            action_obs[5] = 1
        else:
            raise ValueError(f"Unexpected action for observation encoding: {self._action}")

        stateful_obs = np.concatenate([
            self._normalized_belief_distribution_parallel_activation_with_k_words,
            self._sampled_letters_so_far_representation,
            [self._word_len],
            action_obs,
            [self._prior_type]
        ])

        assert len(stateful_obs) == self._num_stateful_obs, (
            f"expected {self._num_stateful_obs} but got {len(stateful_obs)}"
        )

        return stateful_obs


    def _calculate_entropy_diff(self):
        """
        Calculate the entropy difference
        """
        self._current_step_entropy = self._calculate_entropy(probability_distribution=self._normalized_belief_distribution_parallel_activation_with_k_words)
        self._entropy_diff = self._previous_step_entropy - self._current_step_entropy
        self._previous_step_entropy = self._current_step_entropy
        self._entropy_diffs_list.append(self._entropy_diff)


    # def _calc_gaze_duration(self):
    #     """
    #     Calculate the gaze duration
    #     """
    #     self._gaze_duration = self.transition_function.calc_gaze_duration_ms(entropy_diffs=self._entropy_diffs_list, kappa=self.param_kappa)


    def _calc_gaze_duration(self):
        """
        Calculate the gaze duration
        """
        self._gaze_duration = self.transition_function.calc_gaze_duration_ms(
            entropy_diffs=self._entropy_diffs_list,
            kappa=self.param_kappa
        )

    
    def get_gaze_and_elapsed_duration_in_ms(self):
        gaze_duration, inflated_gaze_duration = self.transition_function.calc_gaze_related_duration_in_ms(
            entropy_diffs=self._entropy_diffs_list,
            rho_inflation_percentage=self._rho_inflation_percentage,
        )

        saccades_sum_duration = self.transition_function.calc_total_saccades_duration_ms(
            entropy_diffs=self._entropy_diffs_list
        )

        elapsed_time = inflated_gaze_duration + saccades_sum_duration
        return gaze_duration, elapsed_time

    def _get_sum_saccade_duration_for_this_word(self):
        """
        Get the summed saccade duration for this word.
        """
        return self.transition_function.calc_total_saccades_duration_ms(
            entropy_diffs=self._entropy_diffs_list
        )
    
    def _get_logs(self, is_initialization=False, mode="train"):
        if mode == "train":
            return {}

        elif mode in ["debug", "test", "grid_test"]:
            if is_initialization:
                self.log_cumulative_version = {
                    "episode_idnex": "TBD",
                    "word": self._word,
                    "word_len": self._word_len,
                    "prior_type": self._prior_type,
                    "word_prior_prob": self._word_prior_prob,
                    "occurance": self._raw_occurance,
                    "word_frequency": self._word_freq_prob,
                    "word_predictability": self._word_predictability_prob,
                    "word_representation": self._word_representation,
                    "normalized_ground_truth_word_representation": self._normalized_ground_truth_word_representation,
                    "free_param_rho_inflation_percentage": self._rho_inflation_percentage,
                    "fixations": [],
                }
                return self.log_cumulative_version

            else:
                self.log_cumulative_version["fixations"].append({
                    "steps": self._steps,
                    "action": self._action,  # coarse region action
                    "intended_action": self._intended_action,  # coarse region action
                    "executed_action": self._executed_action,  # actual sampled letter index
                    "target_action": self._target_action,
                    "sampled_region_window": None if self._sampled_region_window is None else self._sampled_region_window.copy(),
                    "done": self._done,
                    "word_likelihood": self.lex_manager.get_likelihood_by_sampled_letters_so_far(
                        sampled_letters_so_far=self._sampled_letters_so_far_with_spaces,
                        candidate_word=self._word,
                        original_word=self._word,
                    ),
                    "sampled_letters_so_far": self._sampled_letters_so_far_with_spaces,
                    "sampled_letters_so_far_representation": self._sampled_letters_so_far_representation.copy(),
                    "word_to_activate": self._word_to_activate,
                    "prior_distribution_dict": self._prior_distribution_dict_parallel_activation_with_k_words.copy(),
                    "likelihood_distribution_dict": self._likelihood_dict_parallel_activation_with_k_words.copy(),
                    "normalized_belief_distribution_dict": self._normalized_belief_distribution_dict_parallel_activation_with_k_words.copy(),
                    "normalized_belief_distribution": self._normalized_belief_distribution_parallel_activation_with_k_words.copy(),
                    "current_step_entropy": self._current_step_entropy,
                    "entropy_diff": self._entropy_diff,
                    "gaze_duration_for_this_word": self.gaze_duration_for_this_word,
                    "sum_saccade_duration_for_this_word": self.sum_saccade_duration_for_this_word,
                    "total_time_cost_for_this_word": self.total_elapsed_time_for_this_word,
                    "accurate_recognition": self._word_to_activate == self._word if self._done else None,
                })
                return self.log_cumulative_version
    
    def get_gaze_duration_for_this_word(self):
        return self.gaze_duration_for_this_word

    def get_sum_saccade_duration_for_this_word(self):
        return self.sum_saccade_duration_for_this_word

    def get_total_elapsed_time_for_this_word(self):
        return self.total_elapsed_time_for_this_word