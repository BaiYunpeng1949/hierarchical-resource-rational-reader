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
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        # Get the mode from the config yaml file
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

        print(f"Word Activation (No Vision) Environment V0316 -- Deploying the environment in the {self._config['rl']['mode']} mode.")

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
        self.param_kappa = None     # Out-of-model tunable parameter kappa, does not need to be trained with
    
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
        self._word_to_activate = None

        # Initialize the ground truth representation -- the word to be recognize is encoded as:
        self._normalized_ground_truth_word_representation = self.transition_function.get_normalized_ground_truth_word_representation(target_word=self._word)
        # This is only used for identifying words and numerical computations

        # Check whether there are out-of-model tunable parameters
        if params is not None:
            self.param_kappa = params['kappa']
        else:
            self.param_kappa = 3.75     # An default value from the literature

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
            self._calc_gaze_duration()

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
    
    def _get_region_window(self, region_action: int):
        """
        Return a 3-letter window as valid integer indices within the word.

        Regions:
            0 = beginning
            1 = mid_left
            2 = mid_right
            3 = ending

        Strategy:
        - beginning always starts at 0
        - ending always starts at word_len - 3
        - mid_left and mid_right are placed by centering between 0-middle and middle-end
        - for short words, overlap is allowed naturally

        Examples:
            len=5  -> starts [0,1,1,2]
                    windows: [0,1,2], [1,2,3], [1,2,3], [2,3,4]
            len=6  -> starts [0,1,2,3]
                    windows: [0,1,2], [1,2,3], [2,3,4], [3,4,5]
            len=10 -> starts [0,2,5,7]
                    windows: [0,1,2], [2,3,4], [5,6,7], [7,8,9]
        """
        if self._word_len <= 3:
            return list(range(self._word_len))

        max_start = self._word_len - 3

        # Calculate region starts
        region_starts = [
            0,  # beginning
            int((0 + (self._word_len // 2)) // 2),  # mid_left
            int(((self._word_len // 2) + (self._word_len - 1)) // 2),  # mid_right
            max_start  # ending
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

        return [start, start + 1, start + 2]
    

    def _sample_target_action_from_region(self, region_action: int) -> int:
        """
        Sample an intended target letter uniformly from the selected region.
        """
        window = self._get_region_window(region_action)
        self._sampled_region_window = window.copy()
        return int(np.random.choice(window))

    
    def _terminate_step(self):
        self._word_to_activate = self.transition_function.activate_a_word(
            normalized_belief_distribution_dict=self._normalized_belief_distribution_dict_parallel_activation_with_k_words, 
            deterministic=Constants.DETERMINISTIC_WORD_ACTIVATION
        )
            
        reward = self.reward_function.get_terminate_reward(
            word_to_recognize=self._word,
            word_to_activate=self._word_to_activate
        )       

        done = True

        return reward, done
    
    def _sample_landed_action_from_target(self, target_action: int) -> int:
        """
        Given a sampled target letter, sample the actual landed fixation
        from [target-2, target-1, target, target+1, target+2],
        clipped to valid word boundaries.
        """
        candidates = [
            pos for pos in [
                target_action - 2,
                target_action - 1,
                target_action,
                target_action + 1,
                target_action + 2,
            ]
            if 0 <= pos <= self._word_len - 1
        ]
        return int(np.random.choice(candidates))
    
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


    def _calc_gaze_duration(self):
        """
        Calculate the gaze duration
        """
        self._gaze_duration = self.transition_function.calc_gaze_duration_ms(entropy_diffs=self._entropy_diffs_list, kappa=self.param_kappa)

    
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
                    "current_fixation_duration": self._current_fixation_duration,
                    "individual_fixations_durations_list": self._individual_fixations_durations_list.copy(),
                    "gaze_duration": self._gaze_duration,
                    "accurate_recognition": self._word_to_activate == self._word if self._done else None,
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