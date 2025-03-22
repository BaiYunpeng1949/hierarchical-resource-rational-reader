import math
import os
import yaml
import random
import torch
import numpy as np

from gymnasium import Env
from gymnasium.spaces import Box, Discrete, Dict

from .SentencesManager import SentencesManager
from .TransitionFunction import TransitionFunction
from .RewardFunction import RewardFunction
from . import Constants


class SentenceReadingEnv(Env):
    def __init__(self):
        """
        Create on 19 March 2025.
        This is the environment for the RL-based intermediate level -- sentence-level control agent: 
            it controls, word skippings, word revisits, and when to stop reading in a given sentence
        
        Features: predict the word skipping and word revisiting decisions, and when to stop reading.
        Cognitive constraints: (limited time and cognitive resources) 
            Foveal vision (so need multiple fixations), 
            STM (so need to revisit sometimes, provide limited contextual predictability),
            attention resource (so need to skip the unnecessary words),
            Time pressure (so need to finish reading before time runs out)
        """
        # Load configuration
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
        self._mode = self._config["rl"]["mode"]
        
        print(f"Sentence Reading Environment V0319 -- Deploying in {self._mode} mode with simplified scalar signals")

        # Initialize components
        self.sentences_manager = SentencesManager()
        self.transition_function = TransitionFunction()
        self.reward_function = RewardFunction()

        # State tracking
        self._sentence_info = None
        self._sentence_len = None
        self._current_word_index = None
        self._previous_word_index = None
        self._word_beliefs = None  # Store beliefs for each word
        self._read_words = set()  # Track which words have been read

        # Reading behavior tracking
        self._skipped_words_indexes = None      
        self._regressed_words_indexes = None
        self._reading_sequence = None

        # Environment parameters
        self._steps = None
        self.ep_len = 2 * Constants.MAX_SENTENCE_LENGTH
        self._terminate = None
        self._truncated = None
        self._episode_id = None

        # Action space
        self._REGRESS_ACTION = 0
        self._READ_ACTION = 1
        self._SKIP_ACTION = 2
        self._STOP_ACTION = 3    
        self.action_space = Discrete(4)     
        
        # Observation space - simplified to scalar signals
        self._num_stateful_obs = 6
        self.observation_space = Box(low=0, high=1, shape=(self._num_stateful_obs,))
        
    def reset(self, seed=42, sentence_idx=None, episode_id=None):
        """Reset environment and initialize states"""
        super().reset(seed=seed)

        self._steps = 0
        self._terminate = False
        self._truncated = False
        self._episode_id = episode_id

        # Get new sentence
        self._sentence_info = self.sentences_manager.reset(sentence_idx)
        self._sentence_len = len(self._sentence_info['words'])

        # # TODO debug delete later
        # print(f"sentence_info: {self._sentence_info}")
        # print(f"--------------------------------")
        # print(f"the words ranked word integration probabilities: {self._sentence_info['words_ranked_word_integration_probabilities']}")
        # print(f"--------------------------------")
        # print(f"the words predictabilities: {self._sentence_info['words_predictabilities']}")
        
        # Initialize word beliefs from pre-computed data
        self._word_beliefs = [-1] * self._sentence_len
        self._read_words = []
        
        # Reset reading state
        self._current_word_index = -1
        self._previous_word_index = None
        
        # Reset tracking
        self._skipped_words_indexes = []    # only check the first-pass skipped words
        self._regressed_words_indexes = []
        self._reading_sequence = []

        # # TODO: predefine a action sequence to test the environment
        # self._action_sequence = [1, 1, 2, 1, 1, 2, 1, 1, 0, 1, 3]
        
        return self._get_obs(), {}
    
    def step(self, action):
        """Take action and update states"""
        self._steps += 1
        reward = 0

        # # TODO debug delete later -- TODO manipulate the actions to see whether they work properly
        # action = self._action_sequence[self._steps-1]


        if action == self._REGRESS_ACTION:
            self._current_word_index, action_validity = (
                self.transition_function.update_state_regress(
                    self._current_word_index,
                    self._sentence_len
                )
            )
            if action_validity:
                self._regressed_words_indexes.append(self._current_word_index)
                self._reading_sequence.append(self._current_word_index)
                # Reset belief to 1 for regressed word
                self._word_beliefs[self._current_word_index] = 1.0
                self._previous_word_index = self._current_word_index - 1
            reward = self.reward_function.compute_regress_reward()
        
        elif action == self._SKIP_ACTION:
            self._current_word_index, action_validity = (
                self.transition_function.update_state_skip_next_word(
                    self._current_word_index,
                    self._sentence_len
                )
            )
            if action_validity:
                # Use pre-computed ranked integration probability as belief
                skipped_word_index = self._current_word_index - 1
                self._previous_word_index = skipped_word_index
                self._word_beliefs[skipped_word_index] = self._sentence_info['words_predictabilities_for_running_model'][skipped_word_index]
                self._word_beliefs[self._current_word_index] = self._sentence_info['words_ranked_word_integration_probabilities_for_running_model'][self._current_word_index]
                
                # Check if the skipped word is the first-pass skipped word
                if skipped_word_index not in self._reading_sequence:
                    self._skipped_words_indexes.append(skipped_word_index)
                self._reading_sequence.append(skipped_word_index)
                self._reading_sequence.append(self._current_word_index)
            reward = self.reward_function.compute_skip_reward()
        
        elif action == self._READ_ACTION:
            self._current_word_index, action_validity = (
                self.transition_function.update_state_read_next_word(
                    self._current_word_index,
                    self._sentence_len
                )
            )
            if action_validity:
                # Sample from prediction candidates with highest probabilit
                self._reading_sequence.append(self._current_word_index)
                self._previous_word_index = self._current_word_index - 1
                self._word_beliefs[self._current_word_index] = self._sentence_info['words_ranked_word_integration_probabilities_for_running_model'][self._current_word_index]
            reward = self.reward_function.compute_read_reward()

        elif action == self._STOP_ACTION:
            self._terminate = True
            # Compute final comprehension reward
            
            valid_words_beliefs = [b for b in self._word_beliefs if b != -1]

            reward = self.reward_function.compute_terminate_reward(
                sentence_len=self._sentence_len,
                num_words_read=len(valid_words_beliefs),
                words_beliefs=valid_words_beliefs
            )
        
        # Check termination
        if self._steps >= self.ep_len:
            self._terminate = True
            self._truncated = True

        # TODO debug delete later
        if self._terminate: 
            info = self.get_episode_log()
        else:
            info = {}
        
        return self._get_obs(), reward, self._terminate, self._truncated, info
    
    @staticmethod
    def normalise(x, x_min, x_max, a, b):
    # Normalise x (which is assumed to be in range [x_min, x_max]) to range [a, b]
        return (b - a) * ((x - x_min) / (x_max - x_min)) + a
    
    def _get_obs(self):
        """Get observation with simplified scalar signals"""
        # Get current position (normalized)
        current_position = self.normalise(self._current_word_index, 0, self._sentence_len - 1, 0, 1)
        
        valid_words_beliefs = [b for b in self._word_beliefs if b != -1]

        # Get remaining words (normalized)
        remaining_words = self.normalise(self._sentence_len - len(valid_words_beliefs), 0, self._sentence_len, 0, 1)
        
        # Get the previous word's belief
        norm_previous_word_belief = np.clip(self._word_beliefs[self._previous_word_index], 0, 1) if self._previous_word_index is not None and 0 <= self._previous_word_index < self._sentence_len else 1
        norm_current_word_belief = np.clip(self._word_beliefs[self._current_word_index], 0, 1) if self._current_word_index is not None and 0 <= self._current_word_index < self._sentence_len else 1
        norm_next_word_predictability = np.clip(self._sentence_info['words_predictabilities_for_running_model'][self._current_word_index + 1], 0, 1) if self._current_word_index + 1 is not None and 0 <= self._current_word_index + 1 < self._sentence_len else 1
        
        # Get the on-going comprehension scalar
        on_going_comprehension_scalar = np.clip(math.prod(valid_words_beliefs), 0, 1)

        # # TODO debug delete later
        # print(f"valid_words_beliefs: {self._word_beliefs}")
        # print(f"on_going_comprehension_scalar: {on_going_comprehension_scalar}")

        stateful_obs = np.array([
            current_position,
            remaining_words,
            norm_previous_word_belief,
            norm_current_word_belief,
            norm_next_word_predictability,
            on_going_comprehension_scalar
        ])

        assert stateful_obs.shape == (self._num_stateful_obs,)

        return stateful_obs
        
    def get_episode_log(self) -> dict:
        """Get logs for the episode"""
        words_data_list = []
        for word_idx in range(self._sentence_len):
            word_data = {
                "word": self._sentence_info['words'][word_idx],
                "word_clean": self._sentence_info['word_cleans'][word_idx],
                "word_id": self._sentence_info['word_ids'][word_idx],
                "length": self._sentence_info['word_lengths_for_analysis'][word_idx],
                "frequency_per_million": self._sentence_info['word_frequencies_per_million_for_analysis'][word_idx],
                "log_frequency": self._sentence_info['word_log_frequencies_per_million_for_analysis'][word_idx],
                "difficulty": self._sentence_info['word_difficulties_for_analysis'][word_idx],
                "predictability": self._sentence_info['word_predictabilities_for_analysis'][word_idx],
                "logit_predictability": self._sentence_info['word_logit_predictabilities_for_analysis'][word_idx],
                "is_first_pass_skip": word_idx in self._skipped_words_indexes,
                "is_regression_target": word_idx in self._regressed_words_indexes,
                "FFD": [],
                "GD": [],
                "TRT": [],
                "nFixations": [],
            }
            words_data_list.append(word_data)
        
        episode_log = {
            "episode_id": self._episode_id,
            'sentence_id': self._sentence_info['sentence_id'],
            'participant_id': self._sentence_info['participant_id'],
            'sentence_content': self._sentence_info['sentence_content'],
            'sentence_len': self._sentence_len,
            "words": words_data_list
        }

        return episode_log


if __name__ == "__main__":
    def print_state(env, action_name):
        """Helper function to print current state"""
        print(f"\nAction: {action_name}")
        print(f"Current word index: {env._current_word_index}")
        print(f"Previous word index: {env._previous_word_index}")
        print(f"Word beliefs: {env._word_beliefs}")
        print(f"Reading sequence: {env._reading_sequence}")
        print(f"Skipped words: {env._skipped_words_indexes}")
        print(f"Regressed words: {env._regressed_words_indexes}")
        print(f"Observation: {env._get_obs()}")

    def test_normal_reading():
        """Test normal word-by-word reading"""
        print("\n=== Testing Normal Reading ===")
        env = SentenceReadingEnv()
        obs, _ = env.reset(sentence_idx=0)  # Use first sentence for testing
        print(f"Initial observation: {obs}")
        
        # Read first few words
        for i in range(3):
            obs, reward, term, trunc, _ = env.step(env._READ_ACTION)
            print_state(env, "READ")
            print(f"Reward: {reward}")
            print(f"Terminated: {term}")
            print(f"Truncated: {trunc}")

    def test_skipping():
        """Test word skipping behavior"""
        print("\n=== Testing Word Skipping ===")
        env = SentenceReadingEnv()
        obs, _ = env.reset(sentence_idx=0)
        
        # Read first word
        obs, reward, term, trunc, _ = env.step(env._READ_ACTION)
        print_state(env, "READ")
        
        # Skip next word
        obs, reward, term, trunc, _ = env.step(env._SKIP_ACTION)
        print_state(env, "SKIP")
        print(f"Reward: {reward}")

    def test_regression():
        """Test regression behavior"""
        print("\n=== Testing Regression ===")
        env = SentenceReadingEnv()
        obs, _ = env.reset(sentence_idx=0)
        
        # Read first two words
        for _ in range(2):
            obs, reward, term, trunc, _ = env.step(env._READ_ACTION)
            print_state(env, "READ")
        
        # Regress to previous word
        obs, reward, term, trunc, _ = env.step(env._REGRESS_ACTION)
        print_state(env, "REGRESS")
        print(f"Reward: {reward}")

    def test_termination():
        """Test environment termination"""
        print("\n=== Testing Termination ===")
        env = SentenceReadingEnv()
        obs, _ = env.reset(sentence_idx=0)
        
        # Read all words
        while not env._terminate:
            obs, reward, term, trunc, _ = env.step(env._READ_ACTION)
            print_state(env, "READ")
            if term:
                print(f"Terminated with reward: {reward}")
                break
        
        # Try to stop
        obs, reward, term, trunc, _ = env.step(env._STOP_ACTION)
        print_state(env, "STOP")
        print(f"Final reward: {reward}")
        print(f"Terminated: {term}")

    def test_episode_length():
        """Test episode length limit"""
        print("\n=== Testing Episode Length ===")
        env = SentenceReadingEnv()
        obs, _ = env.reset(sentence_idx=0)
        
        # Run until truncation
        step = 0
        while not env._truncated:
            obs, reward, term, trunc, _ = env.step(env._READ_ACTION)
            step += 1
            if step % 10 == 0:
                print(f"Step {step}: Current word index = {env._current_word_index}")
        
        print(f"Episode truncated after {step} steps")
        print(f"Final observation: {obs}")

    # Run all tests
    test_normal_reading()
    test_skipping()
    test_regression()
    test_termination()
    test_episode_length()
        
