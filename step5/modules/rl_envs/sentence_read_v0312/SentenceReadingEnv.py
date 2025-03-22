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
        Create on 13 March 2025.
        This is the environment for the RL-based intermediate level -- sentence-level control agent: 
            it controls, word skippings, word revisits, and when to stop reading in a given sentence
        
        Features: predict the word skipping and word revisiting decisions, and when to stop reading.
        Cognitive constraints: (limited time and cognitive resources) 
            Foveal vision (so need multiple fixations), 
            STM (so need to revisit sometimes, provide limited contextual predictability),
            attention resource (so need to skip the unnecessary words),
            Time pressure (so need to finish reading before time runs out) -- this is an additional trait for sec 2.4/2.5
        Optimality: maximizing overall comprehension.
        State (addtional stuff in resource rationality): continuous monitoring of certainty. 
            (How confident am I that I already got that word? Or that I understood the prior sentence?) 
        Noisy information input (observation).

        NOTE: my understanding of word skipping and revisiting
            1. Word Skipping: "I Already Know Enough."
                High Confidence / Low Utility of Further Fixation
                Rate of Information Gain Falls Below Threshold
                Hence a Single "Skip or Not" Decision
            2. Revisiting (Regressions): "I Don't Understand Enough Yet."
                Detecting a Comprehension or Integration Problem
                Cost-Benefit Trade-Off
                Adaptive Correction (coherence)
        
        Task: each episode read one sentence. 
        Objective: read it as soon as possible, with the highest overall comprehension.
        
        Author: Bai Yunpeng

        NOTE: this version is originated from v0306. I rewrite it for replicating the results from ZuCo 1.0 Task 2 Natural Reading. 
            Here word skipping and regressions are analyzed through proababilities, it is calculated by the same word's skip/regression occurances
            divided by the total number of occurances of the word in the SAME sentence and SAME position.
            Therefore, I need to sample sentences from a fixed sentence dataset, where I can get fixed words in fixed positions to analyze.
            But there is a problem: the RL agent might overfit to the dataset, and perform deterministically to words.
            To avoid this, I need to add noise to the input sentence's predictabilities.
        NOTE: this env is for explaining word features' effect on the word skipping and regressions. 
            For e.g., word length, frequency, and predictability's effect on word skipping; 
                and word difficultiy's effect on regressions; and my custom effect:  
            As for the when to stop and specific sentence-level word skipping and regression replications (with specific values), 
                will be added in the 4th section and maybe 6th section of simulation results (maybe such a section is not needed for a Nature Human Behavior paper).
        
        Formalism: both word skipping and regressions are a result of the resource-rationality. 
            These are results of the optimazation of the trade-off between the information gain and costs of eye movements.
        
        NOTE: empirical data evidence:
            Source: ZuCo 1.0 Task 2 Natural Reading, processed by me, aggregated metrics could be found: /home/baiy4/ScanDL/scripts/data/zuco/bai_processed_task2_NR_ET_aggregated_metrics
            Word skipping, average skip distance: 1.93 words; max skip distance mean: 3.51 words;
            Word regression, average regression distance: 1.59 words; max regression distance mean: 1.96 words;
            NOTE: But there are a lot of outliers with huge jumps. So for the simplicity, we are mainly simulating one word regression and one word skipping first.
        
        NOTE: Sentence reading as a resource-rational agent:
        Begin by positioning human reading as a cognitively demanding, goal-directed task performed under resource constraints 
        (e.g., limited attention, limited memory, or computational resources). Explain that readers cannot process every detail at once. 
        Thus, they must strategically allocate their cognitive resources.
        Key points: 1. Humans aim for a balance between accuracy/comprehension (performance) and cognitive cost (effort/time).
        2. Reading behavior (skipping, regression, stopping) emerges naturally as adaptive solutions to this resource trade-off.

        POMDP: 
            State: comprehension level of the sentence (sent_comprhension_level), uncertainty of meaning, exectations about upcoming words.
            Action: word skipping, word regression, word reading, stopping.
            Observation: comprehension, observed and interpreted words, exectations about upcoming words (predictability or something else).
            Reward: cognitive cost (time, effort, etc.); and performance (accuracy reward: high comprehension, correct interpretation, goal achievement).
            Transition: skip increases uncertainty but reduces cost/time; regressing reduces uncertainty but increases cost/time.
        
        Enhanced sentence reading environment using neural language models for comprehension tracking.
        """
        # Load configuration
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        with open(os.path.join(root_dir, "config.yaml")) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
        self._mode = self._config["rl"]["mode"]
        
        print(f"Sentence Reading Environment V0312 -- Deploying in {self._mode} mode with neural comprehension tracking")

        # Initialize components
        self.sentences_manager = SentencesManager()
        self.transition_function = TransitionFunction()
        self.reward_function = RewardFunction()

        # State tracking
        self._sentence_info = None
        self._sentence_len = None
        self._current_word_index = None
        self._next_word_predictability = None

        # Neural comprehension states
        self._word_states = None  # Will hold embeddings and comprehension states
        self._global_comprehension = None  # Overall sentence understanding
        
        # Reading behavior tracking
        self._skipped_words_indexes = None      
        self._regressed_words_indexes = None
        self._reading_sequence = None

        # Environment parameters
        self._steps = None
        self.ep_len = 2 * Constants.MAX_SENTENCE_LENGTH
        self._terminate = None
        self._truncated = None

        # Action space
        self._REGRESS_ACTION = 0
        self._READ_ACTION = 1
        self._SKIP_ACTION = 2
        self._STOP_ACTION = 3    
        self.action_space = Discrete(4)     
        
        # Observation space - now includes embedding dimensions
        hidden_size = self.transition_function.hidden_size
        self.observation_space = Dict({
            'word_states': Box(low=-np.inf, high=np.inf, shape=(Constants.MAX_SENTENCE_LENGTH, hidden_size)),
            'current_position': Box(low=0, high=1, shape=(1,)),
            'next_word_pred': Box(low=0, high=1, shape=(1,)),
            'global_comprehension': Box(low=-np.inf, high=np.inf, shape=(hidden_size,))
        })
        
    def reset(self, seed=42, sentence_idx=None):
        """Reset environment and initialize neural states
        Args:
            seed: Random seed for reproducibility
            sentence_idx: Optional specific sentence index for controlled testing
        """
        super().reset(seed=seed)

        self._steps = 0
        self._terminate = False
        self._truncated = False
        
        # Get new sentence, using specific index if provided
        self._sentence_info = self.sentences_manager.reset(sentence_idx)
        self._sentence_len = len(self._sentence_info['words'])
        
        # Initialize neural states
        self._word_states = self.transition_function.reset(self._sentence_info['words'])
        self._global_comprehension = torch.zeros(self.transition_function.hidden_size)
        
        # Reset reading state
        self._current_word_index = -1
        self._next_word_predictability = self._sentence_info['word_contextual_predictabilities'][0]

        # Reset tracking
        self._skipped_words_indexes = []
        self._regressed_words_indexes = []
        self._reading_sequence = []
        
        return self._get_obs(), {}
    
    def step(self, action):
        """Take action and update neural comprehension states"""
        self._steps += 1
        reward = 0

        if action == self._REGRESS_ACTION:
            self._word_states, self._current_word_index, action_validity = (
                self.transition_function.update_state_regress(
                    self._word_states,
                    self._current_word_index
                )
            )
            if action_validity:
                self._regressed_words_indexes.append(self._current_word_index)
                self._update_global_comprehension()
            reward = self.reward_function.compute_regress_reward()
        
        elif action == self._READ_ACTION:
            self._word_states, self._current_word_index, action_validity = (
                self.transition_function.update_state_read_next_word(
                    self._word_states,
                    self._current_word_index,
                    self._sentence_len
                )
            )
            if action_validity:
                self._update_global_comprehension()
            reward = self.reward_function.compute_read_reward()
        
        elif action == self._SKIP_ACTION:
            self._word_states, self._current_word_index, action_validity = (
                self.transition_function.update_state_skip_next_word(
                    self._word_states,
                    self._current_word_index,
                    self._sentence_len,
                    self._next_word_predictability
                )
            )
            if action_validity:
                self._skipped_words_indexes.append(self._current_word_index)
                self._update_global_comprehension()
            reward = self.reward_function.compute_skip_reward()

        elif action == self._STOP_ACTION:
            self._terminate = True
            reward = self.reward_function.compute_terminate_reward(self._global_comprehension)
        
        # Update reading sequence
        self._reading_sequence.append(self._current_word_index)
        
        # Check termination
        if self._steps >= self.ep_len:
            self._terminate = True
            self._truncated = True
        
        # Update next word predictability if not at end
        if self._current_word_index < self._sentence_len - 1:
            self._next_word_predictability = self._sentence_info['word_contextual_predictabilities'][self._current_word_index + 1]
        
        return self._get_obs(), reward, self._terminate, self._truncated, {}
    
    def _get_obs(self):
        """Get observation including neural states"""
        word_states_tensor = torch.zeros(Constants.MAX_SENTENCE_LENGTH, self.transition_function.hidden_size)
        for i, state in enumerate(self._word_states):
            if state is not None:
                word_states_tensor[i] = state['comprehension'][-1].squeeze()  # Use last layer's state
                
        return {    # TODO I do not think the agent needs to know so much details. Just some representative appraisals would be enough.
            'word_states': word_states_tensor.detach().numpy(),
            'current_position': np.array([self._current_word_index / self._sentence_len]),
            'next_word_pred': np.array([self._next_word_predictability]),
            'global_comprehension': self._global_comprehension.detach().numpy()
        }
        
    def _update_global_comprehension(self):
        """Update global sentence comprehension state using the latest GRU state
        
        The GRU's hidden state already maintains cumulative comprehension of all words read so far.
        We simply use the latest state which contains the understanding of all previous words.
        """
        # Get the latest word state
        latest_state = None
        for state in reversed(self._word_states):
            if state is not None and not torch.isnan(state['comprehension']).any():
                latest_state = state['comprehension']
                break
                
        if latest_state is not None:
            # Use the last layer's state which has processed all previous words
            self._global_comprehension = latest_state[-1].squeeze().detach()
        else:
            # Initialize with zeros if no valid state found
            self._global_comprehension = torch.zeros(self.transition_function.hidden_size)
    
    def get_episode_logs(self):
        """Get enhanced logs including neural states"""
        # Calculate rates
        total_words = self._sentence_len
        num_skips = len(self._skipped_words_indexes)
        num_regressions = len(self._regressed_words_indexes)
        
        skipping_rate = (num_skips / total_words) * 100 if total_words > 0 else 0
        regression_rate = (num_regressions / len(self._reading_sequence)) * 100 if self._reading_sequence else 0
        
        # Get comprehension states
        word_comprehension_states = []
        word_difficulties = []
        for state in self._word_states:
            if state is not None:
                word_comprehension_states.append(state['comprehension'].squeeze().numpy())
                word_difficulties.append(float(state['difficulty']))
            else:
                word_comprehension_states.append(None)
                word_difficulties.append(None)
        
        return {
            'sentence_length': self._sentence_len,
            'reading_sequence': self._reading_sequence,
            'skipped_words': self._skipped_words_indexes,
            'regressed_words': self._regressed_words_indexes,
            'skipping_rate': skipping_rate,
            'regression_rate': regression_rate,
            'word_comprehension_states': word_comprehension_states,
            'word_difficulties': word_difficulties,
            'global_comprehension': self._global_comprehension.numpy(),
            'final_word_states': [
                (i, s['comprehension'].squeeze().numpy(), float(s['difficulty']))
                for i, s in enumerate(self._word_states)
                if s is not None
            ]
        }


if __name__ == "__main__":

    def print_comprehension_state(env, step, action_name, additional_word_idx=None):
        """Helper to print comprehension metrics
        Args:
            env: The environment instance
            step: Current step number
            action_name: Name of the action (READ/SKIP/REGRESS)
            additional_word_idx: Optional additional word index to print (for skipped words)
        """
        print(f"\nStep {step} - {action_name}")
        if env._current_word_index >= 0:
            word = env._sentence_info['words'][env._current_word_index]
            print(f"Current word: '{word}'")
            if env._word_states[env._current_word_index] is not None:
                state = env._word_states[env._current_word_index]['comprehension']
                norm = torch.norm(state[-1]).item()  # Use last layer's norm
                difficulty = float(env._word_states[env._current_word_index]['difficulty'])
                print(f"Current word state norm: {norm:.4f}")
                print(f"Integration difficulty: {difficulty:.4f}")
                
        # For skipped words, show their predicted state too
        if additional_word_idx is not None and additional_word_idx >= 0:
            skipped_word = env._sentence_info['words'][additional_word_idx]
            if env._word_states[additional_word_idx] is not None:
                skip_state = env._word_states[additional_word_idx]['comprehension']
                skip_norm = torch.norm(skip_state[-1]).item()
                skip_difficulty = float(env._word_states[additional_word_idx]['difficulty'])
                print(f"Skipped word '{skipped_word}' state norm: {skip_norm:.4f}")
                print(f"Skipped word difficulty: {skip_difficulty:.4f}")
                
        print(f"Global comprehension norm: {torch.norm(torch.tensor(env._global_comprehension)).item():.4f}")
        
    def test_reading_patterns(sentence_idx=0):
        """Compare different reading patterns on the same sentence"""
        print("\n=== Testing Reading Patterns on Same Sentence ===")
        
        # Test 1: Normal full reading
        print("\n--- Test 1: Normal Full Reading ---")
        env = SentenceReadingEnv()
        obs = env.reset(sentence_idx=sentence_idx)
        print(f"Sentence: {' '.join(env._sentence_info['words'])}\n")
        
        comprehension_history = []
        reading_sequence = []
        for i in range(min(8, env._sentence_len)):
            obs, reward, term, trunc, info = env.step(env._READ_ACTION)
            comp_norm = torch.norm(torch.tensor(env._global_comprehension)).item()
            word_norm = torch.norm(env._word_states[i]['comprehension'][-1]).item()
            comprehension_history.append((word_norm, comp_norm))
            reading_sequence.append(f"Read({i}:'{env._sentence_info['words'][i]}')")
            print_comprehension_state(env, i, "READ")
            
        print("\nFinal comprehension (full reading):", comprehension_history[-1][1])
        
        # Test 2: Reading with skips
        print("\n--- Test 2: Reading with Skips ---")
        env = SentenceReadingEnv()
        obs = env.reset(sentence_idx=sentence_idx)
        
        # Read-Skip pattern
        actions = [
            (env._READ_ACTION, f"Read(0:'{env._sentence_info['words'][0]}')", None),
            (env._READ_ACTION, f"Read(1:'{env._sentence_info['words'][1]}')", None),
            (env._SKIP_ACTION, f"Skip(2:'{env._sentence_info['words'][2]}') & Read(3:'{env._sentence_info['words'][3]}')", 2),
            (env._SKIP_ACTION, f"Skip(4:'{env._sentence_info['words'][4]}') & Read(5:'{env._sentence_info['words'][5]}')", 4),
            (env._READ_ACTION, f"Read(6:'{env._sentence_info['words'][6]}')", None),
            (env._READ_ACTION, f"Read(7:'{env._sentence_info['words'][7]}')", None),
            (env._READ_ACTION, f"Read(8:'{env._sentence_info['words'][8]}')", None),
            (env._READ_ACTION, f"Read(9:'{env._sentence_info['words'][9]}')", None)
        ]
        
        skip_comprehension_history = []
        skip_sequence = []
        for i, (action, action_desc, skipped_idx) in enumerate(actions):
            obs, reward, term, trunc, info = env.step(action)
            comp_norm = torch.norm(torch.tensor(env._global_comprehension)).item()
            
            # Get current word norm
            curr_word_norm = torch.norm(env._word_states[env._current_word_index]['comprehension'][-1]).item()
            
            # Get skipped word norm if applicable
            skipped_word_norm = None
            if skipped_idx is not None:
                skipped_word_norm = torch.norm(env._word_states[skipped_idx]['comprehension'][-1]).item()
            
            skip_comprehension_history.append((curr_word_norm, skipped_word_norm, comp_norm))
            skip_sequence.append(action_desc)
            print_comprehension_state(env, i, "SKIP" if action == env._SKIP_ACTION else "READ", skipped_idx)
            
        print("\nFinal comprehension (with skips):", skip_comprehension_history[-1][2])
        print("Skipped words:", [f"{i}:'{env._sentence_info['words'][i]}'" for i in env._skipped_words_indexes])
        print("Total words processed:", env._current_word_index + 1)
        
        # Test 3: Reading with regressions
        print("\n--- Test 3: Reading with Regressions ---")
        env = SentenceReadingEnv()
        obs = env.reset(sentence_idx=sentence_idx)
        
        # Read forward then regress pattern
        regression_actions = [
            (env._READ_ACTION, f"Read(0:'{env._sentence_info['words'][0]}')", None),    # Word 1
            (env._READ_ACTION, f"Read(1:'{env._sentence_info['words'][1]}')", None),    # Word 2
            (env._READ_ACTION, f"Read(2:'{env._sentence_info['words'][2]}')", None),    # Word 3
            (env._READ_ACTION, f"Read(3:'{env._sentence_info['words'][3]}')", None),    # Word 4
            (env._REGRESS_ACTION, f"Back(3→2:'{env._sentence_info['words'][3]}'→'{env._sentence_info['words'][2]}')", 2),  # Back to Word 2
            (env._READ_ACTION, f"Read(3:'{env._sentence_info['words'][3]}')", None),    # Word 4 again
            (env._READ_ACTION, f"Read(4:'{env._sentence_info['words'][4]}')", None),    # Word 5
            (env._READ_ACTION, f"Read(5:'{env._sentence_info['words'][5]}')", None)     # Word 6
        ]
        
        regression_comprehension_history = []
        regression_sequence = []
        for i, (action, action_desc, regress_idx) in enumerate(regression_actions):
            obs, reward, term, trunc, info = env.step(action)
            comp_norm = torch.norm(torch.tensor(env._global_comprehension)).item()
            
            # Get current word norm
            curr_word_norm = torch.norm(env._word_states[env._current_word_index]['comprehension'][-1]).item()
            
            # Get regressed word norm if applicable
            regress_word_norm = None
            if regress_idx is not None:
                regress_word_norm = torch.norm(env._word_states[regress_idx]['comprehension'][-1]).item()
            
            regression_comprehension_history.append((curr_word_norm, regress_word_norm, comp_norm))
            regression_sequence.append(action_desc)
            print_comprehension_state(env, i, "REGRESS" if action == env._REGRESS_ACTION else "READ", regress_idx)
            
        print("\nFinal comprehension (with regressions):", regression_comprehension_history[-1][2])
        print("Regression points:", [f"{i}:'{env._sentence_info['words'][i]}'" for i in env._regressed_words_indexes])
        print("Total words processed:", env._current_word_index + 1)
        
        # Compare comprehension patterns
        print("\n=== Comprehension Pattern Comparison ===")
        print("\nNormal reading:")
        for i, (word_comp, global_comp) in enumerate(comprehension_history):
            print(f"Step {i}: {reading_sequence[i]:30} → Word: {word_comp:.4f}, Global: {global_comp:.4f}")
            
        print("\nSkip reading (showing both skipped and read word comprehension):")
        for i, (curr_comp, skip_comp, global_comp) in enumerate(skip_comprehension_history):
            if skip_comp is not None:
                print(f"Step {i}: {skip_sequence[i]:50} → Current: {curr_comp:.4f}, Skipped: {skip_comp:.4f}, Global: {global_comp:.4f}")
            else:
                print(f"Step {i}: {skip_sequence[i]:50} → Current: {curr_comp:.4f}, Global: {global_comp:.4f}")
            
        print("\nRegression reading:")
        for i, (curr_comp, regress_comp, global_comp) in enumerate(regression_comprehension_history):
            if regress_comp is not None:
                print(f"Step {i}: {regression_sequence[i]:50} → Current: {curr_comp:.4f}, Regressed: {regress_comp:.4f}, Global: {global_comp:.4f}")
            else:
                print(f"Step {i}: {regression_sequence[i]:50} → Current: {curr_comp:.4f}, Global: {global_comp:.4f}")
            
        # Show progression comparison
        print("\nProgression Summary:")
        print("Normal reading progression:")
        for i, (word_comp, global_comp) in enumerate(comprehension_history):
            print(f"  Step {i}: Word: {word_comp:.4f}, Global: {global_comp:.4f}")
            
        print("\nSkip reading progression:")
        for i, (curr_comp, skip_comp, global_comp) in enumerate(skip_comprehension_history):
            if skip_comp is not None:
                print(f"  Step {i}: Current: {curr_comp:.4f}, Skipped: {skip_comp:.4f}, Global: {global_comp:.4f}")
            else:
                print(f"  Step {i}: Current: {curr_comp:.4f}, Global: {global_comp:.4f}")
            
        print("\nRegression reading progression:")
        for i, (curr_comp, regress_comp, global_comp) in enumerate(regression_comprehension_history):
            if regress_comp is not None:
                print(f"  Step {i}: Current: {curr_comp:.4f}, Regressed: {regress_comp:.4f}, Global: {global_comp:.4f}")
            else:
                print(f"  Step {i}: Current: {curr_comp:.4f}, Global: {global_comp:.4f}")
        
    # Run controlled tests
    test_reading_patterns(sentence_idx=0)  # Use first sentence for consistent comparison
        
