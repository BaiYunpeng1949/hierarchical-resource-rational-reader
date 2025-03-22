import numpy as np
import torch


class RewardFunction():
    """
    Reward Function
    """

    def __init__(self):
        self._coefficient_eye_movement_cost = 5
        self._coefficeint_comprehension = 1

    def compute_regress_reward(self):
        """
        Compute the reward for the regress action
        """

        return -0.1 * self._coefficient_eye_movement_cost

    def compute_read_reward(self):
        """
        Compute the reward for the read action
        """

        return -0.1 * self._coefficient_eye_movement_cost
    
    def compute_skip_reward(self):
        """
        Compute the reward for the skip action
        """

        return -0.1 * self._coefficient_eye_movement_cost
    
    def compute_terminate_reward(self, sentence_len: int, num_words_read: int, words_beliefs: list[float]):
        """
        Compute reward for terminating reading.
        Rewards high comprehension, penalizes poor comprehension.
        global_comprehension: tensor of shape [hidden_size] or [num_layers, hidden_size]
        """
        # Penalize for not finishing the sentence reading task
        if num_words_read < sentence_len:
            linear_penalty = -100 * (sentence_len - num_words_read) / sentence_len - 10
            return linear_penalty 
        else:
            overall_comprehension_scalar = 1
            for b in words_beliefs:
                overall_comprehension_scalar *= b   
                # NOTE: maybe need to use a sigmoid function to smooth edges; then leave some tolerance 
                # for not regressing; but let me try it later
                
            return 100 * self._coefficeint_comprehension * overall_comprehension_scalar
