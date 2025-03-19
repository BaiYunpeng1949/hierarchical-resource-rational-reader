import numpy as np
import torch


class RewardFunction():
    """
    Reward Function
    """

    def __init__(self):
        self._terminate_reward = 1.0
        self._good_comprehension_threshold = 0.5  # Threshold for good comprehension

    def compute_regress_reward(self):
        """
        Compute the reward for the regress action
        """

        return -0.1

    def compute_read_reward(self):
        """
        Compute the reward for the read action
        """

        return -0.1
    
    def compute_skip_reward(self):
        """
        Compute the reward for the skip action
        """

        return -0.1
    
    def compute_terminate_reward(self, global_comprehension):
        """
        Compute reward for terminating reading.
        Rewards high comprehension, penalizes poor comprehension.
        global_comprehension: tensor of shape [hidden_size] or [num_layers, hidden_size]
        """
        # If global_comprehension is 2D (from GRU layers), use the last layer
        if len(global_comprehension.shape) > 1:
            global_comprehension = global_comprehension[-1]  # Use last layer's state
            
        # Convert to scalar value between 0 and 1
        comprehension_score = torch.mean(torch.abs(global_comprehension)).item()
        
        if comprehension_score > self._good_comprehension_threshold:
            return self._terminate_reward
        else:
            return -self._terminate_reward