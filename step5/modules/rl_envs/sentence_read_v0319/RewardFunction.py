import numpy as np
import torch
import math

class RewardFunction():
    """
    Reward Function with sigmoid-based comprehension saturation
    """

    def __init__(self):
        self._coefficient_eye_movement_cost = 5
        self._coefficeint_comprehension = 1
        # Sigmoid parameters for comprehension saturation
        self._sigmoid_scale = 10.0  # Controls how quickly the sigmoid saturates, larger the agent more easily to please
        self._sigmoid_shift = 0.5  # Controls where the sigmoid saturates, smaller the agent more easily to please

    def _sigmoid(self, x):
        """
        Compute sigmoid function for reward saturation
        Args:
            x: Input value in [0,1]
        Returns:
            Saturated value in [0,1]
        """
        return 1 / (1 + math.exp(-self._sigmoid_scale * (x - self._sigmoid_shift)))

    def compute_regress_reward(self, w_regression_cost: float):
        """
        Compute the reward for the regress action
        """
        # return -0.1 * self._coefficient_eye_movement_cost * 0.2 # NOTE: do this for increasing the regression probability, till ppo_87, no regression leared
        # return 0
        factor_range = [1, 10]       # increase the range from 1 to 5 --> 1 to 10
        factor = np.interp(w_regression_cost, [0, 1], factor_range)
        return -0.1 * self._coefficient_eye_movement_cost * factor

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
        Uses sigmoid saturation to encourage more human-like reading behavior.
        """
        # Penalize for not finishing the sentence reading task
        if num_words_read < sentence_len:
            return -100
        else:
            # Compute geometric mean of word beliefs
            overall_comprehension_log = 0.0
            if len(words_beliefs) > 0:
                for b in words_beliefs:
                    overall_comprehension_log += math.log(max(b, 1e-9))
                # geometric mean
                overall_comprehension_scalar = math.exp(overall_comprehension_log / len(words_beliefs))
            else:
                overall_comprehension_scalar = 0.0
            
            # NOTE: satisfication reward: saturation for the comprehension performance
            # # Apply sigmoid saturation to comprehension
            # saturated_comprehension = self._sigmoid(overall_comprehension_scalar)
            
            # # Scale the final reward (100 is the max reward)
            # final_reward = 100 * self._coefficeint_comprehension * saturated_comprehension

            # NOTE: linear reward: linear scaling for the comprehension performance
            final_reward = 100 * self._coefficeint_comprehension * overall_comprehension_scalar
                
            return final_reward
