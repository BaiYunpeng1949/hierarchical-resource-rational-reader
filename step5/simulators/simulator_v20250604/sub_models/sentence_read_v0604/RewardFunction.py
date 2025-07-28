import numpy as np
import torch
import math

from . import Utilities

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
        factor_range = [0, 6]       # increase the range from 1 to 5 --> 1 to 10
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
    
    def compute_step_wise_comprehension_gain(self, w_comprehension_gain, old_beliefs, new_beliefs):
        """
        Compute the step-wise reward for the comprehension gain
        """
        # Make sure these beliefs are valid
        valid_new_beliefs = [b for b in new_beliefs if b != -1]
        valid_old_beliefs = [b for b in old_beliefs if b != -1]
        comprehension_gain = Utilities.calc_dynamic_sentence_comprehension_score(valid_new_beliefs) - Utilities.calc_dynamic_sentence_comprehension_score(valid_old_beliefs)
        return w_comprehension_gain * comprehension_gain
    
    def compute_terminate_reward(self, sentence_len: int, num_words_read: int, words_beliefs: list[float], remaining_time: float, expected_sentence_reading_time: float, w_comprehension_vs_time_pressure: float):
        """
        Compute reward for terminating reading.
        Uses sigmoid saturation to encourage more human-like reading behavior.
        """
        
        if num_words_read < sentence_len:   # If the agent did not finish the sentence reading, then the reward is 0
            not_finished_penalty = -10
            logs = {
                "comprehension_reward": 0,
                "penalty_for_wasting_time": 0,
                "final_reward": not_finished_penalty,
            }
            return not_finished_penalty, logs
        else:    # If the agent finished the sentence reading, then compute the reward
            final_step_bonus_fundation_value = 1.0
            
            # Compute geometric mean of word beliefs
            overall_comprehension_log = 0.0
            if len(words_beliefs) > 0:
                # Apply the softmin function to calculate the sentence-appraisals, such to stress the importance of the accurate word understandings, i.e., higher appraisals
                overall_comprehension_scalar = Utilities.calc_dynamic_sentence_comprehension_score(words_beliefs, mode="mean")
            else:
                overall_comprehension_scalar = 0.0
            
            comprehension_reward = final_step_bonus_fundation_value * overall_comprehension_scalar

            if remaining_time < 0:    # If the agent finished the sentence reading out of expected time, then apply some penalties
                penalty_for_wasting_time = final_step_bonus_fundation_value * (remaining_time / expected_sentence_reading_time)    # NOTE: see if need a parameter to tune here later (re-use w_comprehension_vs_time_pressure)
            else:
                penalty_for_wasting_time = 0

            # NOTE: linear reward: linear scaling for the comprehension performance
            final_reward = comprehension_reward + w_comprehension_vs_time_pressure * penalty_for_wasting_time

            logs = {
                "comprehension_reward": comprehension_reward,
                "penalty_for_wasting_time": penalty_for_wasting_time,
                "final_reward": final_reward,
            }

            # TODO maybe need to check the metrics -- how come the skipping rate is above 50%? Should not exceed that.
                
            return final_reward, logs
