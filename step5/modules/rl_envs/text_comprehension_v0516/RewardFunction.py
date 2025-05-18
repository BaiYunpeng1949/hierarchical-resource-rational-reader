import numpy as np
import torch
import math

class RewardFunction():
    """
    Reward Function for text comprehension 
    """

    def __init__(self):
        self._coefficient_sentence_selection = 1
        self._coefficeint_comprehension = 1

    def compute_regress_to_sentence_reward(self):
        """
        Compute the reward for the regress action
        """
        return -0.1 * self._coefficient_sentence_selection

    def compute_read_next_sentence_reward(self):
        """
        Compute the reward for the read action
        """
        return -0.1 * self._coefficient_sentence_selection
    
    
    def compute_terminate_reward(self, num_sentences: int, num_sentences_read: int, sentence_appraisal_scores_distribution: list[float]):
        """
        Compute reward for terminating reading.
        Uses sigmoid saturation to encourage more human-like reading behavior.
        """
        # Identify valid scores first
        valid_scores = [a for a in sentence_appraisal_scores_distribution if 0 <= a <= 1]

        # Penalize for not finishing the sentence reading task
        if num_sentences_read < num_sentences:
            return -100
        else:
            # Compute geometric mean of word beliefs
            overall_comprehension_log = 0.0
            if len(valid_scores) > 0:
                for a in valid_scores:
                    overall_comprehension_log += math.log(max(a, 1e-9))
                # geometric mean
                overall_comprehension_scalar = math.exp(overall_comprehension_log / len(valid_scores))
            else:
                overall_comprehension_scalar = 0.0

            # NOTE: linear reward: linear scaling for the comprehension performance
            final_reward = 500 * self._coefficeint_comprehension * overall_comprehension_scalar

            # # TODO debug delete later
            # print(f"The final reward is: {final_reward}")
                
            return final_reward
