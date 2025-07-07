import numpy as np
import torch
import math

from .Utilities import calc_dynamic_text_comprehension_score
from . import Constants

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

            # # Compute geometric mean of word beliefs
            # overall_comprehension_log = 0.0
            # if len(valid_scores) > 0:
            #     for a in valid_scores:
            #         overall_comprehension_log += math.log(max(a, 1e-9))
            #     # geometric mean
            #     overall_comprehension_scalar = math.exp(overall_comprehension_log / len(valid_scores))
            # else:
            #     overall_comprehension_scalar = 0.0

            # NOTE the softmin-based bonus in the reward function might solve the issue of which sentences to regress to, e.g., the weak ones; 
            # but it might not solve the issue of the lowest sentences should be fixed first, because the effect only happens in the very last step.
            # So maybe I need step-wise small incentives for effective regressions.
            
            # Compute the overall comprehension score
            overall_comprehension_scalar = max(0, calc_dynamic_text_comprehension_score(valid_scores, mode=Constants.COMPREHENSION_SCORE_MODE, tau=Constants.TAU))

            # NOTE: linear reward: linear scaling for the comprehension performance
            final_reward = 100 * self._coefficeint_comprehension * overall_comprehension_scalar

            # # TODO debug delete later
            # print(f"The final reward is: {final_reward}")
                
            return final_reward
