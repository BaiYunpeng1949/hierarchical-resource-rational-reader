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
    
    
    def compute_terminate_reward(self, num_sentences: int, num_sentences_read: int, sentence_appraisal_scores_distribution: list[float], coverage_factor: float):
        """
        Compute reward for terminating reading.
        Uses sigmoid saturation to encourage more human-like reading behavior.

        NOTE: might need to change to another reward function design later, where cumulative comprehensions are applied, 
        not just the averaged comprehension score across all sentences that does not account for the reading progress's effect.
        """
        # Identify valid scores first
        valid_scores = [a for a in sentence_appraisal_scores_distribution if 0 <= a <= 1]

        # NOTE: but if there's no cut-off from not finishing the sentence, the agent would never be eager to read more.

        # Compute the overall comprehension score
        overall_comprehension_scalar = 0.0
        if len(valid_scores) > 0:
            overall_comprehension_scalar = max(0, calc_dynamic_text_comprehension_score(valid_scores, mode=Constants.COMPREHENSION_SCORE_MODE, tau=Constants.TAU))
        
        # Since no unfinish penalty, need to tune the quality vs. quantity.
        # NOTE: METHOD 1 times a coverage factor; if it does not work so good, METHOD 2 do the reward shaping by adding the coverage portion.
        assert num_sentences_read == len(valid_scores), f"num_sentences_read={num_sentences_read} != len(valid_scores)={len(valid_scores)}"
        coverage_rate = num_sentences_read / num_sentences

        # NOTE: linear reward: linear scaling for the comprehension performance
        # final_reward = 100 * self._coefficeint_comprehension * overall_comprehension_scalar * coverage_rate
        final_reward = 100 * self._coefficeint_comprehension * overall_comprehension_scalar * (coverage_rate ** coverage_factor)
        # final_reward = 100 * (coverage_factor * coverage_rate + (1 - coverage_factor) * overall_comprehension_scalar)

        return final_reward