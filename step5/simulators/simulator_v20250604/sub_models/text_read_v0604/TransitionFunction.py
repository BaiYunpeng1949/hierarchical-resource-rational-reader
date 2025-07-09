import os
import yaml
import numpy as np
from . import Constants

class TransitionFunction():
    """
    Transition Function for text comprehension
    """

    def __init__(self):
        pass
    
    def update_state_read_next_sentence(self, current_sentence_index, sentence_appraisal_scores_distribution, num_sentences):
        # Initialize with -1s for all possible sentences
        read_sentence_appraisal_scores_distribution = [-1] * Constants.MAX_NUM_SENTENCES
        
        if current_sentence_index < num_sentences - 1:
            # Copy the appraisal scores up to current sentence index
            # Move one sentence forward
            current_sentence_index += 1
            # Get all appraisals
            for i in range(current_sentence_index + 1):
                read_sentence_appraisal_scores_distribution[i] = sentence_appraisal_scores_distribution[i]
            return read_sentence_appraisal_scores_distribution, True
        else:
            # Copy all appraisal scores
            for i in range(num_sentences):
                read_sentence_appraisal_scores_distribution[i] = sentence_appraisal_scores_distribution[i]
            return read_sentence_appraisal_scores_distribution, False

    def update_state_regress_to_sentence(self, revised_sentence_index, furtherest_read_sentence_index, read_sentence_appraisal_scores_distribution):
        try:
            assert revised_sentence_index < Constants.MAX_NUM_SENTENCES
        except:
            print(f"revised_sentence_index: {revised_sentence_index}, furtherest_read_sentence_index: {furtherest_read_sentence_index}, read_sentence_appraisal_scores_distribution: {read_sentence_appraisal_scores_distribution}")
            raise ValueError(f"revised_sentence_index is out of range (>= Constants.MAX_NUM_SENTENCES = {Constants.MAX_NUM_SENTENCES})")
        
        if revised_sentence_index <= furtherest_read_sentence_index:
            read_sentence_appraisal_scores_distribution[revised_sentence_index] = 1.0        # A simple set to 1.0 for the revised sentence
            return read_sentence_appraisal_scores_distribution, True
        else:
            return read_sentence_appraisal_scores_distribution, False
    
    def optimize_select_sentence_to_regress_to(self, current_sentence_index, read_sentence_appraisal_scores_distribution):
        """
        Optimize the sentence to regress to. This is a simple greedy algorithm that selects the sentence with the lowest appraisal score.
        """
        # Get the valid sentences appraisal scores
        valid_sentences_appraisals = [a for a in read_sentence_appraisal_scores_distribution if a != -1]

        if len(valid_sentences_appraisals) == 0:
            return current_sentence_index
        
        # Select the sentence with the lowest appraisal score
        revised_sentence_index = valid_sentences_appraisals.index(min(valid_sentences_appraisals))

        # Guarantee there is a valid sentence to regress to
        assert revised_sentence_index != -1, "No valid sentence to regress to"

        return revised_sentence_index
    
    def apply_time_independent_memory_decay(self, read_sentence_appraisal_scores_distribution, sentence_index_do_not_decay, apply=False):
        """
        Apply the memory decay over time to the read sentence appraisal scores distribution.
        """
        if not apply:
            return read_sentence_appraisal_scores_distribution.copy()
        else:
            # Apply the memory decay over time to the read sentence appraisal scores distribution
            for i in range(len(read_sentence_appraisal_scores_distribution)):
                if i != sentence_index_do_not_decay and read_sentence_appraisal_scores_distribution[i] != -1:
                    read_sentence_appraisal_scores_distribution[i] = np.clip(read_sentence_appraisal_scores_distribution[i] - Constants.MEMORY_DECAY_CONSTANT, 0, 1)
            
            return read_sentence_appraisal_scores_distribution.copy()
    
    def update_state_time(self, elapsed_time, sentence_reading_time, time_condition_value):
        """
        Update the state of the time.
        """
        elapsed_time += sentence_reading_time
        remaining_time = time_condition_value - elapsed_time
        return elapsed_time, remaining_time