from modules.rl_envs.sentence_read_v0306 import Constants


class TransitionFunction():
    """
    Transition Function
    """

    def __init__(self):
        pass

    def reset(self, sentence_length: int):
        """
        Reset the transition function
        """

        init_appraisals_states = [0] * sentence_length
        init_appraisals_states.extend([-1] * (Constants.MAX_SENTENCE_LENGTH - sentence_length))
        return init_appraisals_states
    
    def update_state_regress(self, appraisals: list, current_word_index: int):
        """
        Update the state

        NOTE: we use the very simple mechnism first: when regressing the word, that word's appraisal will be set to 1.0
        NOTE: the second simplicity is we only consider the last word regression 
            (have empirical data as evidence, most of the time the last word is regressed)
        """

        # First check the action's validity, is there a word before to regress to?
        if current_word_index <= 0:
            action_validity = False
        else:
            action_validity = True
            current_word_index -= 1
            appraisals[current_word_index] = 1.0
        
        return appraisals, current_word_index, action_validity
    
    def update_state_read_next_word(self, appraisals: list, current_word_index: int, sentence_length: int): 
        """
        Update the state
        """

        # First check the action's validity, is there a word to read?
        if current_word_index >= sentence_length - 1:
            action_validity = False
        else:
            action_validity = True
            current_word_index += 1
            appraisals[current_word_index] = 1.0

        return appraisals, current_word_index, action_validity
    
    def update_state_skip_next_word(self, appraisals: list, current_word_index: int, sentence_length: int, skip_word_predictability: float):
        """
        Update the state

        NOTE: A key simplicity of this design is that skip the next word makes its appraisal directly inherit its predictability; 
            another simplicity is that we only consider the next word skip (with empirical evidence).
        """

        # First check the action's validity, is there a word to skip? 
        if current_word_index >= sentence_length - 1:       # Cannot skip the last word 
            action_validity = False
        else:       # Normal case: skip the next word
            action_validity = True
            if current_word_index >= sentence_length - 2:
                current_word_index = sentence_length   # Anchor at the end of the sentence <EOS>
            else:       # Normal case: skip the next word and read the following word
                current_word_index += 2
                appraisals[current_word_index] = 1.0
            appraisals[current_word_index-1] = skip_word_predictability

        return appraisals, current_word_index, action_validity
        