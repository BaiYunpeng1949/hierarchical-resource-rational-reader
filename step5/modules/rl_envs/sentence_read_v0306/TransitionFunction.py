class TransitionFunction():
    """
    Transition Function
    """

    def __init__(self):
        pass

    def reset(self):
        """
        Reset the transition function
        """

        pass
    
    def update_state_regress(self, states: list):
        """
        Update the state
        """

        # First check the action's validity, need to punish the agent for non-sensen / non-informative actions
        # So later need to pass a variable to notify this.

        # TODO fix

        return states
    
    def update_state_read_next_word(self, states: list):
        """
        Update the state
        """

        # Check the validity first.

        # TODO fix

        return states
    
    def update_state_skip_next_word(self, states: list):
        """
        Update the state
        """

        # Check the validity first.

        # TODO fix

        return states
    
    def check_terminate_state(self, states: list):
        """
        Check if the state is a terminate state
        """
        
        done = False

        if reading_progress >= len(sentence):
            done = True

        return done
        