from modules.rl_envs.sentence_read_v0306 import Constants


class TransitionFunction():
    """
    Transition Function that models human-like reading behavior including comprehension uncertainty
    and conditions that may trigger regressions.

    TODO: once the current numeric sim env works, show some basic trends; I could try to involve the language model for real-time reading later.
    """

    def __init__(self):
        """
        Comprehension thresholds and weights
        """

        # NOTE: these are the parameters could be tuned for more human-like reading behavior
        self._comprehension_threshold = 0.8  # Increased threshold to make comprehension more difficult, previously 0.7
        self._integration_decay = 0.3  # Increased decay rate to create more need for regressions, previously 0.1
        self._context_weight = 0.2  # Reduced context weight to make skipping less reliable
        
        self._regression_boost = 0.5  # Increased regression boost, previously 0.5
        self._appraisal_decay_baseline = 0.2     # Previously 0.3

        self._skip_word_predictability_weight = 0.3  # Previously 0.3

        self._context_size = 3
        self._initial_comprehension = 0.5  # Reduced initial comprehension, previously 0.7
        self._initial_skip_comprehension = 0.5  # Reduced skip comprehension significantly, previously 0.7

    def reset(self, sentence_length: int):
        """
        Reset the transition function with initial comprehension states
        """
        # TODO maybe read from the pre-read sentence clusters later
        init_appraisals_states = [0] * sentence_length
        init_appraisals_states.extend([-1] * (Constants.MAX_SENTENCE_LENGTH - sentence_length))
        return init_appraisals_states
    
    def update_state_regress(self, appraisals: list, current_word_index: int):
        """
        Update state for regression action. Regression is more likely when:
        1. Previous word has low comprehension (appraisal)
        2. Current word creates integration difficulty
        """
        if current_word_index <= 0:
            action_validity = False
        else:
            action_validity = True
            current_word_index -= 1
            # Regression improves comprehension but not to perfect level
            prev_appraisal = appraisals[current_word_index]
            appraisals[current_word_index] = min(1.0, prev_appraisal + self._regression_boost)
        
        return appraisals, current_word_index, action_validity
    
    def update_state_read_next_word(self, appraisals: list, current_word_index: int, sentence_length: int): 
        """
        Update state for reading action. Reading effectiveness depends on:
        1. Context from previous words
        2. Natural decay of previous word comprehension
        """
        if current_word_index >= sentence_length - 1:
            action_validity = False
        else:
            action_validity = True
            current_word_index += 1
            
            # NOTE: a heuristic comprehension mechanism
            # Apply decay to previous words' appraisals to create potential need for regression
            #  NOTE: this is a simple fixed decay mechanism; try complex ones later when needed.
            for i in range(current_word_index):
                if appraisals[i] > 0:  # Only decay positive appraisals
                    appraisals[i] = max(self._appraisal_decay_baseline, appraisals[i] - self._integration_decay)
            
            # # New word comprehension depends on previous context
            # context_comprehension = sum(appraisals[max(0, current_word_index-self._context_size):current_word_index]) / self._context_size if current_word_index > 0 else 1.0
            # initial_comprehension = self._initial_comprehension + self._context_weight * context_comprehension  # Base comprehension + context effect
            # appraisals[current_word_index] = initial_comprehension
            appraisals[current_word_index] = self.calc_read_word_appraisal(appraisals, current_word_index, sentence_length) 

        return appraisals, current_word_index, action_validity
    
    def update_state_skip_next_word(self, appraisals: list, current_word_index: int, sentence_length: int, skip_word_predictability: float):
        """
        Update state for skipping action. Skipping effectiveness depends on:
        1. Word predictability
        2. Context quality
        3. Risk of comprehension failure
        """
        if current_word_index >= sentence_length - 1:     # Do not skip 
            action_validity = False
        else:       # Normally skip the next word
            action_validity = True
            
            # Calculate context quality from previous words
            context_quality = sum(appraisals[max(0, current_word_index-(self._context_size-1)):current_word_index+1]) / self._context_size if current_word_index >= 0 else 0
            
            # Skipped word comprehension depends on both predictability and context
            skip_comprehension = min(0.7, self._initial_skip_comprehension + self._context_weight * context_quality + self._skip_word_predictability_weight * skip_word_predictability)
            
            # Update skipped word's appraisal
            appraisals[current_word_index-1] = skip_comprehension

            # Update positions
            if current_word_index >= sentence_length - 2:
                current_word_index = sentence_length    # Anchor the fixation position to <EOS>
            else:
                current_word_index += 2

                appraisals[current_word_index] = self.calc_read_word_appraisal(appraisals, current_word_index, sentence_length)
            
            # Higher chance of low comprehension when skipping
            if skip_comprehension < self._comprehension_threshold:
                # Decay previous words' comprehension more to increase regression probability
                for i in range(current_word_index-1):
                    if appraisals[i] > 0:
                        appraisals[i] = max(0.2, appraisals[i] - 2.5 * self._integration_decay)

        return appraisals, current_word_index, action_validity
    
    def calc_read_word_appraisal(self, appraisals: list, current_word_index: int, sentence_length: int):
        """
        Calculate the appraisal for the next word to read
        """
        if current_word_index >= sentence_length - 1:
            return 0
        else:
            # New word comprehension depends on previous context
            context_comprehension = sum(appraisals[max(0, current_word_index-self._context_size):current_word_index]) / self._context_size if current_word_index > 0 else 1.0
            initial_comprehension = self._initial_comprehension + self._context_weight * context_comprehension  # Base comprehension + context effect
            appraisals[current_word_index] = initial_comprehension

            return appraisals[current_word_index]