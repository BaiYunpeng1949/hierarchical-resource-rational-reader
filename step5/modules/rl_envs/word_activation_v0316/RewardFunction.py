class RewardFunction(): 
    """
    Reward Function
    """

    def __init__(self, weight_effort_cost=1.0, weight_recognition_bonus=1.0):
        self._weight_effort_cost = weight_effort_cost
        self._weight_recognition_bonus = weight_recognition_bonus

    def get_step_wise_effort_cost(self, is_action_valid):
        if is_action_valid:
            return -1 * self._weight_effort_cost
        else:
            return -1 * self._weight_effort_cost        # For faster training, could remove

    def get_terminate_reward(self, word_to_recognize, word_to_activate):
        
        Bonus = self._weight_recognition_bonus * 10
        
        if word_to_recognize == word_to_activate:
            return Bonus
        else:
            return -1 * Bonus