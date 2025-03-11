import numpy as np


class RewardFunction():
    """
    Reward Function
    """

    def __init__(self):
        pass

    def compute_regress_reward(self, states: list):
        """
        Compute the reward for the regress action
        """

        return -0.1

    def compute_read_reward(self, states: list):
        """
        Compute the reward for the read action
        """

        return -0.1
    
    def compute_skip_reward(self, states: list):
        """
        Compute the reward for the skip action
        """

        return -0.1
    
    def compute_terminate_reward(self, sentence_appraisals: list):
        """
        Compute the reward for the terminate action
        This should be a positive reward based on the sentence's overall appraisal.

        NOTE: maybe need to make this more explainable for a NHB paper. Now it is tricky. 
            Not really sentence comprehension or coherence.
        """

        # To avoid sampled sentences' length, we apply 10 * average sentence appraisals as the reward.
        # Use the Bernoulli distribution to sample the reward. Each word's reward is independent.
        sampled_words_info_gain = [np.random.choice([0, 1], p=[1 - a, a]) for a in sentence_appraisals]

        # Get the average reward
        average_reward = np.mean(sampled_words_info_gain)

        # Apply the reward
        reward = 10 * average_reward

        return reward