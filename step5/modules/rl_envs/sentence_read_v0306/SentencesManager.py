import random

from modules.rl_envs.sentence_read_v0306 import Constants


class SentencesManager():
    """
    Sentences Manager
    """

    def __init__(self):
        
        self._max_sent_len = Constants.MAX_SENTENCE_LENGTH
        self._min_sent_len = Constants.MIN_SENTENCE_LENGTH

        # TODO since I am dealling with the skip probability and regress probability, they need to be tested on the same words.
        #   So I need to have a fixed set of sentences as the dataset, so all the words are fixed.
        #   The current plan is reading the sentences from the ZuCo 1.0 Task 2 Natural Reading data.

    def reset(self):
        """
        Reset the sentences manager
        """

        # TODO so later here I am not randomizing, but sampling; but this will lose the agent's generalizability. 
        #So try this first, see whether reasonable decisions can be made.
        num_words_of_sampled_sentence = random.randint(self._min_sent_len, self._max_sent_len)
        preset_predictabilities_of_sampled_sentence = [random.uniform(0, 1) for _ in range(num_words_of_sampled_sentence)]

        # TODO in the transition function, need to decide whether change 
        # the predictabilities according to the action.
        
        return num_words_of_sampled_sentence, preset_predictabilities_of_sampled_sentence
        