from modules.rl_envs.sentence_read_v0306 import Constants


class SentencesManager():
    """
    Sentences Manager
    """

    def __init__(self):
        
        self._max_num_words = MAX_NUM_WORDS_PER_SENTENCE
        self._min_num_words = MIN_NUM_WORDS_PER_SENTENCE

        self.num_words_of_sampled_sentencec = None
        self.preset_predictabilities_of_sampled_sentence = None

    def reset(self):
        """
        Reset the sentences manager
        """

        self.num_words_of_sampled_sentence = random.randint(self._min_num_words, self._max_num_words)
        self.preset_predictabilities_of_sampled_sentence = [random.uniform(0, 1) for _ in range(self.num_words_of_sampled_sentence)]

        # TODO in the transition function, need to decide whether change 
        # the predictabilities according to the action.
        
        
        