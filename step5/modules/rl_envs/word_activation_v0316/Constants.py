# Cognitive constraints
FOVEAL_SIZE = 8
WORKING_MEMORY_SIZE = 5
DEFAULT_FIXATION_DURATION = 8
DETERMINISTIC_WORD_ACTIVATION = True

# Heuristic model parameters
GAZE_DURATION_LAMDA = 10
NUM_WORDS_IN_LEXICON = 1000000      # Number of words in the lexicon, 1 million, used for calculating the word frequency
ZIPF_PARAM_PARETO_ALPHA = 1.0
XMIN = 1  

# Variables
NON_WORD = "NON_WORD"
EPSILON = 1e-5
CELEX_LOG_WORD_FREQ_MIN = 0
CELEX_LOG_WORD_FREQ_MAX = 6
PREDICTABILITY_MIN = 1/(2*83)  # Reference: Length, frequency, and predictability effects of words on eye movements in reading   
PREDICTABILITY_MAX = (2*83-1)/(2*83)  # Reference: Length, frequency, and predictability effects of words on eye movements in reading
LOG_FREQ_BINS = {
    "class 1": (1, 10),
    "class 2": (10, 100),
    "class 3": (100, 1000),
    "class 4": (1000, 10000),
    "class 5": (10000, 100000),
}
LOGIT_PRED_BINS = {
    "class 1": (-2.553, -1.5),
    "class 2": (-1.5, -1.0),
    "class 3": (-1.0, -0.5),
    "class 4": (-0.5, 0.0),
    "class 5": (0.0, 2.553),
}

PRIOR_AS_FREQ = 0
PRIOR_AS_PRED = 1

MAX_WORD_LENGTH = 14
MIN_WORD_LENGTH = 1