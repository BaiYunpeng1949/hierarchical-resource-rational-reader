# Cognitive constraints
FOVEAL_SIZE = 5
WORKING_MEMORY_SIZE = 5
DEFAULT_FIXATION_DURATION = 80
DETERMINISTIC_WORD_ACTIVATION = True

# Heuristic model parameters
GAZE_DURATION_LAMDA = 0.1

# Variables
NON_WORD = "NON_WORD"
EPSILON = 1e-5
CELEX_LOG_WORD_FREQ_MIN = 0
CELEX_LOG_WORD_FREQ_MAX = 6
PREDICTABILITY_MIN = 0.07   # Cater to the paper's logit predictability ranges from -2.5 to 1
PREDICTABILITY_MAX = 0.73   # Logit predictability = ln(p/(1-p)), when p=0.07, logit predictability = -2.5; when p=0.73, logit predictability = 1