# These two parameters are set as such because our read_under_time_pressure dataset has 6-11 sentences per text.
MAX_NUM_SENTENCES = 12        # number of the sentences
MIN_NUM_SENTENCES = 5

# Parameters sampled from the real dataset, do the matching later
MAX_SENTENCE_LENGTH = 30
MIN_SENTENCE_LENGTH = 5

# Parameters for the text length
MAX_TEXT_LENGTH = 150
MIN_TEXT_LENGTH = 130

# Approximated reading speed (seconds per word) -- 238 words per minute: 0.252 seconds per word
READING_SPEED = 0.4     # 0.252 is too fast to our participants.

# A new feature: memory decay over time
MEMORY_DECAY_CONSTANT = 0.02

# Time conditions (in seconds)
TIME_CONDITIONS = {
    "30s": 30,
    "60s": 60,
    "90s": 90,
}