# Sentences dataset
# Sentence length
MAX_SENTENCE_LENGTH = 70

# Word frequency
ONE_MILLION = 1000000        # Word frequencies are word occurances per million

# Language model
LANGUAGE_MODEL_NAME = "bert-base-uncased"

# Contexts related
CONTEXT_SIZE = 5

NOISY_OBS_SIGMA = 0.1

#########################
# Time conditions
#########################

TIME_CONDITIONS = {
    "30s": 30,
    "60s": 60,
    "90s": 90,
}

#########################
# Reading speed: approximated from our human participants' preliminary tests
#########################

# READING_SPEED = 0.4     # 0.252 is too fast to our participants. NOTE If wants to have a higher fidelity, sample a from an approximated distribution for training
READING_SPEED = 0.252

#########################
# Datasets
#########################

DATASETS = {
    "Ours": "Bai",
    "ZuCo1.0": "ZuCo",
}