# Constants
# FONT_PATH = r"C:\\Windows\\Fonts\\Arial.ttf"
# FONT_PATH = r"C:\\Windows\\Fonts\\COUR.TTF"
FONT_PATH = "/home/baiy4/reader-agent-zuco/step5/data/assets/fonts/cour.ttf"
RED_TO_GRAY = 0.299
GREEN_TO_GRAY = 0.587
BLUE_TO_GRAY = 0.114

ONE = 1
ZERO = 0
HALF = 0.5
NEGATIVE_ONE = -1
TEN = 10
THREE = 3
FIFTEEN = 15
FIVE = 5
EIGHT = 8
SEVENTY = 70
FORTY = 40
FIFTY = 50
UPDATE_PROCESSING_CYCLE_THRESHOLD = 9    # The number of words in a phrase
NA = "NA"
NO_NEXT_WORD = "NO_NEXT_WORD"
NOT_READ = "NOT_READ"
REGRESS = "REGRESS"
NOT_REGRESS = "NOT_REGRESS"


# New Oculomotor Controller
NO_LETTERS_SAMPLED = "NO_LETTERS_SAMPLED"


# Lexicon and image generation related constants
# ----------------------------------------------------------------------------------------------------------------------
PUNCTUATION_MARKS = "_punctuation_marks"
SYMBOLS = [
    '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@',
    '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~',
    ]
REGEX_PRINTABLE_ASCII = r'[^\x20-\x7E]'
REGEX_ALL_ASCII = r'[^\x00-\x7F]'
END_PUNCTUATION_MARKS = ['.', '!', '?', ';', '"', "'", ')', ']', '}', '>']
PROMPT_DELIMITER = "|" * 10  # The delimiter for the prompt
HIGH_RELEVANCE = "high_relevance"
MEDIUM_RELEVANCE = "medium_relevance"
LOW_RELEVANCE = "low_relevance"
RELEVANCE = "relevance"
MEMORY_LOSS_MARK = "X" * 10

# Dictionary keys
SENTENCE_IDX = "sentence_idx"
SECTION_IDX = "section_idx"

# Pseudo OCR related constants
# ----------------------------------------------------------------------------------------------------------------------
SURROUNDING_NUM_WORDS = 1

# Constants for model terminologies
# ----------------------------------------------------------------------------------------------------------------------
PHNOLOGICAL_INFO = "phonological_info"
VISUOSPATIAL_INFO = "visuospatial_info"
GROUND_TRUTH_READ_CONTENT = "ground_truth_read_content"

# Constants for modes
# ----------------------------------------------------------------------------------------------------------------------
TRAIN = "train"
CONTINUAL_TRAIN = "continual_train"
TEST = "test"
DEBUG = "debug"
SIMULATE = "simulate"
BBOX_STIM = "bounding_boxed_stimulus"

# Constants for the words and the letters
# ----------------------------------------------------------------------------------------------------------------------
MAX_WORD_LEN = 16    # Set to 14 -- Previously 12, then 8; 20 contains a lot of words I generated randomly
MIN_WORD_LEN = 1
MAX_NUM_WORDS = 150
MIN_NUM_WORDS = 120

# Constants for the images recognition
# ----------------------------------------------------------------------------------------------------------------------
OCR_COVERAGE_THRESHOLD = 75     # the circle must cover xx percent of the letter, previously 85
FOVEA_FACTOR = 1.30

# Constants for the files
# ----------------------------------------------------------------------------------------------------------------------
MD_FILE_NAME = "metadata.json"
SIM_DATA_FOLDER_NAME_PREFIX = f"{SIMULATE}_xep_TimeWeight"
SIM_DATA_JS_FILE_NAME = f"{SIMULATE}_xep.json"
SIM_DATA_CSV_FILE_NAME = f"{SIMULATE}_xep.csv"
SIM_DATA_TEXT_LEVEL_W_KINTSCH_JS_FILE_NAME = f"{SIMULATE}_xep_text_level_w_Kintsch.json"
SIM_DATA_TEXT_LEVEL_WO_KINTSCH_MEMORY_RETRIEVAL_JS_FILE_NAME = f"{SIMULATE}_xep_text_level_wo_kintsch_memory_retrieval.json"
SIM_DATA_SENTENCE_LEVEL_JS_FILE_NAME = f"{SIMULATE}_xep_sentence_level.json"
SIM_DATA_WORD_LEVEL_JS_FILE_NAME = f"{SIMULATE}_xep_word_level.json"

# Generator's configuration
LINE_SPACING = 2.5 * 16
LEFT_MARGIN = 400

RIGHT_MARGIN = LEFT_MARGIN
config = {
    "generator": {
        "dft_words": {
            1: ["the", "wind", "whispered", "secrets", "through", "the", "leaves", "of", "the", "ancient", "oak",
                "tree", "as", "a", "curious", "cat", "cautiously", "crept", "along", "the", "narrow", "ledge",
                "its", "eyes", "glittering", "in", "the", "moonlight"],
            # The wind whispered secrets through the leaves of the ancient oak tree as a curious cat cautiously crept along the narrow ledge, its eyes glittering in the moonlight.
        },
        "test_size": 0.1,
    },
    "positions": {
        "sentence_center": "center",
        "sentence_top_center": "top_center",
        "sentence_random": "random",
    },
    "concrete_configs": {
        "img_size": (int(1920), int(1080)),   # W, H: (1920, 1080): meet the size of the Tobii Pro Spectrum monitor.
        "word_size": 16,
        "foveal_size": (int(80), int(45)),    # This variable is only related to word_size NOTE (80, 45) corresponds to 7-8 letters processed by the model;
        "parafoveal_size": (int(160), int(45)),
        "peripheral_size": (int(1920), int(1080)),
        "training_foveal_and_peripheral_size": (int(80), int(45)),  # This is the size of the image that will be used for training the model
        "num_images": 10,
        "num_words": 100,
        "random num_words": True,
        "corpus": False,
    },
    "vis_configs": {
        "amp_ratio": 1,
    }
}

md = {
    "config": "config",
    "words": "words",
    "lexicon file name": "lexicon file name",
    "corpus file name": "corpus file name",
    "corpus json file name": "corpus json file name",
    "corpus json file dir": "corpus json file dir",
    "corpus": "corpus",
    "domain": "domain",
    "img size": "img size",
    "word size": "word size",
    "foveal size": "foveal size",
    "parafoveal size": "parafoveal size",
    "peripheral size": "peripheral size",
    "training foveal and peripheral size": "training foveal and peripheral size",
    "num images": "num images",
    "num words": "num words",
    "background color": "background color",
    "word color": "word color",
    "pos sentences": "pos sentence",
    "x": "x",
    "y": "y",
    "x_norm": "x_norm",
    "y_norm": "y_norm",
    "x_init": "x_init",
    "y_init": "y_init",
    "max_x_offset": "max_x_offset",
    "max_y_offset": "max_y_offset",
    "word_width": "word_width",
    "word_height": "word_height",
    "line_height": "line_height",
    "y_lines": "y_lines",
    "letter width": "letter width",
    "letter height": "letter height",
    "letter left": "letter left",
    "letter top": "letter top",
    "letter right": "letter right",
    "letter bottom": "letter bottom",
    "letter box left": "letter box left",
    "letter box top": "letter box top",
    "letter box right": "letter box right",
    "letter box bottom": "letter box bottom",
    "images": "images",
    "letter index": "letter index",
    "letters": "letters",
    "letter boxes": "letter boxes",
    "word": "word",
    "word_bbox": "word_bbox",
    "word length": "word length",
    "position": "position",
    "index": "index",
    "normalized index": "normalized index",
    "letters metadata": "letters metadata",
    "image index": "image index",
    "filename": "filename",
    "words metadata": "words metadata",
    "selected words": "selected words",
    "selected words indexes": "selected words indexes",
    "selected words norm indexes": "selected words norm indexes",
    "visuospatial info": "visuospatial info",
    "line index": "line index",
    "word index in line": "word index in line",
    "lines number": "lines number",
    "words number in line": "words number in line",
    "normalised_masked_downsampled_peripheral_view": "normalised_masked_downsampled_peripheral_view",
    "normalised_original_image_pixels": "normalised_original_image_pixels",
    "normalised_foveal_patch": "normalised_foveal_patch",
    "relative_bbox_foveal_patch": "relative_bbox_foveal_patch",
    "FOVEAL_PATCH": "FOVEAL_PATCH_",
    "PERIPHERAL_VIEW": "PERIPHERAL_VIEW_",
}

LV_ONE_DASHES = "----------"   # 10
LV_TWO_DASHES = "--------------------"  # 20
LV_THREE_DASHES = "------------------------------"  # 30
LV_FOUR_DASHES = "----------------------------------------"  # 40

# Simulator data save dir
# ----------------------------------------------------------------------------------------------------------------------
SIM_DATA_SAVE_PATH_LINUX = 'data/sim_results/'

# Constants for running the simulator
# ----------------------------------------------------------------------------------------------------------------------
SIMULATION_TASK_MODE = {
    "comprehend": "comprehend",
    "info_search": "info_search",
    # "": "",
}
READ_STRATEGIES = {
    "skim": "skim",
    "normal": "normal",
    "careful": "careful",
}

REGRESS_DECISIONS = {
    "regress": "regress",
    "continue_forward": "continue_forward",
}

SENTENCE_STATES_IN_MEMORY = {
    "forgotten": "forgotten",
    "revisited": "revisited",
    "first_visit": "first_visit",
}

QUESTION_TYPES = {
    "MCQ": "Multiple-Choice Questions",
    "FRS": "Free Recall Summary",
}

READING_STATES = {
    "predefined_time_constraint_in_seconds": "predefined_time_constraint_in_seconds",
    "elapsed_time_in_seconds": "elapsed_time_in_seconds",
    "remaining_time_in_seconds": "remaining_time_in_seconds",
    "total_num_words": "total_num_words",
    "num_words_read": "num_words_read",
    "num_words_remaining": "num_words_remaining",
}

HEURISTIC_APPRAISAL_LEVELS = {
    "skim": 0.3,
    "normal": 0.5,
    "careful": 0.7,
}

TIME_AWARENESS_LEVELS = {
    "very_limited_time": "very_limited_time",
    "sufficient_time": "sufficient_time",
    "ample_time": "ample_time",
}

TIME_CONSTRAINT_LEVELS = {
    "90S": 90,
    "60S": 60,
    "30S": 30,
}

# RL Supervisory Controller Environment related constants
# ----------------------------------------------------------------------------------------------------------------------
MAX_NUM_SENTENCES = 20
MIN_NUM_SENTENCES = 10

MAX_NUM_WORDS_PER_SENTENCE = 32
MIN_NUM_WORDS_PER_SENTENCE = 2

MIN_TEXT_LENGTH = 120
MAX_TEXT_LENGTH = 160

# Constants for the STM and LTM dictionary keys
# ----------------------------------------------------------------------------------------------------------------------
CONTENT = "content"
SENTENCE_ID = "sentence_id"
VISIT_COUNT = "visit_count"
STM_STRENGTH = "stm_strength"
STM_ELAPSED_TIME = "stm_elapsed_time"
SENTENCE_START_INDEX = "sentence_start_index"
ACTIVATED_SCHEMAS = "activated_schemas"
FORGOTTEN_FLAG = "forgotten_flag"
APPRAISAL = "appraisal"

AVERAGE_READ_WORDS_PER_SECOND = 3       # Set as 3 to align with human data I collected. From google: average reading speed is 238 words per minute; -- Fine tune whether the reading speed was calculated correctly

MEMORY_RETAIN_APPRAISAL_LEVEL_THRESHOLD = 0.8

STM_CAPACITY = 4

# Constants for the DRL OCR model
# ----------------------------------------------------------------------------------------------------------------------
WORDS_INTEREST_RANGE = 1

# MCQ metadata file path
# ----------------------------------------------------------------------------------------------------------------------
MCQ_METADATA_PATH = "/home/baiy4/reader-agent-zuco/step5/data/assets/MCQ/mcq_metadata.json"
