import time

import numpy as np
import yaml
import json
import os
import math
import datetime
# import spacy
import textstat
import warnings
from collections import Counter
from openai import OpenAI
from nltk.data import find
from nltk.corpus import brown

from stable_baselines3 import PPO

from step5.modules.llm_envs.TransformerContextPredictor import TransformerContextPredictor
from step5.modules.llm_envs.TransformerLikelihoodCalculator import TransformerLikelihoodCalculator

from step5.modules.rl_envs.SupervisoryControllerEnv_v1101_2 import SupervisoryControllerEnv
from step5.modules.rl_envs.SentenceLevelControllerEnv_v1014 import SentenceLevelControllerEnv
# from step5.modules.rl_envs.GeneralOculomotorControllerEnv_v1010 import GeneralOculomotorControllerEnv
from step5.modules.rl_envs.GeneralOculomotorControllerEnv_v1122 import GeneralOculomotorControllerEnv

from step5.modules.llm_envs.LLMMemories import LLMWorkingMemory, LLMShortTermMemory, LLMLongTermMemory
from step5.STB3RL import RL

import step5.utils.constants as const
import step5.utils.auxiliaries as aux



class RLSupervisoryController:      

    def __init__(self, config = 'config.yaml') -> None:
        """
        The text-level controller to control reading of sentences, including the general reading behaviors, including the reading strategies, reading positions, and time management.
        Controller type: RL
        """
        # Read the configuration file
        with open(config, 'r') as stream:
            self._config = yaml.load(stream, Loader=yaml.FullLoader)
        
        # Initialize the pre-trained model deployment
        self._pretrained_model_deployment = self._config['simulate']['rl_models']['supervisory_controller']

        # Get the pretrained model path
        self._model_path = os.path.join('training', 'saved_models', self._pretrained_model_deployment['checkpoints_folder_name'], self._pretrained_model_deployment['loaded_model_name'])

        # Initialize the environemnt
        self.env = SupervisoryControllerEnv()

        # Load the pre-trained model
        try:
            self._model = PPO.load(self._model_path, self.env, custom_objects={'observation_space': self.env.observation_space, 'action_space': self.env.action_space})
        except (RuntimeError, TypeError) as e:
            warnings.warn(f"Could not deserialize object: {e}")
            raise e
        
        print(f"Successfully loaded the pre-trained {self._pretrained_model_deployment['env_name']} model from {self._model_path}.\n"
              f"{'-'*50}\n")
        
        # Determine rl variables
        self.text_level_steps = None
        self._obs = None
        self._info = None
        self.done = None
        self.score = None
        self.sc_logs = None
        self.action = None
        self._states = None
        self._reward = None
        self._truncated = None
    
    def reset(self, inputs: dict = None) -> None:   
        """
        Reset the environment for each new episode -- each new reading stimulus / reading trial which contains many sentences.
        """ 
        
        assert inputs is not None, "The inputs should be provided by the external environment. Contains information about a task and regarding stimulus information."
        # assert inputted_appraisal_level is not None, "The external actions from the LLM should be provided. Contains the LLM's actions of reading strategies."

        self._obs, self._info = self.env.reset(inputs=inputs)
        self.done = False
        self.score = 0
        self.sc_logs = {}
        self.text_level_steps = 0
    
    def step_part1(self):
        """
        The step function for the Supervisory Controller. Every step is a single sentence-level reading step.
        :param external_llm_actions: LLM's reading strategies.
        :return: The output list for the Supervisory Controller.
        """
        self.text_level_steps += 1
 
        self.action, self._states = self._model.predict(self._obs, deterministic=True)
        # self.action, self._states = self._model.predict(self._obs, deterministic=False)
        self._reward = self.env.step_part1(action=self.action)
    
    def step_part2(self, inputted_read_sentence_appraisal_level: float = None, 
                   inputted_num_words_explicitly_fixated_in_sentence: int = None) -> dict:
        """
        The second step of the Supervisory Controller. Every step is a single sentence-level reading step.
        """
        self._obs, self._reward, self.done, self._truncated, self._info = self.env.step_part2(
            reward=self._reward,
            inputted_read_sentence_appraisal_level=inputted_read_sentence_appraisal_level, 
            inputted_num_words_explicitly_fixated_in_sentence=inputted_num_words_explicitly_fixated_in_sentence
            )


class RLSentenceLevelController:     # TODO check whether a reset function is needed to reset the environment, especially (free) parameters.

    def __init__(self, config = "config.yaml") -> None:
        """
        The word-level controller to control the word skips. Maybe could also control the word-level regressions.
        Controller type: RL
        """
        # Read the configuration file
        with open(config, 'r') as stream:
            self._config = yaml.load(stream, Loader=yaml.FullLoader)
        
        # Initialize the pre-trained model deployment
        self._pretrained_model_deployment = self._config['simulate']['rl_models']['word_controller']

        # Get the pretrained model path
        self._model_path = os.path.join('training', 'saved_models', self._pretrained_model_deployment['checkpoints_folder_name'], self._pretrained_model_deployment['loaded_model_name'])

        # Initialize the environemnt
        self.env = SentenceLevelControllerEnv()

        # Load the pre-trained model
        try:
            self._model = PPO.load(self._model_path, self.env, custom_objects={'observation_space': self.env.observation_space, 'action_space': self.env.action_space})
        except (RuntimeError, TypeError) as e:
            warnings.warn(f"Could not deserialize object: {e}")
            raise e
        
        print(f"Successfully loaded the pre-trained {self._pretrained_model_deployment['env_name']} model from {self._model_path}.\n"
              f"{'-'*50}\n")
        
        # Determine rl variables
        self.sentence_level_steps = None
        self._obs = None
        self._info = None
        self.done = None
        self.score = None
        self.wc_logs = None
        self.action = None
        self._states = None
        self._rewards = None
        self._truncated = None
    
    def reset(self, inputs: dict = None) -> None:
        """
        Reset the environment for each allocated sentence.
        """ 
        
        self._obs, self._info = self.env.reset(inputs=inputs)
        self.done = False
        self.score = 0
        self.sc_logs = {}
        self.sentence_level_steps = 0
    
    def step_part1(self):
        """
        Separate the step function into two parts for reasonable implementation. 
            Step part 1 will determine the word to fixate. 
        """
        self.sentence_level_steps += 1
 
        self.action, self._states = self._model.predict(self._obs, deterministic=True)
        self._reward, self.done = self.env.step_part1(action=self.action)
    
    def step_part2(self, external_llm_next_word_predictability: float = None) -> dict:
        """
        The second step of the Supervisory Controller. Every step is a single sentence-level reading step.
        """
        self._obs, self._reward, self.done, self._truncated, self._info = self.env.step_part2(
            reward=self._reward,
            done=self.done,
            external_llm_next_word_predictability = external_llm_next_word_predictability,
            )


class RLOculomotorController:       # TODO check whether a reset function is needed to reset the environment, especially (free) parameters.

    def __init__(self, config="config.yaml") -> None:
        """
        Step part 2 will feed in the external info -- the predictability of the next word 
            -- regarding the fixated word determined just now in step part 1.
        """
        # Read the configuration file
        with open(config, 'r') as stream:
            self._config = yaml.load(stream, Loader=yaml.FullLoader)
        
        # Initialize the pre-trained model deployment
        self._pretrained_model_deployment = self._config['simulate']['rl_models']['oculomotor_controller']

        # Get the pretrained model path
        self._model_path = os.path.join('training', 'saved_models', self._pretrained_model_deployment['checkpoints_folder_name'], self._pretrained_model_deployment['loaded_model_name'])

        # Load the dataset
        shared_dataset_metadata_of_stimuli, shared_dataset_encoded_lexicon, self._dataset_mode = aux.load_oculomotor_controller_dataset(config=self._config)

        # Create an instance of the environment for use in other methods
        self.env = GeneralOculomotorControllerEnv(
            shared_dataset_metadata_of_image_stimuli=shared_dataset_metadata_of_stimuli,
            shared_dataset_encoded_lexicon=shared_dataset_encoded_lexicon
        )
        # Load the pre-trained model
        try:
            self._model = PPO.load(self._model_path, self.env, custom_objects={'observation_space': self.env.observation_space, 'action_space': self.env.action_space})
        except (RuntimeError, TypeError) as e:
            warnings.warn(f"Could not deserialize object: {e}")
            raise e
        
        print(f"Successfully loaded the pre-trained {self._pretrained_model_deployment['env_name']} model from {self._model_path}.\n"
              f"{'-'*50}\n")
        
        # Determine rl variables
        self.word_or_pixel_level_steps = None      # It is within-word level, pixel-level, or called letter-level steps
        self._obs = None
        self._info = None
        self.done = None
        self.score = None
        self.oc_logs = None
        self.action = None
        self._states = None
        self._rewards = None
        self._truncated = None
    
    def reset(self, inputs: dict = None) -> None:
        """
        Reset the environment for each allocated word (target word OR fixated word).
            Whenever a new word is determined, the oculomotor controller should be reset.
        :param inputs: The input dictionary for the Oculomotor Controller. Should contain the target word index, fixation x and y coordinates.
        :param time_awareness_weight: The weight for the time awareness in the reward function.
        :return: None
        """ 

        self._obs, self._info = self.env.reset(inputs=inputs)    # Initialize the external predictability to 0.0
        self.done = False
        self.score = 0
        self.oc_logs = {}
        self.word_or_pixel_level_steps = 0
    
    def step(self,) -> dict: 
        """
        The step function for the Oculomotor Controller. Every step refers to a single fixation.
        :return: The output list for the Word Controller.
        """

        self.word_or_pixel_level_steps += 1

        self.action, self._states = self._model.predict(self._obs, deterministic=True)
        # self.action, self._states = self._model.predict(self._obs, deterministic=False)
        self._obs, self._rewards, self.done, self._truncated, self._info = self.env.step(action=self.action)
        self.score += self._rewards


class NextWordPredictabilityCalculator:
    def __init__(self):
        """
        Initialize the Bayesian Skipper.
        Bayesian Skipper was inspired by paper:
            A Rational Model of Word Skipping in Reading: Ideal Integration of Visual and Linguistic Information
        where we frame the skip gesture as a probability-dependent decision:
            P(w|I) = normalize(P(w) * P(I|w))
        where P(w) is the prior probability of the next word w, and P(I|w) is the likelihood of the next word w given the input I.
        If P(w|I) is greater than a threshold (determined as "threshold"), we skip the next word.
        Moreover, we consider word frequency, contextual constraint, and parafovea preview's effects on the skipping decision, thus,
        P(w) = w_freq * word_freq (derived from existing dataset) + w_pred * word_contextual_constraint/predictability (generated by LLM) + Konst.
        While parafovea preview's effect (OR "Launch Site Distance" in paper A Rational Model of Word Skipping in Reading)
        is characterized by P(I|w), as shorter the word, higher the likelihood value.

        Some previous empirical findings:
            1. When not reading, but just word searching, the word frequency's effect on identification time is not significant.
            2. Word frequency's effect is much more significant compared to contextual constraint.
            3. No need to include morphological priming effect in the model because it was found not significant.
                Ref: Semantic and morphological cross-word priming during sentence reading
                Link: https://docs.google.com/presentation/d/1_GDyfsmUE4LoKYZTw1CY8x7zVsIpnSeVcFS5l_2rT-k/edit#slide=id.g272b771c0a9_0_14
            4. Word freq is dominant compared to the 5-gram predictability [Yuanyuan Duan, 2019].
        """

        # Define the tunable parameters for human data alignment
        self._word_frequency_weight = None
        self._word_predictability_weight = None
        self._constant = None      # A mutable konst
        self._likelihood_weight = None  # A mutable likelihood weight

        # Define the variables
        self.predictability = None    # P(w|I), if greater than this value, skip the next word, otherwise read it
        self.next_word_frequency = None    # Use existing dataset
        self.next_word_contextual_predictability = None   # Generate by LLM
        self.next_word_parafoveal_preview_likelihood = None         # Handles the parafovea preview's effect

        # Initialize the LLM next word predictor, likelihood calculator, and word frequency calculator
        self.llm_predictor = TransformerContextPredictor()
        self.llm_likelihood_calculator = TransformerLikelihoodCalculator()
        self.word_frequency_calculator = WordFrequencyCalculator()

    def reset(
            self,
            word_frequency_weight: float = 10,
            word_predictability_weight: float = 1,
            likelihood_weight: float = 100,
            constant: float = 1.0
    ):
        """
        Reset the Bayesian Skipper, especially the free tunable parameters.
        :return: None
        """
        self._word_frequency_weight = word_frequency_weight
        self._word_predictability_weight = word_predictability_weight
        self._likelihood_weight = likelihood_weight
        self._constant = constant

    def step(self, inputs: dict = None):
        """
        Skip the next word or not.
        :return:
            skip the next word or not
        """

        # Get the next word frequency
        self.next_word_frequency = self._get_next_word_frequency(inputs=inputs)

        # Get the next word predictability
        # Two methods to determine the window length of the context information that is fed to the LLM:
        #   1. the fixed window length of 5 that is usually used in the literature, e.g., A Rational Model of Word Skipping in Reading
        #   2. the current and previous sentence with a mental representation of sentences, could use LLM to extract key gists to infer
        self.next_word_contextual_predictability = self._get_predictability(inputs=inputs)

        # Get the likelihood
        self.next_word_parafoveal_preview_likelihood = self._get_likelihood(inputs=inputs)

        # Calculate the skip confidence
        self.predictability = self._calculate_next_word_predictability()

        return self.predictability

    def _calculate_next_word_predictability(self):
        """
        Calculate the skip predictability.
        :return:
            the predictability value
        """
        confidence = (self._word_frequency_weight * self.next_word_frequency + self._word_predictability_weight * self.next_word_contextual_predictability + self._constant) * (self._likelihood_weight * self.next_word_parafoveal_preview_likelihood)
        return np.clip(confidence, 0, 1)

    def _get_next_word_frequency(self, inputs: dict = None):
        """
        Get the frequency of the next word.
        :return:
            the frequency of the next word
        """
        return self.word_frequency_calculator.get_probability(word=inputs['next_word'])

    def _get_predictability(self, inputs: dict = None):
        """
        Get the predictability of the next word.
        :return:
            the predictability of the next word
        """
        return self.llm_predictor.predict(context_words=inputs['context_words'], next_word=inputs['next_word'])

    def _get_likelihood(self, inputs: dict = None):
        """
        Get the likelihood of the next word.
        :return:
            the likelihood of the next word
        """
        return self.llm_likelihood_calculator.get_likelihood(
            parafovea_letters=inputs['parafovea_letters'],
            next_word=inputs['next_word'],
            estimated_next_word_len=inputs['estimated_next_word_len']
        )

    def log(self):
        """
        Log the Bayesian Skipper's parameters.
        :return: a dictionary of the Bayesian Skipper's parameters and values
        """
        return {
            'Synthetic Predictability': self.predictability,
            'Word Frequency Weight': self._word_frequency_weight,
            'Word Predictability Weight': self._word_predictability_weight,
            'Constant': self._constant,
            'Next Word Frequency': self.next_word_frequency,
            'Next Word Predictability': self.next_word_contextual_predictability,
            'Likelihood': self.next_word_parafoveal_preview_likelihood,
        }


class WordFrequencyCalculator:

    def __init__(self):
        """
        Initialize the Word Frequency.
        """
        # Check if the Brown corpus is available
        try:
            find('corpora/brown')
        except LookupError:
            print(f"Downloading the Brown corpus...")
            nltk.download('brown', quiet=True)

        # Load the Brown corpus
        words = brown.words()

        # Calculate word frequencies
        word_counts = Counter(words)
        total_words = sum(word_counts.values())

        # Normalize frequencies to probabilities
        self.word_probabilities = {word: count / total_words for word, count in word_counts.items()}

    def get_probability(self, word: str) -> float:
        """
        Get the probability of a word.
        :param word: the word
        :return: the probability of the word
        """
        return self.word_probabilities.get(word, 0.0)


class ReaderAgent:

    def __init__(self, config: str = "config.yaml", mcq_metadata: dict = None,):
        """
        Date: 2024-10-01, Author: Bai Yunpeng
        Simulator name: ReaderAgent simulator
        Version: 1001-linux
        Description: The model has three-level controllers: the supervisory controller, the word controller, and the oculomotor controller.

        Some old but potentially useful notes:
            Argument: Human's reading comprehension is trying to minimize the uncertainty about the texts. Rational-related paper: https://docs.google.com/presentation/d/1BakDe7jPmbqpYHDU5A1bx5X2y0vQXPDvShSRhShYhyc/edit#slide=id.g2157cd1f5c7_0_17
            Argument: reading regression is mainly for fixing oculomotor error in word recognitions and comprehension improvement. Some summarized literature: https://docs.google.com/presentation/d/1BakDe7jPmbqpYHDU5A1bx5X2y0vQXPDvShSRhShYhyc/edit#slide=id.g2157cd1f5c7_0_0
            Text comprehension paper: Toward a model of text comprehension and production, 1978, link: https://www.cl.cam.ac.uk/teaching/1516/R216/Towards.pdf

        Assumptions: (may write in the paper later)
            1. The skim reading does not have a comprehension degradation. See the simulation results of the questions later.

        """

        # Read the configuration file
        with open(config, 'r') as stream:
            self._config = yaml.load(stream, Loader=yaml.FullLoader)
        
        # Ensure the simulation mode is set to "simulate"
        assert self._config['rl']['mode'] == const.SIMULATE, "The simulation mode should be set to 'simulate'."

        # Initialize the NLP model -- OPENAI models     
        # --------------------------------------------------------------------------------------------------------------
        openai_api_key = self._config['llm']['API_key']
        # Specific configurations
        self._gpt_model = self._config['llm']['model']
        self._refresh_interval = self._config['llm']['refresh_interval']
        self._max_num_requests = self._config['llm']['max_num_requests']
        self._retry_delay = self._config['llm']['retry_delay']
        # LLM related variables
        self._llm_client = OpenAI(api_key=openai_api_key)

        # Initialize configs according to the yaml file -- 
        # --------------------------------------------------------------------------------------------------------------
        # Simulation configurations
        self._simulate_config = self._config['simulate']
        # --------------------------------------------------------------------------------------------------------------
        # Episodes specifications
        self._num_ep = self._simulate_config['num_episodes']     # Number of episodes run per environment per model per set of parameters
        self._episode_index = None

        # Initialize the modules -- RL models
        self.sc_env = RLSupervisoryController(config=config)
        self.slc_env = RLSentenceLevelController(config=config)
        self.oc_env = RLOculomotorController(config=config)

        # Initialize the LLM-based memory modeles
        self.wm_env = LLMWorkingMemory(config=config)
        self.stm_env = LLMShortTermMemory(config=config)
        self.ltm_env = LLMLongTermMemory(config=config)

        # Initialize the utility models -- the Next Word Predictability Calculator
        self.nwpc_env = NextWordPredictabilityCalculator()

        # Initialize the task-related variables
        # --------------------------------------------------------------------------------------------------------------
        # Time constraint and the global time-related variables: clock
        # Agent's state related variable -- from the external: the remaining time in seconds and the number of words on stimuli
        # Task-related variables: time constraint
        self._total_time_limit_in_seconds = None
        self._time_spent_in_seconds = None
        self._time_left_in_seconds = None
        # The reading progress
        self._total_num_words_on_stimuli = None
        self._num_words_read_on_stimuli = None
        self._num_words_skipped_on_stimuli = None
        self._num_words_left_on_stimuli = None
        # Initialize the weights for time awareness in the oculomotor controller
        self._oc_time_awareness_weight = None
        # Other time-related variables -- the average reading time per word
        # TODO @parameter_inference: the reading time should also be a tunable parameter
        # --------------------------------------------------------------------------------------------------------------
        # Initialize the reading strategy-related variables -- it is updated at every sentence
        self._reading_strategy_for_the_sentence_to_read = None
        # --------------------------------------------------------------------------------------------------------------
        # Initialize the environment-related variables
        # --------------------------------------------------------------------------------------------------------------
        # Image environments to run in the simulator    TODO check which are deletable
        root_dir = os.path.dirname(os.path.abspath(__file__))
        image_envs_filename = self._config["resources"]["img_env_dir"]
        image_envs_dir = os.path.join(root_dir, "data", "gen_envs", image_envs_filename, const.SIMULATE)
        self._image_envs, self._image_filenames = aux.read_images(image_envs_dir)
        self._image_filenames_indexes = [int(name.split("_")[1].split(".")[0]) for name in self._image_filenames]
        self._num_image_envs = len(self._image_envs)
        self._image_env = None
        self._image_env_pixels = None
        self._image_metadata = None
        # --------------------------------------------------------------------------------------------------------------
        # Read the JSON file (metadata)
        with open(os.path.join(image_envs_dir, const.MD_FILE_NAME), "r") as file:
            self._metadata = json.load(file)
        # --------------------------------------------------------------------------------------------------------------
        # Get image-level metadata
        self._images_metadata = self._metadata[const.md["images"]]

        # Initialize the variables for the reading progress or status
        # --------------------------------------------------------------------------------------------------------------
        # Sections -- out of scope of this paper now
        self._sections = None
        self._num_sections = None
        self._memory_across_sections = None
        # One section -- the scope of this paper -- memory stuff
        self._stm = None  # Short-term memory
        self._ltm_gists = None  # Long-term memory
        # --------------------------------------------------------------------------------------------------------------
        # Comprehension-related variables -- controlled by the LLM Text Comprehend
        self._section_comprehension = None      # TODO: seems no use, check later 
        self._task = None   # The task type -- a placeholder for now, should be consistent with our benchmark, including normal comprehension (and skim reading), scanning, and proofreading
        self._task_specification = None
        self._task_instruction = None   # TODO debug delete later
        self._user_profile = None   # User profile, a placeholder for now
        # Initialize the inputted variables
        self._question = None   # The question for the comprehension task   # TODO check how to use and name this later
        # --------------------------------------------------------------------------------------------------------------
        # Within this paper's scope -- one single section's reading-related variables -- controlled by the three-level controllers: the supervisory controller, the word controller, and the oculomotor controller
        # Section / Stimuli-level
        self._stimulus_index = None
        # Sentence-level
        self._num_sentences_in_stimulus = None
        self._first_words_indexes_across_sentences = None
        self._num_words_per_sentences = None
        self._reading_sentence_index = None     # The index of the sentence being read
        self._latest_read_sentence_index = None   # The index of the latest sentence being read
        self._sentence_just_read = None   # The sentence just read
        # Word-level
        self._words_in_stimulus = None
        self._num_words_in_stimulus = None
        self._target_word_index_in_stimulus = None
        self._target_word_in_stimulus = None
        self._target_word_len_in_stimulus = None
        self._next_word_index_in_stimulus = None
        self._next_word_in_stimulus = None
        self._next_word_len_in_stimulus = None
        self._next_word_predictability = None
        self._n_gram_context = None
        self._fixated_word_index_in_sentence = None   
        # Letter-level TODO revisit later
        self._norm_fix_x = None
        self._norm_fix_y = None
        self._parafoveal_preview_letters_for_next_word = None
        self._norm_fix_x_for_reset = None
        self._norm_fix_y_for_reset = None

        # Initialize the MCQ metadata
        # --------------------------------------------------------------------------------------------------------------
        if mcq_metadata is not None:
            self._mcq_metadata = mcq_metadata
        else:
            with open(const.MCQ_METADATA_PATH, "r") as file:
                self._mcq_metadata = json.load(file)

        # Initialize the data loggers
        # --------------------------------------------------------------------------------------------------------------
        # The data logger for the simulation results
        self._simulation_data_log_dict = None
        # self._sim_data_logs = []    # Across different sections
        self._all_text_level_logs: list = []   # Text level logs
        self._episodic_info: dict = None   # Within one reading stimulus
        self._text_level_steps: dict = None   # Text level 
        self._text_level_logs: dict = None   # Text level logs
        self._all_sentence_level_logs: list = []   # Sentence level
        self._sentence_level_steps: dict = None   # Sentence level
        self._sentence_level_logs: dict = None   # Sentence level logs
        self._all_word_level_logs: list = []   # Word level
        self._word_level_steps: dict = None   # Word level
        self._word_level_logs: dict = None   # Word level logs
        # self._simulation_step_logs = []   # Within one reading stimulus
        # self._simulation_data_json_dir = simulation_data_save_dir
        self._simulation_data_json_dir = None
        self._simulation_data_text_level_json_filename = None
        self._simulation_data_sentence_level_json_filename = None
        self._simulation_data_word_level_json_filename = None

    def reset(
            self,
            episode_index: int = 0,     # The index of the episode, each episode should run through all the image environments, 3 conditions each
            image_env_index: int = 0,   # The index of the image environment -- the index of the stimulus
            task_specification: str = "Please summarize what you have read",
            user_profile: dict = None,
            question: str = None,
            total_time_limit_in_seconds: int = 30,   # It should be either 30, 60, or 90 seconds.
            oc_time_awareness_weight: float = 1.0,
            simulation_data_save_dir: str = "data/sim_results",
    ):
        """
        Reset the simulator with one batch of free parameter in one parameter-inference trial.
        :return: None
        """

        # Reset the episode index
        self._episode_index = int(episode_index)
        
        # --------------------------------------------------------------------------------------------------------------
        # TODO debug delete later
        print(f"Episode index: {self._episode_index}")

        # Reset the image environment
        image_env_index_from_images = self._image_filenames_indexes.index(image_env_index)  # Get the correct index of the image environment
        self._image_env = self._image_envs[image_env_index_from_images]
        self._image_env_pixels = np.array(self._image_env)
        self._image_metadata = self._images_metadata[image_env_index]
        self._words_in_stimulus = self._image_metadata[const.md["selected words"]]
        self._num_words_in_stimulus = self._image_metadata[const.md["num words"]]
        self._num_sentences_in_stimulus, self._first_words_indexes_across_sentences, self._num_words_per_sentences = self._get_text_level_sentences_info_in_stimulus()  
        self._reading_sentence_index: int = const.ZERO      # Default start from the first sentence -- TODO check maybe this should be determined by the supervisory controller
        self._latest_read_sentence_index: int = const.NEGATIVE_ONE   # Default start from the first sentence -- TODO check maybe this should be determined by the supervisory controller
        self._sentence_just_read = ""

        # Reset the reading progress or states related variables -- word level manipulation -- skip or not
        self._stimulus_index = image_env_index
        
        # Initialize the target word -- by default, the first word in the stimulus
        self._target_word_index_in_stimulus = const.ZERO    # Default start from the first word -- As an initialization, it has to be the first word in the whole stimulus -- TODO: check later whether need to be determined by the word controller
        self._target_word_in_stimulus = self._image_metadata[const.md["words metadata"]][self._target_word_index_in_stimulus][const.md["word"]]
        self._target_word_len_in_stimulus = len(self._target_word_in_stimulus)
        # Initialize the next word -- by default, the second word in the stimulus if there is any
        if self._target_word_index_in_stimulus + 1 <= self._image_metadata[const.md["num words"]]:
            self._next_word_index_in_stimulus = self._target_word_index_in_stimulus + 1
            self._next_word_in_stimulus = self._image_metadata[const.md["words metadata"]][self._next_word_index_in_stimulus][const.md["word"]]
            self._next_word_len_in_stimulus = len(self._next_word_in_stimulus)
        else:
            self._next_word_index_in_stimulus = const.NEGATIVE_ONE
            self._next_word_in_stimulus = const.NO_NEXT_WORD
            self._next_word_len_in_stimulus = const.NEGATIVE_ONE
        
        # Reset the word skip related variables
        self._n_gram_context = []
        self._parafoveal_preview_letters_for_next_word = ""
        self._next_word_predictability = 0.0

        # Reset the word-level-related variables
        self._fixated_word_index_in_sentence = const.NEGATIVE_ONE

        # Reset the comprehension-related variables
        self._task_instruction = task_specification
        self._user_profile = user_profile
        # Reset the inputted question
        self._question = question

        # Reset the actions
        self._norm_fix_x = 0.0
        self._norm_fix_y = 0.0
        self._norm_fix_x_for_reset = 0.0
        self._norm_fix_y_for_reset = 0.0

        # Rest the flag
        self._flag_all_words_are_read = False

        # Reset the time penalty weight
        self._oc_time_awareness_weight = oc_time_awareness_weight

        # Reset the time constraint-related variables
        self._total_time_limit_in_seconds = total_time_limit_in_seconds
        total_time_limit_key_consistent_with_constants = f'{self._total_time_limit_in_seconds}S'
        self._time_spent_in_seconds = 0
        self._time_left_in_seconds = self._total_time_limit_in_seconds

        # Reset the reading progress-related variables
        self._total_num_words_on_stimuli = self._image_metadata[const.md["num words"]]
        self._num_words_read_on_stimuli = const.ZERO
        self._num_words_skipped_on_stimuli = const.ZERO
        self._num_words_left_on_stimuli = self._total_num_words_on_stimuli

        # Reset the reading strategy-related variable -- set as the default reading strategy for the first sentence
        self._reading_strategy_for_the_sentence_to_read = const.READ_STRATEGIES["normal"]

        # Reset the models/modules
        # --------------------------------------------------------------------------------------------------------------
        # Reset the RL modules
        # Supervisory Controller
        sc_inputs = {
            'total_num_sentences_in_stimulus': self._num_sentences_in_stimulus,
            'num_words_per_sentences': self._num_words_per_sentences,
            'total_time_in_seconds': self._total_time_limit_in_seconds,
        }
        sc_external_llm_actions = {'reading_strategy_for_this_sentence': self._reading_strategy_for_the_sentence_to_read}
        self.sc_env.reset(inputs=sc_inputs)
        # Word Controller
        wc_inputs = {'num_words_in_sentence': self._num_words_per_sentences[self._reading_sentence_index], 'time_constraint_level': total_time_limit_key_consistent_with_constants}
        self.slc_env.reset(inputs=wc_inputs,) 
        # Oculomotor Controller
        oc_inputs = {'target_word_index': self._target_word_index_in_stimulus, 'image_index': self._stimulus_index, 'norm_fix_x': self._norm_fix_x_for_reset, 
                     'norm_fix_y': self._norm_fix_y_for_reset, 'time_awareness_weight': self._oc_time_awareness_weight,}
        self.oc_env.reset(inputs=oc_inputs)
        # --------------------------------------------------------------------------------------------------------------
        # Reset the LLM memory modules
        self.ltm_env.reset()    # Reset the LTM
        self.stm_env.reset()    # Reset the STM
        self.wm_env.reset()     # Reset the WM
        # --------------------------------------------------------------------------------------------------------------
        # Reset the utility models
        self.nwpc_env.reset()

        # Reset the simulation data save directory
        # --------------------------------------------------------------------------------------------------------------
        self._simulation_data_json_dir = simulation_data_save_dir

        # Reset the data save-related variables
        # --------------------------------------------------------------------------------------------------------------
        self._init_logs()
    
    def _init_logs(self,):
        """
        Initialize the data log dictionary.
        """
        # Reset the json file name
        self._simulation_data_text_level_json_filename = os.path.join(self._simulation_data_json_dir, const.SIM_DATA_TEXT_LEVEL_WO_KINTSCH_MEMORY_RETRIEVAL_JS_FILE_NAME)   # The questions are answered without Kintsch's memory retrieval mechanism, serve as a baseline / ablation study to compare with the full model
        self._simulation_data_sentence_level_json_filename = os.path.join(self._simulation_data_json_dir, const.SIM_DATA_SENTENCE_LEVEL_JS_FILE_NAME)
        self._simulation_data_word_level_json_filename = os.path.join(self._simulation_data_json_dir, const.SIM_DATA_WORD_LEVEL_JS_FILE_NAME)
        
        # Reset the data logger
        self._episodic_info = {     # This should be only updated once per episode
            "episode_index": self._episode_index,
            "stimulus": {
                "stimulus_index": self._stimulus_index,
                "words_in_section": ' '.join(self._words_in_stimulus),  # self._words_in_stimulus, Make them in one line
                "stimulus_width": self._metadata["config"]["img size"][0],      
                "stimulus_height": self._metadata["config"]["img size"][1],
            },
            "task": {
                "time_constraint": self._total_time_limit_in_seconds,
                "task_type": "comprehension",       # Enable more task types later
            },
            "cognitive_bouds": {    
                "foveal_width": self._metadata["config"]["foveal size"][0],
                "foveal height": self._metadata["config"]["foveal size"][1],
                "parafoveal_width": self._metadata["config"]["parafoveal size"][0],
                "parafoveal_height": self._metadata["config"]["parafoveal size"][1],
                "peripheral_width": self._metadata["config"]["peripheral size"][0],
                "peripheral_height": self._metadata["config"]["peripheral size"][1],
            },
        }

        # --------------------------------------------------------------------------------------------------------------
        # Reset the text level logs
        self._text_level_logs = {}
        self._text_level_logs['episodic_info'] = self._episodic_info.copy()
        self._text_level_logs['episodic_info']['text_level_steps'] = []     # Initialize the text level steps
        # Reset the step logs at text level -- Save sentence by sentence
        self._update_text_level_one_step_logs()

        # --------------------------------------------------------------------------------------------------------------
        # Reset the sentence level logs
        self._sentence_level_logs = {}
        self._sentence_level_logs['episodic_info'] = self._episodic_info.copy()
        self._sentence_level_logs['episodic_info']['sentence_level_steps'] = []     # Initialize the sentence level steps
        # Reset the step logs at sentence level -- Save word by word
        self._update_sentence_level_one_step_logs()

        # --------------------------------------------------------------------------------------------------------------
        # Reset the word level logs
        self._word_level_logs = {}
        self._word_level_logs['episodic_info'] = self._episodic_info.copy()
        self._word_level_logs['episodic_info']['word_level_steps'] = []     # Initialize the word level steps
        # Reset the step logs at word level -- Save fixation by fixation
        self._update_word_level_one_step_logs()
        
    def _update_text_level_one_step_logs(self,):
        text_level_one_step_logs = {
            "stimulus_index": self._stimulus_index,
            "-----------------------------------------------------------------------":'',
            "text_level_steps": self.sc_env.text_level_steps, 
            "reading_sentence_index": self._reading_sentence_index,
            "sentence_just_read": self._sentence_just_read,
            "latest_read_sentence_index": self._latest_read_sentence_index,
            "reading_strategy_for_this_sentence": self._reading_strategy_for_the_sentence_to_read,
            "revisit_data": {
                "num_revisit_sentences": self.sc_env.env.num_revisit_sentences,   
                "num_total_sentences_read": self.sc_env.env.num_total_sentences_read,   
                "num_revisit_saccades_on_word_level": self.sc_env.env.num_revisit_saccades_on_word_level,  
                "num_total_saccades_on_word_level": "This should be calculated by the word controller.",
            },
            "reading_progress": {
                "num_words_read_on_stimuli": self._num_words_read_on_stimuli,
                "num_words_left_on_stimuli": self._num_words_left_on_stimuli,
                "time_spent_in_seconds": self._time_spent_in_seconds,
                "time_left_in_seconds": self._time_left_in_seconds, 
            },
            "memories": {   # Use copies to avoid overwriting
                "STM": self.stm_env.STM.copy(),         # This was an OrderedDict
                "LTM": self.ltm_env.gists,       # This was a list
                "Schemas": self.ltm_env.main_schemas,    # This was a list
            },
        }
        # Update the text level logs
        self._text_level_logs['episodic_info']['text_level_steps'].append(text_level_one_step_logs)

    def _update_sentence_level_one_step_logs(self,):
        sentence_level_one_step_logs = {
            "stimulus_index": self._stimulus_index,
            "text_level_steps": self.sc_env.text_level_steps,
            "sentence_index_in_stimulus": self._reading_sentence_index,
            "-----------------------------------------------------------------------":'',
            "sentence_level_steps": self.slc_env.sentence_level_steps,
            "given_sentence_reading_time_in_steps": self.slc_env.env.ep_len,
            "target_word_info": {
                "target_word_index_in_stimulus": self._target_word_index_in_stimulus,
                "target_word_in_stimulus": self._target_word_in_stimulus,
                "target_word_len_in_stimulus": self._target_word_len_in_stimulus,
            },
            "next_word_info": {
                "next_word_index_in_stimulus": self._next_word_index_in_stimulus,
                "next_word_in_stimulus": self._next_word_in_stimulus,
            },
            "predictability_info": {
                "parafoveal_preview_letters_for_next_word": self._parafoveal_preview_letters_for_next_word,
                "next_word_predictability": self._next_word_predictability,
                "next_word_frequency": self.nwpc_env.next_word_frequency,
                "next_word_contextual_predictability": self.nwpc_env.next_word_contextual_predictability,
                "next_word_parafoveal_preview_likelihood": self.nwpc_env.next_word_parafoveal_preview_likelihood,
                "context_words": self._n_gram_context,
            },
            "word_skipping_data": {
                "num_words_read_in_sentence": self.slc_env.env.num_words_read_in_sentence,
                "num_words_skipped_in_sentence": self.slc_env.env.num_words_skipped_in_sentence,
                "num_saccades_on_word_level": self.slc_env.env.num_saccades_on_word_level,
            }
        }
        # Update the sentence level logs
        self._sentence_level_logs['episodic_info']['sentence_level_steps'].append(sentence_level_one_step_logs)

    def _update_word_level_one_step_logs(self,):
        word_level_one_step_logs = {
            "stimulus_index": self._stimulus_index,
            "text_level_steps": self.sc_env.text_level_steps,
            "sentence_index_in_stimulus": self._reading_sentence_index,
            "sentence_level_steps": self.slc_env.sentence_level_steps,
            "-----------------------------------------------------------------------":'',
            "word_level_or_fixation_steps": self.oc_env.word_or_pixel_level_steps,
            "target_word_index_in_stimulus": self._target_word_index_in_stimulus,
            "target_word_in_stimulus": self._target_word_in_stimulus,
            "fixation_info": {
                "norm_fix_x": self._norm_fix_x,       
                "norm_fix_y": self._norm_fix_y,
                "is_terminate_step": self.oc_env.env.is_terminate_step,
                "sampled_letters": self.oc_env.env.foveal_seen_letters[self._target_word_index_in_stimulus],
                "parafoveal_preview_letters_for_next_word": self.oc_env.env.parafovea_seen_letters[self._target_word_index_in_stimulus + 1] if (self._target_word_index_in_stimulus + 1) <= (self._num_words_in_stimulus - 1) else const.NO_NEXT_WORD,
            },
        }
        # Update the word level logs
        self._word_level_logs['episodic_info']['word_level_steps'].append(word_level_one_step_logs)
    
    def _get_text_level_sentences_info_in_stimulus(self):     # TODO check the validity later
        """
        Get the number of sentences in the current stimulus/reading material.
        :return: The number of sentences in the given stimulus, the start/first word indexes across sentences,
                and the number of words in each sentence.
        """
        num_sentences = 0
        first_word_indexes_across_sentences = [0]
        num_words_per_sentences = []
        words_in_current_sentence = 0

        for idx, word in enumerate(self._words_in_stimulus):
            words_in_current_sentence += 1
            if word[-1] in const.END_PUNCTUATION_MARKS:
                num_sentences += 1
                num_words_per_sentences.append(words_in_current_sentence)
                words_in_current_sentence = 0  # Reset for the next sentence
                if idx + 1 < len(self._words_in_stimulus):
                    first_word_indexes_across_sentences.append(idx + 1)

        # Add the last sentence if it doesn't end with a punctuation mark
        if words_in_current_sentence > 0:
            num_sentences += 1
            num_words_per_sentences.append(words_in_current_sentence)

        return num_sentences, first_word_indexes_across_sentences, num_words_per_sentences

    def simulate(self,):
        """
        Run the simulation.
        """
        # Run the simulation
        self._simulate_text_level_reading()
        # Close the simulation
        self._save_data()

    def _update_memories_by_sentences(self, sentence_just_read: dict = {const.CONTENT: const.NA, const.SENTENCE_ID: const.ZERO,},):  # TODO we need to specify the reading_sentence's format here
        """
        This is a processing cycle integrating the working memory, short-term memory, and long-term memory.
        :param read_sentence: The sentence being read by the agent.
        :return:
        """
        # Step 1: activate the schemas in the LTM
        self.ltm_env.activate_schemas(raw_sentence=sentence_just_read[const.CONTENT])
        # Step 2: update the short-term memory
        self.stm_env.extract_microstructure_and_update_stm(
            raw_sentence=sentence_just_read[const.CONTENT],
            spatial_info=sentence_just_read[const.SENTENCE_ID],
            activated_schemas=self.ltm_env.activated_schemas,
            main_schemas=self.ltm_env.main_schemas,
        )
        # Step 3: update the long-term memory
        self.ltm_env.generate_macrostructure(stm=self.stm_env.STM)
    
    def _update_reading_progress_and_time_by_sentence(self,):
        """
        Update the reading progress and the time-related variables.
        """
        num_words_just_read_in_sentence = self._num_words_per_sentences[self._reading_sentence_index]

        new_sentence_read = False
        # Determine the reading progress
        if self._reading_sentence_index > self._latest_read_sentence_index:   # New sentence has been read
            self._latest_read_sentence_index = self._reading_sentence_index
            new_sentence_read = True
        
        # Update the reading progress-related variables
        if new_sentence_read:   # Only update the reading progress-related variables when a new sentence is read
            self._num_words_read_on_stimuli += num_words_just_read_in_sentence
            self._num_words_left_on_stimuli = self._total_num_words_on_stimuli - self._num_words_read_on_stimuli

        # Update the time-related states directly from the RL models for consistency
        self._time_spent_in_seconds = self.sc_env.env.time_spent_in_seconds
        self._time_left_in_seconds = self.sc_env.env.time_left_in_seconds       # TODO this maybe needed to be updated considering word skipper and ocular controller as well. TODO TODO!!!
    
    def _simulate_text_level_reading(self,):
        """
        Run the text-level simulation. It is composed of both RL and LLM models, especially LLLM-based memory modules.
        """

        # Reset the environment
        self.sc_env.reset(inputs={'total_num_sentences_in_stimulus': self._num_sentences_in_stimulus, 'num_words_per_sentences': self._num_words_per_sentences, 'total_time_in_seconds': self._total_time_limit_in_seconds,}, )

        while not self.sc_env.done:     # Read the sentences until the time is up
            
            # --------------------------------------------------------------------------------------------------------------
            # Determine which sentece to read using the RL model
            self.sc_env.step_part1()
            self._reading_sentence_index = self.sc_env.env.reading_sentence_index
            # # Determine the reading strategy for the next sentence to read using LLM-based memory model
            # reading_strategy_for_the_read_sentence = self._determine_the_sentence_reading_strategy_using_llm_working_memory()

            # --------------------------------------------------------------------------------------------------------------
            # Read the sentence
            # Run the sentence-level simulation for the given sentence -- read the given sentence
            self._simulate_sentence_level_reading()    
            # Get the average appraisal levels of the read sentences
            avg_appraisal_levels_of_the_read_sentence = self.slc_env.env.avg_words_appraisals
            # Get the given sentence content, and the words indexes in the sentence
            self._sentence_just_read, words_sampled_from_sentence = self._get_sentence_content_in_stimulus_using_sentence_index(
                sentence_index=self._reading_sentence_index, sampled_words_indexes_in_sentence=self.slc_env.env.sampled_indexes_in_sentence,
                )
            # Update the reading progress and the time-related variables by the just read sentence
            self._update_reading_progress_and_time_by_sentence()
            # Update the appraisal levels of the read sentence -- IMPORTANT: it has to be before the memories updating, especially the LTM gisting
            individual_sentence_appraisal_level = self._calculate_individual_sentence_appraisal_level(  
                actual_sentence_content=self._sentence_just_read, 
                words_sampled_from_sentence= words_sampled_from_sentence,
                appraisal_of_info_sampling=avg_appraisal_levels_of_the_read_sentence
                ) 
            # Update memories using LLM-based memory modules -- TODO solve this 
            self._update_memories_by_sentences(sentence_just_read={const.CONTENT: self._sentence_just_read, const.SENTENCE_ID: self._reading_sentence_index,},)

            # --------------------------------------------------------------------------------------------------------------
            # # Update the reading-related states
            # individual_sentence_appraisal_level = self._calculate_individual_sentence_appraisal_level(  
            #     actual_sentence_content=self._sentence_just_read, 
            #     words_sampled_from_sentence= words_sampled_from_sentence,
            #     appraisal_of_info_sampling=avg_appraisal_levels_of_the_read_sentence
            #     ) 
            # Update the explicit fixated words in the sentence for better infer the given sentence's reading time
            num_words_explicitly_fixated_in_sentence = self.slc_env.env.num_words_in_sentence - self.slc_env.env.num_words_skipped_in_sentence
            # Step2
            self.sc_env.step_part2(inputted_read_sentence_appraisal_level=individual_sentence_appraisal_level, 
                                   inputted_num_words_explicitly_fixated_in_sentence=num_words_explicitly_fixated_in_sentence,)
            
            # TODO debug delete later
            print(f"(Supervisory Controller) The calculated appraisal state of the current reading sentence is: {self.sc_env.env.appraisal_states[self._reading_sentence_index]['appraisal_level']}; The current reading sentence index is: {self._reading_sentence_index}; \n")
            # print(f"(Supervisory Controller) The appraisal states are: {self.sc_env.env.appraisal_states}; \n")
            
            # Update the text level logs
            self._update_text_level_one_step_logs()
        
        # Finish the reading simulation, finialized the gists in the LTM, and output some conditions and results
        # --------------------------------------------------------------------------------------------------------------
        self.ltm_env.finalize_gists()

        print(f"*****************************************************************************************\n"
              f"(Simulation) The reading simulation is finished because the time is up.\n"
              f"*****************************************************************************************\n")  

        # Answer the MCQ and free recall questions
        # --------------------------------------------------------------------------------------------------------------
        self._evaluate_comprehensions()

    def _get_sentence_content_in_stimulus_using_sentence_index(self, sentence_index: int = 0, sampled_words_indexes_in_sentence: list = []):
        """
        Get the sentence content using the sentence index.
        :param sentence_index: The index of the sentence
        :return: The sentence content
        """
        # Get the start word index of the sentence
        start_word_index = self._first_words_indexes_across_sentences[sentence_index]
        # Get the number of words in the sentence
        num_words_in_sentence = self._num_words_per_sentences[sentence_index]
        # Get the sentence content
        sentence_content = ' '.join(self._words_in_stimulus[start_word_index: start_word_index + num_words_in_sentence])

        # Get the sampled words indexes in the sentence according to the input list
        if len(sampled_words_indexes_in_sentence) > 0:
            words_sampled_from_sentence = ' '.join([self._words_in_stimulus[start_word_index + idx] for idx in sampled_words_indexes_in_sentence])
        else:
            words_sampled_from_sentence = const.NA
            print(f"Warning: no words sampled in the sentence. \n")

        return sentence_content, words_sampled_from_sentence
    
    #@parameter_inference   
    def _calculate_individual_sentence_appraisal_level(     
            self, actual_sentence_content, words_sampled_from_sentence,
            appraisal_of_info_sampling, 
            tunable_readability_weight=0.4, tunable_info_appraisal_weight=0.3, tunable_understanding_by_memory_context_weight=0.3,
            ):     
        """
        Calculate the appraisal level of the sentence to be read.
        We jointly consider:
            - The readability of the sentence;
            - The appraisal of information sampling (from the sentence-level simulation).
            - The understanding of the sampled information according to the memory context. But if the weights here is set to too high, the agent can easily saturates to 1.0, and it will cause troubles.
                TODO: handle agent's saturation problem in the supervisory controller. Issues reason -- the agent's testing dataset distribution (appraisal levels) is different from training dataset distribution.
                Fix mathods: 1. retrain a SC with more aligned data distribution; 2. lower the weights of the understanding. Try to keep the sentence away from being saturated to 1.0
        What can be added here:
            - The importance of the sentence;
            - The reader's subjective interest level;
            - The reader's proficiency level;

        Simplified version: joint consideration of the sentence's readability, the reading strategy's effect, and the understanding of the sampled information according to the memory context.

        This is a method with tunable parameter
        """
        # Step 1: Calculate the readability score of the sentence
        readability_score = textstat.flesch_reading_ease(actual_sentence_content)
        readability_score_clipped = np.clip(readability_score, 0, 100)
        inverted_readability_score_clipped = 100 - readability_score_clipped
        norm_readability_score = aux.normalise(inverted_readability_score_clipped, 0, 100, 0, 1)

        # Step 2: Get the appraisal of information sampling
        appraisal_of_info_sampling = appraisal_of_info_sampling

        # Step 3: Calculate the sampled information's understanding according to memories
        appraisal_of_understanding_by_memory_context = self.wm_env.calculate_appraisal_of_understanding_by_memory_context(
            ltm_gists=self.ltm_env.gists, sampled_sentence_content=actual_sentence_content,
            )
        
        # Step 4: Manipulate weights
        # Ensure the weights sum to 1
        weight_readability = tunable_readability_weight
        weight_info_appraisal = tunable_info_appraisal_weight
        weight_understanding_by_memory_context = tunable_understanding_by_memory_context_weight
        total_weight = weight_readability + weight_info_appraisal + weight_understanding_by_memory_context
        weight_readability /= total_weight
        weight_info_appraisal /= total_weight
        weight_understanding_by_memory_context /= total_weight

        # Calculate the final appraisal level 
        individual_sentence_appraisal_level = weight_readability * norm_readability_score + weight_info_appraisal * appraisal_of_info_sampling + weight_understanding_by_memory_context * appraisal_of_understanding_by_memory_context   

        # Clip the appraisal level to [0, 1]
        debug_tuning_weight = 1.0       # TODO debug delete later
        individual_sentence_appraisal_level = debug_tuning_weight * np.clip(individual_sentence_appraisal_level, 0, 1)

        # TODO the appraisal_of_understanding_by_memory_context is always fed as 1.0, which might have issues, check later after the parameter tuninig, not a priority
        rounded_individual_sentence_appraisal_level = round(individual_sentence_appraisal_level * 10) / 10

        # TODO debug delete later
        print(f"The sampled sentence is: {words_sampled_from_sentence}; \n"
              f"The norm readability score is: {norm_readability_score}; \n"  
              f"The appraisal of information sampling is: {appraisal_of_info_sampling}; \n"     
              f"the appraisal of understanding by memory context is: {appraisal_of_understanding_by_memory_context}")
        print(f"========== The individual sentence appraisal level is: {individual_sentence_appraisal_level}; The rounded value is: {rounded_individual_sentence_appraisal_level}\n")
        
        return rounded_individual_sentence_appraisal_level
    
    def _simulate_sentence_level_reading(self,):
        """
        Run the word-level simulation. It is composed of both RL and LLM models, especially LLLM-based word predictability modules.     
        This level simulation is terminated when the given sentence is finished reading.
        """
        # Reset word-level variables
        self._fixated_word_index_in_sentence = const.NEGATIVE_ONE

        # Reset the environment
        self.slc_env.reset(inputs={'num_words_in_sentence': self._num_words_per_sentences[self._reading_sentence_index], 'time_constraint_level': f'{self._total_time_limit_in_seconds}S',})

        # Run the word-level simulation
        while not self.slc_env.done:     # Read the words until the allocated time to the sentence is up
            # Run the word controller -- determine the next word to read -- continue or skip
            # self.slc_env.step(external_llm_predictability=self._next_word_predictability)   # Step once, indexes are updated within the environment
            self.slc_env.step_part1()   # Step part one, update states, get the word to read by the OMC
            # Update the target word-related variables      
            self._update_target_word_info()
            # Get the letter-level/pixel-level fixations and reading stuff -- read the given target word -- the parafoveal preview letters are updated here as well
            self._simulate_word_level_reading()
            # Update the n gram context buffer
            self._update_n_gram_context()   
            # Update the next word-related variables
            self._update_next_word_info()

            # Update the next word predictability -- Determine the next word's predictability using different models -- LLM models and the RL Oculomotor controller model, the parafoveal preview was updated in the letter-level simulation
            self._next_word_predictability = self._get_next_word_predictability(
                context_words=self._n_gram_context, next_word=self._next_word_in_stimulus, 
                parafoveal_preview_letters=self._parafoveal_preview_letters_for_next_word, next_word_len=self._next_word_len_in_stimulus,)
            # Update the following word's predictability after the OMC reading the given target word assigned by the step part1
            self.slc_env.step_part2(external_llm_next_word_predictability=self._next_word_predictability)

            # Update the sentence level logs
            self._update_sentence_level_one_step_logs()
    
    #@parameter_inference  
    def _get_next_word_predictability(self, context_words: str = None, next_word: str = None, parafoveal_preview_letters: str = None, next_word_len: int = 0,):  
        """
        Get the predictability of the next word using the LLM model.
        I use a linear function to consider factors: word frequency, predictability according to the context, and parafoveal preview.

        :param next_word: The next word, could be used to infer the word frequency
        :param context: The context that could be used to predict the next word using the LLM model, usually it is several words (within 5)
        :param parafoveal_preview: The parafoveal preview that could be used to predict the next word using the LLM model, usually it is several letters
        :return: The predictability of the next word
        """
        predicatability = self.nwpc_env.step(inputs={'context_words': context_words, 'next_word': next_word, 'parafovea_letters': parafoveal_preview_letters, 'estimated_next_word_len': next_word_len,})
        return predicatability

    def _update_target_word_info(self,):
        """
        Update the target word-related variables.
        """
        # Word index in the sentence
        self._fixated_word_index_in_sentence = self.slc_env.env.fixated_word_index   # The index of the word in the given sentence
        # Word index in the stimulus
        self._target_word_index_in_stimulus = self._first_words_indexes_across_sentences[self._reading_sentence_index] + self._fixated_word_index_in_sentence
        # Target word
        self._target_word_in_stimulus = self._image_metadata[const.md["words metadata"]][self._target_word_index_in_stimulus][const.md["word"]]
    
    def _update_n_gram_context(self, buffer_size: int = 4,):
        """
        Apply the n-gram prediction. As indicated by the prior work "A Rational Model of Word Skipping in Reading: Ideal Integration of Visual and Linguistic Information", Yuanyuan Duan, 2019,
            I set the buffer size to 4, which is the same as the word unigram model and the 5-gram model.
        "we compare two representations of the prior: a word unigram model (i.e., using word frequency information), which ignores any context information, 
            and a 5-gram model, which conditions on the previous four words of context." 
        """
        self._n_gram_context.append(self._target_word_in_stimulus)
        if len(self._n_gram_context) > buffer_size:
            self._n_gram_context.pop(0)
    
    def _update_next_word_info(self,):
        """
        Update the next word-related variables. Check whether there is a next word in the given sentence.
        Use the sentence-level info and metadata to infer the next word.
        """
        if self._fixated_word_index_in_sentence + 1 < self.slc_env.env.num_words_in_sentence:    # There is still a next word
            self._next_word_index_in_stimulus = self._target_word_index_in_stimulus + 1
            self._next_word_in_stimulus = self._image_metadata[const.md["words metadata"]][self._next_word_index_in_stimulus][const.md["word"]]
            self._next_word_len_in_stimulus = len(self._next_word_in_stimulus)
        else:   # There is no next word
            self._next_word_index_in_stimulus = const.NEGATIVE_ONE
            self._next_word_in_stimulus = const.NO_NEXT_WORD
            self._next_word_len_in_stimulus = const.NEGATIVE_ONE
            # TODO maybe add a function to make the loop break from here, check it later
    
    def _simulate_word_level_reading(self,):
        """
        Run the letter-level simulation by implementing the oculomotor controller.
        """
        
        # Reset the pixel or letter-level oculomotor controller first -- primarily updating the target word index and the fixation position
        self.oc_env.reset(inputs={'target_word_index': self._target_word_index_in_stimulus, 'image_index': self._stimulus_index, 'norm_fix_x': self._norm_fix_x_for_reset, 'norm_fix_y': self._norm_fix_y_for_reset, 'time_awareness_weight': self._oc_time_awareness_weight,})
        
        # Simulate to read the given word through a lot of fixations
        while not self.oc_env.done:     # Read the word from the text display until the agent thinks a word is activated
            # Run the oculomotor controller
            self.oc_env.step()
            # Log fixation data somewhere -- TODO check later
            self._norm_fix_x = self.oc_env.env.norm_fix_x
            self._norm_fix_y = self.oc_env.env.norm_fix_y

            # Update the word-level logs
            self._update_word_level_one_step_logs()
        # The word is read, update the some variables for reset
        # Update the norm fixations x and y for reset
        self._norm_fix_x_for_reset = self.oc_env.env.norm_fix_x
        self._norm_fix_y_for_reset = self.oc_env.env.norm_fix_y
        # Update the parafoveal preview letters
        self._parafoveal_preview_letters_for_next_word = self.oc_env.env.parafovea_seen_letters[self._target_word_index_in_stimulus + 1] if (self._target_word_index_in_stimulus + 1) <= (self._num_words_in_stimulus - 1) else const.NO_NEXT_WORD

    def _evaluate_comprehensions(self,):
        """
        Answer the MCQs and free recall questions with corresponding stimulus idx using the LLm-based memory modules and MCQ metadata.
        """
        # Answer the MCQs
        self._answer_mcqs()
        # Answer the free recall questions
        self._answer_free_recall_questions()
    
    def _answer_mcqs(self,):
        """
        Answer the MCQs with corresponding stimulus idx using the LLm-based memory modules and MCQ metadata.
        """
        # Get the MCQs for the given stimulus
        mcqs = self._mcq_metadata[str(self._stimulus_index)]
        # Initialize the log dict
        mcq_logs = []
        # Answer the MCQs
        for mcq_idx, mcq in mcqs.items():
            # Answer the MCQ
            question = mcq["question"]
            options = mcq["options"]
            mcq_answer = self.wm_env.retrieve_memory(
                question_type=const.QUESTION_TYPES["MCQ"],
                question=question,
                options=options,
                ltm_gists=self.ltm_env.gists
            )
            mcq_logs.append({
                "mcq_idx": mcq_idx,
                "question": question,
                "options": options,
                "answer": mcq_answer,
                "correct_answer": mcq["correct_answer"],
                })
           
            print(f"The answer to the question '{question}' is: {mcq_answer}. \n")

        self._text_level_logs['episodic_info']['mcq_logs'] = mcq_logs

    def _answer_free_recall_questions(self,):
        free_recall_answer = self.wm_env.retrieve_memory(
            question_type=const.QUESTION_TYPES["FRS"],
            ltm_gists=self.ltm_env.gists
        )
        
        print(f"The free recall results are: \n{free_recall_answer}. \n")

        self._text_level_logs['episodic_info']['free_recall_answer'] = free_recall_answer
    
    def _save_data(self):
        """
        Close the simulator.
        :return: None
        """
        # Wrap up the lists of dictionaries -- Saving all the data in one batch run, a good practice for now, fix it later if have memory issues
        self._all_text_level_logs.append(self._text_level_logs)
        self._all_sentence_level_logs.append(self._sentence_level_logs)
        self._all_word_level_logs.append(self._word_level_logs)

        # Save the simulation results
        self._save_logs_to_json()
    
    def _save_logs_to_json(self):
        """
        Write the simulation data to a JSON file.
        :return: None
        """
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {convert_numpy_types(key): convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(element) for element in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(element) for element in obj)
            elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float_, np.float16,
                                np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj  # Return as is if it doesn't match any known type

        # Convert data types in the logs
        converted_text_level_logs = convert_numpy_types(self._all_text_level_logs)
        converted_sentence_level_logs = convert_numpy_types(self._all_sentence_level_logs)
        converted_word_level_logs = convert_numpy_types(self._all_word_level_logs)
            
        # Save the logs to JSON files
        with open(self._simulation_data_text_level_json_filename, 'w') as file:
            json.dump(converted_text_level_logs, file, indent=4)
        print(f"The text-level sim data logs are stored at: {self._simulation_data_text_level_json_filename}")      

        with open(self._simulation_data_sentence_level_json_filename, 'w') as file:
            json.dump(converted_sentence_level_logs, file, indent=4)
        print(f"The sentence-level sim data logs are stored at: {self._simulation_data_sentence_level_json_filename}")      

        with open(self._simulation_data_word_level_json_filename, 'w') as file:
            json.dump(converted_word_level_logs, file, indent=4)
        print(f"The word-level sim data logs are stored at: {self._simulation_data_word_level_json_filename}")


if __name__ == '__main__':

    pass