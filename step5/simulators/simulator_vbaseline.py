import time

import numpy as np
import yaml
import json
import os
import datetime
import spacy
from openai import OpenAI

from step5.modules.OculomotorController import OculomotorController
from step5.modules.BayesianSkipper import BayesianSkipper
from step5.modules.llm_envs.LLMSupervisoryController import LLMSupervisoryController
from step5.modules.llm_envs.LLMMemories import LLMShortTermMemory, LLMLongTermMemory, LLMWorkingMemory

import step5.utils.constants as const
import step5.utils.auxiliaries as aux


class ReaderAgent:

    def __init__(
            self,
            config: str = r'D:\Users\91584\PycharmProjects\reading-model\step5\config.yaml',
            sim_data_save_dir: str = r'D:\Users\91584\PycharmProjects\reading-model\step5\data\sim_results',
    ):
        """
        Initialize the UIReader. This is the simulator running pipeline / workflow.
        Argument: Human's reading comprehension is trying to minimize the uncertainty about the texts.
            Rational-related paper: https://docs.google.com/presentation/d/1BakDe7jPmbqpYHDU5A1bx5X2y0vQXPDvShSRhShYhyc/edit#slide=id.g2157cd1f5c7_0_17

        Argument: reading regression is mainly for fixing oculomotor error in word recognitions and comprehension improvement.
            Some summarized literature: https://docs.google.com/presentation/d/1BakDe7jPmbqpYHDU5A1bx5X2y0vQXPDvShSRhShYhyc/edit#slide=id.g2157cd1f5c7_0_0

        Text comprehension paper:
            Toward a model of text comprehension and production, 1978, link: https://www.cl.cam.ac.uk/teaching/1516/R216/Towards.pdf

        Latest diagram:
            - Version 16 July https://www.figma.com/board/T9YEXqz2NukOfcLLmHUP4L/18-June-Architecture?node-id=0-1&t=lX2j28YyOl2CPR1m-0
        """

        # Read the configuration file
        with open(config, 'r') as stream:
            self._config = yaml.load(stream, Loader=yaml.FullLoader)
        self._sim_data_save_dir = sim_data_save_dir

        # Initialize the NLP model -- OPENAI models
        # --------------------------------------------------------------------------------------------------------------
        openai_api_key = self._config['llm']['API_key']
        # Specific configurations
        self._gpt_model = self._config['llm']['ltm']['model']
        self._refresh_interval = self._config['llm']['ltm']['refresh_interval']
        self._max_num_requests = self._config['llm']['ltm']['max_num_requests']
        self._retry_delay = self._config['llm']['ltm']['retry_delay']
        # LLM related variables
        self._client = OpenAI(api_key=openai_api_key)

        # Initialize configs according to the yaml file
        # --------------------------------------------------------------------------------------------------------------
        # Modules
        self._sim_config = self._config['simulate']
        # self._sim_llm_config = self._sim_config['llm_model']
        self._sim_bs_config = self._sim_config['bayesian_model']
        # self._sim_crm_config = self._sim_config['computationally_rational_model']
        # --------------------------------------------------------------------------------------------------------------
        # Specifications
        self._num_ep = self._sim_config['num_episodes']     # Number of episodes run per environment per model per set of parameters

        # Initialize the modules
        self.oc_env = OculomotorController(config=config)
        self.bs_env = BayesianSkipper()
        self.wm_env = LLMWorkingMemory(config=config)
        self.stm_env = LLMShortTermMemory(config=config)
        self.ltm_env = LLMLongTermMemory(config=config)

        # Initialize the task-related variables
        # --------------------------------------------------------------------------------------------------------------
        # Time constraint and the global time-related variables: clock
        # Agent's state related variable -- from the external: the remaining time in seconds and the number of words on stimuli
        # Task-related variables: time constraint
        self._predefined_fixed_time_constraint_in_seconds = None
        self._elapsed_time_in_seconds = None
        self._remaining_time_in_seconds = None
        # The reading progress
        self._num_words_on_stimuli = None
        self._num_read_words_on_stimuli = None
        self._num_skipped_words_on_stimuli = None
        self._num_remaining_words_on_stimuli = None
        # Initialize the weights for time penalty
        self._time_penalty_weight = None
        # Other time-related variables
        self._AVG_READ_TIME_PER_WORD = 0.40  # Roughly set the reading time per word to be 0.25 seconds -- update using the EMMA model later
        # --------------------------------------------------------------------------------------------------------------
        # Initialize the reading strategy-related variables
        self._reading_strategy = None
        # --------------------------------------------------------------------------------------------------------------
        # Initialize the environment-related variables
        # --------------------------------------------------------------------------------------------------------------
        # Image environments to run in the simulator
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
        # Sections related -- controlled by the Supervisory Controller (all memory-related memories)
        self.STM = None  # Short-term memory, a list of tuples of (micro-gists, spatial information)
        self.gists_in_LTM = None  # Long-term memory, a list of tuples of (macro-gists, spatial information)
        self._stm_elapsed_time_since_last_update = None # The elapsed time since the last update -- assume one word takes 0.25 seconds to process
        self._sections = None
        self._num_sections = None
        self._memory_gists_across_all_sections = None
        self._memory_gists_in_the_current_section = None
        self._gists_aligned_visuospatial_info_in_the_current_section = None
        # --------------------------------------------------------------------------------------------------------------
        # Comprehension-related variables -- controlled by the LLM Text Comprehend
        self._section_comprehension = None
        self._task_instruction = None
        self._user_profile = None
        # --------------------------------------------------------------------------------------------------------------
        # Single section's reading-related variables -- controlled by the Bayesian Skipper and Oculomotor Controller
        self._stim_index = None
        self._words_in_section = None
        self._sentence_num_in_section = None
        self._start_words_indexes_in_sentences = None
        self._current_sentence_index = None
        self._num_words_in_section = None
        self._target_word_index_in_section = None
        self._target_word_in_stimuli = None
        self._target_word_len_in_section = None
        self._next_word_index_in_section = None
        self._next_word_in_section = None
        self._next_word_len_in_section = None
        # --------------------------------------------------------------------------------------------------------------
        # Memory-related variables -- related to the Bayesian Skipper
        self._pseudo_short_term_memory = None
        self._working_memory_len = None
        self._read_content_in_section = None    # Refer to the phonological loop stored information
        self._read_positions_in_section = None  # Refer to the visuospatial sketchpad stored information
        # --------------------------------------------------------------------------------------------------------------
        # Initialize the agent's action space
        # --------------------------------------------------------------------------------------------------------------
        # Actions on reading the section -- controlled by all the modules
        self._stop_reading_trial_by_time = None
        self._skip_next_word_in_section = None
        self._regress_decision = None
        # --------------------------------------------------------------------------------------------------------------
        # Flag of reading till the end
        self._flag_read_to_the_end = None

        # Initialize the data loggers
        # --------------------------------------------------------------------------------------------------------------
        # The data logger for the simulation results
        self._sim_data_log_dict = None
        self._sim_data_logs = []    # Across different sections
        self._fixed_sim_data_log_dict = None   # Within one section / one section's reading trial
        self._dynamic_sim_data_log_dicts = []   # Within one section / one section's reading trial
        self._sim_data_json_dir = sim_data_save_dir
        self._sim_data_json_filename = None

        # Initialize the steps (word-wise steps, not actual fixation steps)
        self._word_steps = None

        # Initialize the inputted variables
        self._question = None

    def reset(
            self,
            image_env_index: int = 0,   # The index of the image environment -- the index of the section
            skip_activation_threshold: float = 0.5,
            word_frequency_weight: float = 10,
            word_predictability_weight: float = 0.5,
            konst: float = 1.0,
            working_memory_len: int = 5,
            task_instruction: str = "Please summarize what you have read",
            user_profile: dict = None,
            question: str = None,
            time_constraint_in_seconds: int = 30,   # It should be either 30, 60, or 90 seconds.
            time_penalty_weight: float = 1.0,
    ):
        """
        Reset the simulator with one batch of free parameter in one parameter-inference trial.
        :return: None
        """
        # Reset the overall sections-related variables  # TODO now we only have one section
        self._memory_gists_across_all_sections: dict = {}

        # Reset the one-section-related variables
        self._memory_gists_in_the_current_section: list = []
        self._gists_aligned_visuospatial_info_in_the_current_section: list = []

        # Reset the steps
        self._word_steps = 0

        # Reset the image environment
        image_env_index_from_images = self._image_filenames_indexes.index(image_env_index)  # Get the correct index of the image environment
        self._image_env = self._image_envs[image_env_index_from_images]
        self._image_env_pixels = np.array(self._image_env)
        self._image_metadata = self._images_metadata[image_env_index]
        self._words_in_section = self._image_metadata[const.md["selected words"]]
        self._sentence_num_in_section, self._start_words_indexes_in_sentences = self._get_num_sentences_in_section()
        self._current_sentence_index: int = const.ZERO

        # Reset the memory-related variables
        self.STM: list = []
        self.gists_in_LTM: list = []

        # Reset the reading progress or states related variables -- word level manipulation -- skip or not
        self._stim_index = image_env_index
        self._working_memory_len = working_memory_len
        self._pseudo_short_term_memory = {}
        self._read_content_in_section = []
        self._read_positions_in_section = []
        self._target_word_index_in_section = 0  # Always start from the first word in the section
        self._target_word_in_stimuli = self._image_metadata[const.md["words metadata"]][self._target_word_index_in_section][const.md["word"]]
        self._target_word_len_in_section = len(self._target_word_in_stimuli)
        if self._target_word_index_in_section + 1 <= self._image_metadata[const.md["num words"]]:
            self._next_word_index_in_section = self._target_word_index_in_section + 1
            self._next_word_in_section = self._image_metadata[const.md["words metadata"]][self._next_word_index_in_section][const.md["word"]]
            self._next_word_len_in_section = len(self._next_word_in_section)
        else:
            self._next_word_index_in_section = const.NEGATIVE_ONE
            self._next_word_in_section = const.NO_NEXT_WORD
            self._next_word_len_in_section = const.NEGATIVE_ONE

        # Reset the comprehension-related variables
        self._section_comprehension = const.NA
        self._task_instruction = task_instruction
        self._user_profile = user_profile
        # Reset the inputted variables
        self._question = question

        # Reset the actions
        self._stop_reading_trial_by_time = False
        self._skip_next_word_in_section = False
        self._regress_decision = False

        # Rest the flag
        self._flag_read_to_the_end = False

        # Reset the modules
        # --------------------------------------------------------------------------------------------------------------
        # Reset the Bayesian Skipper
        self.bs_env.reset(
            skip_activation_threshold=skip_activation_threshold,
            word_frequency_weight=word_frequency_weight,
            word_predictability_weight=word_predictability_weight,
            konst=konst,
        )
        # --------------------------------------------------------------------------------------------------------------
        # Oculomotor Controller has no free parameter to reset
        # --------------------------------------------------------------------------------------------------------------
        # Reset the Supervisory Controllers
        self.wm_env.reset(
            task_specification=self._task_instruction,
            user_profile=self._user_profile,
        )
        # Reset the STM environment
        self.stm_env.reset()
        # Reset the LTM environment
        self.ltm_env.reset(
            task_specification=self._task_instruction,
            user_profile=self._user_profile,
            question=self._question,
        )

        # Reset the time penalty weight
        self._time_penalty_weight = time_penalty_weight
        # Reset the time constraint-related variables
        self._predefined_fixed_time_constraint_in_seconds = time_constraint_in_seconds
        self._elapsed_time_in_seconds = 0
        self._remaining_time_in_seconds = self._predefined_fixed_time_constraint_in_seconds
        # Reset the reading progress-related variables
        self._num_words_on_stimuli = self._image_metadata[const.md["num words"]]
        self._num_read_words_on_stimuli = const.ZERO
        self._num_skipped_words_on_stimuli = const.ZERO
        self._num_remaining_words_on_stimuli = self._num_words_on_stimuli

        # Reading strategy-related variable
        self._reading_strategy = const.READ_STRATEGIES["normal"]

        # Reset the json file name
        self._sim_data_json_filename = os.path.join(self._sim_data_json_dir, const.SIM_DATA_JS_FILE_NAME)

        # Reset the data logger
        # --------------------------------------------------------------------------------------------------------------
        self._fixed_sim_data_log_dict = {
            "episode_index": 0,
            "section_info": {
                "section_index": self._stim_index,
                "section_metadata": self._image_metadata.copy(),
                "words_in_section": self._words_in_section.copy(),
                "section_comprehension": {
                    "task_specification": const.NA,
                    "read_content": const.NA,
                    "comprehension": const.NA,
                }
            },
        }
        # Reset the dynamic data logger
        dynamic_sim_log_dict = {
            "step_info": {
                "steps": self._word_steps,
                "working_memory": self._pseudo_short_term_memory.copy(),
                "read_content_in_section": self._read_content_in_section.copy(),
                "read_positions_in_section": self._read_positions_in_section.copy(),
                "gists_in_section": self._memory_gists_in_the_current_section.copy(),
                "terminate_reading": self._stop_reading_trial_by_time,
                "flag_read_to_the_end": self._flag_read_to_the_end,
                "target_word_info": {
                    "target_word_index_in_section": self._target_word_index_in_section,
                    "target_word_in_section": self._target_word_in_stimuli,
                    "target_word_len_in_section": self._target_word_len_in_section,
                    "target_word_visuospatial_info": const.NA,
                },
                "next_word_info": {
                    "next_word_index_in_section": self._next_word_index_in_section,
                    "next_word_in_section": self._next_word_in_section,
                    "next_word_len_in_section": self._next_word_len_in_section,
                },
                "skip_next_word_info": {
                    "skip_decision": self._skip_next_word_in_section,
                    "context_words": const.NA,
                    "parafoveal_seen_letters": const.NA,
                    "estimated_next_word_len": const.NA,
                    "skipping_details": const.NA,
                },
                "fixations_info": {
                    "init_norm_fix_x": const.NA,
                    "init_norm_fix_y": const.NA,
                    "norm_fix_x_batch": const.NA,
                    "norm_fix_y_batch": const.NA,
                    "parafoveal_seen_letters_batch": const.NA,
                    "oc_logs_batch": const.NA,
                },
            },
        }
        self._dynamic_sim_data_log_dicts.append(dynamic_sim_log_dict)

    def _get_num_sentences_in_section(self):
        """
        Get the number of sentences in the current section.
        :return: The number of sentences in the given section.
        """
        sentence_num = 0
        start_words_indexes_in_sentences = [0]
        for idx, word in enumerate(self._words_in_section):
            if word[-1] in const.END_PUNCTUATION_MARKS:
                sentence_num += 1
                if idx + 1 < len(self._words_in_section):
                    start_words_indexes_in_sentences.append(idx + 1)

        return sentence_num, start_words_indexes_in_sentences

    def run_memory_modules(
            self,
            read_sentence: dict,    # sentence, spatial information, sentence start word index
            # question: str = None,
    ):
        """
        This is a prototype method to testify my pipelines of the working memory, short-term memory, and long-term memory.
        This is a processing cycle within the working memory, short-term memory, and long-term memory.
        :return:
        """

        # Get the elapsed time since the last update first
        self._stm_elapsed_time_since_last_update = self._AVG_READ_TIME_PER_WORD * len(read_sentence[const.CONTENT].split())

        # Step 1: activate the schemas in the LTM
        self.ltm_env.activate_schemas(raw_sentence=read_sentence[const.CONTENT])

        # Step 2: update the short-term memory
        self.stm_env.extract_microstructure_and_update_stm(
            raw_sentence=read_sentence[const.CONTENT],
            spatial_info=read_sentence[const.SENTENCE_ID],
            activated_schemas=self.ltm_env.activated_schemas,
            main_schemas=self.ltm_env.main_schemas,
        )

        # Step 3: update the long-term memory
        self.ltm_env.generate_macrostructure(stm=self.stm_env.STM)

        # # Update the short-term memory using the lately read sentence
        # self.stm_env.extract_micro_gist_and_update_stm(
        #     raw_sentence=read_sentence[cons.CONTENT],
        #     spatial_info=read_sentence[cons.SPATIAL_INFO],
        #     sentence_start_word_index=read_sentence[cons.SENTENCE_START_INDEX],
        #     elapsed_time_since_last_update=self._stm_elapsed_time_since_last_update,
        #     reading_strategy=self._reading_strategy,
        # )
        #
        # # Update the gists in the LTM frequently at each processing cycle
        # self.ltm_env.generate_ltm_gists(
        #     stm=self.stm_env.STM,
        #     # question=question,
        # )

    # Check whether regress first
    def _check_regress_when_info_search(self, question: str = None):
        """
        Check whether to regress or not.
        :return: None
        """
        sentence_idx, start_word_idx_in_sentence = self.wm_env.generate_sentence_planning_decision_for_info_search_task(
            stm=self.stm_env.STM,
            ltm_gists=self.ltm_env.gists,
            question=question,
            num_sentences=self._sentence_num_in_section,
            start_words_indexes_in_sentences=self._start_words_indexes_in_sentences,
        )

        self._current_sentence_index = sentence_idx

        # Update the target word index -- regress to the previous word position
        self._target_word_index_in_section = start_word_idx_in_sentence
        self._target_word_in_stimuli = self._image_metadata[const.md["words metadata"]][self._target_word_index_in_section][const.md["word"]]
        self._target_word_len_in_section = len(self._target_word_in_stimuli)
        # Update the next word index
        self._next_word_index_in_section = self._target_word_index_in_section + 1
        self._next_word_in_section = self._image_metadata[const.md["words metadata"]][self._next_word_index_in_section][const.md["word"]]
        self._next_word_len_in_section = len(self._next_word_in_section)

    def _step_word_level_reading(self):

        # Forward a word level step
        self._word_steps += 1

        # Update the time-related states
        self._elapsed_time_in_seconds += self._AVG_READ_TIME_PER_WORD
        self._remaining_time_in_seconds = self._predefined_fixed_time_constraint_in_seconds - self._elapsed_time_in_seconds

        # Update the reading progress-related variables
        self._num_read_words_on_stimuli += 1
        self._num_remaining_words_on_stimuli -= 1

        # Calculate the state information -- step once
        print(f"The current word step is: {self._word_steps}. The current word is: {self._target_word_in_stimuli}. \n"
              f"The elapsed time is: {self._elapsed_time_in_seconds} seconds, the remaining time is: {self._remaining_time_in_seconds} seconds. "
              f"The predefined time constraint is: {self._predefined_fixed_time_constraint_in_seconds} seconds. \n"
              f"The number of words read is: {self._num_read_words_on_stimuli}, "
              f"the number of skipped words is: {self._num_skipped_words_on_stimuli}, "
              f"the number of remaining words is: {self._num_remaining_words_on_stimuli}."
              f"\n")

    def _check_termination_by_time_constraint(self, all_read_words_buffer: list = None):
        """
        Check whether to terminate the reading by the time constraint.
        :return: None
        """
        if self._remaining_time_in_seconds <= 0:
            self._stop_reading_trial_by_time = True
            # self._stop_reading_by_agent = True

            self.ltm_env.finalize_gists()

            # Print the termination information
            print(f"\n"
                  f"The reading trail is stopped due to the time constraint. The remaining time is: {self._remaining_time_in_seconds} seconds. \n"
                  f"The reading progress is: {self._num_read_words_on_stimuli} words read, {self._num_remaining_words_on_stimuli} words remaining. \n"
                  f"All the words in the section have been read: {all_read_words_buffer} \n")

    def _update_memories_and_update_strategy_by_sentences(      # TODO this should in the supervisory controller -- the working memory -- maybe remove it there
            self,
            word: str = None,
            word_index: int = None,
            section_index: int = 0,
            temp_sentence_buffer: list = None,
            temp_sentence_words_indexes_buffer: list = None,
            all_read_words_buffer: list = None,
            question: str = None,
    ):
        """
        Update the memories with the incoming word.
        First determine whether to update the memories.
        If yes, push the sentence buffer's content as one instance of raw reading material for the STM.
        :param word: The incoming word
        :return: None
        """
        # Reset the regression decision
        self._regress_decision = False

        # Update the all-read words buffer
        all_read_words_buffer.append(word)

        # Update the sentence buffer first
        temp_sentence_buffer.append(word)
        # Update the words indexes buffer
        temp_sentence_words_indexes_buffer.append(word_index)

        # Determine whether to update the memories: if this sentence is longer than the threshold,
        #   and it is ended with a non-alphabet character, and this ending character has to be a full stop.
        # If len(temp_sentence_buffer) > update_processing_cycle_threshold and word[-1] in cons.END_PUNCTUATION_MARKS:
        if word[-1] in const.END_PUNCTUATION_MARKS:
            # Convert the sentence buffer to a sentence
            sentence = ' '.join(temp_sentence_buffer)
            # print(f"The sentence is: {sentence}")
            # Update the memories, start the processing cycle again
            self.run_memory_modules(
                read_sentence={
                    const.CONTENT: sentence,
                    const.SENTENCE_ID: {const.SENTENCE_IDX: self._current_sentence_index,
                                       const.SECTION_IDX: section_index},
                    const.SENTENCE_START_INDEX: temp_sentence_words_indexes_buffer[0],
                },
                # question=question,
            )
            # Update the current reading sentence --> going to read the next sentence
            self._current_sentence_index += 1
            # Clear the sentence buffer
            temp_sentence_buffer.clear()
            # Clear the words indexes buffer
            temp_sentence_words_indexes_buffer.clear()

            # Determine whether to regress or not
            regress_action_by_llm = self.wm_env.regress_or_not(
                stm=self.stm_env.STM,
                ltm_gists=self.ltm_env.gists,
                predefined_time_constraint_in_seconds=self._predefined_fixed_time_constraint_in_seconds,
                elapsed_time_in_seconds=self._elapsed_time_in_seconds,
                remaining_time_in_seconds=self._remaining_time_in_seconds,
                total_num_words=self._num_words_on_stimuli,
                num_words_read=self._num_read_words_on_stimuli,
                remaining_num_words=self._num_remaining_words_on_stimuli,
            )
            if regress_action_by_llm == const.REGRESS_DECISIONS["regress"]:
                self._regress_decision = True
            else:
                self._regress_decision = False

            # TODO: debug delete later
            print(f"The regress decision is: {self._regress_decision}. "
                  f"The regress action by llm is: {regress_action_by_llm}\n")

            # # Determine whether to stop reading in this section everytime a sentence is finished
            # self._stop_reading_in_section = self.sc_env.generate_stop_reading_decision_for_info_search_task(
            #     task_specification=self._task_instruction,
            #     user_profile=self._user_profile,
            #     stm=self.stm_env.STM,
            #     ltm_gists=self.ltm_env.gists_in_LTM,
            #     question=question,
            # )

            # Update the reading strategy
            self._decide_reading_strategies()

        return temp_sentence_buffer, temp_sentence_words_indexes_buffer, all_read_words_buffer

    @staticmethod
    def _update_verbal_buffer_for_word_prediction(
            word: str,
            temp_verbal_buffer: list,
            temp_verbal_buffer_size: int,
    ):
        """
        Update the verbal buffer with the incoming word.
            Strictly constraint the buffer size.
        :param word:
        :return: None
        """
        temp_verbal_buffer.append(word)
        if len(temp_verbal_buffer) > temp_verbal_buffer_size:
            temp_verbal_buffer.pop(0)

        return temp_verbal_buffer

    def _decide_reading_strategies(self):
        """
        Decide the reading strategies using the Supervisory Controller.
        :return:
        """
        raw_decision = self.wm_env.decide_reading_strategy_by_llm(
            predefined_time_constraint_in_seconds=self._predefined_fixed_time_constraint_in_seconds,
            elapsed_time_in_seconds=self._elapsed_time_in_seconds,
            remaining_time_in_seconds=self._remaining_time_in_seconds,
            total_num_words=self._num_words_on_stimuli,
            num_words_read=self._num_read_words_on_stimuli,
            remaining_num_words=self._num_remaining_words_on_stimuli,
        )

        # Update the reading strategies
        if raw_decision == const.READ_STRATEGIES["normal"]:
            self._reading_strategy = const.READ_STRATEGIES["normal"]
        elif raw_decision == const.READ_STRATEGIES["skim"]:
            self._reading_strategy = const.READ_STRATEGIES["skim"]
        elif raw_decision == const.READ_STRATEGIES["careful"]:
            self._reading_strategy = const.READ_STRATEGIES["careful"]
        else:
            self._reading_strategy = const.READ_STRATEGIES["normal"]
            print(f"Warning: the raw_decision is: {raw_decision}, cannot be recognized. Use the normal reading strategy. \n")

    def _check_target_word_and_next_word_validity(self):
        # Check the validity of the target word index
        if self._target_word_index_in_section <= self._image_metadata[const.md["num words"]] - 1:
            self._target_word_in_stimuli = self._image_metadata[const.md["words metadata"]][self._target_word_index_in_section][const.md["word"]]
            self._target_word_len_in_section = len(self._target_word_in_stimuli)
        else:
            self._flag_read_to_the_end = True
            # break
        # Check the validity of the next word index
        if self._next_word_index_in_section <= self._image_metadata[const.md["num words"]] - 1:
            self._next_word_in_section = self._image_metadata[const.md["words metadata"]][self._next_word_index_in_section][const.md["word"]]
            self._next_word_len_in_section = len(self._next_word_in_section)
        else:
            self._next_word_index_in_section = const.NEGATIVE_ONE
            self._next_word_in_section = const.NO_NEXT_WORD
            self._next_word_len_in_section = const.NEGATIVE_ONE

    def _update_target_word_and_next_word_in_serial_reading(self):
        if self._skip_next_word_in_section == True:  # Skip the next word
            self._target_word_index_in_section += 2
        else:  # Read the next word
            self._target_word_index_in_section += 1
        # Determine the new next word according to the updated target word index
        self._next_word_index_in_section = self._target_word_index_in_section + 1

    def _update_target_word_and_next_word_in_regression_reading(self):

        # Call the supervisory controller to determine where to jump to
        sentence_idx, start_word_idx_in_sentence = self.wm_env.determine_regression_target(
            stm=self.stm_env.STM,
            ltm_gists=self.ltm_env.gists,
            # num_sentences=self._sentence_num_in_section,
            start_words_indexes_in_sentences=self._start_words_indexes_in_sentences,
        )

        # TODO debug delete later
        print(f"The sentence index is: {sentence_idx}, the start word index in the sentence is: {start_word_idx_in_sentence}.\n")

        self._current_sentence_index = sentence_idx

        # Update the target word index -- regress to the previous word position
        self._target_word_index_in_section = start_word_idx_in_sentence
        self._target_word_in_stimuli = self._image_metadata[const.md["words metadata"]][self._target_word_index_in_section][const.md["word"]]
        self._target_word_len_in_section = len(self._target_word_in_stimuli)
        # Update the next word index
        self._next_word_index_in_section = self._target_word_index_in_section + 1
        self._next_word_in_section = self._image_metadata[const.md["words metadata"]][self._next_word_index_in_section][const.md["word"]]
        self._next_word_len_in_section = len(self._next_word_in_section)

    def simulate(
            self,
            section_index: int = 0,     # We only deal with one section for now.
            question: str = "NO QUESTION AT THE READING STAGE",
            simulation_task_type: str = const.SIMULATION_TASK_MODE["comprehend"],  # The task type of the simulation
    ):
        """
        This is a prototype method to testify the reading control,
            including both the Bayesian reader and Oculomotor Controller.
        :return:
        """
        # Determine the reading strategy first
        self._decide_reading_strategies()

        # Reset the reading action control in case the agent regresses and re-reads
        self._flag_read_to_the_end = False

        # Initialize one verbal buffer that could be used to predict the next word
        temp_verbal_buffer = []
        # The size of the verbal buffer is set to 5 according to paper (reference):
        #   A Rational Model of Word Skipping in Reading: Ideal Integration of Visual and Linguistic Information
        temp_verbal_buffer_size = 5

        # Sentence buffer: These are used to directly update memories
        temp_sentence_buffer = []
        temp_sentence_words_indexes_buffer = []

        # All-read words buffer
        all_read_words_buffer = []

        # Check the regress first, ONLY if in the info search task
        if simulation_task_type == const.SIMULATION_TASK_MODE["info_search"]:
            if self._target_word_index_in_section > 0:
                self._check_regress_when_info_search(question=question)
                print(f"Finish checking the regression decision. Now the target word index has been "
                      f"re-directed to {self._target_word_index_in_section}.\n")

        # Initialize the variables for the reading progress or status
        init_norm_fix_x = np.random.uniform(-1, 1)
        init_norm_fix_y = np.random.uniform(-1, 1)

        # Start to read the give stimulus (section)
        # --------------------------------------------------------------------------------------------------------------
        # While not reaching the end of the section, keep reading
        while self._stop_reading_trial_by_time is False:
            # Forward a word level step
            self._step_word_level_reading()
            # ----------------------------------------------------------------------------------------------------------
            # Read the target word with the Oculomotor Controller
            oc_outputs = self.oc_env.step(inputs={
                "image_index": self._stim_index,
                "norm_fix_x": init_norm_fix_x,
                "norm_fix_y": init_norm_fix_y,
                "target_word_idx": self._target_word_index_in_section,
                "w_penalty": self._time_penalty_weight,
            })
            # ----------------------------------------------------------------------------------------------------------
            # Update the verbal buffer and memories with the incoming read word
            temp_verbal_buffer = self._update_verbal_buffer_for_word_prediction(
                word=self._target_word_in_stimuli,
                temp_verbal_buffer=temp_verbal_buffer.copy(),
                temp_verbal_buffer_size=temp_verbal_buffer_size,
            )
            # Check and update the processing cycles in the memories
            temp_sentence_buffer, temp_sentence_words_indexes_buffer, all_read_words_buffer = self._update_memories_and_update_strategy_by_sentences(
                word=self._target_word_in_stimuli,
                word_index=self._target_word_index_in_section,
                temp_sentence_buffer=temp_sentence_buffer.copy(),
                temp_sentence_words_indexes_buffer=temp_sentence_words_indexes_buffer.copy(),
                all_read_words_buffer=all_read_words_buffer,
                question=question,
            )

            # ----------------------------------------------------------------------------------------------------------
            # Determine whether to skip the next word or not using the Bayesian Skipper
            context_words = ' '.join(temp_verbal_buffer)
            parafoveal_seen_letters = oc_outputs[-1]['parafoveal seen letters']
            estimated_next_word_len = self._next_word_len_in_section
            if self._next_word_index_in_section != const.NEGATIVE_ONE:   # If the next word exists
                # TODO we can skip more than one word, or skip with a lower threshold for higher skipping rate.
                #  Try the second one first. We could let the agent output a binary choice: with the current state of
                #  time constraint and the current word (contextual understanding),
                #  reason what strategy to read next: increase the reading speed or not.
                #  <maybe increased reading speed will ocupy the memory gisting ability, tune this later if needed in terms of the comprehension data>
                # TODO: other things to change: the skipped words should not be stored in the verbal buffer to be processed into the memory.
                self._skip_next_word_in_section = self.bs_env.step(
                    inputs={
                        'context_words': context_words,
                        'next_word': self._next_word_in_section,
                        # 'fovea_letters': fovea_seen_letters,        # TODO if the next word is also processed in the fovea, we should skip it as well, not only parafoveal seen letters.
                        'parafovea_letters': parafoveal_seen_letters,   # TODO, check the oculomotor controller's output, since the parafoveal already covered the fovea, it should be fine.
                        'estimated_word_len': estimated_next_word_len,
                    },
                    # strategy="skim", # TODO use this to change the skipping behavior of the agent.
                )
            else:
                # If there is no next word, we should not skip the current word
                self._skip_next_word_in_section = False     # TODO now we only talk about skipping one word at a time
                self._flag_read_to_the_end = True
            # print(
            #     f"{cons.LV_TWO_DASHES}Skipping decision: {self._skip_next_word_in_section} the target word is: {self._target_word_in_stimuli}, the next word is: {self._next_word_in_section}\n"
            #     f"{cons.LV_THREE_DASHES}The skipping details are: {self.bs_env.log()}")

            # ----------------------------------------------------------------------------------------------------------
            # Update the short-term memory and gist the information (in an episodic manner) if there is the next word and asked to skip
            # TODO maybe we should not update the verbal buffer with the skipped word, skipping is skipping
            target_word_spatial_info = oc_outputs[-1]['target word visuospatial info']
            if self._skip_next_word_in_section == True:
                self._num_skipped_words_on_stimuli += 1
                target_word_spatial_info = oc_outputs[-1]['next word visuospatial info']
                # Update the verbal buffer and memories with the incoming read word
                temp_verbal_buffer = self._update_verbal_buffer_for_word_prediction(
                    word=self._next_word_in_section,
                    temp_verbal_buffer=temp_verbal_buffer.copy(),
                    temp_verbal_buffer_size=temp_verbal_buffer_size,
                )
                # Check and update the processing cycles in the memories
                temp_sentence_buffer, temp_sentence_words_indexes_buffer, all_read_words_buffer = self._update_memories_and_update_strategy_by_sentences(
                    word=self._next_word_in_section,
                    word_index=self._next_word_index_in_section,    # TODO to cover more important information in the text, the agent should be able to skip more
                    temp_sentence_buffer=temp_sentence_buffer.copy(),
                    temp_sentence_words_indexes_buffer=temp_sentence_words_indexes_buffer.copy(),
                    all_read_words_buffer=all_read_words_buffer,
                    question=question,
                )
            # TODO when under time pressure, the agent should attempt to infer the most important parts.
            # TODO some literature finding: Principles of Skim Reading: This study finds that when skimming under time pressure, readers' ability to recognize important, unimportant, and inferable information declines equally. The process is less about effectively focusing on critical information and more about managing limited cognitive resources.
            #  Ref: "How Much Do We Understand When Skim Reading?" (Duggan & Payne, 2006)
            #  Skim reading primarily affects cognitive understanding rather than being directly correlated with detailed word skipping.
            #  The study highlights a cognitive bias where skimmers over-interpret complex information as consistent with the text,
            #  suggesting that skimming is more about the cognitive struggle to maintain coherence under time pressure.
            #  Impact of Time Pressure on Understanding: Time pressure leads to a general decline in comprehension,
            #  with skimmers more likely to misinterpret or overgeneralize information due to cognitive biases.
            #  Time Pressure and Word Skipping: The study does not specifically address how time pressure affects word skipping but suggests that under time pressure,
            #  skimmers are more likely to misinterpret complex information, indirectly implying that word skipping might contribute to these misunderstandings.

            # ----------------------------------------------------------------------------------------------------------
            # Prepare for the next simulation step
            norm_fix_x_batch = [oc_output['norm_fix_x'] for oc_output in oc_outputs]
            norm_fix_y_batch = [oc_output['norm_fix_y'] for oc_output in oc_outputs]
            # target_word_visuospatial_info = oc_outputs[-1]['target word visuospatial info']
            parafoveal_seen_letters_batch = [oc_output['parafoveal seen letters'] for oc_output in oc_outputs]
            dynamic_sim_data_log_dict = self._update_log(
                init_norm_fix_x=init_norm_fix_x,
                init_norm_fix_y=init_norm_fix_y,
                norm_fix_x_batch=norm_fix_x_batch,
                norm_fix_y_batch=norm_fix_y_batch,
                parafoveal_seen_letters_batch=parafoveal_seen_letters_batch,
                oc_logs_batch=oc_outputs,
                context_words=context_words,
                parafoveal_seen_letters=parafoveal_seen_letters,
                target_word_visuospatial_info=target_word_spatial_info,
                estimated_word_len=estimated_next_word_len,
                skipping_details=self.bs_env.log(),
            )
            self._dynamic_sim_data_log_dicts.append(dynamic_sim_data_log_dict)
            # ----------------------------------------------------------------------------------------------------------
            # # Check whether to stop reading the section
            # if self._stop_reading_by_agent is True:
            #     break

            # ----------------------------------------------------------------------------------------------------------
            # Finished one step, determine whether to regress or not
            # Update the states -- init eye movements
            init_norm_fix_x = oc_outputs[-1]['norm_fix_x']
            init_norm_fix_y = oc_outputs[-1]['norm_fix_y']

            # ----------------------------------------------------------------------------------------------------------
            # Move to the next step -- Update whether to regress or not when updating the memory
            # Determine the next word to read based on the regression decision
            if self._regress_decision == True:
                # Regress to the previous word
                self._update_target_word_and_next_word_in_regression_reading()
            else:
                # Without regressing to other words, the agent will move to the next word
                self._update_target_word_and_next_word_in_serial_reading()

            # # ----------------------------------------------------------------------------------------------------------
            # # Without regressing to other words, the agent will move to the next word
            # # Update the target word index and memory according to the skip decision
            # # Skip or not --> prepare for moving to the next fixations
            # self._update_target_word_and_next_word_in_serial_reading()

            # ----------------------------------------------------------------------------------------------------------
            # Check the target word and the next word's validity
            self._check_target_word_and_next_word_validity()

            # ----------------------------------------------------------------------------------------------------------
            # Finish preparations
            # Check whether to finish reading the section
            self._check_termination_by_time_constraint(all_read_words_buffer=all_read_words_buffer)

    def _update_log(
            self,
            init_norm_fix_x: float,
            init_norm_fix_y: float,
            norm_fix_x_batch: list,
            norm_fix_y_batch: list,
            parafoveal_seen_letters_batch: list,
            oc_logs_batch: list,
            context_words: str,
            parafoveal_seen_letters: str,
            target_word_visuospatial_info: list,
            estimated_word_len: int,
            skipping_details: dict = None,
    ) -> dict:
        """
        Update the data logger dictionary with new values. Only update the step-related dynamic values.
        :return: Simulated data logger dictionary -- the dynamic part
        """
        # Update the dynamic part of the dictionary
        step_info = {
            "steps": self._word_steps,
            "working_memory": self._pseudo_short_term_memory.copy(),
            "read_content_in_section": self._read_content_in_section.copy(),
            "read_positions_in_section": self._read_positions_in_section.copy(),
            "gists_in_section": self._memory_gists_in_the_current_section.copy(),
            "terminate_reading": self._stop_reading_trial_by_time,
            "stop_reading_in_section": self._flag_read_to_the_end,
            "target_word_info": {
                "target_word_index_in_section": self._target_word_index_in_section,
                "target_word_in_section": self._target_word_in_stimuli,
                "target_word_len_in_section": self._target_word_len_in_section,
                "target_word_visuospatial_info": target_word_visuospatial_info,
            },
            "next_word_info": {
                "next_word_index_in_section": self._next_word_index_in_section,
                "next_word_in_section": self._next_word_in_section,
                "next_word_len_in_section": self._next_word_len_in_section,
            },
            "skip_next_word_info": {
                "skip_decision": self._skip_next_word_in_section,
                "context_words": context_words,
                "next_word": self._next_word_in_section,
                "parafoveal_seen_letters": parafoveal_seen_letters,
                "estimated_next_word_len": estimated_word_len,
                "skipping_details": skipping_details,
            },
            "fixations_info": {
                "init_norm_fix_x": init_norm_fix_x,
                "init_norm_fix_y": init_norm_fix_y,
                "norm_fix_x_batch": norm_fix_x_batch,
                "norm_fix_y_batch": norm_fix_y_batch,
                "parafoveal_seen_letters_batch": parafoveal_seen_letters_batch,
                "oc_logs_batch": oc_logs_batch,
            },
        }

        # Update the comprehension of the given section
        if self._flag_read_to_the_end is True:
            self._fixed_sim_data_log_dict["section_info"]["section_comprehension"]["task_specification"] = "Please summarize what you have read"
            self._fixed_sim_data_log_dict["section_info"]["section_comprehension"]["read_content"] = ' '.join(self._read_content_in_section)
            self._fixed_sim_data_log_dict["section_info"]["section_comprehension"]["comprehension"] = self._section_comprehension

        return step_info

    def _save_logs_to_json(self):
        """
        Write the simulation data to a JSON file.
        :return: None
        """

        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(element) for element in obj]
            else:
                return obj

        # Assuming self._sim_data_logs is your list of logs
        self._dynamic_sim_data_log_dicts = [convert_numpy_types(log) for log in self._dynamic_sim_data_log_dicts]

        # Combine the fixed and dynamic logs
        combined_logs = {
            "fixed_info": self._fixed_sim_data_log_dict,
            "dynamic_steps": self._dynamic_sim_data_log_dicts,
        }

        # Save the logs to a JSON file
        with open(self._sim_data_json_filename, 'w') as file:
            json.dump(combined_logs, file, indent=4)

        print(f"The sim data logs are stored at: {self._sim_data_json_filename}")

    def close(self):
        """
        Close the simulator.
        :return: None
        """

        # Save the simulation results
        self._save_logs_to_json()


if __name__ == '__main__':

    # ------------------------------------------------------------------------------------------------------------------
    # Write and test example here
    read_info = [
        ("I am Bai Yunpeng (白云鹏), a PhD candidate in Computer Science, NUS, under the supervision of Prof. Shengdong Zhao and Prof. Antti Oulasvirta.", "first sentence in the first paragraph"),
        ("My research interests lie in the fields of Human-Computer Interaction, modeling, and artificial intelligence.", "second sentence in the first paragraph"),
        ("Specifically, I focus on Computationally Rational Modeling using Reinforcement Learning and Heads-Up Computing. ", "third sentence in the first paragraph"),
        ("My work involves modeling human users and developing simulations to aid researchers and designers in understanding user behavior and building simulators.", "fourth sentence in the first paragraph"),
        ("This helps enhance efficiency and innovation within the HCI community.", "fifth sentence in the first paragraph"),
        ("In addition to my primary research areas, I have expertise in signal processing, machine learning (specifically in activity detection), 3D modeling, and manufacturing.", "first sentence in the second paragraph"),
        ("I am deeply passionate about engineering and technology.", "second sentence in the second paragraph"),
        ("Originally from Tai Yuan (太原), Shan Xi (山西), China – a city known for its rich history and cultural heritage, including being the birthplace of the Tang Dynasty.", "first sentence in the third paragraph"),
        ("I enjoy spending my leisure time playing video games, basketball, and football.", "second sentence in the third paragraph"),
    ]
    task_instruction = (
        "You are a reader engaged in reading a text with the goal of answering comprehension questions and completing free recall tests afterward. However, your reading time is limited. "
        "Your task is to balance your reading speed and comprehension accuracy effectively. "
        "The comprehension questions are based on the entire text, from beginning to end. To answer these questions accurately, "
        "you should aim to read the entire text if possible."
    )
    # _question = "What was he awarded the Nobel Prize for?"
    STIMULI_MCQs = {    # TODO move this to a json file later
        0: {},  # Indexed by the image idx
        1: {},
        2: {},
        3: {},
        4: {
            0: {
                "question": "What was one of the first sustainable changes the speaker made?",
                "options": {
                    "A": "Switching to public transportation regularly.",
                    "B": "Cutting down on the use of single-use plastics.",
                    "C": "Starting to grow their own vegetables at home.",
                    "D": "Installing solar panels for renewable energy.",
                    "E": "Do not know / Cannot remember",
                },
            },
            1: {
                "question": "How has meal planning benefited the speaker?",
                "options": {
                    "A": "It has made meal preparation more efficient.",
                    "B": "It has allowed them to shop less frequently.",
                    "C": "It has minimized food waste and cut down on expenses.",
                    "D": "It has led to more thoughtful and organized grocery shopping.",
                    "E": "Do not know / Cannot remember",
                },
            },
            2: {
                "question": "What did the speaker set up in their backyard?",
                "options": {
                    "A": "A vegetable garden for growing fresh produce.",
                    "B": "A compost bin to recycle organic waste.",
                    "C": "A rainwater collection system for sustainability.",
                    "D": "A bird feeder to attract local wildlife.",
                    "E": "Do not know / Cannot remember",
                },
            },
            3: {
                "question": "Why does the speaker find composting gratifying?",
                "options": {
                    "A": "It cuts down on the amount of food they need to purchase.",
                    "B": "It produces natural fertilizer for their garden.",
                    "C": "It keeps food scraps out of landfills.",
                    "D": "It brings more wildlife into their yard.",
                    "E": "Do not know / Cannot remember",
                },
            },
            4: {
                "question": "What is the speaker encouraging others to do?",
                "options": {
                    "A": "Start incorporating organic products into their routine.",
                    "B": "Think about ways to make their community more sustainable.",
                    "C": "Share tips and ideas for adopting greener habits.",
                    "D": "Explore ways to reduce waste and live more sustainably.",
                    "E": "Do not know / Cannot remember",
                },
            },
        },
        5: {},
        6: {},
        7: {},
        8: {},
        9: {},
    }
    # _questions = [
    #     # "What additional feature will the new bikes include?	A) Built-in navigation systems; B) GPS tracking and improved safety features; C) Larger baskets for carrying items; D) Automated self-repair systems"
    #     # "What will the city do to promote safe biking practices?	A) Increase fines for unsafe biking; B) Build more bike lanes; C) Launch a public awareness campaign; D) Offer free bicycle maintenance"
    #     # "What has been the trend in daily ridership of the public bicycle program over the past year?	A) It has decreased by 10%; B) It has remained the same; C) It has increased by 20%; D) It has doubled",
    #     # "please summarise what you have read, a free recall test, write down anything you remember from the LTM",
    #     # "What is the primary goal of the new robotics program?	A) To compete with other schools; B) To inspire students to explore STEM careers; C) To reduce screen time for students; D) To promote physical fitness",
    #     # "What kind of experience does the robotics program offer to students?	A) Hands-on experience in designing and building robots; B) Theory-based learning; C) Online courses; D) Physical education",
    #     # "Who partnered with the school to provide resources for the program?	A) Local universities; B) Government agencies; C) Tech companies; D) Non-profit organizations",
    #     # "What opportunities will students have as part of the program?	A) Internships at tech companies; B) Scholarships for college; C) Competing in regional and national robotics competitions; D) Access to private tutoring",
    #     # "What are the future plans for the robotics program?	A) To introduce advanced courses and summer camps; B) To limit participation to seniors; C) To replace the computer science curriculum; D) To focus on sports training",
    #     # "Wind is the weakest force for A. mass movement. B. abrasions. C. erosion. D. sand deposition.",
    #     # "Plants can combat erosion because they A. can't be picked up by wind. B. hold the soil in place. C. embed themselves in rock. D. keep the soil moist.",
    #     # "Wind's primary means of erosion is by A. deflation. B. inflation. C. abrasion. D. excision.",
    #     # "Wind carries mostly small sediment particles, such as A. clay. B. silt. C. both A & B. D. neither A nor B.",
    #     # "When the wind leaves large, heavy rocks behind, it is called A. a blowout. B. desert pavement. C. attrition. D. a hollow.",
    #     # "Most deserts' stone shapes are the result of weathering and A. polishing. B. deflation. C. water erosion. D. abrasion.",
    #     # "Some sand dunes can reach ______ meters tall. A. 25 B. 100 C. 250 D. 500",
    #     # "Loess help to generate A. sand dunes. B. coarse sediment particles. C. fertile soil. D. water reservoirs.",
    #     # "please summarise what you have read, a free recall test, write down anything you remember from the LTM",
    #     # "Who was having a birthday? A. Sally B. Billy C. Tom D. The man",
    #     # "Who gave Tom the balloon? A. Sally B. The man C. Bill D. His mother",
    #     # "Who saved the balloon? A. Tom B. Bill C. Sally D. The man",
    #     # "What was on the front of the balloon? A. Tom's name B. Tom's picture C. A happy face D. Nothing",
    #     # "What is the main idea of this story? A. How to distinguish a mirage from another type of image B. The many forms of mirages C. The nature of mirages D. Superior and inferior images",
    #     # "'Refraction' is a term used to denote a light stream's A. reflection. B. bending. C. absorption. D. condensation.",
    #     # "The 'water' you see in a highway mirage is actually A. water molecules in the air. B. the blue sky. C. the shimmering blacktop. D. a reflection from a nearby water source, such as a lake.",
    #     # "A Fata Morgana is also known as what phenomenon? A. A mirage at sea B. The reflection of a distant city C. A superior mirage of a woman D. An inferior mirage in tropical jungles",
    #     # "When are you most likely to see a Fata Morgana? A. Around dusk or dawn B. At night C. At noon D. During a rainstorm",
    #     # "When was the Silent City first described? A. 1897 B. 1905 C. 1927 D. 1953",
    #     # "What do scientists think the Silent City was? A. A cold air front in Alaska B. A mirage of Bristol, England C. A mirage of the Pacific Ocean D. Heat rising from the ground surface",
    #     # "The people who saw a city floating in the air saw an example of a[n] A. Fata Morgana. B. inferior mirage. C. superior mirage. D. mirage at sea.",
    #
    #     # "What will the city do to promote safe biking practices?	A) Increase fines for unsafe biking; B) Build more bike lanes; C) Launch a public awareness campaign; D) Offer free bicycle maintenance",
    #     # "What opportunities will students have as part of the program?	A) Internships at tech companies; B) Scholarships for college; "
    #     # "C) Competing in regional and national robotics competitions; D) Access to private tutoring; E) Do not know / Cannot remember,"
    #
    #     # "How many garden sites currently exist in the city as part of the community garden project?	A) 5; B) 15; C) 20; D) Over 20",
    #     # "What is a key feature of the new gardens being added?	A) Automated watering systems; B) Raised beds for planting; C) Vertical gardens; D) Hydroponic systems",
    #     # "Who will help teach gardening classes at the new community gardens?	A) Professional gardeners; B) Local volunteers; C) School teachers; D) City officials",
    #     # "What environmental benefit do the community gardens provide?	A) Increase in wildlife population; B) Reduction of the city’s carbon footprint; C) Creation of new parking spaces; D) Improvement of air quality",
    #     # "How is the expansion of the community garden project being funded?	A) Federal grants; B) Private investments; C) City grants and donations from local businesses; D) Membership fees",
    #
    #     # "What was one of the first sustainable changes the speaker made?	A) Using public transportation; B) Reducing reliance on single-use plastics; C) Growing their own vegetables; D) Installing solar panels",
    #     # "How has meal planning benefited the speaker?	A) It has helped save time; B) It has reduced the need for shopping; C) It has minimized food waste and saved money; D) It has increased the amount of food they buy",
    #     # "What did the speaker set up in their backyard?	A) A vegetable garden; B) A compost bin; C) A rainwater collection system; D) A bird feeder",
    #     # "Why does the speaker find composting gratifying?	A) It reduces the amount of food they need to buy; B) It creates fertilizer for their garden; C) It prevents food scraps from going to a landfill; D) It attracts wildlife to their yard",
    #     # "What is the speaker encouraging others to do?	A) Purchase organic products; B) Move to a more sustainable community; C) Exchange ideas for a greener lifestyle; D) Adopt a zero-waste lifestyle",
    #
    #     # "What has been essential for the speaker in maintaining consistency with their workout regimen?	A) Working out daily; B) Setting realistic, attainable goals; C) Lifting heavier weights; D) Running long distances",
    #     # "How has having a workout partner benefited the speaker?	A) Provided competition; B) Added accountability and enjoyment; C) Increased the intensity of workouts; D) Reduced the time needed to work out",
    #     # "What routine did the speaker and their workout partner establish?	A) Regularly checking in with each other; B) Running together every morning; C) Tracking their diets; D) Competing in races",
    #     # "What aspect of fitness has the speaker come to appreciate?	A) Working out as much as possible; B) The importance of rest and recovery; C) Lifting heavy weights; D) High-intensity interval training (HIIT)",
    #     # "What advice does the speaker offer to those starting a new workout routine?	A) Work out every day without fail; B) Focus only on intense workouts; C) Prioritize consistency and balance; D) Start with heavy weights",
    #
    #     # "What is the title of the book the speaker recently finished?	A) The Journey Within; B) The Road Ahead; C) The Path to Peace; D) The Inner Quest",
    #     # "What themes does the novel primarily explore?	A) Adventure and travel; B) Self-discovery and personal growth; C) Romance and relationships; D) Mystery and suspense",
    #     # "How did the book impact the speaker personally?	A) It was a fun read but had little impact; B) It challenged their perspectives and prompted introspection; C) It made them want to travel more; D) It inspired them to write a novel",
    #     # "What aspect of the book did the speaker find particularly authentic?	A) The dialogue; B) The setting; C) The portrayal of the characters’ internal struggles; D) The action scenes",
    #     # "What does the speaker encourage others to do at the end of the passage?	A) Write their own book; B) Share and discuss their favorite books; C) Read only classic literature; D) Avoid reading introspective novels",
    #
    #     # "What is the primary purpose of a digital detox according to the passage?	A) To abandon technology entirely; B) To find a healthy balance with technology; C) To avoid all social media; D) To eliminate screen time completely",
    #     # "What was one of the first changes the speaker made for their digital detox?	A) Deleting all social media accounts; B) Establishing tech-free zones in the home; C) Limiting phone use to weekends; D) Using only a flip phone",
    #     # "Which areas of the home did the speaker designate as tech-free zones?	A) Living room and kitchen; B) Bedroom and dining area; C) Office and garage; D) Bathroom and hallway",
    #     # "What activities did the speaker choose for their mornings during the digital detox?	A) Checking emails; B) Reading a physical book or taking a walk; C) Watching TV; D) Listening to podcasts",
    #     # "What advice does the speaker give to those considering a digital detox?	A) Start by eliminating all screens; B) Set small, manageable goals; C) Avoid using technology for a week; D) Purchase a digital detox guidebook",
    #
    #     # "How did the speaker originally view cooking?	A) As a relaxing activity; B) As a daily necessity; C) As a social event; D) As a hobby",
    #     # "What does mindful cooking involve according to the speaker?	A) Cooking quickly to save time; B) Paying full attention and enjoying each step; C) Following recipes exactly; D) Using only organic ingredients",
    #     # "What does the speaker do to start the mindful cooking process?	A) Sets a timer; B) Chooses fresh ingredients and notices their qualities; C) Prepares all the ingredients in advance; D) Cleans the kitchen thoroughly",
    #     # "How has mindful cooking changed the speaker's perspective on making food?	A) It has become a creative and self-care activity; B) It is now seen as a competition; C) It is more stressful; D) It takes longer than before",
    #     # "Why does the speaker believe the food tastes better now?	A) They use more spices; B) The food is cooked faster; C) Each dish is made with care and attention; D) They follow recipes exactly",
    #
    #     # "What has the speaker learned about packing over the years?	A) More is better; B) Less is more; C) Always bring extra clothes; D) Pack for every possible scenario",
    #     # "What is the key to minimalist travel according to the speaker?	A) Bringing as many items as possible; B) Careful planning; C) Always checking luggage; D) Traveling with friends",
    #     # "What does the speaker focus on when creating a packing list?	A) Fashionable clothing; B) Souvenirs; C) Versatile clothing and multipurpose items; D) Electronics",
    #     # "How does traveling light affect the speaker's travel experience?	A) It adds stress; B) It encourages a more mindful experience; C) It makes travel less enjoyable; D) It requires more planning",
    #     # "What does the speaker value most from their travels?	A) Souvenirs; B) Material items; C) Experiences and memories; D) Luxury accommodations",
    # ]
    user_profile = {
        "proficiency": "good",
        "interest level": "high",
        "background knowledge": "specific",
    }
    _image_env_index = 4
    time_penalty_weight = 5
    predefined_fixed_time_constraint = 90
    sim_data_save_path = const.SIM_DATA_SAVE_PATH_LINUX
    # Create a folder name related to time
    time_str = datetime.datetime.now().strftime("%m_%d_%H_%M")
    folder_name = f"sim_raw_data_{time_str}_image_{_image_env_index}_time_penalty_weight_{time_penalty_weight}"
    sim_data_dir = os.path.join(sim_data_save_path, folder_name)

    # Create this folder is not exists
    if not os.path.exists(sim_data_dir):
        os.makedirs(sim_data_dir)

    # Test the UIReader
    reader_agent = ReaderAgent(sim_data_save_dir=sim_data_dir)
    reader_agent.reset(
        image_env_index=_image_env_index,
        task_instruction=task_instruction,
        question="NA for now",
        user_profile=user_profile,
        time_penalty_weight=time_penalty_weight,
        time_constraint_in_seconds=predefined_fixed_time_constraint,
    )
    reader_agent.simulate(simulation_task_type=const.SIMULATION_TASK_MODE["comprehend"])
    print(f"The question's STM is {reader_agent.stm_env.STM}\n"
          f"The question's gists in the LTM is {reader_agent.ltm_env.gists}\n")

    # Answer MCQ questions
    # for mcq in MCQs[_image_env_index]:
    stim_mcqs = STIMULI_MCQs[_image_env_index]
    for idx, mcq in enumerate(stim_mcqs):
        question = stim_mcqs[mcq]["question"]
        options = stim_mcqs[mcq]["options"]
        _answer = reader_agent.wm_env.retrieve_memory(question_type=const.QUESTION_TYPES["MCQ"], question=question, options=options, ltm_gists=reader_agent.ltm_env.gists)
        print(f"The answer to the question '{question}' is: {_answer}. \n")
    # Answer free-recall questions
    _free_recall_answer = reader_agent.wm_env.retrieve_memory(question_type=const.QUESTION_TYPES["FRS"], ltm_gists=reader_agent.ltm_env.gists)
    print(f"The free recall results are: {_free_recall_answer}. \n")
    # Close the simulator
    reader_agent.close()
