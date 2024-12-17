import ast
from typing import List

from openai import OpenAI
import time
import yaml
import numpy as np
import spacy

import step5.utils.constants as const


class LLMShortTermMemory:

    def __init__(
            self,
            config: str = r'D:\Users\91584\PycharmProjects\reading-model\step5\config.yaml',
            stm_capacity: int = 4,
    ):
        """
        Initialize the Short-Term Memory.

        Part of the model -- 08 July version
        - Link: https://www.figma.com/board/T9YEXqz2NukOfcLLmHUP4L/18-June-Architecture?node-id=0-1&t=6eo2ZwlExnvK4qnC-0

        Model's structure reference: Baddeley and Hitch's models on working memory
        - Developments in the concept of working memory. Https://psycnet-apa-org.libproxy1.nus.edu.sg/record/1995-04539-001
        - Working Memory. https://www.jstor.org/stable/pdf/2876819.pdf

        The short-term memory is responsible for both phonological and visuospatial information. In the given reading task,
            we abstract the phonological information as the read words from the Rational Reader. And the visuospatial information
            is abstracted as the current reading position in the text. Specifically, the line index and word index in the line.

        We model both the phonological and visuospatial information as parts of the STM, thus they could only contain 7+-2 items/words.
            Without further attention from the supervisory controller to comprehend and summarize them as gists to the episodic buffer,
            it will be forgotten very soon. If the STM is full, the oldest item will be replaced by the new one.
            If the information is comprehended and gisted, then detailed content information will be dumped.

        Reference to the gisting method, and how to integrate content + visuospatial information:
            Paper: A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts, https://arxiv.org/pdf/2402.09727

        When the STM is full, the over-shootings are either forgotten or summarized as gists.

        ----------- Update on 16 July -----------
        Model's new structure -- version 15 July; link: https://www.figma.com/board/T9YEXqz2NukOfcLLmHUP4L/18-June-Architecture?node-id=0-1&t=lX2j28YyOl2CPR1m-0
        The Short-Term Memory Module here should be responsible for the following tasks:
            1. With the working memory (the actual executor), use the LLM convert the read raw words/sentences into micro-gists -- called propositional notations.
            2. Manage the STM storage:
                1) Store the micro-gisted content and spatial information as chunks into the STM storage. We store four chunks according to literature.
                2) Remove the previous micro-gists: the STM storage has a limited capacity. Although the literature says
                    human make-decisions on retaining the most relevant and important chunks, we simplify this process
                    by queueing all chunks. And removing all older chunks automatically. This is because we automatically call the
                    macro-operator to evaluate and store actual gists into the LTM all the time. Another point of discrepancy
                    with literature is usually there will be more than one propositional notations in one chunk, but in our case, since we
                    have the LLM, we do not have to really encode micro-gists into propositional notations with a very concise format of
                    ACTION(SUBJECT, OBJECT), thus we can reserve more information. So we regard one chunk equals to one micro-gists.

        Literature evidence on
        - How short-term memory works when reading text and comprehending: Toward a model of text comprehension and production, 1978, link: https://www.cl.cam.ac.uk/teaching/1516/R216/Towards.pdf
        - The magic number of chunks in the short-term memory:
            1. The number is 7+-2: The Magical Number Seven, Plus or Minus Two. Some Limits on Our Capacity for Processing Information, https://psycnet-apa-org.libproxy1.nus.edu.sg/record/1957-02914-001
            2. The number should be 4: Seven plus or minus two, 2011, https://onlinelibrary.wiley.com/doi/abs/10.1002/piq.20099

        :param stm_capacity: The length of the Short-Term Memory (in our model, STM is part of the Working Memory).
        """
        print(f"{const.LV_ONE_DASHES}Short-Term Memory -- Initialize the Short-Term Memory.")

        # Read the configuration file
        with open(config, 'r') as stream:
            self._config = yaml.load(stream, Loader=yaml.FullLoader)
        openai_api_key = self._config['llm']['API_key']

        # Specific configurations
        self._gpt_model = self._config['llm']['stm']['model']
        self._refresh_interval = self._config['llm']['stm']['refresh_interval']
        self._max_num_requests = self._config['llm']['stm']['max_num_requests']
        self._retry_delay = self._config['llm']['stm']['retry_delay']

        # LLM related variables
        self._client = OpenAI(api_key=openai_api_key)

        self._role = None
        self._micro_gist_prompt = None
        self._response = None

        # Define the variables
        self._ground_truth_read_content = None  # The read content so far -- ground truth
        self._one_content = None  # The phonological information, one instance of reading content, e.g., one sentence
        self._one_spatial_info = None  # The visuospatial information, i.e., the current reading position

        self._reading_strategy = None  # The reading strategy, e.g., normal, skimming, scanning, etc.

        # Define the length of the STM
        self._STM_capacity = stm_capacity  # The length of the Short-Term Memory is set 4 instead of 7+-2

        # Initialize the STM
        # The Short-Term Memory, it will be a list storing a list of dictionaries
        # [{"content": xxx, "spatial info": xx, "memory strength": xx, "elapsed time": xx}, {}, ...]
        self.STM = None
        # Initialize the STM with strength
        # The Short-Term Memory with strength, it will be a list storing a lot of (content str, spatial info str, strength) tuples
        # The strength will decay over time; the more recent the information is, the stronger the strength is;
        #   when the strength decays to a certain value, the memory will be removed from the STM, and left with some marks as indicators of memory loss (if they are not poped out)
        self._INIT_STRENGTH = 1.0  # The initial strength of the memory
        self._MEMORY_LOSS_THRESHOLD = 0.4  # The threshold of the memory loss
        self._MEMORY_LOSS_MARK = const.MEMORY_LOSS_MARK  # The mark of the memory loss

        # Initialize the micro-gist extractor -- spacy NLP model
        self._nlp = spacy.load("en_core_web_sm")

    def reset(self):
        """
        Reset all the Short-Term Memory-related variables.
        :return: None
        """
        self._role: str = ""
        self._micro_gist_prompt: str = ""
        self._response: str = ""

        self._ground_truth_read_content: list = []
        self._one_content: str = ""
        self._one_spatial_info: str = ""

        self.STM: list = []

        self._reading_strategy: str = ""

    def _process_text_by_reading_strategies(self, text: str,):
        """
        Process the text just read differently according to the reading strategies.
        :param text: The text to be processed, the content just read
        :return: None
        """
        # These parameters might be tunable in the future
        if self._reading_strategy == const.READ_STRATEGIES["skim"]:
            TEXT_SAMPLE_RATE = 1.0
        elif self._reading_strategy == const.READ_STRATEGIES["normal"]:
            TEXT_SAMPLE_RATE = 1.0
        elif self._reading_strategy == const.READ_STRATEGIES["careful"]:
            TEXT_SAMPLE_RATE = 1.0
        else:
            raise ValueError(f"The reading strategy {self._reading_strategy} is not supported.")
        # SKIM_READING_SAMPLE_RATE = 0.5
        # NORMAL_READING_SAMPLE_RATE = 0.75
        # CAREFUL_READING_SAMPLE_RATE = 1.0

        # Partition the text into words
        words = text.split(" ")

        # randomly sample the words according to the reading strategy
        num_sampled_words = int(np.ceil(TEXT_SAMPLE_RATE * len(words)))
        if num_sampled_words <= 0:
            sampled_words = [const.NA]
        else:
            # Randomly sample the words
            sampled_words = np.random.choice(words, num_sampled_words, replace=False)

        return " ".join(sampled_words)

    # def extract_micro_gist_and_update_stm(
    #         self,
    #         text: str,  # The text to be processed, the content just read
    #         spatial_info: str,  # The spatial information, e.g., the current reading position
    #         sentence_start_word_index: int,  # The start word index of the sentence in the text
    #         elapsed_time_since_last_update: float,  # The time elapsed since the last reading update
    #         reading_strategy: str = cons.READ_STRATEGIES["normal"],
    #         activated_schemas: List[str] = None  # New parameter for schemas
    # ):
    #     """
    #     Extract the micro-gist from the text and update the STM.
    #     The reason for using a Spacy NLP model to extract key information: entity, noun, and verb is consistent with the literature.
    #     Reference: Toward a model of text comprehension and production, 1978, link: https://www.cl.cam.ac.uk/teaching/1516/R216/Towards.pdf
    #
    #     :param text: The text to be processed, the content just read
    #     :param spatial_info: The spatial information, e.g., the current reading position
    #     :param sentence_start_word_index: The start word index of the sentence in the text
    #     :param elapsed_time_since_last_update: The time elapsed since the last reading update, the unit is seconds
    #     :param reading_strategy: The reading strategy, e.g., normal, skimming, scanning, etc.
    #     :param activated_schemas: The activated schemas in the LTM
    #     :return: None (updated STM)
    #     """
    #
    #     # Assign defaults if activated_schemas is None
    #     if activated_schemas is None:
    #         # TODO the schemas are part of the LTM, which should be an existing knowledge database,
    #         #  assuming to cover all reading materials in the experiment -- meaning the users could understand them anyway.
    #         activated_schemas = []
    #
    #     # Get the one-time content and spatial information
    #     self._one_content = text
    #     self._one_spatial_info = spatial_info
    #
    #     # Initialize the reading strategy
    #     self._reading_strategy = reading_strategy
    #
    #     # Process the text just read differently according to the reading strategies
    #     # resampled_words = self._process_text_by_reading_strategies(text=text)
    #
    #     micro_gist_content = ""
    #
    #     # Run GPT API for the macro-operator
    #     for attempt in range(self._max_num_requests):
    #         try:
    #             # print(f"{cons.LV_ONE_DASHES}LLM-based STM micro-gist extraction:\n"  # 20*'-'
    #             #       f"{cons.LV_TWO_DASHES}Attempt {attempt} is running...\n"
    #             #       f"{cons.LV_TWO_DASHES}STM content: '{self.STM}' \n")
    #
    #             raw_response = self._get_response(
    #                 role="You are a reader processing information into short-term memory chunks.",
    #                 prompt=(
    #                     f"Read the following sentence and create a memory chunk that captures its main idea:\n"
    #                     f"Sentence: '{text}'\n\n"
    #                     f"You have the following schemas (background knowledge) activated:\n"
    #                     f"{', '.join(activated_schemas)}\n\n"
    #                     f"- Use these schemas to interpret and summarize the sentence.\n"
    #                     f"- Focus on the core message or primary action relevant to the schemas.\n"
    #                     f"- Group related concepts together into a meaningful unit.\n"
    #                     f"- Keep the chunk concise, using simple language.\n"
    #                     f"- Avoid unnecessary details, but retain essential information.\n"
    #                     f"- Do not add any information not present in the sentence.\n"
    #                     f"- Imagine you're storing this chunk in your short-term memory to recall it later."
    #                 )
    #             )
    #
    #             print(f"{cons.LV_TWO_DASHES}Raw response for STM: {raw_response}")
    #
    #             if raw_response:
    #                 # Loop to validate and parse the response
    #                 try:
    #                     # Separate the content and spatial information from the response: split by the delimiter
    #                     micro_gist_content = raw_response.strip()
    #                     if isinstance(micro_gist_content, str):
    #                         break
    #
    #                 except (SyntaxError, ValueError, AssertionError) as e:
    #                     print(f"Error parsing response for the LLM STM micro-gist extraction: {e}\n"
    #                           f"The current incorrect-format STM is: {raw_response}")
    #                     continue
    #
    #         except Exception as e:
    #             print(f"Attempt {attempt} failed with error: {e}")
    #             time.sleep(self._retry_delay)
    #
    #     # Get the gist
    #     # Dictionary format:
    #     gist = {
    #         # cons.CONTENT: ' '.join(micro_gist_content),
    #         cons.CONTENT: micro_gist_content,
    #         cons.SPATIAL_INFO: spatial_info,
    #         # cons.SENTENCE_START_INDEX: sentence_start_word_index,
    #         cons.STM_STRENGTH: self._INIT_STRENGTH,
    #         cons.STM_ELAPSED_TIME: 0,
    #     }
    #     # Push to the STM
    #     self.STM.append(gist)
    #
    #     # Update the STM by removing the oldest item if the STM is full
    #     while len(self.STM) > self._STM_capacity:
    #         self.STM.pop(0)
    #
    #     # Update other gists' elapsed time and strength
    #     for stm_chunk_idx in range(len(self.STM)):
    #         if stm_chunk_idx != len(self.STM) - 1:
    #             chunk = self.STM[stm_chunk_idx]
    #             chunk[cons.STM_ELAPSED_TIME] = chunk[cons.STM_ELAPSED_TIME] + elapsed_time_since_last_update
    #             chunk[cons.STM_STRENGTH] = self.memory_decay(chunk[cons.STM_ELAPSED_TIME])
    #             # self.STM_w_strength[stm_chunk_idx] = (chunk[0], chunk[1], chunk[2], chunk[3])
    #         # Remove chunks with strength less than the threshold and mark them as memory loss
    #         if self.STM[stm_chunk_idx][cons.STM_STRENGTH] < self._MEMORY_LOSS_THRESHOLD:
    #             self.STM[stm_chunk_idx][cons.CONTENT] = self._MEMORY_LOSS_MARK
    #
    #     return

    def extract_micro_gist_and_update_stm(
            self,
            raw_sentence: str,  # The text to be processed, the content just read
            spatial_info: str,  # The spatial information, e.g., the current reading position
            sentence_start_word_index: int,  # The start word index of the sentence in the text
            elapsed_time_since_last_update: float,  # The time elapsed since the last reading update
            reading_strategy: str = const.READ_STRATEGIES["normal"],
            activated_schemas: List[str] = None,  # New parameter for schemas
            main_schemas: List[str] = None  # New parameter for main schemas
    ):
        """
        Extract the micro-gist from the text and update the STM.

        :param raw_sentence: The text to be processed, the content just read
        :param spatial_info: The spatial information, e.g., the current reading position
        :param sentence_start_word_index: The start word index of the sentence in the text
        :param elapsed_time_since_last_update: The time elapsed since the last reading update
        :param reading_strategy: The reading strategy, e.g., normal, skimming, scanning, etc.
        :return: None (updated STM)
        """

        # Assume self._activated_schemas and self._main_schemas are updated
        activated_schemas = activated_schemas
        main_schemas = main_schemas

        # Adjust the prompt based on activated schemas
        schemas_text = ""
        if activated_schemas:
            schemas_text += f"You have the following schemas activated by the current sentence:\n{', '.join(activated_schemas)}\n\n"
        if main_schemas:
            schemas_text += f"Additionally, consider the main schemas identified so far:\n{', '.join(main_schemas)}\n\n"

        if not schemas_text:
            schemas_text = "No specific schemas are activated. Interpret the sentence based on its content.\n\n"

        prompt = (
            f"Read the following sentence and create a memory chunk that captures its main idea:\n"
            f"Sentence: '{raw_sentence}'\n\n"
            f"{schemas_text}"
            f"Instructions:\n"
            f"- Use these schemas to interpret and summarize the sentence.\n"
            f"- Focus on the core message or primary action relevant to the schemas.\n"
            f"- Keep the chunk concise, using simple language.\n"
            f"- Do not add any information not present in the sentence.\n"
            f"- Imagine you're storing this chunk in your short-term memory to recall it later."
        )

        micro_gist_content = ""
        # Run LLM API for the micro-gist extraction with multiple attempts
        for attempt in range(self._max_num_requests):
            try:
                print(f"Attempt {attempt + 1}/{self._max_num_requests}: Generating micro-gist for STM.")

                raw_response = self._get_response(
                    role="You are a reader processing information into short-term memory chunks.",
                    prompt=prompt
                )

                print(f"Raw response for STM: {raw_response}\n\n")

                if raw_response:
                    # Process the raw_response
                    micro_gist_content = raw_response.strip()
                    if isinstance(micro_gist_content, str) and micro_gist_content != "":
                        break  # Successful response obtained
                    else:
                        print("Invalid response format. Retrying...")
                else:
                    print("Empty response received. Retrying...")

            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                time.sleep(self._retry_delay)

        else:
            # All attempts failed
            print("Failed to generate micro-gist after maximum attempts.")
            micro_gist_content = self._MEMORY_LOSS_MARK  # e.g., 'xxxxx'

        # Create the STM chunk with the current sentence's schemas
        stm_chunk = {
            const.CONTENT: micro_gist_content,
            const.SENTENCE_ID: spatial_info,
            # cons.STM_STRENGTH: self._INIT_STRENGTH,   # We disable them for now
            # cons.STM_ELAPSED_TIME: 0,
            const.ACTIVATED_SCHEMAS: activated_schemas,  # Store current sentence's schemas
            const.FORGOTTEN_FLAG: False  # Flag to indicate if content is forgotten
        }

        # Append the new chunk to STM
        self.STM.append(stm_chunk)

        # Update the STM
        self._update_stm()

    def _update_stm(self):
        """
        Update the STM by handling forgetting when the capacity is exceeded.
        """
        # Check if the STM exceeds capacity
        if len(self.STM) > self._STM_capacity:
            # Calculate how many chunks exceed the capacity
            excess_chunks = len(self.STM) - self._STM_capacity
            # Iterate over the oldest chunks and mark them as forgotten
            for i in range(excess_chunks):
                stm_chunk = self.STM[i]
                if not stm_chunk['forgotten']:
                    stm_chunk[const.CONTENT] = self._MEMORY_LOSS_MARK  # e.g., 'xxxxx'
                    stm_chunk['forgotten'] = True
        # No need to remove any chunks from the STM

    def _get_response(self, role, prompt):
        response = self._client.chat.completions.create(
            model=self._gpt_model,
            messages=[
                {
                    "role": "system",
                    "content": f""" {role} """
                },
                {
                    "role": "user",
                    "content": f"""{prompt} """
                }
            ],
            temperature=.25,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content

    @staticmethod
    def _memory_decay(delta_time: float) -> float:
        """
        Decay the memory strength over time using delta_time ^ (-0.5).
        Reference: Paper Heads-Up Multitasker
        :param delta_time: The time elapsed since the last reading update
        :return: The decayed memory strength
        """
        return float(np.clip(delta_time ** (-0.5), 0.0, 1.0))
