import json
import math
import httpx
import os
import re
from typing import List

from openai import OpenAI
import openai
import time
import yaml
import numpy as np
import spacy
from collections import OrderedDict

import step5.utils.constants as const


class LLMLongTermMemory:
    def __init__(
            self,
            config: str = '/home/baiy4/reading-model/step5/config.yaml',
    ):
        """
        This is the long-term memory (LTM) hosted by the LLM and updated during trials.

        Key Features:
        - Hosts macro-operated propositional notations (memory gists) generated from the STM and raw inputs.
        - Updates LTM using new gisted information.
        - Acts as an observation source for the LLM agent to make decisions.
        Reference:
        - Paper: Toward a model of text comprehension and production, 1978, link: https://www.cl.cam.ac.uk/teaching/1516/R216/Towards.pdf
        - My Rationality Architecture Diagram link (Version 15 July): https://www.figma.com/board/T9YEXqz2NukOfcLLmHUP4L/18-June-Architecture?node-id=0-1&t=s9tskcJ0c2c1IAWE-0

        OpenAI API:
        - Openai api documentation: https://platform.openai.com/docs/quickstart?context=python
        - api service: https://platform.openai.com/api-keys
        - tutorial: https://github.com/analyticswithadam/Python/blob/main/OpenAI_API_in_Python.ipynb

        Function: Host configured background knowledge and update using new gisted information. Specifically:
        - Macro-operator (deal with information stored in the STM):
            1. Delete non-relevant/non-critical details
            2. Generalize the retained information in STM, summarize them to gists
            3. Construct gists when there is some missing information in the STM/have some gaps
        - LTM updating: with the given user profile (especially the configured background knowledge), update the
            LTM with macro-gists from the macro-operator.
        - Serve as an observation source for the LLM agent to make decisions, especially determining the schema --
            relevance, importance, and critical-level.
        """

        print(f"{const.LV_TWO_DASHES}LLM LTM -- Initialize the LLM LTM module.")

        # Read the configuration file
        with open(config, 'r') as stream:
            self._config = yaml.load(stream, Loader=yaml.FullLoader)
        
        self.use_aalto_openai_api = self._config['llm']['use_aalto_openai_api']
        
        if self.use_aalto_openai_api:
            os.environ['AALTO_OPENAI_API_KEY'] = self._config['llm']['AALTO_OPENAI_API_KEY']

            assert (
                "AALTO_OPENAI_API_KEY" in os.environ and os.environ.get("AALTO_OPENAI_API_KEY") != ""
            ), "you must set the `AALTO_OPENAI_API_KEY` environment variable."
            self._client = OpenAI(
                base_url="https://aalto-openai-apigw.azure-api.net",
                api_key=False, # API key not used, and rather set below
                default_headers = {
                    "Ocp-Apim-Subscription-Key": os.environ.get("AALTO_OPENAI_API_KEY"),
                },
                http_client=httpx.Client(
                    event_hooks={"request": [update_base_url] }
                ),
            )
        else:
            openai_api_key = self._config['llm']['API_key']
            # LLM related variables
            self._client = OpenAI(api_key=openai_api_key)
        
        
        # Specific configurations
        self._gpt_model = self._config['llm']['model']
        self._refresh_interval = self._config['llm']['refresh_interval']
        self._max_num_requests = self._config['llm']['max_num_requests']
        self._retry_delay = self._config['llm']['retry_delay']

        self._role = None
        self._macro_operator_prompt = None      # The prompt for the macro-operator
        self._integrate_gists_prompt = None     # The prompt for integrating stacked gists
        self._response = None

        # Task related variables
        self._LTM = None    # The long-term memory, apart from the LLM's base knowledge regulated by the user profile, has the re-integrated gists, similar gists are combined
        self._STM = None    # The short-term memory
        self._current_single_gist = (None, None)  # The single gist generated from the STM and the Macro-operator
        self.gists = None   # The gisted information in the LTM, contain both the content (phonological information) and the spatial information (visuospatial information), stacked from separate gists
        self.activated_schemas = None  # The activated schemas based on the STM chunk
        self.all_activated_schemas = None  # All activated schemas from the STM chunks
        self.main_schemas = None  # The main schemas or themes extracted from the content
        self._schema_frequency = None  # The frequency of each schema in the activated schemas
        self._macrostructure_in_LTM = None  # The macrostructure in the LTM

        # Initialize the user profile and task-related variables
        self._user_profile = None
        self._task_specification = None
        self._question = None

    def reset(
            self,
            task_specification: str = "Please summarize what you have read",
            user_profile: dict = None,
            question: str = "What is the main idea of the text?",
    ):
        """
        Reset the LLM LTM module for each new trial and new participant.

        :param question: The question for the task.
        :param task_specification: The task specification.
        :param user_profile: The user profile dictionary.
        :return: None
        """
        self._user_profile = user_profile
        self._task_specification = task_specification
        self._question = question
        self._current_single_gist = (None, None)    # (content, spatial information)
        self.gists = []
        self._LTM = []
        self._STM = OrderedDict()
        self.activated_schemas = []
        self.all_activated_schemas = []
        self.main_schemas = []
        self._schema_frequency = {}
        self._macrostructure_in_LTM = MacrostructureNode(content='[Root]', schemas=self.main_schemas.copy())
        # TODO debug delete later
        print(f"Initialization\n"
              f"The macrostructure content is: {self._macrostructure_in_LTM.content}\n"
              f"The main schemas are: {self.main_schemas}"
              f"The chirldren are: {self._macrostructure_in_LTM.children}")

    def activate_schemas(self, raw_sentence: str) -> list:
        """
        Dynamically activate schemas based on the content of the raw sentence using the LLM.
        Update the main schemas based on activated schemas.

        :param raw_sentence: Raw sentence just read.
        :return: A list of activated schemas.
        """
        for attempt in range(self._max_num_requests):
            try:
                schema_raw_response = self._get_response(
                    role="You are a cognitive agent extracting themes from content.",
                    prompt=(
                        f"Read the following content and identify the main schemas or themes it relates to.\n"
                        f"Content: '{raw_sentence}'\n\n"
                        f"Instructions:\n"
                        f"- List up to 3 schemas or themes that are relevant to the content.\n"  # Limit the number of schemas
                        f"- Focus on general concepts or categories.\n"
                        f"- Each schema should be no more than 3 words.\n"  # Limit words per schema
                        f"- Provide the schemas as a comma-separated list.\n"
                        f"- **Do not** include any additional explanation or information."
                    )
                )

                if schema_raw_response:
                    # Process the response
                    activated_schemas = [schema.strip() for schema in schema_raw_response.strip().split(',')]
                    # Remove any empty strings
                    activated_schemas = [schema for schema in activated_schemas if schema]

                    # Update the activated schemas for the current sentence
                    self.activated_schemas = activated_schemas

                    # Append to the global list of all activated schemas
                    self.all_activated_schemas.extend(activated_schemas)

                    # Update schema frequencies
                    for schema in activated_schemas:
                        self._schema_frequency[schema] = self._schema_frequency.get(schema, 0) + 1

                    # Update the main schemas
                    self._update_main_schemas()

                    return activated_schemas

            except Exception as e:
                print(f"Activate Schemas -- Attempt {attempt} failed with error: {e}")
                time.sleep(self._retry_delay)

        print("Failed to activate schemas after maximum attempts")
        return []

    def _update_main_schemas(self):
        """
        Update the main schemas based on all activated schemas using the LLM.
        """
        # Combine all activated schemas into a single string
        all_schemas_text = ', '.join(self.all_activated_schemas)

        for attempt in range(self._max_num_requests):
            try:
                main_schemas_raw_response = self._get_response(
                    role="You are a cognitive agent identifying main themes from a list of schemas.",
                    prompt=(
                        f"Given the following list of schemas activated so far:\n"
                        f"{all_schemas_text}\n\n"
                        f"Instructions:\n"
                        f"- Identify up to 5 main schemas or themes that represent the overall content.\n"
                        f"- Focus on the most frequently occurring or central concepts.\n"
                        f"- Each schema should be no more than 3 words.\n"
                        f"- Provide the main schemas as a comma-separated list.\n"
                        f"- **Do not** include any additional explanation or information."
                    )
                )

                if main_schemas_raw_response:
                    # Process the response
                    main_schemas = [schema.strip() for schema in main_schemas_raw_response.strip().split(',')]
                    # Remove any empty strings
                    main_schemas = [schema for schema in main_schemas if schema]

                    # Update the main schemas
                    self.main_schemas = main_schemas

                    # # Debug print
                    # print(f"The main schemas are now: {self.main_schemas}\n")

                    return

            except Exception as e:
                print(f"Update main schemas -- Attempt {attempt} failed with error: {e}")
                time.sleep(self._retry_delay)

        print("Failed to update main schemas after maximum attempts")
        self.main_schemas = []  # Reset main schemas if failed

    def _determine_relevance(self, stm_chunk, main_schemas):
        """
        Determine the relevance of an STM chunk based on the main schemas.
        """
        activated_schemas = stm_chunk['activated_schemas']
        prompt = (
            f"Assess the relevance of the following STM chunk based on the main schemas.\n"
            f"STM Chunk Content: '{stm_chunk[const.CONTENT]}'\n"
            f"Activated Schemas: {', '.join(activated_schemas)}\n"
            f"Main Schemas: {', '.join(main_schemas)}\n\n"
            f"Instructions:\n"
            f"- Compare the activated schemas with the main schemas.\n"
            f"- Determine if the chunk is highly relevant or not.\n"
            f"- Respond with {const.HIGH_RELEVANCE} or {const.LOW_RELEVANCE} **only**.\n"
            f"- Do not provide any explanations or additional text.\n"
            f"- Your response should be exactly one of these two words, and nothing else."
        )
        for attempt in range(self._max_num_requests):
            try:
                raw_response = self._get_response(
                    role=f"You are an assistant evaluating the relevance of information. When given a task, you should only output {const.HIGH_RELEVANCE} or {const.LOW_RELEVANCE}, and nothing else. Do not provide any explanations or additional text.",
                    prompt=prompt
                )
                if raw_response:
                    relevance = raw_response.strip()
                    if relevance in [const.HIGH_RELEVANCE, const.LOW_RELEVANCE]:
                        return relevance
                    else:
                        print(f"Invalid raw response '{relevance}'. Retrying...")
                else:
                    print("Empty response. Retrying...")
            except Exception as e:
                print(f"Determine relevance -- Attempt {attempt + 1} failed with error: {e}")
                time.sleep(self._retry_delay)
        else:
            print("Failed to determine relevance after maximum attempts.")
            return 'Low Relevance'  # Default to low relevance if all attempts fail

    def _update_macrostructure(self, relevant_stm_chunks):
        """
        Update the macrostructure (gist) in LTM with new relevant information.

        :param relevant_stm_chunks: List of STM chunks deemed highly relevant.
        """
        # Collect current macrostructure content, do not get the root node, or there will be a duplicate root node when parsing
        current_gist = self._collect_macrostructure_content(self._macrostructure_in_LTM)

        # Contents of relevant STM chunks
        new_information = [chunk[const.CONTENT] for chunk in relevant_stm_chunks]

        # Adjust the prompt for the LLM
        prompt = (
                f"Update the existing hierarchical gist with the following new information.\n"
                f"Existing Gist:\n{current_gist}\n\n"
                f"New Information:\n" + "\n".join(f"- {info}" for info in new_information) + "\n\n"
                f"Instructions:\n"
                f"- Integrate the new information into the existing gist hierarchically.\n"
                f"- Update the root node to reflect the overarching theme or general idea based on the new information.\n"
                f"- Organize the main themes at the top level, with supporting details beneath.\n"
                f"- Use an indented outline format to represent the hierarchy.\n"
                f"- Each line should start with '-' followed by the content.\n"
                f"- Indent child nodes with 2 spaces per level.\n"
                f"- Ensure the updated gist remains coherent and focused on the main themes.\n"
                f"- Use clear and concise language.\n"
                f"- Do **not** include irrelevant details."
                f"- Do **not** improvise or add anything beyond the input new information.\n"
                f"- Do **not** include a '[Root]' node or duplicate existing nodes in your output.\n"
        )

        for attempt in range(self._max_num_requests):
            try:
                raw_response = self._get_response(
                    role="You are summarizing and updating a hierarchical gist with new information.",
                    prompt=prompt
                )
                if raw_response:

                    # Process the raw_response to create/update the macrostructure
                    updated_macrostructure = self._parse_macrostructure_response(raw_response.strip())
                    if updated_macrostructure:
                        # Update the macrostructure_in_LTM
                        self._macrostructure_in_LTM = updated_macrostructure
                        self.gists = self._collect_macrostructure_content(self._macrostructure_in_LTM)

                        self.gists = self._collect_macrostructure_content(self._macrostructure_in_LTM)

                        return
                    else:
                        print("Failed to parse macrostructure. Retrying...")
                else:
                    print("Empty response. Retrying...")
            except Exception as e:
                print(f"Update macrostructure -- Attempt {attempt + 1} failed with error: {e}")
                time.sleep(self._retry_delay)
        else:
            print("Failed to update macrostructure after maximum attempts.")

    def _summarize_root_node_as_theme(self, gists):
        """
        Summarize the new information along with the existing gist to update the root node of the macrostructure.

        :param gists: List of new information chunks.
        :return: A summary string representing the overarching theme.
        """
        summary_prompt_template = (
            f"Please summarize the following points into a single overarching theme or main idea:\n"
            f"{{content}}\n"
            f"Instructions:\n"
            f"- Focus on the main idea that ties these points together.\n"
            f"- Include both the existing gist and the new information.\n"
            f"- Ensure the summary is concise and within 10 words.\n"
            f"- Use clear and concise language."
        )

        # Attempt to generate a summary multiple times
        for attempt in range(self._max_num_requests):
            try:
                summary_prompt = summary_prompt_template.format(content=gists)
                raw_summary = self._get_response(
                    role="You are summarizing information.",
                    prompt=summary_prompt
                )
                if raw_summary:
                    return raw_summary.strip()
                else:
                    print(f"Empty response in attempt {attempt + 1}. Retrying...")
            except Exception as e:
                print(f"Summarise root node -- Attempt {attempt + 1} failed with error: {e}")
                time.sleep(self._retry_delay)

        # Fallback to a default summary if all attempts fail
        print("Failed to generate a summary after maximum attempts.")
        return "[Root]"

    def _collect_macrostructure_content(self, node, level=0):

        indent = '  ' * level
        content = f"{indent}- {node.content}\n"
        for child in node.children:
            content += self._collect_macrostructure_content(child, level + 1)
        return content

    def _parse_macrostructure_response(self, raw_response_text):
        """
        Parse the LLM's response into a MacrostructureNode tree.
        """
        # print(f"Parsing macrostructure response: \n{raw_response_text}\n")        # TODO debug delete later

        lines = raw_response_text.split('\n')
        root_node = MacrostructureNode(content='[root]')
        stack = [(root_node, -1)]  # Stack of (node, level)

        for line in lines:
            stripped_line = line.lstrip()
            if not stripped_line.startswith('-'):
                continue  # Skip invalid lines
            level = (len(line) - len(stripped_line)) // 2
            content = stripped_line.lstrip('- ').strip()
            new_node = MacrostructureNode(content=content)
            # Find parent node
            while stack and stack[-1][1] >= level:
                stack.pop()
            if stack:
                parent_node = stack[-1][0]
                parent_node.add_child(new_node)
            else:
                # Should not happen if the response is well-formed
                print("Malformed response structure.")
                return None
            stack.append((new_node, level))

        return root_node

    def generate_macrostructure(
            self,
            stm: OrderedDict,
            question: str = "What is the main idea of the text?",
    ):
        """
        Generate or update the macrostructure (gist) in LTM based on the STM.

        :param stm: The short-term memory. Contains both content and spatial information.
        :param question: The question for the task (optional).
        """
        # Update the STM
        self._STM = stm

        # Update the question (if we use it in prompts)
        self._question = question

        # Determine relevance of each STM chunk
        relevant_stm_chunks = []
        for chunk in stm.values():  # Use stm.values() to iterate over chunks
            if chunk['content'] == const.MEMORY_LOSS_MARK:
                continue  # Skip forgotten chunks
            relevance = self._determine_relevance(chunk, self.main_schemas)
            if relevance == const.HIGH_RELEVANCE:
                relevant_stm_chunks.append(chunk)

        # Update the macrostructure with relevant STM chunks
        if relevant_stm_chunks:
            self._update_macrostructure(relevant_stm_chunks)

    def _config_role(self):
        """
        Configure the role for the LLM API.

        :return: The role for the LLM API.
        """
        proficiency = self._user_profile.get('proficiency', 'average')
        interest_level = self._user_profile.get('interest level', 'medium')
        background_knowledge = self._user_profile.get('background knowledge', 'general')

        role_proficiency = {
            'good': "You are a normal (or native) English speaker with average reading abilities.",
            'poor': "You are not a native English speaker with below average reading abilities."
        }.get(proficiency, "You are an English speaker with average reading abilities.")

        role_interest = {
            'high': "You are interested in the given topic.",
            'low': "You are not interested in the given topic."
        }.get(interest_level, "You have a moderate interest in the given topic.")

        role_background = {
            'specific': "You have specific background knowledge in the given topic.",
            'empty': "You do not have any background knowledge in the given topic."
        }.get(background_knowledge, "You have general background knowledge in the given topic.")

        return f"{role_proficiency} {role_interest} {role_background}"

    def _get_response(self, role, prompt):
        
        if self.use_aalto_openai_api:
            messages=[
                {
                    "role": "system",
                    "content": f""" {role} """
                },
                {
                    "role": "user",
                    "content": f"""{prompt} """
                }
            ]
            completion = self._client.chat.completions.create(
                model="no_effect", # the model variable must be set, but has no effect, model selection done with URL
                messages=messages,
            )
        else:
            completion = self._client.chat.completions.create(
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
        return completion.choices[0].message.content

    def finalize_gists(self):
        """
        Finalize the gists in the LTM.

        This method is called at the end of each trial to finalize the gists in the LTM.
        """
        # Update the LTM with the finalized gists
        updated_root_node = self._summarize_root_node_as_theme(self.gists)
        # Update the root node of the macrostructure
        self._macrostructure_in_LTM.content = updated_root_node
        # Print the updated LTM
        self.gists = self._collect_macrostructure_content(self._macrostructure_in_LTM)
        print(f"The updated LTM is: \n{self.gists}\n")


class MacrostructureNode:
    def __init__(self, content='', schemas=None, children=None):
        self.content = content
        self.schemas = schemas if schemas is not None else []
        self.children = children if children is not None else []

    def add_child(self, node):
        self.children.append(node)


class LLMShortTermMemory:

    def __init__(   # TODO add two things to each sentences: 1. the latest visiting index, 2. the number of visitings
            self,
            config: str = '/home/baiy4/reading-model/step5/config.yaml',
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

        self.use_aalto_openai_api = self._config['llm']['use_aalto_openai_api']
        
        if self.use_aalto_openai_api:
            os.environ['AALTO_OPENAI_API_KEY'] = self._config['llm']['AALTO_OPENAI_API_KEY']

            assert (
                "AALTO_OPENAI_API_KEY" in os.environ and os.environ.get("AALTO_OPENAI_API_KEY") != ""
            ), "you must set the `AALTO_OPENAI_API_KEY` environment variable."
            self._client = OpenAI(
                base_url="https://aalto-openai-apigw.azure-api.net",
                api_key=False, # API key not used, and rather set below
                default_headers = {
                    "Ocp-Apim-Subscription-Key": os.environ.get("AALTO_OPENAI_API_KEY"),
                },
                http_client=httpx.Client(
                    event_hooks={"request": [update_base_url] }
                ),
            )
        else:
            openai_api_key = self._config['llm']['API_key']
            # LLM related variables
            self._client = OpenAI(api_key=openai_api_key)
        
        
        # Specific configurations
        self._gpt_model = self._config['llm']['model']
        self._refresh_interval = self._config['llm']['refresh_interval']
        self._max_num_requests = self._config['llm']['max_num_requests']
        self._retry_delay = self._config['llm']['retry_delay']

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

        # self.STM: list = []
        self.STM = OrderedDict()

        self._reading_strategy: str = ""

    def _degrade_text_comprehension_by_reading_strategies(self, text: str, ):
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

    def extract_microstructure_and_update_stm(
            self,
            raw_sentence: str,
            spatial_info: str,
            activated_schemas: list = None,
            main_schemas: list = None,
            reading_strategy: str = const.READ_STRATEGIES["normal"],
    ):
        """
        Extract the micro-gist from the text and update the STM.
        """
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
            f"- **Focus on one key idea or concept.**\n"
            f"- **Keep the chunk to a maximum of 10 words.**\n"
            f"- Use simple language.\n"
            f"- Do not add any information not present in the sentence.\n"
            f"- Imagine you're storing this chunk in your short-term memory to recall it later."
        )

        micro_gist_content = ""
        # Run LLM API for the micro-gist extraction with multiple attempts
        for attempt in range(self._max_num_requests):
            try:
                print(f"(STM Micro-gists) Attempt {attempt + 1}/{self._max_num_requests}: Generating micro-gist for STM.")

                raw_response = self._get_response(
                    role="You are a reader processing information into short-term memory chunks.",
                    prompt=prompt
                )

                if raw_response:
                    micro_gist_content = raw_response.strip()
                    if isinstance(micro_gist_content, str) and micro_gist_content != "":
                        print(f"STM generation succeeded. The micro_gist_content is: {micro_gist_content}\n")
                        break  # Successful response obtained
                    else:
                        print("Invalid response format. Retrying...")
                else:
                    print("Empty response received. Retrying...")

            except Exception as e:
                print(f"Extract microstructure and update stm -- Attempt {attempt + 1} failed with error: {e}")
                time.sleep(self._retry_delay)

        else:
            # All attempts failed
            print("Failed to generate micro-gist after maximum attempts.")
            micro_gist_content = self._MEMORY_LOSS_MARK  # e.g., 'xxxxx'

        # Check if the chunk already exists in the STM
        if spatial_info in self.STM:
            # Update the existing chunk
            stm_chunk = self.STM[spatial_info]
            stm_chunk[const.VISIT_COUNT] += 1
            stm_chunk[const.FORGOTTEN_FLAG] = False
            # Update the content and appraisal
            stm_chunk[const.CONTENT] = micro_gist_content
            stm_chunk[const.APPRAISAL] = self._calculate_appraisal(reading_strategy, stm_chunk[const.VISIT_COUNT], stm_chunk[const.APPRAISAL])
            print(f"(STM) Reinforced and updated existing STM chunk for sentence {spatial_info}.")
        else:
            # Create a new chunk
            stm_chunk = {
                const.CONTENT: micro_gist_content,
                const.SENTENCE_ID: spatial_info,
                const.ACTIVATED_SCHEMAS: activated_schemas,
                const.FORGOTTEN_FLAG: False,
                const.VISIT_COUNT: const.ONE,  # Initialize visit count
                const.APPRAISAL: self._calculate_appraisal(reading_strategy, const.ONE, const.ZERO)
            }
            # Add the new chunk to STM -- guarantee the order of the STM by using OrderedDict
            self.STM[spatial_info] = stm_chunk
            print(f"(STM) Added new STM chunk for sentence {spatial_info}.")

        # Move the chunk to the end to mark it as most recently used
        self.STM.move_to_end(spatial_info)

        # Update the STM
        self._update_stm()

    def _update_stm(self):      # TODO fix this later
        """
        Update the STM by handling forgetting when the capacity is exceeded.
        """
        if len(self.STM) > self._STM_capacity:
            # Calculate how many items need to be forgotten
            excess = len(self.STM) - self._STM_capacity
            # Get the keys (sentence indices) of the least recently used items
            lru_items = list(self.STM.keys())[:excess]
            for spatial_info in lru_items:
                stm_chunk = self.STM[spatial_info]
                # Skip the chunk if it has a very high appraisal score
                if stm_chunk[const.APPRAISAL] > const.MEMORY_RETAIN_APPRAISAL_LEVEL_THRESHOLD:
                    continue
                # Normally forgets the chunk if it is too early
                if not stm_chunk[const.FORGOTTEN_FLAG]:
                    stm_chunk[const.CONTENT] = self._MEMORY_LOSS_MARK  # e.g., 'xxxxx'
                    stm_chunk[const.FORGOTTEN_FLAG] = True
                    stm_chunk[const.APPRAISAL] = self._decay_appraisal(stm_chunk[const.APPRAISAL], stm_chunk[const.VISIT_COUNT])
                    # TODO could make this elapsed-time based if needed
                    print(f"Forgotten STM chunk for sentence {spatial_info}.")

    @staticmethod
    def _decay_appraisal(current_appraisal, visit_count):
        """
        Decay the appraisal score for a forgotten chunk.
        TODO this is a heuristic function, need to adjust later for being more solid

        :param current_appraisal: The current appraisal score of the chunk.
        :param visit_count: The number of times the chunk has been visited.
        :return: The decayed appraisal score.
        """
        decay_rate = 0.75  # Adjust this value between 0 and 1
        decayed_appraisal = current_appraisal * decay_rate
        return decayed_appraisal

    @staticmethod
    def _calculate_appraisal(current_sentence_reading_strategy, visit_count, init_appraisal) -> float:
        """
        Calculate the appraisal score for a given STM chunk using an exponential method.
        TODO here the model is still heuristic, need to revisit later for being more solid
            FLAG: solve when first version paper is finished

        :param current_sentence_reading_strategy: The current sentence's reading strategy.
        :param visit_count: The number of times the chunk has been visited.
        :return: A float between 0 and 1 representing the appraisal score.
        """
        # Base appraisal score based on reading strategy
        if current_sentence_reading_strategy == const.READ_STRATEGIES['skim']:
            base_appraisal = max(const.HEURISTIC_APPRAISAL_LEVELS['skim'], init_appraisal)
        elif current_sentence_reading_strategy == const.READ_STRATEGIES['normal']:
            base_appraisal = max(const.HEURISTIC_APPRAISAL_LEVELS['normal'], init_appraisal)
        elif current_sentence_reading_strategy == const.READ_STRATEGIES['careful']:
            base_appraisal = max(const.HEURISTIC_APPRAISAL_LEVELS['careful'], init_appraisal)
        else:
            raise ValueError(f"Invalid reading strategy: {current_sentence_reading_strategy}")

        # Exponential increase in appraisal score with each visit
        k = 1.5  # Adjust this value to control the rate of increase
        appraisal_increment = (1 - base_appraisal) * (1 - math.exp(-k * (visit_count - 1)))
        appraisal_score = base_appraisal + appraisal_increment

        # Ensure appraisal_score does not exceed 1
        appraisal_score = min(appraisal_score, 1.0)

        return appraisal_score

    def _get_response(self, role, prompt):
    
        if self.use_aalto_openai_api:
            messages=[
                {
                    "role": "system",
                    "content": f""" {role} """
                },
                {
                    "role": "user",
                    "content": f"""{prompt} """
                }
            ]
            completion = self._client.chat.completions.create(
                model="no_effect", # the model variable must be set, but has no effect, model selection done with URL
                messages=messages,
            )
        else:
            completion = self._client.chat.completions.create(
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
        return completion.choices[0].message.content

    @staticmethod
    def _memory_decay(delta_time: float) -> float:
        """
        Decay the memory strength over time using delta_time ^ (-0.5).
        Reference: Paper Heads-Up Multitasker
        :param delta_time: The time elapsed since the last reading updates
        :return: The decayed memory strength
        """
        return float(np.clip(delta_time ** (-0.5), 0.0, 1.0))


class LLMWorkingMemory:
    def __init__(
            self,
            config: str = '/home/baiy4/reading-model/step5/config.yaml',
    ):
        """
        Updated on 20 July
        The Supervisory Controller controls the high-level decision-makings of reading, including reasoning, planning, and comprehension.

        We start from simple action:
        - Whether to continue reading or not. Based on the current reading progress (no spatial info) and task (information search).

        Openai api documentation: https://platform.openai.com/docs/quickstart?context=python
        api service: https://platform.openai.com/api-keys
        tutorial: https://github.com/analyticswithadam/Python/blob/main/OpenAI_API_in_Python.ipynb
        """

        print(f"{const.LV_TWO_DASHES}LLM Text Comprehend -- Initialize the LLM Text Comprehend module.")

        # Read the configuration file
        with open(config, 'r') as stream:
            self._config = yaml.load(stream, Loader=yaml.FullLoader)
        
        self.use_aalto_openai_api = self._config['llm']['use_aalto_openai_api']
        
        if self.use_aalto_openai_api:
            os.environ['AALTO_OPENAI_API_KEY'] = self._config['llm']['AALTO_OPENAI_API_KEY']

            assert (
                "AALTO_OPENAI_API_KEY" in os.environ and os.environ.get("AALTO_OPENAI_API_KEY") != ""
            ), "you must set the `AALTO_OPENAI_API_KEY` environment variable."
            self._client = OpenAI(
                base_url="https://aalto-openai-apigw.azure-api.net",
                api_key=False, # API key not used, and rather set below
                default_headers = {
                    "Ocp-Apim-Subscription-Key": os.environ.get("AALTO_OPENAI_API_KEY"),
                },
                http_client=httpx.Client(
                    event_hooks={"request": [update_base_url] }
                ),
            )
        else:
            openai_api_key = self._config['llm']['API_key']
            # LLM related variables
            self._client = OpenAI(api_key=openai_api_key)
        
        
        # Specific configurations
        self._gpt_model = self._config['llm']['model']
        self._refresh_interval = self._config['llm']['refresh_interval']
        self._max_num_requests = self._config['llm']['max_num_requests']
        self._retry_delay = self._config['llm']['retry_delay']

        self._role = None
        self._prompt = None
        self._response = None

        # Task related variables
        self._task_spec = None
        self._stm = None
        self._ltm_gists = None

        # Action related variables
        # The action to take -- for the simple version, a binary decision: continue reading (True) or not (False)
        self.action_stop_reading = None

        # Initialize the user profile related variables
        self._user_profile = None

        # Initialize the question related variables
        self._question = None

    def reset(
            self,
            task_specification: str = "Please summarize what you have read",
            stm: list = None,
            ltm_gists: list = None,
            user_profile: dict = None,
            question: str = None,
    ):
        """
        Reset the Supervisory Controller.
        :return: None
        """
        self._task_spec: str = task_specification
        self._stm: list = stm
        self._ltm_gists: list = ltm_gists
        self._user_profile: dict = user_profile
        self._question: str = question

        self.action_stop_reading: bool = False

    def process(
            self,
            ltm: LLMLongTermMemory,
            stm: LLMShortTermMemory,
            read_info: dict,
            reading_states: dict,
            reading_strategy: str,

    ) -> (LLMLongTermMemory, LLMShortTermMemory, str, int):
        """
        Process the information -- in our case, we process sentence by sentence.
        Update the LTM, STM, regression decision, regression position, and reading strategy.
        :return:
        """

        # Step 1: activate the schemas in the LTM
        ltm.activate_schemas(raw_sentence=read_info[const.CONTENT])

        # step2: update the short-term memory
        stm.extract_microstructure_and_update_stm(
            raw_sentence=read_info[const.CONTENT],
            spatial_info=read_info[const.SENTENCE_ID],
            activated_schemas=ltm.activated_schemas,
            main_schemas=ltm.main_schemas,
            reading_strategy=reading_strategy,
        )

        # Step 3: update the long-term memory
        ltm.generate_macrostructure(stm=stm.STM)

        # Update the sentence id, but not here

        # Step 4: determine whether to regress or not
        raw_regression_decision_by_llm = self.decide_regression(
            stm=stm.STM,
            ltm_gists=ltm.gists,
            reading_states=reading_states,
        )

        # Step 5: decide the regressing position if necessary
        if type(raw_regression_decision_by_llm) == tuple:
            regression_decision_by_llm, _regression_sentence_id, regression_reason = raw_regression_decision_by_llm
        else:
            raise ValueError(f"Invalid regression decision: {raw_regression_decision_by_llm}, "
                             f"invalid regression desicion type: {type(raw_regression_decision_by_llm)}.")

        if const.REGRESS_DECISIONS["regress"] in regression_decision_by_llm:
            _regression_decision = const.REGRESS_DECISIONS["regress"]
            regression_target_position = _regression_sentence_id
        elif const.REGRESS_DECISIONS["continue_forward"] in regression_decision_by_llm:
            _regression_decision = const.REGRESS_DECISIONS["continue_forward"]
            regression_target_position = _regression_sentence_id
        else:
            raise ValueError(f"Invalid regression decision: {raw_regression_decision_by_llm}")

        return ltm, stm, _regression_decision, regression_target_position

    def calculate_appraisal_of_understanding_by_memory_context(
            self,
            ltm_gists: list = None,
            sampled_sentence_content: str = None,
    ) -> float:
        """
        Calculate the appraisal of understanding based on the memory context.
        :param ltm_gists: The gists in the LTM.
        :param sampled_sentence_content: The content of the sampled sentence.
        :return: The appraisal score as a float.
        """

        # Run GPT API for the macro-operator
        for attempt in range(self._max_num_requests):
            try:
                prompt = (
                    f"Based on the following Long-term Memory (LTM) content:\n\n{ltm_gists}\n\n"
                    f"Evaluate how understandable and coherent the following sentence is considering the LTM context. "
                    f"Provide a score between 0 and 1 (inclusive), where 0 means not understandable at all, and 1 means completely understandable and coherent. "
                    f"Respond with a JSON object containing the score, like '{{\"score\": 0.75}}', and do **not** include any additional text.\n\nSentence:\n{sampled_sentence_content}\n\nScore:"
                )

                raw_response = self._get_response(
                    role="Please evaluate the appraisal level/understanding of the given sentence regarding the LTM content.",
                    prompt=prompt,
                )

                if raw_response:
                    # Parsing logic
                    try:
                        response_json = json.loads(raw_response)
                        if isinstance(response_json, dict) and "score" in response_json:
                            score = float(response_json["score"])
                        else:
                            # If response_json is a float/int or not a dict
                            score = float(response_json)
                        return score
                    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                        # If JSON parsing fails, try to extract the number using regex
                        print(f"Calculate appraisal level -- Failed to parse JSON response: {raw_response}, try to extract the number using regex")
                        match = re.search(r"[-+]?\d*\.\d+|\d+", raw_response)
                        if match:
                            score = float(match.group())
                            return score
                        else:
                            # If parsing fails, default to 0
                            print(f"Invalid response format: {raw_response}. Defaulting score to 0.")
                            return 0.0

            except Exception as e:
                print(f"Calculate appraisal level -- Attempt {attempt} failed with error: {e}")
                time.sleep(self._retry_delay)

        print("Failed to generate a valid single memory retrieval after maximum attempts")
        return 0.0  # Return 0.0 instead of a string for consistency

    def retrieve_memory(
            self,
            question_type: str = None,  # Either MCQ or FreeRecall
            question: str = None,
            options: dict = None,
            ltm_gists: list = None,
    ) -> str:
        """
        Information retrieval from the STM and LTM to complete given task, e.g., answering questions.
        :param question_type: The type of the question, either MCQ or FreeRecall.
        :param question: A question to test the comprehension according to the agent's LTM and STM.
        :param options: The options for the multiple-choice question.
        :param ltm_gists: The gists in the LTM.
        :return: Answer to the question.
        """

        # Get the prompt based on the question type
        if question_type == const.QUESTION_TYPES['MCQ']:
            prompt = (
            f"Based **EXCLUSIVELY** on the following LTM content: '{ltm_gists}', "
            f"answer the question: '{question}' "
            f"from the options: {options}. "
            f"Respond with the letter of the correct answer ('A', 'B', 'C', or 'D') only. "
            f"If the correct answer is not **explicitly stated** in the LTM content, or if you are unsure, respond with 'E'. "
            f"Do **NOT** provide any explanations or additional text. "
            f"Do **NOT** use any outside knowledge or make any inferences. "
            f"Your answer should be based **solely** on the information provided above."
        )

        elif question_type == const.QUESTION_TYPES['FRS']:
            prompt = (
                f"Based **ONLY** on the Long-term Memory (LTM) content: '{ltm_gists}', "
                f"please provide a narrative summary in the form of a continuous paragraph. "
                f"Do not use bullet points, lists, or any other formatting. "
                f"Do **NOT** add any additional information, interpretations, or inferences. "
                f"Ensure that every detail in your summary directly corresponds to information explicitly stated in the LTM content.\n"
            )

        else:
            raise ValueError(f"Invalid question type: {question_type}")

        # Run GPT API for the macro-operator
        for attempt in range(self._max_num_requests):
            try:

                raw_response = self._get_response(
                    role="Please answer the question based strictly on the provided Long-term Memory (LTM) content.",
                    prompt=prompt,
                )

                if raw_response:
                    # Loop to validate and parse the response
                    try:
                        # Separate the content and spatial information from the response: split by the delimiter
                        answer = raw_response.strip()

                        # Get the correct answer format to the question
                        if question_type == const.QUESTION_TYPES['MCQ']:
                            if answer.upper() in ['A', 'B', 'C', 'D', 'E']:
                                return answer.upper()
                            else:
                                raise ValueError(f"Invalid answer format: {answer}, should be 'A', 'B', 'C', 'D', or 'E'")
                        elif question_type == const.QUESTION_TYPES['FRS']:
                            if type(answer) == str and answer != "":
                                return answer
                            else:
                                raise ValueError(f"Invalid answer format: {answer}")
                        else:
                            raise ValueError(f"Invalid question type: {question_type}")
                        # return answer

                    except (SyntaxError, ValueError, AssertionError) as e:
                        print(f"Error parsing response for the LLM memory retrieval: {e}\n"
                              f"The current incorrect-format memory retrieval is: {raw_response}")
                        continue

            except Exception as e:
                print(f"Retrieve memory -- Attempt {attempt} failed with error: {e}")
                time.sleep(self._retry_delay)

        print("Failed to generate a valid single memory retrieval after maximum attempts")
        return "Failed to answer. Please try again."
    
    def _get_response(self, role, prompt):
        
        if self.use_aalto_openai_api:
            messages=[
                {
                    "role": "system",
                    "content": f""" {role} """
                },
                {
                    "role": "user",
                    "content": f"""{prompt} """
                }
            ]
            completion = self._client.chat.completions.create(
                model="no_effect", # the model variable must be set, but has no effect, model selection done with URL
                messages=messages,
            )
        else:
            completion = self._client.chat.completions.create(
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
        return completion.choices[0].message.content

def determine_time_constraint_level(remaining_time, estimated_time_to_finish):
    """
    Determine the time constraint level.
    TODO: a heuristic function here, may need to revisit later for being more solid
    The LLM agent's understanding/time awareness should be consistent with this function.
    :param remaining_time: Remaining time in seconds.
    :param estimated_time_to_finish: Estimated time to finish in seconds.
    :return: A string indicating the time constraint level ('very_limited_time', 'sufficient_time', 'ample_time').
    """
    if remaining_time < estimated_time_to_finish:
        return const.TIME_AWARENESS_LEVELS['very_limited_time']
    elif remaining_time >= estimated_time_to_finish * 1.5:
        return const.TIME_AWARENESS_LEVELS['ample_time']
    else:
        return const.TIME_AWARENESS_LEVELS['sufficient_time']
    
def update_base_url(request: httpx.Request) -> None:
        if request.url.path == "/chat/completions":
            request.url = request.url.copy_with(path="/v1/chat") # chat/gpt4-8k /chat
            # request.url = request.url.copy_with(path="/v1/openai/gpt4-turbo/chat/completions")


if __name__ == '__main__':

    # Pre-defined materials
    task_specification = (
        "summarise the main idea of the text and be ready for comprehension tests and recall tests."
    )
    user_profile = {
        "proficiency": "good",
        "interest level": "high",
        "background knowledge": "specific",
    }
    raw_sentences = [
        ("Hi everyone! ", ""),
        ("Lately, I’ve been making a concerted effort to incorporate more sustainable practices into my daily routine. ", ""),
        ("One of the first changes I implemented was reducing my reliance on single-use plastics. ", ""),
        ("Switching to reusable bags, containers, and even produce bags has significantly decreased the amount of waste I generate. ", ""),
        ("I’ve also been focusing on minimizing food waste, which has had the dual benefit of helping the environment and saving money. ", ""),
        ("Meal planning has been a game-changer. I only buy what I need, which means fewer leftovers going to waste. ", ""),
        ("Additionally, I set up a small compost bin in my backyard. ", ""),
        ("It’s gratifying to know that my food scraps are being repurposed into something beneficial for the garden rather than ending up in a landfill. ", ""),
        ("I’m curious to hear how others are working towards a more sustainable lifestyle. ", ""),
        ("What small changes have you made that have had a big impact? ", ""),
        ("Let’s exchange ideas and support each other in making our community a little greener. ", ""),
        ("Remember, even small actions can contribute to a larger positive change!", ""),
    ]

    def calculate_num_words(sentences: list=None, single_sentence: str=None):
        num_words = 0

        if sentences is not None:
            for sentence in sentences:
                num_words += len(sentence[0].split())
            return num_words
        elif single_sentence is not None:
            return len(single_sentence.split())

    def update_reading_states(
            predefined_time_constraint_in_seconds: int = None,
            elapsed_time_in_seconds: int = None,
            total_num_words: int = None,
            num_words_read: int = None,
    ):
        """
        Update the reading states based on the current reading progress.
        :return: The updated reading states.
        """
        # Calculate the remaining time in seconds
        remaining_time_in_seconds = predefined_time_constraint_in_seconds - elapsed_time_in_seconds
        # Calculate the remaining number of words
        num_words_remaining = total_num_words - num_words_read

        return {
            const.READING_STATES['predefined_time_constraint_in_seconds']: predefined_time_constraint_in_seconds,
            const.READING_STATES['elapsed_time_in_seconds']: elapsed_time_in_seconds,
            const.READING_STATES['remaining_time_in_seconds']: remaining_time_in_seconds,
            const.READING_STATES['total_num_words']: total_num_words,
            const.READING_STATES['num_words_read']: num_words_read,
            const.READING_STATES['num_words_remaining']: num_words_remaining,
        }

    # A playground for being a small simulator:
    # Initialize the LTM module
    ltm_agent = LLMLongTermMemory()
    # Each participant for each trial, this will be reset once
    ltm_agent.reset()

    # Initialize the STM module
    stm_agent = LLMShortTermMemory(stm_capacity=2)
    stm_agent.reset()

    # Initialize the working memory module
    wm_agent = LLMWorkingMemory()
    wm_agent.reset()

    # Initialize the reading states related variables
    # AVERAGE_READ_WORDS_PER_SECOND = 2.5
    # Temporal variables for the reading states
    predefined_time_constraint_in_seconds = 90
    elapsed_time_in_seconds = const.ZERO
    remaining_time_in_seconds = predefined_time_constraint_in_seconds - elapsed_time_in_seconds
    # Spatial variables for the reading states -- reading progress
    num_words_on_stimuli = calculate_num_words(raw_sentences)
    num_words_read = const.ZERO
    num_words_remaining = num_words_on_stimuli - num_words_read

    # Initialize the reading states
    _reading_states = update_reading_states(
        predefined_time_constraint_in_seconds=predefined_time_constraint_in_seconds,
        elapsed_time_in_seconds=elapsed_time_in_seconds,
        total_num_words=num_words_on_stimuli,
        num_words_read=num_words_read,
    )

    sentence_idx = const.ZERO
    # Iteratively run the memory modules
    while remaining_time_in_seconds >= const.ZERO:       # TODO differentiate between the words read and words read in the stimuli

        # # Determine the reading strategy
        # _reading_strategy = wm_agent.decide_reading_strategy(reading_states=_reading_states)
        _reading_strategy = const.READ_STRATEGIES["normal"]

        # Read the words using oculomotor controller
        # Skip here, assuming we've already done that
        sentence = raw_sentences[sentence_idx][0]
        print(f"The current sentence is: \n{sentence}\n")

        # Get the number of words read in total
        num_words_read += calculate_num_words(single_sentence=sentence)
        # Get the updated time
        elapsed_time_in_seconds = (1 / const.AVERAGE_READ_WORDS_PER_SECOND) * num_words_read
        remaining_time_in_seconds = predefined_time_constraint_in_seconds - elapsed_time_in_seconds

        # Update the reading states
        _reading_states = update_reading_states(
            predefined_time_constraint_in_seconds=predefined_time_constraint_in_seconds,
            elapsed_time_in_seconds=elapsed_time_in_seconds,
            total_num_words=num_words_on_stimuli,
            num_words_read=num_words_read,
        )

        # Process
        ltm_agent, stm_agent, regression_decision, regression_sentence_id = wm_agent.process(
            ltm=ltm_agent,
            stm=stm_agent,
            read_info={const.CONTENT: sentence, const.SENTENCE_ID: sentence_idx},
            reading_states=_reading_states,
            reading_strategy=_reading_strategy,
        )

        if regression_decision == const.REGRESS_DECISIONS['regress']:
            sentence_idx = regression_sentence_id
        elif regression_decision == const.REGRESS_DECISIONS['continue_forward']:
            sentence_idx += 1
        else:
            raise ValueError(f"Invalid regression decision: {regression_decision}")

        print(f"The current state is: \n{_reading_states}\n")
        print(f"The STM is now: {stm_agent.STM}\n")
        print(f"The LTM is now: \n{ltm_agent.gists}\n")
        print(f"The regression decision is: \n{regression_decision}\n, and the regression target position is: \n{regression_sentence_id}\n, "
              f"the reading strategy is: \n{_reading_strategy}\n")
        print(f"---------------------------------------------------------------------------------------------------------------------\n")

        if sentence_idx >= len(raw_sentences):
            print(f"Finished reading all sentences, while the agent thinks it is not necessary to regress but continues to read. The reading has to stop.\n"
                  f"=====================================================================================================================\n")
            break

    # When the trial finishes, update the LTM
    ltm_agent.finalize_gists()





