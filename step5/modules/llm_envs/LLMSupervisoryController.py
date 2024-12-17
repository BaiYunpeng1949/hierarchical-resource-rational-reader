import ast

import numpy as np
from openai import OpenAI
import time
import yaml

import step5.utils.constants as const


class LLMSupervisoryController:
    def __init__(
            self,
            config: str = r'D:\Users\91584\PycharmProjects\reading-model\step5\config.yaml',
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
        openai_api_key = self._config['llm']['API_key']

        # Specific configurations
        self._gpt_model = self._config['llm']['wm']['model']
        self._refresh_interval = self._config['llm']['wm']['refresh_interval']
        self._max_num_requests = self._config['llm']['wm']['max_num_requests']
        self._retry_delay = self._config['llm']['wm']['retry_delay']

        # LLM related variables
        self._client = OpenAI(api_key=openai_api_key)

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
        # Run GPT API for the macro-operator
        for attempt in range(self._max_num_requests):
            try:
                # print(f"{cons.LV_ONE_DASHES}LLM-based STM and LTM Memory Retrieval working:\n"  # 20*'-'
                #       f"{cons.LV_TWO_DASHES}Attempt {attempt} is running...\n"
                #       f"{cons.LV_TWO_DASHES}STM content: '{self._stm}' \n"
                #       f"{cons.LV_TWO_DASHES}Gists in the LTM: '{self._ltm_gists}'\n")

                if question_type == const.QUESTION_TYPES['MCQ']:
                    prompt = (
                        f"Only base on your Long-term Memory (LTM) content: {ltm_gists}, "
                        f"Please select only one option from the following multiple-choice question: '{question}'."
                        f"The options are {options}.\n"
                    )
                elif question_type == const.QUESTION_TYPES['FRS']:
                    # contents = [item['content'] for item in self._ltm_gists]
                    prompt = (
                        f"Please summarise ONLY based on the LTM's content: '{ltm_gists}'. Do not improvise. \n"
                    )
                else:
                    raise ValueError(f"Invalid question type: {question_type}")

                raw_response = self._get_response(
                    role="Please answer the questions ONLY based on your Long-term Memory (LTM) content.",
                    prompt=prompt,
                )

                # print(f"{cons.LV_TWO_DASHES}Raw response: {raw_response}")

                if raw_response:
                    # Loop to validate and parse the response
                    try:
                        # Separate the content and spatial information from the response: split by the delimiter
                        answer = raw_response.strip()
                        return answer

                    except (SyntaxError, ValueError, AssertionError) as e:
                        print(f"Error parsing response for the LLM memory retrieval: {e}\n"
                              f"The current incorrect-format memory retrieval is: {raw_response}")
                        continue

            except Exception as e:
                print(f"Attempt {attempt} failed with error: {e}")
                time.sleep(self._retry_delay)

        print("Failed to generate a valid single memory retrieval after maximum attempts")
        return "Failed to answer. Please try again."

    def decide_reading_strategy(
            self,
            predefined_time_constraint_in_seconds: int = None,
            elapsed_time_in_seconds: int = None,
            remaining_time_in_seconds: int = None,
            total_num_words: int = None,
            num_words_read: int = None,
            remaining_num_words: int = None,
    ):
        """
        The high-level decision-making of the Supervisory Controller. Skim reading or in-depth (careful) reading.
        :return:
        """
        # Run GPT API for the macro-operator
        for attempt in range(self._max_num_requests):
            try:
                state = {
                    "predefined_time_constraint_in_second": predefined_time_constraint_in_seconds,
                    "elapsed_time_in_second": elapsed_time_in_seconds,
                    "remaining_time_in_second": remaining_time_in_seconds,
                    "total_num_words": total_num_words,
                    "num_words_read": num_words_read,
                    "remaining_num_words": remaining_num_words,
                }
                raw_response = self._get_response(
                    role="You are a reader trying to read a text passage.",
                    prompt=f"The task instruction: {self._task_spec}. \n"
                           f"Based on the current state: {state}, please determine whether to skim reading, normally read, or carefully read. \n"
                           f"You must consider the trade-off between reading speed and comprehension. "
                           f"By estimating your reading speed, remaining time, and content, choose the best option.\n"
                           f"Only output one of the three options: {const.READ_STRATEGIES['skim']}, {const.READ_STRATEGIES['normal']}, "
                           f"or {const.READ_STRATEGIES['careful']}.",
                )

                if raw_response:
                    # Loop to validate and parse the response
                    try:
                        answer = raw_response.strip()
                        # Check if the answer is one of the three valid options
                        if answer in {const.READ_STRATEGIES['skim'], const.READ_STRATEGIES['normal'], const.READ_STRATEGIES['careful']}:
                            return answer
                        else:
                            print(f"Invalid answer received: '{answer}'. Retrying...")
                            continue  # Retry if the answer is not valid

                    except (SyntaxError, ValueError, AssertionError) as e:
                        print(f"Error parsing response for the LLM memory retrieval: {e}\n"
                              f"The current incorrect-format memory retrieval is: {raw_response}")
                        continue

            except Exception as e:
                print(f"Attempt {attempt} failed with error: {e}")
                time.sleep(self._retry_delay)

        print("Failed to generate a valid single memory retrieval after maximum attempts")
        return "Failed to answer. Please try again."

    # def regress_or_not(
    #         self,
    #         stm: list = None,
    #         ltm_gists: list = None,
    #         predefined_time_constraint_in_seconds: int = None,
    #         elapsed_time_in_seconds: int = None,
    #         remaining_time_in_seconds: int = None,
    #         total_num_words: int = None,
    #         num_words_read: int = None,
    #         remaining_num_words: int = None,
    # ):
    #     """
    #     The high-level decision-making of the Supervisory Controller. Skim reading or in-depth (careful) reading.
    #     :return:
    #     """
    #     # Run GPT API for the macro-operator
    #     for attempt in range(self._max_num_requests):
    #         try:
    #             state = {
    #                 "predefined_time_constraint_in_second": predefined_time_constraint_in_seconds,
    #                 "elapsed_time_in_second": elapsed_time_in_seconds,
    #                 "remaining_time_in_second": remaining_time_in_seconds,
    #                 "total_num_words": total_num_words,
    #                 "num_words_read": num_words_read,
    #                 "remaining_num_words": remaining_num_words,
    #             }
    #             raw_response = self._get_response(
    #                 role="You are a reader tasked with deciding whether to regress or not during your reading.",
    #                 prompt=f"The task instruction: {self._task_spec}. \n"
    #                        f"Based on the current state: {state}, \n"
    #                        f"STM: {stm}, \n"
    #                        f"and LTM: {ltm_gists}, \n"
    #                        f"please determine whether to regress (go back to reread) or continue forward. \n"
    #                        f"Consider the trade-off between reading speed and comprehension: regressing may slow your progress but can enhance understanding, "
    #                        f"especially if the current content is complex or critical. \n"
    #                        f"Evaluate your current reading progress, the remaining time, understanding to the current content accorting to memories, "
    #                        f"and the content's importance to decide the best course of action.\n"
    #                        f"Only output one of the two options: {cons.REGRESS_DECISIONS['regress']} or {cons.REGRESS_DECISIONS['not_regress']}."
    #             )
    #
    #             if raw_response:
    #                 # Loop to validate and parse the response
    #                 try:
    #                     answer = raw_response.strip()
    #                     # Check if the answer is one of the three valid options
    #                     if answer in {cons.REGRESS_DECISIONS['regress'], cons.REGRESS_DECISIONS['not_regress']}:
    #                         return answer
    #                     else:
    #                         print(f"Invalid answer received: '{answer}'. Retrying...")
    #                         continue  # Retry if the answer is not valid
    #
    #                 except (SyntaxError, ValueError, AssertionError) as e:
    #                     print(f"Error parsing response for the LLM memory retrieval: {e}\n"
    #                           f"The current incorrect-format memory retrieval is: {raw_response}")
    #                     continue
    #
    #         except Exception as e:
    #             print(f"Attempt {attempt} failed with error: {e}")
    #             time.sleep(self._retry_delay)
    #
    #     print("Failed to generate a valid single memory retrieval after maximum attempts")
    #     return "Failed to answer. Please try again."

    def regress_or_not(
            self,
            stm: list = None,
            ltm_gists: list = None,
            predefined_time_constraint_in_seconds: int = None,
            elapsed_time_in_seconds: int = None,
            remaining_time_in_seconds: int = None,
            total_num_words: int = None,
            num_words_read: int = None,
            remaining_num_words: int = None,
    ):
        """
        The high-level decision-making of the Supervisory Controller.
        Determine whether to regress (go back to reread) or continue forward based on the current state of reading.

        :return: Decision to regress or not, and the reasoning behind that decision.
        """
        # Run GPT API for the macro-operator
        for attempt in range(self._max_num_requests):
            try:
                state = {
                    "predefined_time_constraint_in_second": predefined_time_constraint_in_seconds,
                    "elapsed_time_in_second": elapsed_time_in_seconds,
                    "remaining_time_in_second": remaining_time_in_seconds,
                    "total_num_words": total_num_words,
                    "num_words_read": num_words_read,
                    "remaining_num_words": remaining_num_words,
                }
                raw_response = self._get_response(      # TODO need to debug the prompts for motivating the regressions
                    role="You are a reader tasked with deciding whether to regress or not during your reading.",
                    prompt=f"The task instruction: {self._task_spec}. \n"
                           f"Based on the current state: {state}, \n"
                           f"STM: {stm}, \n"
                           f"and LTM: {ltm_gists}, \n"
                           f"please determine whether to regress (go back to reread) or continue forward. \n"
                           f"Consider the trade-off between reading speed and comprehension: regressing may slow your progress but can enhance understanding, "
                           f"especially if the current content is complex or critical. \n"
                           f"Evaluate your current reading progress, the remaining time, understanding of the current content according to memories, "
                           f"and the content's importance to decide the best course of action.\n"
                           f"Please output your decision in the following format: decision {const.PROMPT_DELIMITER} reasoning. "
                           f"The decision must be either {const.REGRESS_DECISIONS['regress']} or {const.REGRESS_DECISIONS['continue_forward']}\n"
                           f"Example: regress {const.PROMPT_DELIMITER} I chose to regress because the content is complex and understanding is critical."
                )
                # TODO solve this, maybe need to determine whether to regress and where to regress simultaneously
                if raw_response:
                    # Loop to validate and parse the response
                    try:
                        # Split the response into decision and reasoning
                        decision, reasoning = raw_response.strip().split(const.PROMPT_DELIMITER)
                        decision = decision.strip()

                        # TODO debug delete later
                        print(f"Decision: {decision}, Reasoning: {reasoning}")

                        # Check if the decision is one of the valid options
                        if decision in {const.REGRESS_DECISIONS['regress'], const.REGRESS_DECISIONS['continue_forward']}:
                            return decision, reasoning.strip()
                        else:
                            print(f"Invalid decision received: '{decision}'. Retrying...")
                            continue  # Retry if the decision is not valid

                    except (SyntaxError, ValueError, AssertionError) as e:
                        print(f"Error parsing response for the LLM memory retrieval: {e}\n"
                              f"The current incorrect-format memory retrieval is: {raw_response}")
                        continue

            except Exception as e:
                print(f"Attempt {attempt} failed with error: {e}")
                time.sleep(self._retry_delay)

        print("Failed to generate a valid decision after maximum attempts")
        return "Failed to answer. Please try again."


    def determine_regression_target(
            self,
            stm: list = None,
            ltm_gists: list = None,
            num_sentences: int = None,
            start_words_indexes_in_sentences: list = None,
            regression_reason: str = "clarifying missing information and strengthening understanding"
    ) -> (int, int):
        """
        The high-level decision-making of the Supervisory Controller.
        Determine whether to regress to previous sentences for reasons like clarifying information or strengthening understanding.

        :param stm: The Short-term Memory (STM) content.
        :param ltm_gists: The Long-term Memory (LTM) content.
        :param num_sentences: The number of sentences that have been read.
        :param start_words_indexes_in_sentences: The start word indexes of the sentences in the section.
        :param regression_reason: The reason for regressing (e.g., "clarifying missing information", "strengthening understanding").
        :return: Sentence index to regress to, and the corresponding start word index in the section.
        """
        # Reset the variables
        self.reset(
            stm=stm,
            ltm_gists=ltm_gists,
        )

        sentence_index = const.NEGATIVE_ONE
        start_word_index_in_section = const.NEGATIVE_ONE

        # Run GPT API for the regression decision
        for attempt in range(self._max_num_requests):
            try:

                regress_target_raw_response = self._get_response(
                    role=f"You are a reader who needs to decide where to regress to strengthen understanding or clarify missing information.",
                    prompt=(
                        f"You have been reading and storing information in your Short-term Memory (STM) and Long-term Memory (LTM). "
                        f"STM is a list of micro-gists and spatial information: {self._stm}. "
                        f"LTM is a list of macro-gists and spatial information: {self._ltm_gists}. "
                        f"Your goal now is to regress and reread a previous sentence to better understand the material or to clarify any gaps in your memory.\n\n"
                        f"Based on the current state of your STM and LTM, determine which sentence is most likely to help with {regression_reason}.\n"
                        f"Please output your decision in the following format: sentence index {const.PROMPT_DELIMITER} Your reasoning.\n"
                        f"Note that the sentence index should be drawn from the 'spatial_info' field in LTM or STM, specifically the {const.SENTENCE_IDX} field.\n"
                        f"Example: "
                        "If you want the sentence with this content: {'content': 'BALABALA.', 'spatial_info': {'sentence_idx': m, 'section_idx': n}, xxx}, "
                        f"Please output: m {const.PROMPT_DELIMITER} I believe sentence index m contains the relevant information because xxx."
                    )
                )

                if regress_target_raw_response:
                    # Validate and parse the response
                    try:
                        sentence_index, reasoning = regress_target_raw_response.strip().split(const.PROMPT_DELIMITER)
                        sentence_index = int(sentence_index.strip())

                        # TODO: Check if the sentence index is valid
                        print(f"Sentence index: {sentence_index}, Reasoning: {reasoning}")
                              # f"Reasoning: {reasoning}")

                        start_word_index_in_section = start_words_indexes_in_sentences[sentence_index]
                        if isinstance(start_word_index_in_section, int):
                            break

                    except (SyntaxError, ValueError, AssertionError) as e:
                        print(f"Error parsing response for Regression Target Decision: {e}\n"
                              f"The current incorrect-format decision is: {regress_target_raw_response}")
                        continue  # Retry if parsing fails

            except Exception as e:
                print(f"Attempt {attempt} failed with error: {e}")
                time.sleep(self._retry_delay)

        return sentence_index, start_word_index_in_section

    def generate_sentence_planning_decision_for_info_search_task(
            self,
            stm: list = None,
            ltm_gists: list = None,
            question: str = None,
            num_sentences: int = None,
            start_words_indexes_in_sentences: list = None,
    ) -> (int, int):
        """
        The high-level decision-making of the Supervisory Controller.
        Determine whether to regress to the previous sentences to answer the question.
        This is a pioneer version of the section-level reading planning.

        Regress's reasons: the past sentences could answer the question, but the details are forgotten.

        :param stm: The Short-term Memory (STM) content.
        :param ltm_gists: The Long-term Memory (LTM) content.
        :param question: The question to answer.
        :param num_sentences: The number of sentences to regress.
        :param start_words_indexes_in_sentences:  The start word indexes of the sentences in the section.
        :return: Action, whether to regress to the previous sentences (True) or not (False) -- continue to read.
                 Integer, the index of the sentence (start word index) to regress.
        """
        # Reset the variables
        self.reset(
            stm=stm,
            ltm_gists=ltm_gists,
            question=question,
        )

        sentence_index = const.NEGATIVE_ONE
        start_word_index_in_section = const.NEGATIVE_ONE

        # Run GPT API for the stop reading decision
        for attempt in range(self._max_num_requests):
            try:
                print(f"{const.LV_TWO_DASHES}LLM-based Working Memory Supervisory Controller working -- Sentence Planning Decision -- {const.REGRESS} or {const.NOT_REGRESS}:\n"  # 20*'-'
                      f"{const.LV_THREE_DASHES}Attempt {attempt} is running...\n"
                      f"{const.LV_THREE_DASHES}STM content: '{self._stm}' \n"
                      f"{const.LV_THREE_DASHES}Gists in the LTM: '{self._ltm_gists}'")

                regress_decision_raw_response = self._get_response(
                    role=f"You are a reader trying read {num_sentences} sentences in a section to answer the question '{self._question}'.",
                    prompt=(
                        f"You are a reader trying to read {num_sentences} sentences in a section to answer the question '{self._question}'.\n"
                        f"The sentences you have read are stored in both Short-term Memory (STM) and Long-term Memory (LTM).\n"
                        f"STM is a list of dictionaries containing micro-gists and spatial information: {self._stm}.\n"
                        f"Some information may be lost due to memory decay, indicated by {const.MEMORY_LOSS_MARK}.\n"
                        f"LTM is a list of dictionaries containing macro-gists and spatial information: {self._ltm_gists}.\n"
                        f"These lists may be empty if no sentences have been read yet.\n\n"
                        f"Which sentence is the most likely to contain information related to the question: '{question}'?\n"
                        f"Please output your decision in the following format: sentence index {const.PROMPT_DELIMITER} Your reasoning.\n"
                        f"Note that the sentence index comes from the 'spatial_info' field in LTM or STM, specifically the {const.SENTENCE_IDX} field, not the direct index from STM or LTM.\n"
                        f"Example:\n"
                        "If you want the sentence with this content: {'content': 'BALABALA.', 'spatial_info': {'sentence_idx': m, 'section_idx': n}, xxx},\n"
                        f"Please output: m {const.PROMPT_DELIMITER} I believe sentence index m contains the relevant information because xxx."
                    )
                )

                print(f"{const.LV_THREE_DASHES}Raw response: {regress_decision_raw_response}")

                if regress_decision_raw_response:
                    # Loop to validate and parse the response
                    try:
                        # Get the decision and index from the raw response
                        sentence_index, reasoning = regress_decision_raw_response.strip().split(const.PROMPT_DELIMITER)
                        sentence_index = int(sentence_index.strip())

                        start_word_index_in_section = start_words_indexes_in_sentences[sentence_index]
                        if isinstance(start_word_index_in_section, int):
                            break

                    except (SyntaxError, ValueError, AssertionError) as e:
                        print(f"Error parsing response for the LLM Supervisory Controller -- Sentence Planning Decision -- {const.REGRESS} or {const.NOT_REGRESS}: {e}\n"
                              f"The current incorrect-format decision is: {regress_decision_raw_response}")
                        continue  # If parsing fails, continue to the next attempt

            except Exception as e:
                print(f"Attempt {attempt} failed of {const.REGRESS} or {const.NOT_REGRESS} with error: {e}")
                time.sleep(self._retry_delay)

        return sentence_index, start_word_index_in_section

    def generate_stop_reading_decision_for_info_search_task(
            self,
            task_specification: str = None,
            stm: list = None,
            ltm_gists: list = None,
            user_profile: dict = None,
            question: str = None,
    ) -> bool:
        """
        The high-level decision-making of the Supervisory Controller.
        Level 1: Whether to continue reading or not.
            Based on the current reading progress (no spatial info) and task (information search).

        :return: Action, whether to continue reading (True) or not (False).
        """

        # Reset the variables
        self.reset(
            task_specification=task_specification,
            stm=stm,
            ltm_gists=ltm_gists,
            user_profile=user_profile,
            question=question,
        )

        # # Configure the role and prompt
        # self._role = self._config_role()
        # self._prompt = self._config_prompt(question=question)

        # Run GPT API for the stop reading decision
        for attempt in range(self._max_num_requests):
            try:
                print(f"{const.LV_TWO_DASHES}LLM-based Working Memory Supervisory Controller working -- Stop Reading Decision:\n"  # 20*'-'
                      f"{const.LV_THREE_DASHES}Attempt {attempt} is running...\n"
                      f"{const.LV_THREE_DASHES}STM content: '{self._stm}' \n"
                      f"{const.LV_THREE_DASHES}Gists in the LTM: '{self._ltm_gists}', and task specification: '{self._task_spec}'")

                raw_response = self._get_response(
                    role=self._config_role(),
                    prompt=self._config_prompt(question=question),
                )

                print(f"{const.LV_THREE_DASHES}Raw response: {raw_response}")
                print("\n\n")

                if raw_response:
                    # Loop to validate and parse the response
                    try:
                        # get the answer from the raw response
                        answer = raw_response.strip()

                        if const.NOT_READ in answer:  # A more general way to check if the answer is empty
                            self.action_stop_reading = False
                            return self.action_stop_reading
                        else:
                            self.action_stop_reading = True
                            return self.action_stop_reading

                    except (SyntaxError, ValueError, AssertionError) as e:
                        print(f"Error parsing response for the LLM Supervisory Controller -- Stop Reading Decision: {e}\n"
                              f"The current incorrect-format decision is: {raw_response}")
                        continue  # If parsing fails, continue to the next attempt

            except Exception as e:
                print(f"Attempt {attempt} failed with error: {e}")
                time.sleep(self._retry_delay)

        print("Failed to generate a valid decision after maximum attempts")

    def _config_role(self):
        """
        Configure the role for the LLM API.
        :return:
            the role for the LLM API
        """

        if self._user_profile['proficiency'] == 'good':
            role_proficiency = "You are a normal (or native) english speaker with average reading abilities."
        elif self._user_profile['proficiency'] == 'poor':
            role_proficiency = "You are not a native english speaker with below average reading abilities."
        else:
            raise ValueError("Invalid user proficiency level.")

        if self._user_profile['interest level'] == 'high':
            role_interest = "You are interested in the given topic."
        elif self._user_profile['interest level'] == 'low':
            role_interest = "You are not interested in the given topic."
        else:
            raise ValueError("Invalid user interest level.")

        if self._user_profile['background knowledge'] == 'specific':
            role_background = "You have specific background knowledge in the given topic."
        elif self._user_profile['background knowledge'] == 'empty':
            role_background = "You do not have any background knowledge in the given topic."
        else:
            raise ValueError("Invalid user background knowledge level.")

        return f"{role_proficiency} {role_interest} {role_background}"

    def _config_prompt(self, question: str = None):
        """
        Configure the prompt for the LLM API.
        :return:
            the prompt for the LLM API
        """
        return (
            f"You just finished reading some texts and stored information in both the Short-term Memory (STM) and Long-term Memory (LTM). "
            f"The STM is a list of tuples containing micro-gists and their spatial information: {self._stm}.\n"
            f"The LTM consists of a knowledge base and gists, which are stored as a list of tuples containing macro-gists and their spatial information: {self._ltm_gists}.\n"
            f"Did STM and LTM has any specific information about the question '{question}'? If yes, output the information, if not, output {const.NOT_READ}.\n"
        )

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


if __name__ == '__main__':
    llm = LLMSupervisoryController()
    _task_specification = (
        "You will be asked to read several paragraphs, and your task is to search for specific information within each paragraph. "
        "Once you find the information you are looking for, you must stop reading immediately. "
        "Instructions: Read the paragraphs and search for the specific information. "
        "Stop reading once you have found a good enough required information."
        "For this trial: Why was he Ebert Einstein awared Nobal Prize?"
    )
    _user_profile = {
        "proficiency": "good",
        "interest level": "high",
        "background knowledge": "specific",
    }
    _stm = [
        ('Albert Einstein eighteen eighteen seventynine Germany renowne his contributions theoretical physics', 'The 1st sentence in the 1st paragraph.'),
        ('His theory relativity particularly the massenergy equivalence equation mass energy revolutionize our understanding the speed light', 'The 2st sentence in the 1st paragraph.'),
        ('nineteen twentyone he award the Nobel Prize in Physics the Nobel Prize Physics his explanation the photoelectric effect', 'The 3st sentence in the 1st paragraph.'),
        ("difficulties flee Nazi Germany Einstein's work Einstein a lasting impact last science modern technology fields quantum mechanics cosmology", 'The 4st sentence in the 1st paragraph.')
    ]
    _ltm_gists = [
        ('Albert Einstein, renowned for theoretical physics, born 1879, Germany', 'the first sentence'),
        ("'Einstein's relativity theory revolutionized understanding of mass-energy and light speed.", "the first paragraph'"),
        ("'Albert Einstein won the 1921 Nobel Prize for his photoelectric effect explanation.", "the first paragraph'"),
        ('"Einstein fled Nazi Germany, impacting science, quantum mechanics, and cosmology.', 'the first paragraph"')
    ]
    for i in range(len(_stm)):
        dynamic_stm = _stm[:i + 1]
        dynamic_ltm_gists = _ltm_gists[:i + 1]
        _decision = llm.generate_stop_reading_decision_for_info_search_task(
            task_specification="Please read the texts and prepare for a comprehension later.",
            stm=dynamic_stm,
            ltm_gists=dynamic_ltm_gists,
            user_profile=_user_profile,
        )
        print(f"Continue reading or not: {_decision}")
        if not _decision:
            break
