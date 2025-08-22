from openai import OpenAI
import openai
import time
import os
import httpx

class LLMAgent:

    def __init__(self, model_name, api_key):

        self.use_aalto_openai_api = True

        if self.use_aalto_openai_api:
            os.environ['AALTO_OPENAI_API_KEY'] = "b5de1b1587e04ee187293168b540136a"

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
            raise ValueError("Aalto Key does not work, need to use our own API key")
            # openai_api_key = xxx    # Use our own API key
            # # LLM related variables
            # self._client = OpenAI(api_key=openai_api_key)
        
        # gpt configurations
        self._gpt_model = "gpt-4o"
        self._refresh_interval = 20
        self._max_num_requests = 10
        self._retry_delay = 20
    
    def get_free_recall(self, ltm_gist: list=None):
        """
        Generate the free recall based on the ltm gist.
        """

        role="You are a reader with given ltm gists (integrated micro-structural propositions), please generate a free recall based only on that."
        prompt = (
            f"Based **ONLY** on the Long-term Memory (LTM) gist: '{ltm_gist}', "
            f"please provide a narrative summary in the form of a continuous paragraph. "
            f"Do not use bullet points, lists, or any other formatting. "
            f"Do **NOT** add any additional information, interpretations, or inferences. "
            f"Ensure that every detail in your summary directly corresponds to information explicitly stated in the LTM gist.\n"
        )
        
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

        # --- Raw response from the LLM ---
        free_recall = completion.choices[0].message.content

        return free_recall
    
    def get_mcq_answers(self, ltm_gist: list=None, question: str=None, options: dict=None):
        """
        Use GPT to answer MCQ questions based on generated LTM gists
        """
        role="You are a reader with given ltm gists (integrated micro-structural propositions), please answer some multiple-choice questions based on that."
        prompt = (
            f"Based **EXCLUSIVELY** on the following LTM content: '{ltm_gist}', "
            f"answer the question: '{question}' "
            f"from the options: {options}. "
            f"Respond with the letter of the correct answer ('A', 'B', 'C', or 'D') only. "
            f"If the correct answer is not **explicitly stated** in the LTM content, or if you are unsure, respond with 'E'. "
            f"Do **NOT** provide any explanations or additional text. "
            f"Do **NOT** use any outside knowledge or make any inferences. "
            f"Your answer should be based **solely** on the information provided above."
        )
        
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

        # --- Raw response from the LLM ---
        response = completion.choices[0].message.content

        if response:
            try: 
                answer = response.strip()

                if answer.upper() in ['A', 'B', 'C', 'D', 'E']:
                    return answer.upper()
                else:
                    raise ValueError(f"Invalid answer format: {answer}")
            except (SyntaxError, ValueError, AssertionError) as e:
                print(
                    f"Error parsing response for the LLM MCQ question answering. The current incorrect-format answer is: {response}"
                )

        print(f"Failed generating proper answers. Maybe should loop this generation process, Bai Yunpeng :> ")
        return

def update_base_url(request: httpx.Request) -> None:
        if request.url.path == "/chat/completions":
            # request.url = request.url.copy_with(path="/v1/chat") # chat/gpt4-8k /chat
            request.url = request.url.copy_with(path="/v1/openai/gpt4-turbo/chat/completions")

if __name__ == "__main__":
    
    #########################################################
    # Configure the propositions and the question
    #########################################################
    ltm_gist_metadata = {
      "program(billboard)": {
        "visits": 2,
        "first_step": 1,
        "last_step": 5,
        "total_strength": 2.05,
        "last_relevance": 1.482075,
        "est_recall": 0.5774999999999999
      },
      "billboard(public)": {
        "visits": 2,
        "first_step": 1,
        "last_step": 5,
        "total_strength": 2.05,
        "last_relevance": 1.2795750000000001,
        "est_recall": 0.5774999999999999
      },
      "bikes(docking stations))": {
        "visits": 2,
        "first_step": 2,
        "last_step": 10,
        "total_strength": 2.05,
        "last_relevance": 0.7402336050000001,
        "est_recall": 0.5774999999999999
      },
      "has(program, seen)": {
        "visits": 1,
        "first_step": 3,
        "last_step": 3,
        "total_strength": 1.0,
        "last_relevance": 0.9525,
        "est_recall": 0.35
      },
      "years(three, ago)": {
        "visits": 1,
        "first_step": 3,
        "last_step": 3,
        "total_strength": 1.0,
        "last_relevance": 0.75,
        "est_recall": 0.35
      },
      "trips(short))))))": {
        "visits": 1,
        "first_step": 4,
        "last_step": 4,
        "total_strength": 1.0,
        "last_relevance": 0.5,
        "est_recall": 0.35
      },
      "birth(gps tracking)": {
        "visits": 2,
        "first_step": 6,
        "last_step": 12,
        "total_strength": 2.15,
        "last_relevance": 1.26108075,
        "est_recall": 0.5774999999999999
      },
      "birth(improved safety feature)": {
        "visits": 2,
        "first_step": 6,
        "last_step": 12,
        "total_strength": 2.15,
        "last_relevance": 1.26108075,
        "est_recall": 0.5774999999999999
      },
      "helmets(free))": {
        "visits": 2,
        "first_step": 7,
        "last_step": 11,
        "total_strength": 2.1,
        "last_relevance": 0.8780500000000001,
        "est_recall": 0.5774999999999999
      },
      "healthy(lifestyle)))": {
        "visits": 1,
        "first_step": 8,
        "last_step": 8,
        "total_strength": 1.0,
        "last_relevance": 0.5,
        "est_recall": 0.35
      },
      "in(transportation(urban)))": {
        "visits": 2,
        "first_step": 9,
        "last_step": 13,
        "total_strength": 2.2,
        "last_relevance": 0.92805,
        "est_recall": 0.5774999999999999
      }
    }

    ltm_gist = list(ltm_gist_metadata.keys())

    #########################################################
    # Test the LLM agent
    #########################################################
    llm_agent = LLMAgent(model_name="gpt-4o", api_key="")
    
    #########################################################
    # Get the free recall
    #########################################################
    response = llm_agent.get_free_recall(ltm_gist=ltm_gist)
    print(f"The raw response is: {response}")
    
    #########################################################
    # Get the MCQ
    #########################################################
    question = "What aspect of fitness has the speaker come to appreciate?"
    options = {
        "A": "Working out as much as possible",
        "B": "The importance of rest and recovery",
        "C": "Lifting heavy weights",
        "D": "High-intensity interval training (HIIT)",
        "E": "Do not know / Cannot remember"
    }
    mcq_answer = llm_agent.get_mcq_answers(ltm_gist=ltm_gist, question=question, options=options)
    print(f"The mcq answer is: {mcq_answer}")
