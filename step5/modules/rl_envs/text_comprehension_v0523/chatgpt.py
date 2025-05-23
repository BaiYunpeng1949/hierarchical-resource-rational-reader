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

def update_base_url(request: httpx.Request) -> None:
        if request.url.path == "/chat/completions":
            # request.url = request.url.copy_with(path="/v1/chat") # chat/gpt4-8k /chat
            request.url = request.url.copy_with(path="/v1/openai/gpt4-turbo/chat/completions")

if __name__ == "__main__":
    
    #########################################################
    # Configure the propositions and the question
    #########################################################
    propositions = [
        "The text is about love.",
        "The text is about hate.",
        "The text is about life.",
    ]           # TODO debug check later   
    question = "What is the main theme of the text?" # TODO debug check later

    #########################################################
    # Test the LLM agent
    #########################################################
    llm_agent = LLMAgent(model_name="gpt-4o", api_key="")
    # llm_agent._demo_get_response("what is love?")
    llm_agent._get_response(
        role="You are a cognitive agent holding a lot of micro-structural propositions about a text. Now you need to answer some questions about the text only based on the propositions you have.",
        prompt=(
            f"The propositions are: {propositions}. \n\n Please answer the question: {question}."
        )
    )