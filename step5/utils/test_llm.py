import os
import httpx
import yaml

import random
import numpy as np

from openai import OpenAI

class TestLLM:
    def __init__(
            self,
            config: str = '/home/baiy4/reading-model/step5/config.yaml',
    ):
        # Read the configuration file
        with open(config, 'r') as stream:
            self._config = yaml.load(stream, Loader=yaml.FullLoader)

        os.environ['AALTO_OPENAI_API_KEY'] = self._config['llm']['AALTO_OPENAI_API_KEY']

        assert (
            "AALTO_OPENAI_API_KEY" in os.environ and os.environ.get("AALTO_OPENAI_API_KEY") != ""
        ), "you must set the `AALTO_OPENAI_API_KEY` environment variable."
        self.client = OpenAI(
            base_url="https://aalto-openai-apigw.azure-api.net",
            api_key=False, # API key not used, and rather set below
            default_headers = {
                "Ocp-Apim-Subscription-Key": os.environ.get("AALTO_OPENAI_API_KEY"),
            },
            http_client=httpx.Client(
                event_hooks={"request": [self.update_base_url] }
            ),
        )
    
    def update_base_url(self, request: httpx.Request) -> None:
        if request.url.path == "/chat/completions":
            request.url = request.url.copy_with(path="/v1/chat") # chat/gpt4-8k /chat
            # request.url = 'https://aalto-openai-apigw.azure-api.net/v1/chat'
            # request.url = request.url.copy_with(path="/v1/openai/gpt4-turbo/chat/completions")
            # request.url = request.url.copy_with(path="/v1/openai/gpt4o/chat/completions")
        
    def test_llm(self):

        message = [
            {
                "role": "system", 
                "content": "You are looking at, and your task is. Your goal is to read a given question and answer it based on the information in the text."
            },
            {
                "role": "user", 
                "content": "Your current memory is I woke up at 9 AM in the morning. Please summarize the memory into one paragraph."
            }
        ]

        completion = self.client.chat.completions.create(
            model="no_effect", # the model variable must be set, but has no effect, model selection done with URL
            messages=message,
        )

        answer = completion.choices[0].message.content

        print(answer)

if __name__ == "__main__":
    test_llm = TestLLM()
    test_llm.test_llm()