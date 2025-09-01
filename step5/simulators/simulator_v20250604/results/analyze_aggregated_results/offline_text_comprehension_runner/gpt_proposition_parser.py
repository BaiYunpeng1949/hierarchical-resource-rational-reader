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
    
    def get_micro_structural_propositions(self, role, prompt):
        
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

        # --- Clean and split response into proposition groups ---
        # Assume one line per group (optional) and split each line by commas
        raw_lines = response.split("\n")
        proposition_groups = []

        for line in raw_lines:
            # Remove any brackets or extra characters, just sanitize
            cleaned = line.strip().rstrip(",")  # remove trailing comma
            if not cleaned:
                continue
            group = [p.strip() for p in cleaned.split(",") if p.strip()]
            if group:
                proposition_groups.append(group)

        return proposition_groups
    
    def get_facet_summaries(self, role, prompt):

        if self.use_aalto_openai_api:
            messages = [{"role":"system","content":role},{"role":"user","content":prompt}]
            completion = self._client.chat.completions.create(model="no_effect", messages=messages)
            text = completion.choices[0].message.content or ""
            # one facet per non-empty line; NO comma splitting
            return [ln.strip() for ln in text.split("\n") if ln.strip()]
        else:
            raise ValueError(f"Please use an institute API, e.g., Aalto's.")
    
    def get_schema_assignments(self, role: str, prompt: str):
        """Return a Python object parsed from the model's JSON output."""
        messages=[{"role":"system","content":role},{"role":"user","content":prompt}]
        comp = self._client.chat.completions.create(model="no_effect", messages=messages)
        txt = comp.choices[0].message.content or "[]"
        # robust JSON load
        try:
            return json.loads(txt)
        except Exception:
            # if the model wrapped JSON in code fences, strip them
            m = re.search(r"\{.*\}|\[.*\]", txt, flags=re.DOTALL)
            return json.loads(m.group(0)) if m else []

def update_base_url(request: httpx.Request) -> None:
        if request.url.path == "/chat/completions":
            # request.url = request.url.copy_with(path="/v1/chat") # chat/gpt4-8k /chat
            request.url = request.url.copy_with(path="/v1/openai/gpt4-turbo/chat/completions")

if __name__ == "__main__":
    
    #########################################################
    # Configure the propositions and the question
    #########################################################
    sentence = "The heart is the hardest working organ in the body."

    #########################################################
    # Test the LLM agent
    #########################################################
    llm_agent = LLMAgent(model_name="gpt-4o", api_key="")
    # llm_agent._demo_get_response("what is love?")
    response = llm_agent.get_micro_structural_propositions(
        role="You are a reader with **high prior knowledge** about **heart disease**. ",
        # role="You are a reader with **low prior knowledge** about **heart disease**. ",
        # prompt=(
        #     f"Now please parse the given sentence into micro-structural propositions that matches Kintsch's model of text comprehension. \n\n The sentence is: {sentence}.\n\n"
        #     f"The propositions should be in the following format: \n\n"
        #     f"A(B, C), or nested A(B, C(D, E))."
        #     f"Please output the propositions only, no other text, separated by comma."
        # )
        prompt = (
            "Parse this sentence into micro-structural propositions (Kintsch-style).\n"
            "STRICT OUTPUT: comma-separated propositions only; NO extra text.\n"
            "Each proposition must be of the form A(B, C) or nested A(B, C(D)).\n\n"
            "Coverage requirements — be EXHAUSTIVE but avoid duplicates:\n"
            "- Actions/events: use predicates like do(agent, action(object)), event(subject, object)\n"
            "- Attributives/modifiers: has_attr(entity, attribute)\n"
            "- Numbers/measurements: quantity(entity, value unit)\n"
            "- Time/temporal: time_at(event_or_state, time_expr)\n"
            "- Location: location(entity_or_event, place)\n"
            "- Causal/conditional: cause(x, y), condition(x, y)\n"
            "- Purpose/goal: purpose(x, y)\n"
            "- Membership/part-whole: part_of(x, y)\n"
            "- Coreference: coref(mention, canonical_entity)\n"
            "- Negation: negate(proposition_signature, reason)\n\n"
            "Prefer canonical nouns/verbs; keep arguments short and consistent.\n"
            "Aim for 8-15 propositions if the sentence is information rich.\n"
            f"Sentence: {sent}"
        )
    )

    # TODO debug check later
    print(response)