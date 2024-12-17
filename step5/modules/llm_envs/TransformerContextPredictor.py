import ast
import numpy as np
import yaml
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import step5.utils.constants as const


class TransformerContextPredictor:
    def __init__(
            self,
            config: str = r'D:\Users\91584\PycharmProjects\reading-model\step5\config.yaml'
    ):
        """
        Initialize the LLM context predictor. Its function is predicting the next word with given information.
        The LLM should generate 5 most likely words with estimated probabilities. And calculate the given word's probability
        according to the generation.

        openai api documentation: https://platform.openai.com/docs/quickstart?context=python
        api service: https://platform.openai.com/api-keys
        tutorial: https://github.com/analyticswithadam/Python/blob/main/OpenAI_API_in_Python.ipynb

        Updated on 18 July,
        I found that the OpenAI API is not stable and returned results always have parsing errors.
        So I will try to use more stable transformer models to predict the next word.
        """

        print(f"{const.LV_TWO_DASHES}Transformer-based Context Predictor -- Initialize the Transformer-based context predictor.")

        # Older but stable Transformer models
        self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self._model = GPT2LMHeadModel.from_pretrained('gpt2')

        # Check if GPU is available and move the model to GPU
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        # Task related variables
        self._context = None
        self._next_word = None
        self._predictability = None

    def _reset(
            self,
            context_words: str = "Hello Bai Yunpeng",
            next_word: str = "Dududa",
            user_profile: dict = None,
    ):
        """
        Reset the LLM context predictor.
        :return: None
        """
        self._context = context_words
        self._next_word = next_word

    def predict(self, context_words: str = "Hello Bai Yunpeng", next_word: str = "Dududa", user_profile: dict = None):
        """
        Predict the next word with given information.
        :param context_words: Context words.
        :param next_word: Next word to predict.
        :param user_profile: User profile dictionary.
        :return:
            5 most likely words with estimated probabilities
        """
        # Reset
        self._reset(context_words=context_words, next_word=next_word, user_profile=user_profile)

        # Predict next words
        next_words_with_probs = self._predict_next_word(self._context)

        # Compute the predictability of the given word
        self._predictability = self._compute_given_word_predictability(next_words_with_probs)

        return self._predictability

    def _predict_next_word(self, context, top_k=50):
        """
        Predict the next word using GPT-2.
        :param context: Context words.
        :param top_k: Number of top predictions to consider.
        :return:
            5 most likely words with estimated probabilities
        """
        # Tokenize the input context
        input_ids = self._tokenizer.encode(context, return_tensors='pt').to(self._device)

        # Predict the next word
        with torch.no_grad():
            outputs = self._model(input_ids)
            next_word_logits = outputs.logits[:, -1, :].squeeze()

        # Apply softmax to get probabilities
        probabilities = F.softmax(next_word_logits, dim=-1)

        # Get the top_k predictions and their probabilities
        top_k_probs, top_k_indices = torch.topk(probabilities, top_k)
        top_k_probs = top_k_probs.cpu().numpy()
        top_k_indices = top_k_indices.cpu().numpy()

        # Decode the top_k predictions
        top_k_words = [self._tokenizer.decode([idx], clean_up_tokenization_spaces=True).strip() for idx in top_k_indices]

        # Combine words with probabilities
        words_with_probs = [(word, prob) for word, prob in zip(top_k_words, top_k_probs)]

        # Return the top 5 words with their probabilities
        return words_with_probs[:10]

    def _compute_given_word_predictability(self, next_words_with_probs):
        """
        Compute the predictability of the given word.
        :param next_words_with_probs: List of predicted words with probabilities.
        :return:
            the predictability of the given word
        """
        for word, probability in next_words_with_probs:
            if word == self._next_word:
                return np.clip(probability, 0, 1)
        return 0


if __name__ == '__main__':
    llm = TransformerContextPredictor()
    predictability = llm.predict(
        context_words="He comes with a",
        next_word="smile",
    )
    print(f"The predictability of the word 'smile' is {predictability}. The type is {type(predictability)}.")
