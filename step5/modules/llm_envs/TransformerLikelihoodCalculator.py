import ast
import numpy as np
import yaml
import time
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import step5.utils.constants as const


class TransformerLikelihoodCalculator:
    def __init__(
            self,
            config: str = 'config.yaml'
    ):
        """
        Initialize the LLM likelihood calculator. Its function is calculating the likelihood of the word with certain
            given information, i.e., P(I|w).
        """

        # Read the configuration file
        with open(config, 'r') as stream:
            self._config = yaml.load(stream, Loader=yaml.FullLoader)

        print(f"{const.LV_TWO_DASHES}Transformer-based Likelihood Calculator -- Initialize the Transformer-based likelihood calculator.")

        # Initialize GPT-2 model and tokenizer
        self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self._model = GPT2LMHeadModel.from_pretrained('gpt2')

        # Check if GPU is available and move the model to GPU
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        # Task related variables
        self._parafovea_letters = None
        self._next_word = None
        self._estimated_word_len = None
        self._likelihood = None  # Handles the parafovea preview's effect, i.e., P(I|w)

    def _reset(
            self,
            parafovea_letters: str = "Hel",
            next_word: str = "Hello",
            estimated_word_len: int = 5,
            user_profile: dict = None,
    ):
        """
        Reset the LLM context predictor.
        :return: None
        """
        self._parafovea_letters = parafovea_letters
        self._next_word = next_word
        self._estimated_word_len = estimated_word_len

    def get_likelihood(
            self,
            parafovea_letters: str = "Hel",
            next_word: str = "Hello",
            estimated_next_word_len: int = 5,
            user_profile: dict = None,
    ):
        """
        Predict the next word with given information.
        :return:
            Likelihood of the given word
        """

        # Reset
        self._reset(
            parafovea_letters=parafovea_letters,
            next_word=next_word,
            estimated_word_len=estimated_next_word_len,
        )

        # Predict word likelihood
        self._likelihood = self._calculate_word_likelihood()

        return self._likelihood

    def _calculate_word_likelihood(self):
        """
        Calculate the likelihood of the target word given the starting letters and word length.
        :return:
            The likelihood of the target word.
        """
        # Construct the context with parafovea letters
        context = f"{self._parafovea_letters}"

        # Tokenize the input context
        input_ids = self._tokenizer.encode(context, return_tensors='pt').to(self._device)

        # Predict the next word
        with torch.no_grad():
            outputs = self._model(input_ids)
            next_word_logits = outputs.logits[:, -1, :].squeeze()

        # Apply softmax to get probabilities
        probabilities = F.softmax(next_word_logits, dim=-1)

        # Get all predictions and their probabilities
        all_probs = probabilities.cpu().numpy()
        all_indices = np.arange(len(all_probs))

        # Decode all predictions
        all_words = [self._tokenizer.decode([idx]).strip() for idx in all_indices]

        # Filter candidates by starting letters and word length
        filtered_words_probs = [(word, prob) for word, prob in zip(all_words, all_probs)
                                if word.startswith(self._parafovea_letters) and len(word) == self._estimated_word_len]

        # Merge probabilities of repeated words
        merged_probs = {}
        for word, prob in filtered_words_probs:
            if word in merged_probs:
                merged_probs[word] += prob
            else:
                merged_probs[word] = prob

        # Convert merged probabilities back to list format
        merged_words_probs = list(merged_probs.items())

        # Find the probability of the target word
        for word, probability in merged_words_probs:
            if word == self._next_word:
                return probability

        return 0


if __name__ == '__main__':
    llm = TransformerLikelihoodCalculator()
    likelihood = llm.get_likelihood(
        parafovea_letters="in",
        next_word="in",
        estimated_next_word_len=2,
    )
    print(f"The likelihood of the word 'physics' is {likelihood}. The type is {type(likelihood)}.")
