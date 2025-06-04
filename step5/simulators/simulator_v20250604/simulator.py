from sub_models.sentence_read_v0604 import SentenceReader
from sub_models.text_comprehension_v0604 import TextReader


class ReaderAgent:
    def __init__(self):
        self.sentence_reader = SentenceReader()
        self.text_reader = TextReader()

    def run(self):
        pass