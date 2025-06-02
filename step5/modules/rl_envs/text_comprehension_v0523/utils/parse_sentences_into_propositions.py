import json
import os
import nltk
from nltk.tokenize import sent_tokenize
import PropositionsGenerator as pg

# Download Punkt tokenizer if not already present
nltk.download('punkt')

def parse_sentences(text):
    sentences = sent_tokenize(text)
    formatted_sentences = [{"id": i + 1, "texts": s} for i, s in enumerate(sentences)]
    return formatted_sentences

def process_json_file():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, '..', 'assets', 'original_texts_v0526.json')

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in data:
        if 'texts' in entry:
            entry['text_into_sentences'] = []
            sentences = parse_sentences(entry['texts'])

            for sent_obj in sentences:
                sentence_id = sent_obj["id"]
                sentence_text = sent_obj["texts"]

                propositions = pg.extract_propositions_spacy(sentence_text)
                sent_obj["propositions"] = propositions

                # Append sentence object with both knowledge cases populated
                entry['text_into_sentences'].append(sent_obj)

    # Save updated JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":

    # Process sentences into propositions
    process_json_file()