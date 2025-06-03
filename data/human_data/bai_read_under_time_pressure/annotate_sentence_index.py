import os
import re
import json
from pathlib import Path

CORPUS_NAME = 'corpus_10_27.txt'

def process_stimulus(text):
    # Split text into sentences using regex
    # This pattern matches sentences ending with .!? followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Process each sentence to get word indices
    sentence_indices = []
    current_index = 0  # Reset index for each stimulus
    
    for sentence_idx, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
            
        # Get words in the sentence
        words = sentence.split()
        # Calculate start and end indices
        start_idx = current_index
        end_idx = current_index + len(words) - 1
        
        sentence_indices.append({
            'sentence': sentence,
            'sentence_idx': sentence_idx,  # Add sentence index within the text
            'start_idx': start_idx,
            'end_idx': end_idx,
            'word_count': len(words)
        })
        
        current_index += len(words)
    
    return sentence_indices

def main():
    # Read the corpus file
    corpus_path = os.path.join(os.path.dirname(__file__), 'stimuli', '10_27_15_58_100_images_W1920H1080WS16_LS40_MARGIN400', 'assets', CORPUS_NAME)
    with open(corpus_path, 'r', encoding='utf-8') as f:
        # Read lines and filter out empty lines
        stimuli = [line.strip() for line in f.readlines() if line.strip()]
    
    # Process each stimulus
    results = []
    for i, stimulus in enumerate(stimuli):
        sentence_indices = process_stimulus(stimulus)
        results.append({
            'stimulus_id': i,
            'text': stimulus,
            'sentences': sentence_indices,
            'total_words': sum(s['word_count'] for s in sentence_indices)  # Add total word count for verification
        })
    
    # Save results to JSON file
    output_path = os.path.join(os.path.dirname(__file__), 'stimuli', '10_27_15_58_100_images_W1920H1080WS16_LS40_MARGIN400', 'assets', 'metadata_sentence_indeces.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(results)} stimuli")
    print(f"Results saved to {output_path}")
    
    # Print some statistics for verification
    for result in results:
        print(f"\nStimulus {result['stimulus_id']}:")
        print(f"Total words: {result['total_words']}")
        for sent in result['sentences']:
            print(f"Sentence: {sent['sentence'][:50]}...")
            print(f"Indices: {sent['start_idx']} to {sent['end_idx']}")

if __name__ == '__main__':
    main()
