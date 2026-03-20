import json
import os
import numpy as np
import string
from tqdm import tqdm

def clean_word(word):
    """Clean word by removing punctuation and converting to lowercase"""
    # Remove punctuation from start and end of word
    word = word.lower().strip()
    # Remove all punctuation marks
    word = ''.join(char for char in word if char not in string.punctuation)
    return word

def calculate_word_difficulty(freq_per_million, alpha=1.0, beta=1.0, F=11):
    """Calculate word difficulty based on SWIFT model"""
    if freq_per_million <= 0:
        return alpha
    log_freq = np.log10(freq_per_million)
    difficulty = alpha * (1 - beta * (log_freq / F))
    return max(0, difficulty)

def create_sentences_dataset():
    """Create comprehensive sentences dataset combining word features"""
    # Setup paths
    raw_sentences_file = "/home/baiy4/ScanDL/scripts/data/raw_sentences.json"
    word_frequencies_file = "/home/baiy4/ScanDL/scripts/data/word_frequencies.json"
    word_predictabilities_file = "/home/baiy4/ScanDL/scripts/data/word_predictabilities.json"
    output_file = "/home/baiy4/ScanDL/scripts/data/sentences_dataset.json"
    
    print("Loading data files...")
    
    try:
        # Load raw sentences
        with open(raw_sentences_file, 'r', encoding='utf-8') as f:
            raw_sentences = json.load(f)
        
        # Load word frequencies
        with open(word_frequencies_file, 'r', encoding='utf-8') as f:
            word_frequencies = json.load(f)
        
        # Load word predictabilities
        with open(word_predictabilities_file, 'r', encoding='utf-8') as f:
            word_predictabilities = json.load(f)
        
        # Initialize dataset
        sentences_dataset = {}
        
        # Process each sentence
        for sentence_id, sentence_data in tqdm(raw_sentences.items(), desc="Processing sentences"):
            words = sentence_data['words']
            word_indices = sentence_data['word_indices']
            
            # Get predictabilities for this sentence
            pred_data = word_predictabilities.get(sentence_id, {})
            word_preds = pred_data.get('word_predictabilities', [])
            word_logit_preds = pred_data.get('word_logit_predictabilities', [])
            
            # Process each word
            processed_words = []
            for i, (word, word_idx) in enumerate(zip(words, word_indices)):
                # Clean word
                word_clean = clean_word(word)
                
                # Get word frequency info
                freq_info = word_frequencies.get(word_clean, {
                    'freq_per_million': 1.0,
                    'log_freq_per_million': 0.0
                })
                
                # Get word predictability
                predictability = word_preds[i] if i < len(word_preds) else 0.0
                logit_predictability = word_logit_preds[i] if i < len(word_logit_preds) else -2.553
                
                # Calculate word features
                word_features = {
                    'word_id': word_idx,
                    'word': word,
                    'word_clean': word_clean,
                    'length': len(word_clean),
                    'frequency': freq_info.get('freq_per_million', 1.0),
                    'log_frequency': freq_info.get('log_freq_per_million', 0.0),
                    'difficulty': calculate_word_difficulty(freq_info.get('freq_per_million', 1.0)),
                    'predictability': predictability,
                    'logit_predictability': logit_predictability
                }
                
                processed_words.append(word_features)
            
            # Store sentence information
            sentences_dataset[sentence_id] = {
                'sentence_content': sentence_data['sentence_content'],
                'words': processed_words
            }
        
        # Print summary
        print(f"\nProcessing complete!")
        print(f"Total sentences processed: {len(sentences_dataset)}")
        print(f"Total words across all sentences: {sum(len(s['words']) for s in sentences_dataset.values())}")
        
        # Print example
        print("\nExample sentence:")
        first_sent = list(sentences_dataset.values())[0]
        print(f"Sentence: {first_sent['sentence_content']}")
        print("Words with features:")
        for word in first_sent['words']:
            print(f"Word: {word['word']}")
            print(f"  Length: {word['length']}")
            print(f"  Frequency: {word['frequency']:.2f}")
            print(f"  Log Frequency: {word['log_frequency']:.2f}")
            print(f"  Difficulty: {word['difficulty']:.2f}")
            print(f"  Predictability: {word['predictability']:.3f}")
            print(f"  Logit Predictability: {word['logit_predictability']:.3f}")
            print()
        
        # Save results
        print(f"\nSaving dataset to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sentences_dataset, f, indent=2)
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    create_sentences_dataset() 