import json
import os
import numpy as np
from tqdm import tqdm
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, KneserNeyInterpolated
from nltk.util import ngrams
from collections import defaultdict

def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('punkt')
        nltk.download('brown')  # We'll use Brown corpus for training
        print("Successfully downloaded NLTK data")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

def train_language_model():
    """Train a simple n-gram language model using NLTK"""
    print("Training language model...")
    
    # Get training data from Brown corpus
    from nltk.corpus import brown
    sentences = brown.sents()
    
    # Convert to lowercase and clean sentences
    sentences = [[word.lower() for word in sent] for sent in sentences]
    
    # Create and train trigram model with Kneser-Ney smoothing
    train_data, padded_sents = padded_everygram_pipeline(3, sentences)
    model = KneserNeyInterpolated(3)  # Using better smoothing
    model.fit(train_data, padded_sents)
    
    print(f"Model trained on {len(sentences)} sentences")
    return model

def calculate_word_predictability(model, words, word_index):
    """
    Calculate predictability of a word given all preceding words in the sentence
    Returns probability between 0 and 1
    """
    # Convert all words to lowercase
    words = [w.lower() for w in words]
    target_word = words[word_index]
    
    if word_index == 0:
        # For first word, use unigram probability
        try:
            prob = model.score(target_word)
        except:
            prob = 0.0001  # small default probability
    else:
        # Use all previous words as context
        context = words[:word_index]
        try:
            prob = model.score(target_word, context)
        except:
            prob = 0.0001
    
    # Ensure probability is between 0 and 1
    prob = max(0.0001, min(0.9999, prob))  # Clip to avoid log(0) or log(1)
    return float(prob)

def calculate_logit_predictability(pred):
    """
    Calculate logit predictability as defined in the paper:
    logit = 0.5 * ln(pred/(1-pred))
    For pred=0, use 1/(2*83)
    For pred=1, use (2*83-1)/(2*83)
    where 83 is the number of complete predictability protocols
    """
    N = 83  # number of complete predictability protocols
    
    # Handle edge cases
    if pred <= 0:
        pred = 1.0 / (2 * N)
    elif pred >= 1:
        pred = (2 * N - 1) / (2 * N)
    
    # Calculate logit
    logit = 0.5 * np.log(pred / (1 - pred))
    return float(logit)

def process_sentences(sentences, model):
    """Process sentences to get word predictabilities"""
    all_predictabilities = {}
    
    for sentence_id, sentence_data in tqdm(sentences.items(), desc="Processing sentences"):
        words = sentence_data['words']
        word_indices = sentence_data['word_indices']
        
        # Calculate predictability for each word
        word_predictabilities = []
        word_logit_predictabilities = []
        
        for i in range(len(words)):
            # Calculate probability given all preceding words
            pred = calculate_word_predictability(model, words, i)
            logit_pred = calculate_logit_predictability(pred)
            word_predictabilities.append(float(pred))
            word_logit_predictabilities.append(float(logit_pred))
        
        # Store results
        all_predictabilities[sentence_id] = {
            'sentence': sentence_data['sentence_content'],
            'words': words,
            'word_indices': word_indices,
            'word_predictabilities': word_predictabilities,
            'word_logit_predictabilities': word_logit_predictabilities
        }
    
    return all_predictabilities

def main():
    # Setup paths
    input_file = "/home/baiy4/ScanDL/scripts/data/raw_sentences.json"
    output_file = "/home/baiy4/ScanDL/scripts/data/word_predictabilities.json"
    
    # Download NLTK data and train model
    print("Setting up NLTK...")
    download_nltk_data()
    model = train_language_model()
    
    # Read raw sentences
    print(f"Reading sentences from {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
            
        # Process sentences
        print("\nCalculating word predictabilities...")
        predictabilities = process_sentences(sentences, model)
        
        # Validate results
        print("\nValidating results...")
        total_sentences = len(predictabilities)
        total_words = sum(len(sent['words']) for sent in predictabilities.values())
        avg_predictability = np.mean([
            np.mean(sent['word_predictabilities'])
            for sent in predictabilities.values()
        ])
        
        print(f"\nProcessing complete!")
        print(f"Total sentences processed: {total_sentences}")
        print(f"Total words processed: {total_words}")
        print(f"Average word predictability: {avg_predictability:.3f}")
        
        # Print example
        print("\nExample predictabilities for first sentence:")
        first_sent = list(predictabilities.values())[0]
        words = first_sent['words']
        preds = first_sent['word_predictabilities']
        for word, pred in zip(words, preds):
            print(f"{word}: {pred:.3f}")
        
        # Save results
        print(f"\nSaving predictabilities to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictabilities, f, indent=2)
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 