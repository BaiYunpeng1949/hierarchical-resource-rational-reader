import json
import os
import numpy as np
import string
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import KneserNeyInterpolated
from tqdm import tqdm
import urllib.request
import zipfile
import io
import pandas as pd

def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('punkt')
        nltk.download('brown')
        print("Successfully downloaded NLTK data")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

def download_subtlex():
    """Download SUBTLEX-US word frequencies"""
    url = "https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/subtlexus2.zip/at_download/file"
    print("Downloading SUBTLEX-US word frequencies...")
    
    try:
        response = urllib.request.urlopen(url)
        zip_data = io.BytesIO(response.read())
        
        with zipfile.ZipFile(zip_data) as zip_ref:
            zip_ref.extractall("temp_subtlex")
        
        df = pd.read_csv("temp_subtlex/SUBTLEXus74286wordstextversion.txt", sep="\t")
        return df
        
    except Exception as e:
        print(f"Error downloading SUBTLEX: {str(e)}")
        try:
            if os.path.exists("temp_subtlex/SUBTLEXus74286wordstextversion.txt"):
                print("Found local SUBTLEX file, reading it...")
                df = pd.read_csv("temp_subtlex/SUBTLEXus74286wordstextversion.txt", sep="\t")
                return df
        except Exception as e2:
            print(f"Error reading local file: {str(e2)}")
        return None

def process_frequencies(df):
    """Process SUBTLEX frequencies into our format"""
    frequencies = {}
    
    print("\nProcessing frequencies...")
    skipped = 0
    processed = 0
    
    for _, row in df.iterrows():
        if pd.isna(row['Word']) or not isinstance(row['Word'], str):
            skipped += 1
            continue
            
        word = row['Word'].lower().strip()
        
        if not word or (len(word) == 1 and word not in ['a', 'i']):
            skipped += 1
            continue
        
        try:
            freq = int(row['FREQcount'])
            cd = int(row['CDcount'])
            freq_per_million = float(row['SUBTLWF'])
            
            frequencies[word] = {
                'freq_count': freq,
                'contextual_diversity': cd,
                'cd_percent': float(row['SUBTLCD']),
                'freq_per_million': freq_per_million,
                'log_freq_per_million': np.log10(freq_per_million) if freq_per_million > 0 else 0,
                'length': len(word)
            }
            processed += 1
            
        except (ValueError, TypeError):
            skipped += 1
            continue
    
    print(f"Processed {processed} words successfully")
    print(f"Skipped {skipped} invalid entries")
    
    return frequencies

def train_language_model():
    """Train a trigram language model using NLTK"""
    print("Training language model...")
    
    from nltk.corpus import brown
    sentences = brown.sents()
    sentences = [[word.lower() for word in sent] for sent in sentences]
    
    train_data, padded_sents = padded_everygram_pipeline(3, sentences)
    model = KneserNeyInterpolated(3)
    model.fit(train_data, padded_sents)
    
    print(f"Model trained on {len(sentences)} sentences")
    return model

def calculate_word_difficulty(freq_per_million, alpha=1.0, beta=1.0, F=11):
    """Calculate word difficulty based on SWIFT model"""
    if freq_per_million <= 0:
        return alpha
    log_freq = np.log10(freq_per_million)
    difficulty = alpha * (1 - beta * (log_freq / F))
    return max(0, difficulty)

def calculate_word_predictability(model, words, word_index):
    """Calculate predictability of a word given preceding words"""
    words = [w.lower() for w in words]
    target_word = words[word_index]
    
    if word_index == 0:
        try:
            prob = model.score(target_word)
        except:
            prob = 0.0001
    else:
        context = words[:word_index]
        try:
            prob = model.score(target_word, context)
        except:
            prob = 0.0001
    
    prob = max(0.0001, min(0.9999, prob))
    return float(prob)

def calculate_logit_predictability(pred, N=83):
    """Calculate logit predictability"""
    if pred <= 0:
        pred = 1.0 / (2 * N)
    elif pred >= 1:
        pred = (2 * N - 1) / (2 * N)
    
    logit = 0.5 * np.log(pred / (1 - pred))
    return float(logit)

def clean_word(word):
    """Clean word by removing punctuation and converting to lowercase"""
    word = word.lower().strip()
    word = ''.join(char for char in word if char not in string.punctuation)
    return word

def process_stimulus_data(stimulus_file, metadata_file, output_file):
    """Process stimulus data into word-level features dataset"""
    print("Loading data files...")
    
    # Load stimulus and metadata
    with open(stimulus_file, 'r', encoding='utf-8') as f:
        stimulus_data = json.load(f)
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Download and process word frequencies
    df = download_subtlex()
    if df is None:
        raise Exception("Failed to load word frequencies")
    word_frequencies = process_frequencies(df)
    
    # Train language model
    download_nltk_data()
    model = train_language_model()
    
    # Process each stimulus
    processed_data = {}
    
    for stimulus in tqdm(stimulus_data, desc="Processing stimuli"):
        stimulus_id = stimulus['stimulus_id']
        metadata_entry = next((m for m in metadata if m['stimulus_id'] == stimulus_id), None)
        
        if metadata_entry is None:
            print(f"Warning: No metadata found for stimulus {stimulus_id}")
            continue
        
        # Process each sentence
        processed_sentences = []
        for sentence in metadata_entry['sentences']:
            words = sentence['sentence'].split()
            processed_words = []
            
            for i, word in enumerate(words):
                word_clean = clean_word(word)
                
                # Get word frequency info
                freq_info = word_frequencies.get(word_clean, {
                    'freq_per_million': 1.0,
                    'log_freq_per_million': 0.0
                })
                
                # Calculate word features
                pred = calculate_word_predictability(model, words, i)
                logit_pred = calculate_logit_predictability(pred)
                
                word_features = {
                    'word_id': i,
                    'word': word,
                    'word_clean': word_clean,
                    'length': len(word_clean),
                    'frequency': freq_info.get('freq_per_million', 1.0),
                    'log_frequency': freq_info.get('log_freq_per_million', 0.0),
                    'difficulty': calculate_word_difficulty(freq_info.get('freq_per_million', 1.0)),
                    'predictability': pred,
                    'logit_predictability': logit_pred
                }
                
                processed_words.append(word_features)
            
            processed_sentences.append({
                'sentence_idx': sentence['sentence_idx'],
                'sentence': sentence['sentence'],
                'words': processed_words
            })
        
        processed_data[stimulus_id] = {
            'stimulus_id': stimulus_id,
            'text_content': stimulus['text_content'],
            'sentences': processed_sentences
        }
    
    # Save processed data
    print(f"\nSaving processed data to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)
    
    print("Processing complete!")
    print(f"Total stimuli processed: {len(processed_data)}")
    
    # Print example
    print("\nExample processed stimulus:")
    first_stimulus = list(processed_data.values())[0]
    print(f"Stimulus ID: {first_stimulus['stimulus_id']}")
    print("\nFirst sentence with word features:")
    first_sentence = first_stimulus['sentences'][0]
    print(f"Sentence: {first_sentence['sentence']}")
    print("\nWord features:")
    for word in first_sentence['words'][:3]:  # Show first 3 words
        print(f"\nWord: {word['word']}")
        print(f"  Length: {word['length']}")
        print(f"  Frequency: {word['frequency']:.2f}")
        print(f"  Log Frequency: {word['log_frequency']:.2f}")
        print(f"  Difficulty: {word['difficulty']:.2f}")
        print(f"  Predictability: {word['predictability']:.3f}")
        print(f"  Logit Predictability: {word['logit_predictability']:.3f}")

if __name__ == "__main__":

    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    parent_dir = os.path.dirname(current_dir)
    
    # Setup paths
    stimulus_file = os.path.join(parent_dir, "assets", "processed_my_stimulus_for_text_reading.json")
    metadata_file = os.path.join(parent_dir, "assets", "metadata_sentence_indeces.json")
    output_file = os.path.join(parent_dir, "assets", "processed_my_stimulus_with_word_features.json")
    
    process_stimulus_data(stimulus_file, metadata_file, output_file)
