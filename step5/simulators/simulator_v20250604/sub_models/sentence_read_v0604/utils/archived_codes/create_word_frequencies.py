import pandas as pd
import json
import os
import urllib.request
import zipfile
import io
import numpy as np

def download_subtlex():
    """Download SUBTLEX-US word frequencies"""
    url = "https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/subtlexus2.zip/at_download/file"
    print("Downloading SUBTLEX-US word frequencies...")
    
    try:
        # Download the zip file
        response = urllib.request.urlopen(url)
        zip_data = io.BytesIO(response.read())
        
        # Extract the zip file
        with zipfile.ZipFile(zip_data) as zip_ref:
            zip_ref.extractall("temp_subtlex")
        
        # Read the tab-separated text file
        df = pd.read_csv("temp_subtlex/SUBTLEXus74286wordstextversion.txt", sep="\t")
        return df
        
    except Exception as e:
        print(f"Error downloading SUBTLEX: {str(e)}")
        # Try to read local file if it exists
        try:
            if os.path.exists("temp_subtlex/SUBTLEXus74286wordstextversion.txt"):
                print("Found local SUBTLEX file, reading it...")
                df = pd.read_csv("temp_subtlex/SUBTLEXus74286wordstextversion.txt", sep="\t")
                return df
        except Exception as e2:
            print(f"Error reading local file: {str(e2)}")
        
        print("Using backup method: Please download manually from:")
        print("https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus")
        return None

def calculate_log_frequency(freq_per_million):
    """Calculate log10 of frequency per million, handling zero values"""
    if freq_per_million <= 0:
        return 0  # or some small value like np.log10(0.5)
    return np.log10(freq_per_million)

def process_frequencies(df):
    """Process SUBTLEX frequencies into our format"""
    # Create dictionary with word frequencies and other metrics
    frequencies = {}
    
    print("\nProcessing frequencies...")
    print(f"Columns available: {df.columns.tolist()}")
    print(f"Total entries to process: {len(df)}")
    
    skipped = 0
    processed = 0
    
    for _, row in df.iterrows():
        # Skip if word is NaN or not a string
        if pd.isna(row['Word']) or not isinstance(row['Word'], str):
            skipped += 1
            continue
            
        word = row['Word'].lower().strip()  # Convert to lowercase and remove whitespace
        
        # Skip empty strings or single characters (except 'a' and 'i')
        if not word or (len(word) == 1 and word not in ['a', 'i']):
            skipped += 1
            continue
        
        try:
            freq = int(row['FREQcount'])
            cd = int(row['CDcount'])
            freq_per_million = float(row['SUBTLWF'])
            
            # Store multiple frequency measures
            frequencies[word] = {
                'freq_count': freq,                    # Raw frequency count
                'contextual_diversity': cd,            # Number of unique movies/TV shows the word appears in
                'cd_percent': float(row['SUBTLCD']),  # Percentage of movies/TV shows containing the word
                'freq_per_million': freq_per_million,  # Frequency per million words (like in the paper)
                'log_freq_per_million': calculate_log_frequency(freq_per_million),  # Log10 of freq per million (paper's approach)
                'length': len(word)                    # Word length (useful for analysis)
            }
            processed += 1
            
        except (ValueError, TypeError):
            skipped += 1
            continue
    
    print(f"\nProcessed {processed} words successfully")
    print(f"Skipped {skipped} invalid entries")
    
    return frequencies

def main():
    output_dir = "/home/baiy4/ScanDL/scripts/data"
    output_file = os.path.join(output_dir, "word_frequencies.json")
    
    # Try to download SUBTLEX
    df = download_subtlex()
    
    if df is None:
        print("\nAlternative options for word frequencies:")
        print("1. SUBTLEX-US (manual download)")
        print("2. British National Corpus (via NLTK)")
        print("3. Google Books Ngram")
        print("\nPlease download one of these and convert to JSON format:")
        print("{'word': frequency, ...}")
        return
    
    # Process frequencies
    frequencies = process_frequencies(df)
    
    # Save to JSON
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(frequencies, f, indent=2)
    
    print(f"\nSaved word frequencies to {output_file}")
    print(f"Total words: {len(frequencies)}")
    
    # Print some statistics
    freq_values = [d['freq_per_million'] for d in frequencies.values()]
    log_freq_values = [d['log_freq_per_million'] for d in frequencies.values()]
    cd_values = [d['contextual_diversity'] for d in frequencies.values()]
    
    print("\nFrequency per million statistics:")
    print(f"Mean: {np.mean(freq_values):.2f}")
    print(f"Median: {np.median(freq_values):.2f}")
    print(f"Max: {max(freq_values):.2f}")
    print(f"Min: {min(freq_values):.2f}")
    
    print("\nLog10 frequency per million statistics:")
    print(f"Mean: {np.mean(log_freq_values):.3f}")
    print(f"Median: {np.median(log_freq_values):.3f}")
    print(f"Max: {max(log_freq_values):.3f}")
    print(f"Min: {min(log_freq_values):.3f}")
    
    print("\nContextual diversity statistics:")
    print(f"Mean movies/shows: {np.mean(cd_values):.2f}")
    print(f"Median movies/shows: {np.median(cd_values):.2f}")
    print(f"Max movies/shows: {max(cd_values)}")
    print(f"Min movies/shows: {min(cd_values)}")
    
    # Print some example entries
    print("\nExample entries (top 10 most frequent words):")
    sorted_words = sorted(frequencies.items(), key=lambda x: x[1]['freq_per_million'], reverse=True)[:10]
    for word, stats in sorted_words:
        print(f"{word}:")
        print(f"  Length: {stats['length']}")
        print(f"  Frequency per million: {stats['freq_per_million']:.2f}")
        print(f"  Log10 frequency: {stats['log_freq_per_million']:.3f}")
        print(f"  Appears in {stats['contextual_diversity']} movies/shows ({stats['cd_percent']:.1f}%)")

if __name__ == "__main__":
    main() 