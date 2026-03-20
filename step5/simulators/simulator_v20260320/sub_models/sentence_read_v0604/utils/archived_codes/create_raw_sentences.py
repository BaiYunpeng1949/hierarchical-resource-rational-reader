import json
import pandas as pd
import os
from tqdm import tqdm

def create_raw_sentences():
    """Create raw_sentences.json from relations_labels_task2.csv"""
    # Setup paths
    input_file = "/home/baiy4/ScanDL/scripts/data/zuco/task_materials/relations_labels_task2.csv"
    output_file = "/home/baiy4/ScanDL/scripts/data/raw_sentences.json"
    
    print(f"Reading sentences from {input_file}")
    
    try:
        # Read CSV file
        df = pd.read_csv(input_file)
        
        # Initialize dictionary to store sentences
        sentences = {}
        
        # Process each row
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing sentences"):
            # Use row index as sentence ID
            sentence_id = str(idx)
            sentence_content = row['sentence']
            
            # Split sentence into words and create word indices
            words = sentence_content.split()
            word_indices = list(range(len(words)))
            
            # Store sentence information
            sentences[sentence_id] = {
                'sentence_content': sentence_content,
                'words': words,
                'word_indices': word_indices
            }
        
        # Print summary
        print(f"\nProcessing complete!")
        print(f"Total sentences processed: {len(sentences)}")
        print(f"Total words across all sentences: {sum(len(s['words']) for s in sentences.values())}")
        
        # Print example
        print("\nExample sentence:")
        first_sent = list(sentences.values())[0]
        print(f"Sentence: {first_sent['sentence_content']}")
        print("Words with indices:")
        for word, idx in zip(first_sent['words'], first_sent['word_indices']):
            print(f"{idx}: {word}")
        
        # Save results
        print(f"\nSaving sentences to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sentences, f, indent=2)
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    create_raw_sentences() 