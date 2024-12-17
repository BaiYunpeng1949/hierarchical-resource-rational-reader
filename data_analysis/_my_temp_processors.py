import json
import pandas as pd
import json
import os
import re

def remove_participants_from_scanpath_data(scanpath_data_path, participant_indices_to_remove, save_path):
    # Step 1: Read the JSON file
    with open(scanpath_data_path, 'r') as file:
        data = json.load(file)

    # Step 2: Filter out entries with participant_index in participant_indices_to_remove
    filtered_data = [
        entry for entry in data
        if entry.get('participant_index') not in participant_indices_to_remove
    ]

    # Step 3: Write the filtered data back to a JSON file
    with open(save_path, 'w') as file:
        json.dump(filtered_data, file, indent=4)
# # Step 1: Read the JSON file
# with open('/home/baiy4/reading-model/data_analysis/human_data/processed_data/11_05_19_13/processed_human_scanpath_wo_p1_to_p4.json', 'r') as file:
#     data = json.load(file)

# # Step 2: Filter out entries with participant_index in [1, 2, 3, 4]
# participant_indices_to_remove = [1, 2, 3, 4]
# filtered_data = [
#     entry for entry in data
#     if entry.get('participant_index') not in participant_indices_to_remove
# ]

# # Step 3: Write the filtered data back to a JSON file
# with open('/home/baiy4/reading-model/data_analysis/human_data/processed_data/11_05_19_13/processed_human_scanpath_wo_p1_to_p4.json', 'w') as file:
#     json.dump(filtered_data, file, indent=4)

# ********************************************************************************************************************************************

def convert_human_comprehension_csv_to_json(csv_path, corpus_json_path, comprehension_answers_json_path, save_path):

    # Function to map 'Stim ID' to 'stimulus_index'
    def stim_id_to_index(stim_id):
        # Extract the number from 'stimuliX' and subtract 1 to get index
        stim_number = int(stim_id.replace('stimuli', ''))
        return stim_number - 1  # Assuming stimuli1 corresponds to index 0

    # Function to clean text by removing non-ASCII characters
    def clean_text(text):
        # Remove non-ASCII characters
        cleaned_text = text.encode('ascii', 'ignore').decode('ascii')
        return cleaned_text

    # Get the dir name
    dir_name = '/home/baiy4/reading-model/data_analysis/human_data/comprehension_data/'

    # Read the CSV file
    df = pd.read_csv(os.path.join(dir_name, 'comprehension_raw_data_p1_to_p32.csv'))  #, sep='\t')

    # Read the corpus JSON file
    with open('/home/baiy4/reading-model/step5/data/gen_envs/10_27_15_58_100_images_W1920H1080WS16_LS40_MARGIN400/simulate/metadata.json', 'r') as f:
        corpus_data = json.load(f)

    corpus = corpus_data['config']['corpus']

    # Read the comprehension answers JSON file
    with open('/home/baiy4/reading-model/step5/data/assets/MCQ/mcq_metadata.json', 'r') as f:
        comprehension_answers = json.load(f)

    processed_data = []

    for index, row in df.iterrows():
        participant_id = row['Participant ID']
        stim_id = row['Stim ID']
        trial_condition_raw = row['Trial Condition']
        mcq_score = float(row['MCQ'])  # Assuming it's a float representing proportion correct
        recall_score = row['recall score']
        free_recall_answer_raw = row['recall']
        
        # Map 'Stim ID' to 'stimulus_index'
        stimulus_index = stim_id_to_index(stim_id)
        
        # Get 'words_in_section' from 'corpus' JSON
        words_in_section = corpus.get(str(stimulus_index + 1), '')
        
        # Get 'mcq_logs' from 'comprehension_answers' JSON
        mcq_data = comprehension_answers.get(str(stimulus_index), {})
        
        # Create 'mcq_logs'
        mcq_logs = []
        total_questions = len(mcq_data)
        num_correct = int(mcq_score * total_questions + 0.5)  # Round to nearest integer
        # Since we don't have participant's individual answers, we can assume they got the first N correct
        # This is an assumption; in reality, you would need the individual answers
        question_indices = list(mcq_data.keys())
        question_indices.sort(key=int)
        
        # Initialize 'answer' and 'correct_answer' for each question
        for i, mcq_idx in enumerate(question_indices):
            question_info = mcq_data[mcq_idx]
            correct_answer = question_info['correct_answer']
            if i < num_correct:
                answer = correct_answer  # Assume participant got this one correct
            else:
                answer = 'E'  # Assume participant did not know/cannot remember
            mcq_log = {
                'mcq_idx': mcq_idx,
                'answer': answer,
                'correct_answer': correct_answer
            }
            mcq_logs.append(mcq_log)
        
        # # Extract time_constraint from 'Trial Condition', e.g., 'A.30s'
        # time_constraint_str = trial_condition.split('.')[1]
        # time_constraint = int(time_constraint_str.replace('s', ''))
        # Extract time_constraint from 'Trial Condition', e.g., 'A.30s'
        # Clean 'trial_condition' to contain only the time constraint number
        # Handle cases where 'Trial Condition' might be malformed
        trial_condition_match = re.search(r'\.(\d+)s', trial_condition_raw)
        if trial_condition_match:
            time_constraint = int(trial_condition_match.group(1))
        else:
            # Default value or handle error
            time_constraint = None  # or assign a default value like 30
            print(f"Warning: Could not parse time constraint from '{trial_condition_raw}'")

        # Update trial_condition to be clean
        trial_condition = str(time_constraint) if time_constraint is not None else ''
        
        # Clean the free_recall_answer by removing non-ASCII characters
        free_recall_answer = clean_text(free_recall_answer_raw)
        
        # Create the episodic_info structure
        episodic_info = {
            'episode_index': index,  # Or use another index if appropriate
            'participant_id': participant_id,
            'trial_condition': trial_condition,  # Now contains only the time constraint
            'stimulus': {
                'stimulus_index': stimulus_index,
                'words_in_section': words_in_section,
                'stimulus_width': 1920,
                'stimulus_height': 1080
            },
            'task': {
                'time_constraint': int(time_constraint),
                'task_type': 'comprehension'
            },
            'mcq_logs': mcq_logs,
            'free_recall_answer': free_recall_answer
        }
        
        # Append to processed_data
        processed_data.append({'episodic_info': episodic_info})

    # Write to JSON file
    save_file_path = os.path.join(dir_name, 'processed_human_comprehension_data_p1_to_p32.json')
    with open(save_file_path, 'w') as f:
        json.dump(processed_data, f, indent=4)
    

# ============================================================================================================================================
def levenshtein_distance(seq1, seq2):
    """
    Compute the Levenshtein distance between two sequences.

    Parameters:
    - seq1: First sequence (list of integers).
    - seq2: Second sequence (list of integers).

    Returns:
    - dist: The Levenshtein distance (int).
    """
    len_seq1 = len(seq1)
    len_seq2 = len(seq2)

    # Initialize matrix
    dp = [[0] * (len_seq2 + 1) for _ in range(len_seq1 + 1)]
    for i in range(len_seq1 + 1):
        dp[i][0] = i
    for j in range(len_seq2 + 1):
        dp[0][j] = j

    # Compute distances
    for i in range(1, len_seq1 + 1):
        for j in range(1, len_seq2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    return dp[len_seq1][len_seq2]

def normalized_levenshtein_distance(seq1, seq2):
    """
    Compute the Normalized Levenshtein Distance between two sequences.

    Parameters:
    - seq1: First sequence (list of integers).
    - seq2: Second sequence (list of integers).

    Returns:
    - nld: Normalized Levenshtein distance (float between 0 and 1).
    """
    lev_dist = levenshtein_distance(seq1, seq2)
    max_len = max(len(seq1), len(seq2))
    if max_len == 0:
        return 0.0  # Both sequences are empty
    nld = lev_dist / max_len
    return nld

def calculate_nld_using_word_indices(sequence1, sequence2):
    """
    Calculate the Normalized Levenshtein Distance (NLD) using word indices.

    Parameters:
    - sequence1: First sequence of word indices (list of integers).
    - sequence2: Second sequence of word indices (list of integers).

    Returns:
    - nld: Normalized Levenshtein distance (float between 0 and 1).
    """
    nld = normalized_levenshtein_distance(sequence1, sequence2)
    return nld
# ============================================================================================================================================

if __name__ == '__main__':
    
    # Test the functions of normalized levenshtein distance
    seq1 = [0, 4, 5, 7, 9, 12, 14, 19]
    seq2 = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10]

    print(calculate_nld_using_word_indices(seq1, seq2))