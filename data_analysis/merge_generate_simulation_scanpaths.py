import json
import step5.utils.auxiliaries as aux
import pandas as pd
import os
import math
import constants as const
import numpy as np

class ReadSimulationScanpath:

    def __init__(self, sim_raw_data_json_dir):
        """
        Initialize paths and load raw simulation data.
        """
        self._sim_raw_data_json_dir = sim_raw_data_json_dir
        self._sim_raw_data_json_path = os.path.join(sim_raw_data_json_dir, 'simulate_xep_word_level.json')
        self._sim_processed_scanpath_json_path = os.path.join(sim_raw_data_json_dir, 'sim_processed_scanpath.json')
    
    def read_and_process_scanpath(self,):
        """
        Read simulation scanpath data, process it to extract necessary information,
        convert coordinates to pixels, and save cleaned data in a new JSON file.
        """
        # Load raw data
        raw_data = load_data(self._sim_raw_data_json_path)

        # Initialize a list to store different trials fixation data
        processed_fixation_data_across_trials = []
        processed_merge_filter_fixation_data_across_trials = []

        # Create the 'trial_wise_scanpaths' directory
        trial_scanpaths_dir = os.path.join(self._sim_raw_data_json_dir, 'simulation_trial_wise_scanpaths')
        os.makedirs(trial_scanpaths_dir, exist_ok=True)
        
        for trial_idx, trial in enumerate(raw_data):
            episode_index = trial.get('episode_index')
            episodic_info = trial.get('episodic_info', {})
            stimulus = episodic_info.get('stimulus', {})
            stimulus_index = stimulus.get('stimulus_index')
            stimulus_width = stimulus.get('stimulus_width')
            stimulus_height = stimulus.get('stimulus_height')
            time_constraint = episodic_info['task']['time_constraint']

            # ==================================================================================================
            # Reset a dictionary to store processed fixation data
            trial_wise_metadata = {
                'stimulus_index': stimulus_index,
                'episode_index': episode_index,
                'sim_trial_index': trial_idx,
                'time_constraint': time_constraint,
                'fixation_data': []
            }

            # Extract fixation points
            word_level_steps = episodic_info.get('word_level_steps', [])
            fixation_points = get_fixation_points(word_level_steps)

            # Map normalized fixation points to pixel coordinates using the normalise function
            pixel_fixation_points = map_to_pixel_coordinates(fixation_points, stimulus_width, stimulus_height)

            # Store each fixation point with required details
            for fixation in pixel_fixation_points:
                trial_wise_metadata['fixation_data'].append({
                    'word_index': fixation['word_index'],
                    'fix_x': fixation['fix_x'],
                    'fix_y': fixation['fix_y'],
                    'norm_fix_x': fixation['norm_fix_x'],
                    'norm_fix_y': fixation['norm_fix_y'],
                })

            # Append processed fixation data to the list
            processed_fixation_data_across_trials.append(trial_wise_metadata)
            
            # ==================================================================================================
            # Merge consecutive fixations on the same word within the distance threshold
            merged_fixations = merge_fixations(pixel_fixation_points, distance_threshold=const.MERGE_PX_THRESHOLD, stimulus_width=stimulus_width, stimulus_height=stimulus_height)
            
            # Reset a dictionary to store processed fixation data
            trial_wise_merge_filter_metadata = {
                'stimulus_index': stimulus_index,
                'episode_index': episode_index,
                'sim_trial_index': trial_idx,
                'time_constraint': episodic_info['task']['time_constraint'],
                'fixation_data': []
            }

            trial_wise_merge_fix8_data = {
                "fixations": [
                    [fixation['fix_x'], fixation['fix_y'], fixation['fix_duration']] for fixation in merged_fixations
                ],
            }
            
            # Store each fixation point with required details
            for fixation in merged_fixations:
                trial_wise_merge_filter_metadata['fixation_data'].append({
                    'word_index': fixation['word_index'],
                    'fix_x': fixation['fix_x'],
                    'fix_y': fixation['fix_y'],
                    'norm_fix_x': fixation['norm_fix_x'],
                    'norm_fix_y': fixation['norm_fix_y'],
                })

            # Append processed fixation data to the list
            processed_merge_filter_fixation_data_across_trials.append(trial_wise_merge_filter_metadata)

            # Save individual trial scanpath data to JSON file for merge-filtered data
            filename = f"trial{trial_idx}_stimid{stimulus_index}_tc{episodic_info['task']['time_constraint']}_merged_scanpath.json"
            output_file_path = os.path.join(trial_scanpaths_dir, filename)
            with open(output_file_path, 'w') as f:
                json.dump(trial_wise_merge_fix8_data, f, indent=4)
            print(f"Merge filtered scanpath data for trial {trial_idx}, stimulus_index {stimulus_index}, and time constraint {trial_wise_merge_filter_metadata['time_constraint']} saved to {output_file_path}")

            
        # Save processed data to a JSON file -- original
        with open(self._sim_processed_scanpath_json_path, 'w') as f:
            json.dump(processed_fixation_data_across_trials, f, indent=4)
        # Get the original average number of fixations per trial
        avg_fixations_per_trial = np.round(sum([len(trial['fixation_data']) for trial in processed_fixation_data_across_trials]) / len(processed_fixation_data_across_trials), 2)
        
        # Save processed data to a JSON file -- merge filter
        merge_filter_file_path = self._sim_processed_scanpath_json_path.replace('.json', '_merge_filter.json')
        with open(merge_filter_file_path, 'w') as f:
            json.dump(processed_merge_filter_fixation_data_across_trials, f, indent=4)
        # Get the average number of fixations per trial after merge filter
        avg_fixations_per_trial_merge_filter = np.round(sum([len(trial['fixation_data']) for trial in processed_merge_filter_fixation_data_across_trials]) / len(processed_merge_filter_fixation_data_across_trials), 2)

        print(f"Original processed scanpath data saved to {self._sim_processed_scanpath_json_path}")
        print(f"Merge filter processed scanpath data saved to {merge_filter_file_path}. \n      The overall number of fixations reduced from {avg_fixations_per_trial} to {avg_fixations_per_trial_merge_filter}")


# Helper functions
# ------------------------------------------------------------------------------------------------------------------------------
def load_data(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data

def get_fixation_points(word_level_steps, max_fixations=None):
    fixation_points = []
    for step in word_level_steps:
        fixation_info = step.get('fixation_info', {})
        norm_fix_x = fixation_info.get('norm_fix_x')
        norm_fix_y = fixation_info.get('norm_fix_y')
        fix_duration = fixation_info.get('fixation_duration', 200)  # Assuming a default duration if not provided
        sampled_letters = fixation_info.get('sampled_letters', 'NA')
        is_terminate_step = fixation_info.get('is_terminate_step', False)
        sentence_level_steps = step.get('sentence_level_steps', 'NA')
        word_step = step.get('word_level_or_fixation_steps', 'NA')
        word_index = step.get('target_word_index_in_stimulus', 'NA')  # Store the word index

        # Exclude fixation points where 'sampled_letters' is empty or 'NA' or where 'is_terminate_step' is True
        if sampled_letters not in ['NA', ''] and not is_terminate_step:
            if norm_fix_x is not None and norm_fix_y is not None:
                # Include coordinates along with sentence, word level steps, and word index
                fixation_points.append({
                    'norm_fix_x': norm_fix_x,
                    'norm_fix_y': norm_fix_y,
                    'sentence_step': sentence_level_steps,
                    'word_step': word_step,
                    'word_index': word_index,  # Include word index
                    'fix_duration': fix_duration
                })
                # Break if we've reached the maximum number of fixations
                if max_fixations is not None and len(fixation_points) >= max_fixations:
                    break
    return fixation_points

def map_to_pixel_coordinates(fixation_points, image_width, image_height):
    pixel_points = []
    for fixation in fixation_points:
        norm_x = fixation['norm_fix_x']
        norm_y = fixation['norm_fix_y']
        sentence_step = fixation['sentence_step']
        word_step = fixation['word_step']
        word_index = fixation['word_index']  # Store the word index
        fix_duration = fixation['fix_duration']

        # Convert normalized coordinates to pixel coordinates
        x_pixel = aux.normalise(norm_x, -1, 1, 0, image_width)
        y_pixel = aux.normalise(norm_y, -1, 1, 0, image_height)
        
        # Append pixel points along with sentence and word steps for annotations
        pixel_points.append({
            'fix_x': x_pixel,
            'fix_y': y_pixel,
            'norm_fix_x': aux.normalise(norm_x, -1, 1, 0, 1),       # Note: normalize from -1 to 1 to 0 to 1 to better compute scanpath similarity metrics, such as ScanMatch
            'norm_fix_y': aux.normalise(norm_y, -1, 1, 0, 1),
            'sentence_step': sentence_step,
            'word_step': word_step,
            'word_index': word_index,
            'fix_duration': fix_duration
        })
    return pixel_points

def euclidean_distance(fix1, fix2):
    return math.hypot(fix1['fix_x'] - fix2['fix_x'], fix1['fix_y'] - fix2['fix_y'])

def merge_fixations(fixations, distance_threshold, stimulus_width, stimulus_height):
    """
    Merge consecutive fixations on the same word if they are within the distance threshold.
    """
    if not fixations:
        return []

    merged_fixations = []
    current_fixation = fixations[0]

    for next_fixation in fixations[1:]:
        # Check if the next fixation is on the same word
        if next_fixation['word_index'] == current_fixation['word_index']:
            # Check if the distance is within the threshold
            distance = euclidean_distance(current_fixation, next_fixation)
            if distance <= distance_threshold:
                # Merge fixations
                total_duration = current_fixation['fix_duration'] + next_fixation['fix_duration']
                # Weighted average of positions based on duration
                fix_x = (current_fixation['fix_x'] * current_fixation['fix_duration'] + next_fixation['fix_x'] * next_fixation['fix_duration']) / total_duration
                fix_y = (current_fixation['fix_y'] * current_fixation['fix_duration'] + next_fixation['fix_y'] * next_fixation['fix_duration']) / total_duration
                # Update current fixation
                current_fixation = {
                    'fix_x': fix_x,
                    'fix_y': fix_y,
                    'norm_fix_x': aux.normalise(fix_x, 0, stimulus_width, 0, 1),
                    'norm_fix_y': aux.normalise(fix_y, 0, stimulus_height, 0, 1),
                    'word_index': current_fixation['word_index'],
                    'fix_duration': total_duration
                }
            else:
                # Add the current fixation to the list and move to the next
                merged_fixations.append(current_fixation)
                current_fixation = next_fixation
        else:
            # Different word, add the current fixation and move to the next
            merged_fixations.append(current_fixation)
            current_fixation = next_fixation

    # Add the last fixation
    merged_fixations.append(current_fixation)
    return merged_fixations


if __name__ == '__main__':
    
    # Process simulation scanpath data
    sim_data_dir = '/home/baiy4/reading-model/data_analysis/simulation_data/sim_raw_data_11_09_17_09_1episodes/stimulus_8_time_constraint_90s'
    # sim_data_dir = '/home/baiy4/reading-model/data_analysis/simulation_data/sim_raw_data_11_26_12_11_1episodes/stimulus_8_time_constraint_90s'
    read_sim_scanpath = ReadSimulationScanpath(sim_raw_data_json_dir=sim_data_dir)
    read_sim_scanpath.read_and_process_scanpath()