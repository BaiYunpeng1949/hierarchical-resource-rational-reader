import os
import json
import re
import constants as const

from step5.utils import auxiliaries as aux

def load_bbox_metadata(metadata_file):
    """Load the bounding box metadata from the JSON file."""
    with open(metadata_file, 'r') as f:
        bbox_metadata = json.load(f)
    return bbox_metadata

def get_trial_metadata(stimulus_index, bbox_metadata):    
    """Get the trial metadata based on the stimulus_index."""
    # Adjust stimulus_index if necessary (e.g., add 1 if image indices start from 1)
    image_index = stimulus_index    # Do not minus one, always check the metadata yourself
    for img_meta in bbox_metadata['images']:
        if img_meta['image index'] == image_index:
            return img_meta
    return None

def get_word_index_from_fixation(fixation_entry, trial_metadata):
    x = fixation_entry['fix_x']
    y = fixation_entry['fix_y']
    min_distance = float('inf')
    nearest_word_idx = -1
    for word_idx, word_info in enumerate(trial_metadata['words metadata']):
        word_bbox = word_info['word_bbox']  # [x_min, y_min, x_max, y_max]

        # Check if the fixation is within the word bbox
        if word_bbox[0] <= x <= word_bbox[2] and word_bbox[1] <= y <= word_bbox[3]:
            return word_idx
        else:
            # Calculate the Euclidean distance from the fixation to the bbox
            dx = max(word_bbox[0] - x, 0, x - word_bbox[2])
            dy = max(word_bbox[1] - y, 0, y - word_bbox[3])
            distance = (dx ** 2 + dy ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_word_idx = word_idx
    # Assign the fixation to the nearest word
    return nearest_word_idx

def aggregate_scanpaths(directory, output_file, bbox_metadata):
    # Initialize the list to hold all scanpath data
    aggregated_data = []

    # Regular expression to match filenames and extract participant and stimulus indices
    # filename_pattern = re.compile(r'p(\d+)_stimid(\d+)_tc(\d+)_scanpath_CORRECTED\.json')
    # filename_pattern = re.compile(r'p(\d+)_stimid(\d+)_tc(\d+)_scanpath\.json')
    # filename_pattern = re.compile(r'p(\d+)_stimid(\d+)_tc(\d+)_scanpath_warp_CORRECTED\.json')
    # filename_pattern = re.compile(r'p(\d+)_stimid(\d+)_tc(\d+)_scanpath_stretch_CORRECTED\.json')
    filename_pattern = re.compile(r'trial(\d+)_stimid(\d+)_tc(\d+)_merged_scanpath_CORRECTED\.json')      # uncomment this for simulation data
    # filename_pattern = re.compile(r'p(\d+)_stimid(\d+)_tc(\d+)_scanpath_(.+?)_CORRECTED\.json')                 # uncomment this for human data

    # Traverse the directory
    for filename in os.listdir(directory):
        match = filename_pattern.match(filename)
        if match:
            participant_index = int(match.group(1))
            stimulus_index = int(match.group(2))
            time_constraint = int(match.group(3))

            # Construct the full path to the file
            file_path = os.path.join(directory, filename)

            # Read the JSON data from the file
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract the fixations
            fixations = data.get('fixations', [])

            # Get the trial metadata for the current stimulus
            trial_metadata = get_trial_metadata(stimulus_index, bbox_metadata)
            if not trial_metadata:
                print(f"No metadata found for stimulus index {stimulus_index}. Skipping file {filename}.")
                continue

            # Build the fixation_data list
            fixation_data = []
            for fixation in fixations:
                fix_x = fixation[0]
                fix_y = fixation[1]
                fix_duration = fixation[2]

                fixation_entry = {
                    "fix_x": fix_x,
                    "fix_y": fix_y,
                    "norm_fix_x": aux.normalise(fix_x, 0, const.SCREEN_RESOLUTION_WIDTH_PX, 0, 1),
                    "norm_fix_y": aux.normalise(fix_y, 0, const.SCREEN_RESOLUTION_HEIGHT_PX, 0, 1),
                    "fix_duration": fix_duration
                }

                # Calculate the word index
                word_index = get_word_index_from_fixation(fixation_entry, trial_metadata)
                fixation_entry["word_index"] = word_index

                fixation_data.append(fixation_entry)

            # Create the entry for this participant and stimulus
            scanpath_entry = {
                "stimulus_index": stimulus_index,
                "participant_index": participant_index,
                "time_constraint": time_constraint,
                "fixation_data": fixation_data
            }

            # Add the entry to the aggregated data list
            aggregated_data.append(scanpath_entry)

            print(f"Processed file: {filename}")

    # Write the aggregated data to the output file
    file_path = os.path.join(directory, output_file)
    with open(file_path, 'w') as f:
        json.dump(aggregated_data, f, indent=4)

    print(f"Aggregated scanpath data saved to {file_path}")


if __name__ == '__main__':
    # Define the directory containing your individual scanpath JSON files
    # directory = '/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_15_13_28_warping'  # Replace with the actual path to your files
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_15_14_14_attach"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_15_14_13_baselines_p5_to_p32_stim0"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_15_15_17_warping_plus_attach"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_15_17_47_attach"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_16_15_51_stim1_original"  
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_16_15_51_stim1_warp_attaching"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_16_15_51_stim1_warp"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_16_15_51_stim1_stretch"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_16_20_00_stimid0"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_17_13_45_stimid1_warp_attach"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_17_13_47_stimid1"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_all_corrected_scanpaths_across_stimuli"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_17_15_12_stimid2"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_17_16_15_stimid3"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_18_10_29_stimid4"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_18_13_54_stimid5"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_18_14_55_stimid6"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_18_15_30_stimid7"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_18_16_11_stimid8"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_18_16_54_stimid0"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_18_17_30_stimid1"
    # directory = "/home/baiy4/reading-model/data_analysis/human_data/corrected_data_by_fix8/11_all_corrected_scanpaths_across_stimuli"       # Human data
    directory = "/home/baiy4/reading-model/data_analysis/simulation_data/sim_raw_data_11_09_17_09_1episodes/stimulus_8_time_constraint_90s/corrected_simulation_data/11_23_21_41_corrected_simulation_trial_wise_scanpaths_35dot3px"        # Simulation data

    # Define the output file path
    output_file = 'integrated_corrected_human_scanpath.json'

    # Load bbox metadata
    metadata_file = const.BBOX_METADATA_DIR  # Ensure this constant is correctly defined
    bbox_metadata = load_bbox_metadata(metadata_file)

    # Call the function to aggregate the scanpaths
    aggregate_scanpaths(directory, output_file, bbox_metadata)
