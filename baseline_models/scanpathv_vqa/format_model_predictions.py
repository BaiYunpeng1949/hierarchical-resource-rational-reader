import json
import os
import data_analysis.constants as const
from step5.utils import auxiliaries as aux

def load_bbox_metadata(metadata_file):
    """Load the bounding box metadata from the JSON file."""
    with open(metadata_file, 'r') as f:
        bbox_metadata = json.load(f)
    return bbox_metadata

def get_trial_metadata(stimulus_index, bbox_metadata):
    """Get the trial metadata based on the stimulus_index."""
    # Adjust stimulus_index if necessary (e.g., add 1 if image indices start from 1)
    image_index = stimulus_index    # Adjust as needed
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

def convert_simulation_data(simulation_data_file, output_file, bbox_metadata):
    with open(simulation_data_file, 'r') as f:
        simulation_data = json.load(f)

    converted_data = []

    for entry in simulation_data:
        img_name = entry['img_names']
        stimulus_index = int(os.path.splitext(img_name)[0])  # Extract the image index from the filename
        qid = entry['qid']  # You may not need this
        X = entry['X']
        Y = entry['Y']
        T = entry['T']
        length = entry['length']

        participant_index = -1  # As per your instruction
        time_constraint = 60  # As per your instruction

        # Get the trial metadata for the current stimulus
        trial_metadata = get_trial_metadata(stimulus_index, bbox_metadata)
        if not trial_metadata:
            print(f"No metadata found for stimulus index {stimulus_index}. Skipping entry with img_name {img_name}.")
            continue

        fixation_data = []
        # Assume T contains fixation durations
        for fix_x, fix_y, fix_duration in zip(X, Y, T):
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

        scanpath_entry = {
            "stimulus_index": stimulus_index,
            "participant_index": participant_index,
            "time_constraint": time_constraint,
            "fixation_data": fixation_data
        }

        converted_data.append(scanpath_entry)

    # Write the converted data to the output file
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=4)

    print(f"Converted data saved to {output_file}")

if __name__ == '__main__':
    # Define the path to your simulation data file
    simulation_data_file = '/home/baiy4/reading-model/baseline_models/ReaderAgent_scanpath/test_predicts.json'  # Replace with your file path

    # Define the output file path
    output_file = '/home/baiy4/reading-model/baseline_models/ReaderAgent_scanpath/formated_model_predictions/converted_simulation_data.json'

    # Load bbox metadata
    metadata_file = const.BBOX_METADATA_DIR  # Ensure this constant is correctly defined
    bbox_metadata = load_bbox_metadata(metadata_file)

    # Call the function to convert the data
    convert_simulation_data(simulation_data_file, output_file, bbox_metadata)
