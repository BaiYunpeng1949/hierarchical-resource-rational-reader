import json
import os
import math
import pandas as pd
import numpy as np
import datetime
import constants as const

import step5.utils.auxiliaries as aux


def calc_px2deg():
    screen_width_cm = const.SCREEN_WIDTH_CM
    screen_height_cm = const.SCREEN_HEIGHT_CM
    viewing_distance_cm = const.VIEWING_DISTANCE_CM
    screen_resolution_width_px = const.SCREEN_RESOLUTION_WIDTH_PX
    screen_resolution_height_px = const.SCREEN_RESOLUTION_HEIGHT_PX
    screen_diagonal_cm = math.sqrt(screen_width_cm**2 + screen_height_cm**2)
    screen_diagonal_px = math.sqrt(screen_resolution_width_px**2 + screen_resolution_height_px**2)
    px2deg = math.degrees(math.atan2(0.5 * screen_diagonal_cm, viewing_distance_cm)) / (0.5 * screen_diagonal_px)
    return px2deg


class TrialBatchDataProcessor:
    def __init__(self, metadata, data_dir):
        """
        Initializes the DataProcessor with the provided metadata and directory.
        """
        self.metadata = metadata
        self.data_dir = data_dir
        self.filename = None
        self.trial_wise_p_name = None
        self.trial_wise_stimuli_name = None
        self.trial_wise_condition = None
        self.trial_wise_df = None
        self.trial_wise_fixation_saccade_detailed_data = {}
        self.aggregated_data = {}
        self.trials_metrics_results = []  # Store results for all trials
        self.trials_fixations_datas = pd.DataFrame()  # Store fixation data for all trials
        self.trials_saccades_datas = pd.DataFrame()  # Store saccade data for all trials
        self.metadata_file = const.BBOX_METADATA_DIR

        # Load the metadata file
        self.bbox_metadata = self.load_bbox_metadata()

    @staticmethod
    def _check_sampling_rate(df):
        """
        Calculate the sampling rate of the eye tracker data.
        Check if the sampling rate is close to the default sampling rate.
        """
        df['Eyetracker time_diff'] = df[const.EYE_TRACKER_TIMESTAMP].diff()
        sampling_rate = 1 / (df['Eyetracker time_diff'].mean() * 1e-6)
        # Check if the sampling rate is close to the default sampling rate
        if abs(sampling_rate - const.SR) > const.SR_EPSILON:
            print(f"Warning: Sampling rate is {sampling_rate} Hz, which is different from the default {const.SR} Hz.")
            return False, sampling_rate
        else:
            print(f"Estimated Sampling Rate: {sampling_rate} Hz. Close to the default sampling rate {const.SR} Hz, okay for further calculation.")
            return True, sampling_rate

    def load_bbox_metadata(self):
        """Load the metadata from the JSON file."""
        with open(self.metadata_file, 'r') as f:
            bbox_metadata = json.load(f)
        return bbox_metadata

    @staticmethod
    def get_image_index_from_stim_id(stim_id):
        """Extract the integer index from the stim_id (e.g., stim_1 -> 1)."""
        return int(stim_id.split('_')[-1])

    def get_trial_metadata(self, stim_id):
        """Get the trial metadata based on the stim_id."""
        image_index = self.get_image_index_from_stim_id(stim_id)

        # Find the corresponding metadata by matching the image index
        for img_meta in self.bbox_metadata['images']:
            if img_meta['image index'] == image_index:
                return img_meta

        return None

    def get_word_index_from_fixation(self, fixation_row, one_stimulus_bbox_metadata):
        x = fixation_row[const.FIX_POINT_X]
        y = fixation_row[const.FIX_POINT_Y]
        min_distance = float('inf')
        nearest_word_idx = -1
        for word_idx, word_info in enumerate(one_stimulus_bbox_metadata['words metadata']):
            word_bbox = word_info['word_bbox']
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
    
    def load_data(self, participant_id, stim_id):
        """
        Loads data for a specific participant and trial based on the metadata.
        """
        if participant_id in self.metadata:
            trial_info = self.metadata[participant_id].get(const.STIMU_NAME, {}).get(stim_id, None)
            if trial_info:
                self.trial_wise_stimuli_name = trial_info[0]
                self.trial_wise_condition = trial_info[1]

                file_path = os.path.join(self.data_dir, self.filename)
                try:
                    whole_data_df = pd.read_csv(file_path, sep='\t', low_memory=False)
                    print(f"Data file '{self.filename}' loaded successfully. Shape: {whole_data_df.shape}\n")
                    
                    # Verify that necessary columns exist
                    required_columns = [const.P_NAME, const.STIMU_NAME, const.EYE_MOV_TYPE]
                    missing_columns = [col for col in required_columns if col not in whole_data_df.columns]
                    if missing_columns:
                        print(f"Error: Missing columns in data file: {missing_columns}")
                        return "Failed to load data due to missing columns."
                    
                    valid_sr, sr = self._check_sampling_rate(whole_data_df.copy())      # TODO BYP solve here
                    self.trial_wise_df = whole_data_df[(whole_data_df[const.P_NAME] == self.trial_wise_p_name) & (whole_data_df[const.STIMU_NAME] == self.trial_wise_stimuli_name)].copy()
                    print(f"Filtered data for participant '{self.trial_wise_p_name}' and stimulus '{self.trial_wise_stimuli_name}'. Shape: {self.trial_wise_df.shape}\n")

                    if self.trial_wise_df.empty:
                        print(f"Warning: No data found for participant '{self.trial_wise_p_name}' and stimulus '{self.trial_wise_stimuli_name}'.")
                        return "No data found for the specified participant and stimulus."

                    return f"Data loaded for {participant_id}, stimulus id {stim_id}, simuli name {self.trial_wise_stimuli_name}, and condition {self.trial_wise_condition}."
                except Exception as e:
                    return f"Failed to load data: {e}"
            else:
                return "Trial ID not found in metadata."
        else:
            return "Participant ID not found in metadata."

    def get_aggregated_data(self, participant_id, stim_id):
        """
        Retrieves the loaded data for a specific participant and trial.
        """
        return self.aggregated_data.get((participant_id, stim_id), None)

    def get_fixation_saccade_detailed_data(self):
        """
        Retrieve the fixation and saccade datasets.
        """
        return self.trial_wise_fixation_saccade_detailed_data.get('df_fixations'), self.trial_wise_fixation_saccade_detailed_data.get('df_saccades')

    def process_data(self, participant_id, stim_id):
        """
        Processes the data for the specified participant and trial.
        """
        # Step 1: Copy the trial-wise DataFrame
        df = self.trial_wise_df.copy()
        if df.empty:
            print(f"Warning: Trial-wise DataFrame is empty for participant '{participant_id}', stimulus '{stim_id}'.")
            return
        print(f"Processing data for participant '{participant_id}', stimulus '{stim_id}'. Initial data shape: {df.shape}")

        # Step 3: Deduplicate based on relevant columns (only once)
        relevant_columns = [const.EYE_MOV_TYPE, const.GAZE_EVENT_DUR, const.EYE_MOV_TYPE_IDX, const.FIX_POINT_X, const.FIX_POINT_Y]
        missing_columns = [col for col in relevant_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns for processing: {missing_columns}")
            return
        
        df_cleaned = df.drop_duplicates(subset=relevant_columns)
        print(f"Data after deduplication. Shape: {df_cleaned.shape}")

        # Step 4: Separate Fixations and Saccades
        unique_eye_mov_types = df_cleaned[const.EYE_MOV_TYPE].unique()
        print(f"Unique values in '{const.EYE_MOV_TYPE}': {unique_eye_mov_types}")

        df_fixations = df_cleaned[df_cleaned[const.EYE_MOV_TYPE] == const.FIXATION].copy()
        df_saccades = df_cleaned[df_cleaned[const.EYE_MOV_TYPE] == const.SACCADE].copy()
        print(f"Number of fixations: {len(df_fixations)}, Number of saccades: {len(df_saccades)}")

        if df_fixations.empty:
            print(f"Warning: No fixations found for participant '{participant_id}', stimulus '{stim_id}'.")
            return

        # Get the trial metadata
        trial_metadata = self.get_trial_metadata(stim_id)
        if not trial_metadata:
            print(f"No metadata found for stim_id {stim_id}. Skipping this trial.")
            return

        # Calculate Fixation Metrics
        fixation_count = len(df_fixations)
        mean_fixation_duration = df_fixations[const.GAZE_EVENT_DUR].mean()
        total_fixation_duration = df_fixations[const.GAZE_EVENT_DUR].sum()

        # Calculate Saccade Metrics
        saccade_count = len(df_saccades)

        # Add Participant ID, Stimulus ID, and Trial Condition columns
        df_fixations['Participant_ID'] = participant_id
        df_fixations['Stimulus_ID'] = stim_id
        df_fixations['Trial_Condition'] = self.trial_wise_condition

        df_saccades['Participant_ID'] = participant_id
        df_saccades['Stimulus_ID'] = stim_id
        df_saccades['Trial_Condition'] = self.trial_wise_condition

        # Map each fixation to a word index
        df_fixations['word_index'] = df_fixations.apply(lambda row: self.get_word_index_from_fixation(row, trial_metadata), axis=1)

        # Step 5: Identify valid saccades (saccades with valid preceding and following fixations)
        df_saccades.loc[:, 'Prev_Type'] = df_cleaned[const.EYE_MOV_TYPE].shift(1)
        df_saccades.loc[:, 'Next_Type'] = df_cleaned[const.EYE_MOV_TYPE].shift(-1)
        df_saccades.loc[:, 'Prev_Fixation_X'] = df_cleaned[const.FIX_POINT_X].shift(1)
        df_saccades.loc[:, 'Prev_Fixation_Y'] = df_cleaned[const.FIX_POINT_Y].shift(1)
        df_saccades.loc[:, 'Next_Fixation_X'] = df_cleaned[const.FIX_POINT_X].shift(-1)
        df_saccades.loc[:, 'Next_Fixation_Y'] = df_cleaned[const.FIX_POINT_Y].shift(-1)

        df_saccades.loc[:, 'Is_Valid_Saccade'] = (
                (df_saccades[const.EYE_MOV_TYPE] == const.SACCADE) &
                (df_saccades['Prev_Type'] == const.FIXATION) &
                (df_saccades['Next_Type'] == const.FIXATION)
        )

        df_valid_saccades = df_saccades[df_saccades['Is_Valid_Saccade']].copy()

        # Store results in self for further use
        self.trial_wise_fixation_saccade_detailed_data['df_fixations'] = df_fixations
        self.trial_wise_fixation_saccade_detailed_data['df_saccades'] = df_valid_saccades

        # Calculate Saccade Length and Speed
        df_valid_saccades.loc[:, 'Saccade_Length'] = np.sqrt(
            (df_valid_saccades['Prev_Fixation_X'] - df_valid_saccades['Next_Fixation_X']) ** 2 +
            (df_valid_saccades['Prev_Fixation_Y'] - df_valid_saccades['Next_Fixation_Y']) ** 2
        )
        df_valid_saccades.loc[:, 'Saccade_Speed'] = df_valid_saccades['Saccade_Length'] / (
                df_valid_saccades[const.GAZE_EVENT_DUR] / 1000)

        # Calculate Regression Frequency
        df_valid_saccades.loc[:, 'Is_Regression_XY'] = (
                (df_valid_saccades['Next_Fixation_X'] < df_valid_saccades['Prev_Fixation_X']) |
                (df_valid_saccades['Next_Fixation_Y'] < df_valid_saccades['Prev_Fixation_Y'])
        )
        df_valid_saccades.loc[:, 'Is_Regression_X'] = (df_valid_saccades['Next_Fixation_X'] < df_valid_saccades['Prev_Fixation_X'])

        # Store results in self for further use
        self.trial_wise_fixation_saccade_detailed_data['df_fixations'] = df_fixations

        # Step 6: Calculate px2deg
        px2deg = calc_px2deg()  # Assuming calc_px2deg is a method in the class
        # print(f"                Calculated px2deg: {px2deg}")

        # Step 7: Output Metrics
        pid = participant_id
        stim_id = stim_id
        trial_condition = self.trial_wise_condition
        fixation_count = fixation_count
        avg_fixation_count_per_second = fixation_count / trial_condition
        total_fixation_duration = total_fixation_duration
        avg_fixation_duration_per_second = total_fixation_duration / trial_condition
        saccade_count = saccade_count
        avg_saccade_count_per_second = saccade_count / trial_condition
        fixation_count_percentage = fixation_count / (fixation_count + saccade_count + const.EPSILON) * 100
        avg_saccade_length_px = df_valid_saccades['Saccade_Length'].mean()
        avg_saccade_length_deg = df_valid_saccades['Saccade_Length'].mean() * px2deg
        avg_saccade_velocity_px = df_valid_saccades['Saccade_Speed'].mean()
        avg_saccade_velocity_deg = df_valid_saccades['Saccade_Speed'].mean() * px2deg

        self.aggregated_data = {
            const.PID: pid,
            const.STIM_ID: stim_id,
            const.TRIAL_COND: trial_condition,
            const.FIX_COUNT: fixation_count,
            const.AVG_FIX_COUNT_PER_SEC: avg_fixation_count_per_second,
            const.TOTAL_FIX_DUR: total_fixation_duration,
            const.AVG_FIX_DUR_PER_SEC: avg_fixation_duration_per_second,
            const.AVG_FIX_DUR_PER_COUNT: mean_fixation_duration,
            const.SACCADE_COUNT: saccade_count,
            const.AVG_SACCADE_COUNT_PER_SEC: avg_saccade_count_per_second,
            const.FIX_COUNT_PERCENTAGE: fixation_count_percentage,
            const.AVG_SACCADE_LENGTH_PX: avg_saccade_length_px,
            const.AVG_SACCADE_LENGTH_DEG: avg_saccade_length_deg,
            const.AVG_SACCADE_VEL_PX: avg_saccade_velocity_px,
            const.AVG_SACCADE_VEL_DEG: avg_saccade_velocity_deg,
        }

        self.trials_metrics_results.append(self.aggregated_data.copy())
        # Concatenate fixation and saccade data with previous data
        self.trials_fixations_datas = pd.concat([self.trials_fixations_datas, df_fixations], ignore_index=True)
        self.trials_saccades_datas = pd.concat([self.trials_saccades_datas, df_valid_saccades], ignore_index=True)

    def run(self, participant_id):
        """
        Generates datasets for all trials associated with a specific participant.
        """
        if participant_id in self.metadata:
            self.filename = self.metadata[participant_id][const.FILENAME]
            self.trial_wise_p_name = self.metadata[participant_id][const.P_NAME]
            for stim_id in self.metadata[participant_id][const.STIMU_NAME]:
                loaded_result = self.load_data(participant_id, stim_id)
                if "Failed" in loaded_result or "No data" in loaded_result:
                    print(f"Skipping processing for participant '{participant_id}', stimulus '{stim_id}' due to previous errors.")
                    continue
                self.process_data(participant_id, stim_id)
            return "All datasets generated."
        else:
            print(f"Participant ID '{participant_id}' not found in metadata.")
            return "Participant ID not found."


class ParticipantsBatchProcessor:
    def __init__(self, metadata, data_dir):
        """
        Initializes the ParticipantsBatchProcessor with metadata and the data directory.
        """
        self.metadata = metadata
        self.data_dir = data_dir
        self.aggregated_metrics_results = []
        self.fixation_detailed_datas_df = pd.DataFrame()
        self.saccade_detailed_datas_df = pd.DataFrame()
    
    def process_participants(self, participant_ids):
        for participant_id in participant_ids:
            processor = TrialBatchDataProcessor(self.metadata, self.data_dir)
            processor.run(participant_id)
            # Get aggregated data for all trials for this participant
            self.aggregate_results(processor.trials_metrics_results)
            # Concatenate fixation and saccade data for all trials for this participant
            self.fixation_detailed_datas_df = pd.concat([self.fixation_detailed_datas_df, processor.trials_fixations_datas], ignore_index=True)
            self.saccade_detailed_datas_df = pd.concat([self.saccade_detailed_datas_df, processor.trials_saccades_datas], ignore_index=True)

        timestamp_dir_name = f"{datetime.datetime.now().strftime('%m_%d_%H_%M')}"
        dir_name = os.path.join(const.HUMAN_PROCESSED_DATA_DIR, timestamp_dir_name)
        # Create a directory to save the processed data
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        # Save files to the directory
        self.save_results_to_csv(dir_name)
        self.save_scanpath_to_json(dir_name)

    def aggregate_results(self, trial_results):
        """
        Aggregates the results from processing each trial for all participants.
        """
        self.aggregated_metrics_results.extend(trial_results)  # Append all trial results

    def save_results_to_csv(self, dir_name):
        """
        Saves the aggregated results to a CSV file.
        """
        # Save aggregated metrics to CSV
        df = pd.DataFrame(self.aggregated_metrics_results)
        df.to_csv(os.path.join(dir_name, 'processed_metrics.csv'), index=False)

        # Concatenate and save fixation data to CSV
        if not self.fixation_detailed_datas_df.empty:
            self.fixation_detailed_datas_df.to_csv(
                os.path.join(dir_name, 'fixation_detailed_data.csv'),
                index=False)

        # Concatenate and save saccade data to CSV
        if not self.saccade_detailed_datas_df.empty:
            self.saccade_detailed_datas_df.to_csv(
                os.path.join(dir_name, 'saccade_detailed_data.csv'), index=False)

        print(f"Metrics saved files in this directory: {dir_name}")
    
    @staticmethod
    def get_image_index_from_stim_id(stim_id):
        """Extract the integer index from the stim_id (e.g., 'stim_1' -> 1)."""
        return int(stim_id.split('_')[-1])
    
    def save_scanpath_to_json(self, dir_name):

        # Create the 'trial_wise_scanpaths' folder inside dir_name
        trial_scanpaths_dir = os.path.join(dir_name, 'trial_wise_scanpaths')
        os.makedirs(trial_scanpaths_dir, exist_ok=True)
        
        scanpath_data = []

        # Group the fixation data by Stimulus_ID, Participant_ID, and Trial_Condition
        grouped = self.fixation_detailed_datas_df.groupby(['Stimulus_ID', 'Participant_ID', 'Trial_Condition'])

        for (stimulus_id, participant_id, trial_condition), group_df in grouped:
            
            # Use get_image_index_from_stim_id to get the integer index from stimulus_id
            stimulus_index = self.get_image_index_from_stim_id(stimulus_id) - 1  # Subtract 1 to start indexing from 0

            # Convert participant_id to integer if it's numeric
            participant_id = int(participant_id)


            # For each group, create the required data structure
            trial_wise_fixation_data_list = group_df.apply(lambda row: [
                float(row[const.FIX_POINT_X]),
                float(row[const.FIX_POINT_Y]),
                float(row[const.GAZE_EVENT_DUR])
            ], axis=1).tolist()
            
            # Create the trial_wise_scanpath_entry
            trial_wise_scanpath_entry = {
                "fixations": trial_wise_fixation_data_list
            }

            # Save individual trial scanpath data to JSON file
            # Create the filename using the pattern 'p<participant_index>_stimid<stimulus_index>_scanpath.json'
            filename = f"p{participant_id}_stimid{stimulus_index}_tc{int(trial_condition)}_scanpath.json"

            # Save the scanpath_entry to a JSON file inside 'trial_wise_scanpaths' folder
            output_file_path = os.path.join(trial_scanpaths_dir, filename)
            with open(output_file_path, 'w') as f:
                json.dump(trial_wise_scanpath_entry, f, indent=4)

            print(f"Scanpath data for stimulus_index {stimulus_id}, participant_id {participant_id}, and time constraint {trial_condition} saved to {output_file_path}")

            # For all trials together
            # For each group, create the required data structure
            fixation_data_list = group_df.apply(lambda row: {
                "word_index": int(row['word_index']),
                "fix_x": float(row[const.FIX_POINT_X]),
                "fix_y": float(row[const.FIX_POINT_Y]), 
                "norm_fix_x": aux.normalise(row[const.FIX_POINT_X], 0, const.SCREEN_RESOLUTION_WIDTH_PX, 0, 1),
                "norm_fix_y": aux.normalise(row[const.FIX_POINT_Y], 0, const.SCREEN_RESOLUTION_HEIGHT_PX, 0, 1),
                "fix_duration": float(row[const.GAZE_EVENT_DUR])
            }, axis=1).tolist()

            # Got the scanpaths for all trials
            scanpath_entry = {
                "stimulus_index": stimulus_index,
                "participant_index": participant_id,
                "time_constraint": int(trial_condition),
                "fixation_data": fixation_data_list
            }

            scanpath_data.append(scanpath_entry)

        # Save to JSON file
        output_file_path = os.path.join(dir_name, 'processed_human_scanpath.json')
        with open(output_file_path, 'w') as f:
            json.dump(scanpath_data, f, indent=4)

        print(f"Scanpath data saved to {output_file_path}")

if __name__ == '__main__':

    metadata = const.METADATA
    participant_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32'] 
    # participant_list = ["30"]
    batch_processor = ParticipantsBatchProcessor(metadata, const.HUMAN_RAW_DATA_DIR)
    batch_processor.process_participants(participant_list)
