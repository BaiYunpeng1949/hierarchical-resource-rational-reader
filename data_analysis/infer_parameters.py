import os
import csv
import json
import constants as const
from calculate_metrics_for_user_study import MCQFreeRecallProcessor, calc_comprehension_metrics

class ParametersInference4ComprehensionTests:
    
    def __init__(self, input_dir, output_dir, baseline_json_file, traverse_json_data=True, traverse_csv_data=True):
        """
        Parameter inference for the comprehension tests. Run it because it is a standalone section apart from the reading process. It is mainly relying on Kintsch's memory retrieval model.
        """
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.baseline_json_file = baseline_json_file

        self.data = None

        self.parameters = [const.TEXT_SIMILARITY_THRESHOLD, const.EXPLORATION_RATE]
        self.metrics = [const.MCQ_ACC, const.FREE_RECALL_SCORE]
        self.conditions = [const.TIME_CONSTRAINTS]

        self.traverse_json_data_flag = traverse_json_data
        self.traverse_csv_data_flag = traverse_csv_data

        self.key_word_for_csv_files = ["comprehension", "summary"]

        self.report_human_baseline_results_flag = True
    
    def load_json_data_to_generate_csv_files(self):

        # Traverse all the json files in the directory
        json_files = [f for f in os.listdir(self.input_dir) if f.endswith('.json')]
        print('Number of json files read: ', len(json_files))
        # print('json files: ', json_files)
        
        # Load json data
        for index, json_file in enumerate(json_files):
            with open(os.path.join(self.input_dir, json_file)) as f:
                calc_comprehension_metrics(
                    sim_json_data_path=os.path.join(self.input_dir, json_file),
                    human_json_data_path=self.baseline_json_file,
                    folder_dir=self.output_dir,
                    sim_output_csv="simulation.csv",
                    report_human_baseline_results=self.report_human_baseline_results_flag,
                )
            print('         Json file processed: ', json_file)
            self.report_human_baseline_results_flag = False     # Only report the first json file once

            # if index >= 3:      # TODO debug delete later
            #     break


    def read_all_csv_files_to_unified_json(self):
        """
        Read all CSV files containing specific keywords from a directory, process the data,
        and save it as a unified JSON file with text_similarity_threshold and exploration_rate
        as separate keys, grouping the rest of the data.
        """
        # Create lists to store simulation and human results
        simulation_csv_results = []
        human_csv_results = []

        # Get all needed CSV files
        csv_files = [
            f for f in os.listdir(self.output_dir)
            if f.endswith('.csv') and all(keyword in f for keyword in self.key_word_for_csv_files)
        ]

        print('Number of csv files read: ', len(csv_files))
        # print('csv files: ', csv_files)

        # Process each CSV file
        for csv_file in csv_files:
            file_path = os.path.join(self.output_dir, csv_file)

            # Open the CSV file
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)

                # Grouped results
                simulation_data = []
                human_data = []
                text_similarity_threshold = None
                exploration_rate = None

                # Process each row
                for row in reader:
                    # Extract shared parameters
                    text_similarity_threshold = (
                        float(row['text_similarity_threshold']) if row['text_similarity_threshold'].strip() else 0.0
                    )
                    exploration_rate = (
                        float(row['exploration_rate']) if row['exploration_rate'].strip() else 0.0
                    )
                    type_ = row['Type']  # Human or Simulation

                    time_const = int(row['time_constraint'])
                    mcq_accuracy_mean = float(row['MCQ Accuracy_mean'])
                    free_recall_score_mean = float(row['Free Recall Score_mean'])
                    mcq_accuracy_std = float(row.get('MCQ Accuracy_std', 0))
                    free_recall_score_std = float(row.get('Free Recall Score_std', 0))

                    # Construct the single result entry
                    single_result = {
                        'time_constraint': time_const,
                        'mcq_acc': {
                            'mean': mcq_accuracy_mean,
                            'std': mcq_accuracy_std
                        },
                        'free_recall_score': {
                            'mean': free_recall_score_mean,
                            'std': free_recall_score_std
                        }
                    }

                    # Append the entry to the correct type
                    if type_ == 'Simulation':
                        simulation_data.append(single_result)
                    elif type_ == 'Human':
                        human_data.append(single_result)

                # Add grouped results for each file
                if simulation_data:
                    simulation_csv_results.append({
                        'text_similarity_threshold': text_similarity_threshold,
                        'exploration_rate': exploration_rate,
                        'data': simulation_data
                    })
                if human_data:
                    human_csv_results.append({
                        'text_similarity_threshold': None,
                        'exploration_rate': None,
                        'data': human_data
                    })

        # Combine results into a single dictionary
        unified_results = {
            'Simulation': simulation_csv_results,
            'Human': human_csv_results
        }

        # Write the unified JSON file
        output_file_path = os.path.join(self.output_dir, 'simulation_results_across_all_params.json')
        with open(output_file_path, 'w') as f:
            json.dump(unified_results, f, indent=4)
        
        print(f"Unified JSON file saved to {output_file_path}")

    
    def discrepancy_function(self, sim_data, human_data):
        """
        Calculate the total discrepancy between simulation data and human data.
        For each time constraint, sum the absolute differences of MCQ accuracy and free recall score.
        """
        total_discrepancy = 0.0

        # Loop over each time constraint in the simulation data
        for sim_result in sim_data['data']:
            time_const = sim_result['time_constraint']

            # Find the corresponding human data for the same time constraint
            human_result = next((item for item in human_data['data'] if item['time_constraint'] == time_const), None)
            if human_result:
                # Calculate the absolute differences for MCQ accuracy and free recall score
                mcq_gap = abs(sim_result['mcq_acc']['mean'] - human_result['mcq_acc']['mean'])
                free_recall_gap = abs(sim_result['free_recall_score']['mean'] - human_result['free_recall_score']['mean'])
                total_discrepancy += mcq_gap + free_recall_gap
            else:
                print(f"No human data found for time constraint {time_const}")
                # If no human data is found for a time constraint, consider it as maximum discrepancy
                total_discrepancy += float('inf')

        return total_discrepancy


    def grid_search(self):
        """
        Perform a grid search over the parameter settings to find the best parameters.
        """
        min_discrepancy = float('inf')
        best_parameters = None

        # Load the unified simulation and human data
        unified_results_path = os.path.join(self.output_dir, 'simulation_results_across_all_params.json')
        with open(unified_results_path, 'r') as f:
            unified_results = json.load(f)

        simulation_results = unified_results['Simulation']
        human_results = unified_results['Human'][0]  # Assuming a single set of human data

        # Iterate over each simulation result
        for sim_data in simulation_results:
            # Calculate discrepancy for the current simulation data
            discrepancy = self.discrepancy_function(sim_data, human_results)
            text_sim_threshold = sim_data['text_similarity_threshold']
            exploration_rate = sim_data['exploration_rate']
            print(f"Parameters: text_similarity_threshold={text_sim_threshold}, exploration_rate={exploration_rate}, discrepancy={discrepancy}")

            # Update the best parameters if current discrepancy is lower
            if discrepancy < min_discrepancy:
                min_discrepancy = discrepancy
                best_parameters = {
                    'text_similarity_threshold': text_sim_threshold,
                    'exploration_rate': exploration_rate
                }

        if best_parameters:
            # Save the best parameters to a JSON file
            best_parameters_path = os.path.join(self.output_dir, 'best_parameters.json')
            with open(best_parameters_path, 'w') as f:
                json.dump(best_parameters, f, indent=4)
            print(f"\nBest Parameters Found: {best_parameters} with total discrepancy {min_discrepancy}")
        else:
            print("No matching simulation data found.")


    def infer_parameters(self):
        # Data preparation
        print('Loading data for comprehension tests......')

        # Load json data to generate csv files
        if self.traverse_json_data_flag:
            self.load_json_data_to_generate_csv_files()
        else:
            print('Skipping json data loaded, using existing csvs......')

        # Read all csv files to unified json
        if self.traverse_csv_data_flag:
            self.read_all_csv_files_to_unified_json()
        else:
            print('Skipping csv data loaded, using existing jsons......')

        # Infer parameters through grid search
        print('Starting parameter inference through grid search...')
        self.grid_search()



if __name__ == '__main__':

    # Comprehension tests parameter inference
    ############################################################################################################################################################################################################################
    # comprehension_test_input_dir = "/home/baiy4/reading-model/data_analysis/simulation_data/sim_raw_data_11_26_12_11_1episodes/stimulus_8_time_constraint_90s/Kintsch_memory_retrieval_simulations_acorss_parameters_11_29_13_18"
    # comprehension_test_input_dir = "/home/baiy4/reading-model/data_analysis/simulation_data/sim_raw_data_11_26_12_11_1episodes/stimulus_8_time_constraint_90s/Kintsch_memory_retrieval_simulations_acorss_parameters_12_01_19_12"
    comprehension_test_input_dir = "/home/baiy4/reading-model/data_analysis/simulation_data/sim_raw_data_11_26_12_11_1episodes/stimulus_8_time_constraint_90s/Kintsch_memory_retrieval_simulations_acorss_parameters_12_02_09_59"
    comprehension_test_output_dir = "/home/baiy4/reading-model/data_analysis/user_study_results_after_parameter_inference/comprehension_tests"
    comprehension_test_human_data_baseline = "/home/baiy4/reading-model/data_analysis/human_data/comprehension_data/processed_human_comprehension_data_p1_to_p32.json"

    comprehension_test_parameter_inference = ParametersInference4ComprehensionTests(
        input_dir=comprehension_test_input_dir, output_dir=comprehension_test_output_dir, baseline_json_file=comprehension_test_human_data_baseline, 
        traverse_json_data=True, traverse_csv_data=True)
    comprehension_test_parameter_inference.infer_parameters()


