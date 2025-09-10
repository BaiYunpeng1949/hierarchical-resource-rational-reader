import yaml
import os
import json
import warnings
from datetime import datetime
import sys

# Add sub_models directory to Python path
current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_file_dir, 'sub_models'))

from stable_baselines3 import PPO

from sub_models.word_recognition_v0807.WordRecognitionEnv import WordRecognitionEnv
from sub_models.sentence_read_v0604.SentenceReadingEnv import SentenceReadingUnderTimePressureEnv
from sub_models.text_read_v0604.TextReadEnv import TextReadingUnderTimePressureEnv
from sub_models.text_read_v0604.Utilities import DictActionUnwrapper

from utils.json_utils import np_to_native
from utils.auxiliary_functions import list_diff_preserve_order

# Constants and Configurations
CONFIG_PATH = "sub_models/config.yaml"
TIME_CONDITIONS = {
    "30s": 30,
    "60s": 60,
    "90s": 90,
}

SIM_RESULTS_DIR = "simulated_results"


class TextReader:
    def __init__(self):
        """
        This is the text reader that reads a sentence, controls which sentence to read. 
        And it returns the states of reading progress and reading time.
        """
        # Read the configuration file
        with open(CONFIG_PATH, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # Pre-trained model infomation
        self._model_info = self.config["simulate"]["rl_models"]["text_reader"]
        
        # Get the pre-trained model path
        self._model_path = os.path.join("sub_models", "training", "saved_models", self._model_info["checkpoints_folder_name"], self._model_info["loaded_model_name"])

        # Initialize the environment
        def make_env():
            env = TextReadingUnderTimePressureEnv()
            env = DictActionUnwrapper(env)
            return env
        self.env = make_env()

        # Load the pre-trained model
        try:
            self._model = PPO.load(self._model_path, self.env, custom_objects={"observation_space": self.env.observation_space, "action_space": self.env.action_space})
        except (RuntimeError, TypeError) as e:
            warnings.warn(f"Could not deserialize object: {e}")
            raise e

        print(f"{'='*200}\n" 
              f"Successfully loaded the pre-trained {self._model_info['env_name']} model from {self._model_path}.\n"
              f"{'='*200}\n")
            
        # Initialize the rl-related variables that are necessary
        self.action = None
        self._states = None
        self._reward = None
        self._truncated = None
        self._obs = None
        self._info = None
        self.done = None
        self.score = None
        self.text_reading_steps = None
        self.text_reading_logs = None
    
    def reset(self, inputs: dict = None):
        """
        Reset the environment for each allocated sentence and time condition.
        """
        # RL-related variables
        self._obs, self._info = self.env.reset(inputs=inputs)
        self.done = False
        self.score = 0

        # Environment-related variables
        self.text_reading_steps = 0
        self.text_reading_logs = {}
    
    def step(self, time_info: dict = None):
        """
        Read the whole text, return corresponding states information.
        NOTE: might need to parse this reader into two parts, because I may use on-the-fly sentence reading content and comprehension score to evaluate the reading appraisals.
        But these are not good because they do not take the contextual information.
        TODO: needs to be separated because the time needs to be returned by the lower-level agent --> NOT Priority, postpone to later
        """
        self.text_reading_steps += 1
        self.action, self._states = self._model.predict(self._obs, deterministic=True)
        self._obs, self._reward, self.done, self._truncated, self._info = self.env.step(action=self.action, time_info=time_info)
        # Get the step-wise log
        self.text_reading_logs = self.env.unwrapped.get_individual_step_log()


class SentenceReader:

    def __init__(self):
        """
        This is the sentence reader that reads a sentence, controls which word to read in a sentence. 
        And it returns the reading progress and the reading time.
        """
        # Read the configuration file
        with open(CONFIG_PATH, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # Pre-trained model infomation
        self._model_info = self.config["simulate"]["rl_models"]["sentence_reader"]

        # Get the pre-trained model path
        self._model_path = os.path.join("sub_models", "training", "saved_models", self._model_info["checkpoints_folder_name"], self._model_info["loaded_model_name"])

        # Initialize the environment
        self.env = SentenceReadingUnderTimePressureEnv()

        # Load the pre-trained model
        try:
            self._model = PPO.load(self._model_path, self.env, custom_objects={"observation_space": self.env.observation_space, "action_space": self.env.action_space})
        except (RuntimeError, TypeError) as e:
            warnings.warn(f"Could not deserialize object: {e}")
            raise e

        print(f"{'='*200}\n"
              f"Successfully loaded the pre-trained {self._model_info['env_name']} model from {self._model_path}.\n"
              f"{'='*200}\n")
            
        # Initialize the rl-related variables that are necessary
        self.action = None
        self._states = None
        self._reward = None
        self._truncated = None
        self._obs = None
        self._info = None
        self.done = None
        self.score = None
        self.sentence_reading_steps = None
        self.sentence_reading_logs = None

    def reset(self, inputs: dict = None):
        """
        Reset the environment for each allocated sentence and time condition.
        """
        # RL-related variables
        self._obs, self._info = self.env.reset(inputs=inputs)
        self.done = False
        self.score = 0

        # Environment-related variables
        self.sentence_reading_steps = 0
        self.sentence_reading_logs = {}
    
    def step(self):
        """
        Read the whole sentence, return corresponding states information.
        """
        self.sentence_reading_steps += 1
        self.action, self._states = self._model.predict(self._obs, deterministic=True)
        self._obs, self._reward, self.done, self._truncated, self._info = self.env.step(action=self.action)
        # Get the step-wise log
        self.sentence_reading_logs = self.env.get_individual_step_log()        


class WordRecognizer:

    def __init__(self):
        """
        This is the word recognizer that recognizes a word, controls which letter to fixate on in a word. 
        And it returns the reading progress and the reading time. Accumulated by saccades duration and gaze durations.
        """
        # Read the configuration file
        with open(CONFIG_PATH, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # Pre-trained model infomation
        self._model_info = self.config["simulate"]["rl_models"]["word_recognizer"]

        # Get the pre-trained model path
        self._model_path = os.path.join("sub_models", "training", "saved_models", self._model_info["checkpoints_folder_name"], self._model_info["loaded_model_name"])

        # Initialize the environment
        self.env = WordRecognitionEnv()

        # Load the pre-trained model
        try:
            self._model = PPO.load(self._model_path, self.env, custom_objects={"observation_space": self.env.observation_space, "action_space": self.env.action_space})
        except (RuntimeError, TypeError) as e:
            warnings.warn(f"Could not deserialize object: {e}")
            raise e

        print(f"{'='*200}\n"
              f"Successfully loaded the pre-trained {self._model_info['env_name']} model from {self._model_path}.\n"
              f"{'='*200}\n")
            
        # Initialize the rl-related variables that are necessary
        self.action = None
        self._states = None
        self._reward = None
        self._truncated = None
        self._obs = None
        self._info = None
        self.done = None
        self.score = None
        self.word_recognition_steps = None
        self.word_recognition_logs = None

    def reset(self, inputs: dict = None):
        """
        Reset the environment for each allocated word.
        """
        # RL-related variables
        self._obs, self._info = self.env.reset(inputs=inputs)
        self.done = False
        self.score = 0

        # Environment-related variables
        self.word_recognition_steps = 0
        self.word_recognition_logs = {}
    
    def step(self):
        """
        Recognize the word (sample and activate), return corresponding states information.
        """
        self.word_recognition_steps += 1
        self.action, self._states = self._model.predict(self._obs, deterministic=True)
        self._obs, self._reward, self.done, self._truncated, self._info = self.env.step(action=self.action)
        # Get the step-wise log
        self.word_recognition_logs = self.env.get_individual_step_log()        


class ReaderAgent:

    def __init__(self):
        """
        ReaderAgent joint place. 
        Created on 12 June 2025.
        Author: Bai Yunpeng

        Version 0612
            Text reader and sentence reader only. And do not have the lexical text comprehension yet. Only vectorized values (normalised appraisal levels). 
                And using the pseudo time consumption.
            Simulation objective: the main metrics: reading speed, regression rate, and skip rate; 
                and comprehension metrics (including the comprehension score and the comprehension time) are compatible with human data.
        
        Version 0612
            Based on the version 0612, but using the real reading time.


        Version 0617
            Using the real reading time done by the simulation.
            NOTE: try later -- use the handcrafted sentence comprehension to guide the text reader.
        
        Version 0808
            Integrating the word recognizer to calculate the actual simulation reading time, instead of using a global constant.
            Objective: 1. get the actual simulation word reading time; 2. enable drawing scanpath from the letter level.
        """

        # Read the configuration file
        with open(CONFIG_PATH, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        mode = self.config["rl"]["mode"]
        assert mode == "simulate", f"Invalid mode: {mode}, should be 'simulate' when running the simulator!"

        # Initialize the readers
        self.text_reader = TextReader()
        self.sentence_reader = SentenceReader()
        self.word_recognizer = WordRecognizer()

        # Time-related variables (states)
        self._total_time = None     # in seconds
        self._elapsed_time = None   # in seconds
        self._remaining_time = None # in seconds

        # Read the assests -- NOTE: not needed here, all managed in the environment, just changing the modes

        # Logs
        self._episode_index = None
        self._stimulus_index = None
        self._time_condition = None
        # NOTE: adhoc variables for now, will be revised later
    
    def reset(
        self, 
        episode_index: int = 0,
        stimulus_index: int = 0,
        time_condition: str = "30s",
        ):
        """
        Reset the simulator with one batch of free parameters in one parameter-inference trial.
        """
        ##########################################################
        # Reset conditions (configurations)
        ##########################################################
        # Reset the episode (trial) index
        self._episode_index = int(episode_index)    

        # Reset the stimulus index
        self._stimulus_index = int(stimulus_index)

        # Reset the time-related states
        self._time_condition = str(time_condition)

        # ----------------------------------------------
        print(f"{'-'*100}")
        print("Get the reset conditions:")
        print(f"The episode index is {self._episode_index}")
        print(f"The stimulus index is {self._stimulus_index}")
        print(f"The time condition is {self._time_condition}")
        print(f"{'-'*100}")
        print()
        # ----------------------------------------------------------------------

        # Reset the time-related variables
        self._total_time = TIME_CONDITIONS[self._time_condition]
        self._elapsed_time = 0
        self._remaining_time = self._total_time - self._elapsed_time

        ##########################################################
        # Reset the RL environments -- configure using the reset variables
        ##########################################################
        # # Reset the readers
        # self.text_reader.reset(inputs=inputs)         
        # self.sentence_reader.reset(inputs=inputs) 

        ##########################################################
        # Reset the logs
        ##########################################################
        # Reset the logs
        self._init_logs()
    
    def run(self):
        """
        Run the simulation.
        """
        # Run the simulation
        self._simulate_text_reading()
        # Close the simulation
        self._save_data()
    
    def _simulate_text_reading(self):
        """
        Simulate the text reading.
        """
        # Reset the environment
        inputs = {
            "episode_id": self._episode_index,
            "stimulus_id": self._stimulus_index,
            "time_condition": self._time_condition,
            "get_data_from_agents": True,
        }

        self.text_reader.reset(inputs=inputs)          

        time_info = {
            "elapsed_time": self._elapsed_time,     
            "remaining_time": self._remaining_time,
        }

        # Start to read the given stimulus -- the text paragraph with many sentences
        while not self.text_reader.done:

            # Determine which sentence to read using the RL model
            self.text_reader.step(time_info=time_info)

            # UPDATE! Get the reading sentence index
            self.current_sentence_index = self.text_reader.env.unwrapped.current_sentence_index
            self.actual_reading_sentence_index = self.text_reader.env.unwrapped.actual_reading_sentence_index

            # Having the sentence index, read the sentence using the sentence reader, and update the time consumption (and maybe also the comprehension score)
            sentence_reading_time_consumed = self._simulate_sentence_reading()

            # Update the time-related variables
            self._elapsed_time += sentence_reading_time_consumed
            self._remaining_time = self._total_time - self._elapsed_time
            # Update text logs
            self._update_text_reading_logs()
            # Update the time-info using the real reading-time data. So reading is fine on the text-level.
            time_info = {
                "elapsed_time": self._elapsed_time,
                "remaining_time": self._remaining_time,
            }
    
    def _simulate_sentence_reading(self):
        """
        Simulate the sentence reading.
        """
        # Reset the environment
        inputs = {
            "episode_id": self._episode_index,
            "stimulus_id": self._stimulus_index,
            "sentence_id": self.actual_reading_sentence_index,
            "time_condition": self._time_condition,
        }

        self.sentence_reader.reset(inputs=inputs)

        # Reset the word index in the sentence
        self.current_word_index = -1          # Always start from the first word when running the simulation

        # Reset the setence reading time in ms
        sentence_reading_time_in_s = 0

        # Reset the individual sentence reading logs
        self._individual_sentence_reading_logs = []
        # Reset the prevoius step's reading words; use this to determine the newly reading words (the number of words is 1 for forward reading and skipping, but for regression it is 2)
        previous_step_local_actual_fixation_sequence_in_sentence = []

        while not self.sentence_reader.done:

            # Determine which word to read using the RL model
            self.sentence_reader.step()

            # Get the words indexes for those need to be read
            current_step_local_actual_fixation_sequence_in_sentence = self.sentence_reader.env.local_actual_fixation_sequence_in_sentence.copy()
            new_words_indexes_list = list_diff_preserve_order(a=previous_step_local_actual_fixation_sequence_in_sentence, b=current_step_local_actual_fixation_sequence_in_sentence)

            # Reset the recognized words list
            recognized_words_list = []

            # Reset the sampled letters indexes dict
            sampled_lettes_indexes_dict = {}

            # Reset the step-wise time consumption in second: total elapsed time
            individual_step_elapsed_time_in_s = 0
            # Reset the step-wise time consumption in second: total gaze duration
            individual_step_gaze_duration_in_s = 0

            # Read word or words one by one
            for (i, new_reading_word_index) in enumerate(new_words_indexes_list):

                word_gt = self.sentence_reader.env._sentence_info['words'][new_reading_word_index]

                # Get the word's meta-data for the word recognizer
                current_word_metadata = {
                    'word': word_gt,
                    'word_freq_prob': self.sentence_reader.env._sentence_info['word_frequencies_per_million_for_analysis'][new_reading_word_index] / 1_000_000, 
                    'word_pred_prob': self.sentence_reader.env._sentence_info['word_predictabilities_for_analysis'][new_reading_word_index],
                }

                # Call the word recognizer for time consumption
                word_reccognition_gaze_time_in_ms, word_recognition_elapsed_time_in_ms, recognized_word, valid_sampled_letters_indexes_list = self._simulate_word_recognition(inputs=current_word_metadata)

                # Update the sampled letters indexes dict
                sampled_lettes_indexes_dict[i] = {
                    "sentence_local_word_index": new_reading_word_index,
                    "letters_indexes": valid_sampled_letters_indexes_list,
                    "word": word_gt,
                    "word_len": len(word_gt),
                }

                # Update the recognized words list
                recognized_words_list.append(recognized_word)
                
                # Update the individual step time consumption
                individual_step_elapsed_time_in_s += word_recognition_elapsed_time_in_ms / 1_000
                individual_step_gaze_duration_in_s += word_reccognition_gaze_time_in_ms / 1_000     

                # # TODO debubg delete later
                # print('\n------------------------------------------------------------------------------------------------------------')
                # print(f"The individual step elapsed time in s is: {individual_step_elapsed_time_in_s}, the gaze duration is: {individual_step_gaze_duration_in_s}. |||| The elapsed time is smaller than the gaze duration: {individual_step_elapsed_time_in_s <= individual_step_gaze_duration_in_s}")
                # print('------------------------------------------------------------------------------------------------------------\n')

                # Sum to the sentence reading time
                sentence_reading_time_in_s += word_recognition_elapsed_time_in_ms / 1_000
            
            # Update sentence logs
            # NOTE NOT Pirority: I can do the cross validation using the reading_sequence and logs from that reader <<PLUS>> check whether the index input is correct -- Error: list index out of range
            self._update_sentence_reading_logs(
                num_words_read=len(new_words_indexes_list),
                sampled_lettes_indexes_dict=sampled_lettes_indexes_dict,
                individual_step_elapsed_time_in_s=individual_step_elapsed_time_in_s,
                individual_step_gaze_duration_in_s=individual_step_gaze_duration_in_s,
                recognized_words_list=recognized_words_list,
            )    
            # Update the previous step local actual fixation sequence
            previous_step_local_actual_fixation_sequence_in_sentence = self.sentence_reader.env.local_actual_fixation_sequence_in_sentence.copy()

        # # TODO debug delete later
        # print(f"The self.actual_reading_sentence_index is: {self.actual_reading_sentence_index}, the sentence reading time in s is: {sentence_reading_time_in_s}")
        
        return sentence_reading_time_in_s
    
    def _simulate_word_recognition(self, inputs: dict=None):
        """
        Simulate the word recognition

        Two primary usage: 1. calculate reading time; 2. draw the scanpath from the letter level
        """
        # Reset the environment
        self.word_recognizer.reset(inputs=inputs)

        # Reset the letter index in the word? NOTE: check whether this is necessary (maybe yes, because we need to plot later)
        self.current_letter_index = -1          # Always start from outside of the word

        # Reset the individual word recognition logs
        self._individual_word_recognition_logs = []

        # Run the frozen controller
        while not self.word_recognizer.done:

            # Determine which letter to center in the foveal vision
            self.word_recognizer.step()

            # Get the letter index
            self.current_letter_index, is_terminate = self.word_recognizer.env.get_current_letter_index()

            # Get the recognized word
            recognized_word = self.word_recognizer.env.get_recognized_word()

            # Get the valid sampled letters indexes list
            valid_sampled_letters_indexes_list = self.word_recognizer.env.get_valid_sampled_letters_indexes_list()

            # Update word logs
            self._update_word_recognition_logs()
        
        # Return the consumed time
        return self.word_recognizer.env.gaze_duration_for_this_word, self.word_recognizer.env.total_elapsed_time_for_this_word, recognized_word, valid_sampled_letters_indexes_list

    ########################################################## Helper functions ##########################################################
    def _init_logs(self):
        """
        Initialize the data log dictionary.
        Contains general information (metadata of the trial)
        """
        self._single_episode_logs = {
            "episode_index": self._episode_index,
            "stimulus_index": self._stimulus_index,
            "time_condition": self._time_condition,
            "total_time": self._total_time,
            "text_reading_logs": [],
        }
        self._individual_sentence_reading_logs = []
    
    def _update_text_reading_logs(self):
        """
        Update the text reading logs.
        """
        # Add the sentence in the text
        self._single_episode_logs["text_reading_logs"].append(self.text_reader.text_reading_logs)    
        # Add the sentence reading summary
        self._single_episode_logs["text_reading_logs"][self.text_reader.text_reading_steps - 1]["sentence_reading_summary"] = self.sentence_reader.env.get_summarised_sentence_reading_logs()
        # Add the words reading in the given sentence (the reading sequence) 
        self._single_episode_logs["text_reading_logs"][self.text_reader.text_reading_steps - 1]["sentence_reading_logs"] = self._individual_sentence_reading_logs
    
    def _update_sentence_reading_logs(self, num_words_read, sampled_lettes_indexes_dict, individual_step_elapsed_time_in_s, individual_step_gaze_duration_in_s, recognized_words_list):
        """
        Update the sentence reading logs.
        """
        self._individual_sentence_reading_logs.append(self.sentence_reader.sentence_reading_logs)
        # Add the word recognition summary
        self._individual_sentence_reading_logs[self.sentence_reader.sentence_reading_steps - 1]["word_recognition_summary"] = {
            "num_words_read_this_step": num_words_read,
            "sampled_letters_indexes_dict": sampled_lettes_indexes_dict,
            "total_elapsed_time_in_s": individual_step_elapsed_time_in_s,
            "individual_step_gaze_duration_in_s": individual_step_gaze_duration_in_s,
            "recognized_words_list": recognized_words_list,
        }      
        # Detailed logs NOTE: I feel this is not necessary
        # self._individual_sentence_reading_logs[self.sentence_reader.sentence_reading_steps - 1]["word_recognition_logs"] = self._individual_word_recognition_logs
    
    def _update_word_recognition_logs(self):
        """
        Update the word recognition logs.
        """
        self._individual_word_recognition_logs.append(self.word_recognizer.word_recognition_logs)
    
    def _save_data(self):
        """
        Save the data.          NOTE: do later, if run multiple trials, stack all data together.
        """
        # Create a folder named after the simulation date, and the time, hour, and minute
        simulation_date = datetime.now().strftime("%Y%m%d")
        simulation_time = datetime.now().strftime("%H%M")
        simulation_folder_name = f"{simulation_date}_{simulation_time}"
        simulation_folder_path = os.path.join(SIM_RESULTS_DIR, simulation_folder_name)
        if not os.path.exists(simulation_folder_path):
            os.makedirs(simulation_folder_path)
        else:
            print(f"The folder {simulation_folder_path} already exists. Will overwrite the results.")

        # Save the logs to json files
        with open(os.path.join(simulation_folder_path, "text_reading_logs.json"), "w") as file:
            json.dump(self._single_episode_logs, file, indent=4, default=np_to_native)
        print(f"The text-level sim data logs are stored at: {os.path.join(simulation_folder_path, 'text_reading_logs.json')}")      

        # Save the metadata about the simulation configurations
        metadata_dict = self.config["simulate"]
        with open(os.path.join(simulation_folder_path, "metadata.json"), "w") as file:
            json.dump(metadata_dict, file, indent=4, default=np_to_native)
        print(f"The metadata about the simulation configurations are stored at: {os.path.join(simulation_folder_path, 'metadata.json')}")


def run_batch_simulations(
    stimulus_ids: list = None,
    time_conditions: list = None,
    num_trials: int = 1,
    output_dir: str = None
) -> dict:
    """
    Run multiple trials across specified stimuli and time conditions.
    
    Args:
        stimulus_ids (list): List of stimulus IDs to simulate. If None, uses default range.
        time_conditions (list): List of time conditions to test. If None, uses default conditions.
        num_trials (int): Number of trials per stimulus-condition combination.
        output_dir (str): Directory to save results. If None, creates timestamped directory.
    
    Returns:
        dict: Dictionary containing all simulation results and metadata.
    """
    # Initialize the simulator
    simulator = ReaderAgent()

    # Set default values if not provided
    if stimulus_ids is None:
        stimulus_ids = list(range(0, 9))  # Default range
    if time_conditions is None:
        time_conditions = ["30s", "60s", "90s"]

    # Create output directory if not provided
    if output_dir is None:
        simulation_date = datetime.now().strftime("%Y%m%d")
        simulation_time = datetime.now().strftime("%H%M")
        simulation_folder_name = f"{simulation_date}_{simulation_time}_trials{num_trials}_stims{len(stimulus_ids)}_conds{len(time_conditions)}"
        output_dir = os.path.join(SIM_RESULTS_DIR, simulation_folder_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Store all results
    all_results = []

    # Run simulations for each combination
    for stimulus_id in stimulus_ids:
        for time_condition in time_conditions:
            for trial in range(num_trials):
                print(f"\nRunning trial {trial + 1}/{num_trials} for stimulus {stimulus_id} under {time_condition} condition")
                
                # Reset the simulator with current parameters
                simulator.reset(
                    episode_index=trial,
                    stimulus_index=stimulus_id,
                    time_condition=time_condition
                )

                # Run the simulation
                simulator.run()

                # Add the results to our collection
                all_results.append(simulator._single_episode_logs)

    # Save all results to a single JSON file
    results_file = os.path.join(output_dir, "all_simulation_results.json")
    with open(results_file, "w") as file:
        json.dump(all_results, file, indent=4, default=np_to_native)
    print(f"\nAll simulation results have been saved to: {results_file}")

    # Save the metadata about the simulation configurations
    metadata_dict = simulator.config["simulate"]
    metadata_dict.update({
        "num_trials": num_trials,
        "stimulus_ids": stimulus_ids,
        "time_conditions": time_conditions,
        "total_simulations": len(stimulus_ids) * len(time_conditions) * num_trials,
        "simulation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w") as file:
        json.dump(metadata_dict, file, indent=4, default=np_to_native)
    print(f"Simulation metadata has been saved to: {metadata_file}")

    return {
        "results": all_results,
        "metadata": metadata_dict,
        "output_dir": output_dir
    }


if __name__ == "__main__":
    """
    Example usage of the batch simulation function.
    """
    # Run simulations with default parameters
    results = run_batch_simulations(num_trials=1)
    
    # # Example of running with custom parameters:
    # custom_stimuli = [0] # [0, 1, 2]
    # custom_conditions = ["30s", "60s"]
    # custom_trials = 1
    # custom_output = "custom_simulation_results"
    # results = run_batch_simulations(
    #     stimulus_ids=custom_stimuli,
    #     time_conditions=custom_conditions,
    #     num_trials=custom_trials,
    #     output_dir=custom_output
    # )