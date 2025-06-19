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

from sub_models.sentence_read_v0604.SentenceReadingEnv import SentenceReadingUnderTimePressureEnv
from sub_models.text_read_v0604.TextReadEnv import TextReadingUnderTimePressureEnv

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
        self.env = TextReadingUnderTimePressureEnv()

        # Load the pre-trained model
        try:
            self._model = PPO.load(self._model_path, self.env, custom_objects={"observation_space": self.env.observation_space, "action_space": self.env.action_space})
        except (RuntimeError, TypeError) as e:
            warnings.warn(f"Could not deserialize object: {e}")
            raise e

        print(f"{'='*50}\n" 
              f"Successfully loaded the pre-trained {self._model_info['env_name']} model from {self._model_path}.\n"
              f"{'='*50}\n")
            
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
        TODO: needs to be separated because the time needs to be returned by the lower-level agent.
        """
        self.text_reading_steps += 1
        self.action, self._states = self._model.predict(self._obs, deterministic=True)
        self._obs, self._reward, self.done, self._truncated, self._info = self.env.step(action=self.action, time_info=time_info)
        # Get the step-wise log
        self.text_reading_logs = self.env.get_individual_step_log()


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

        print(f"{'='*50}\n"
              f"Successfully loaded the pre-trained {self._model_info['env_name']} model from {self._model_path}.\n"
              f"{'='*50}\n")
            
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
        self.sentence_reading_logs = self.env.get_individual_step_log()        # TODO code in the original env


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
            NOTE: if want to compare the gaze duration vs. time conditions. Then I need to retrain that model to be time-awared, then integrate here to run simulations.
        
        Version 0612
            Based on the version 0612, but using the real reading time.


        Version 0617
            Using the real reading time done by the simulation.
            TODO: train a faster reading speed.
            NOTE: try later -- use the handcrafted sentence comprehension to guide the text reader.
        """

        # Read the configuration file
        with open(CONFIG_PATH, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # Initialize the readers
        self.text_reader = TextReader()
        self.sentence_reader = SentenceReader()

        # Time-related variables (states)
        self._total_time = None     # in seconds
        self._elapsed_time = None   # in seconds
        self._remaining_time = None # in seconds

        # Read the assests -- NOTE: not needed here, all managed in the environment, just changing the modes

        # Logs
        self._episode_index = None
        self._stimulus_index = None
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
            self.current_sentence_index = self.text_reader.env.current_sentence_index
            self.actual_reading_sentence_index = self.text_reader.env.actual_reading_sentence_index

            # Having the sentence index, read the sentence using the sentence reader, and update the time consumption (and maybe also the comprehension score)
            sentence_reading_time_consumed = self._simulate_sentence_reading()

            # Update the time-related variables
            self._elapsed_time += sentence_reading_time_consumed
            self._remaining_time = self._total_time - self._elapsed_time
            # Update text logs
            self._update_text_reading_logs()
            # Update the time-info
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

        # Reset the individual sentence reading logs
        self._individual_sentence_reading_logs = []

        while not self.sentence_reader.done:

            # Determine which word to read using the RL model
            self.sentence_reader.step()

            # Get the reading word index
            self.current_word_index = self.sentence_reader.env.current_word_index     # TODO need to differentiate from the word that is being actually read NOTE: the regressed and skipped words will not be tracked down; so better check the reading sequence
            
            # Update sentence logs
            self._update_sentence_reading_logs()        
            # TODO I can do the cross validation using the reading_sequence and logs from that reader
            # TODO check whether the index input is correct -- Error: list index out of range

        # Return the time consumed by the sentence reading
        return self.sentence_reader.env.elapsed_time


    ########################################################## Helper functions ##########################################################
    def _init_logs(self):
        """
        Initialize the data log dictionary.    NOTE: modify later if needed.
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
    
    def _update_sentence_reading_logs(self):
        """
        Update the sentence reading logs.
        """
        self._individual_sentence_reading_logs.append(self.sentence_reader.sentence_reading_logs)
        # self._single_episode_logs["text_reading_logs"][text_reading_step]["sentence_reading_logs"].append(self.sentence_reader.sentence_reading_logs)     # TODO debug this later -- check whether the text_reading_step is correct
    
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
            json.dump(self._single_episode_logs, file, indent=4)
        print(f"The text-level sim data logs are stored at: {os.path.join(simulation_folder_path, 'text_reading_logs.json')}")      

        # Save the metadata about the simulation configurations
        metadata_dict = self.config["simulate"]
        with open(os.path.join(simulation_folder_path, "metadata.json"), "w") as file:
            json.dump(metadata_dict, file, indent=4)
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
        json.dump(all_results, file, indent=4)
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
        json.dump(metadata_dict, file, indent=4)
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