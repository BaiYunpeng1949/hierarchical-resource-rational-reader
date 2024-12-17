import os
import datetime
import json
import yaml

import utils.constants as const
from simulator_v1001_linux import ReaderAgent

class BatchRunSimulator:
    def __init__(self, mcq_metadata_path, save_path, time_constraints):

        # Read the configuration
        with open("config.yaml") as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
        
        # Get the number of episodes to run
        self._num_episodes = self._config["simulate"]["num_episodes"]
        # Log out
        print(f"\n\n********************************************************************************")
        print(f"Start to run {self._num_episodes} episodes...")
        print(f"********************************************************************************\n\n")

        self.mcq_metadata_path = mcq_metadata_path
        self.save_path = save_path
        self.time_constraints = time_constraints
        self.mcqs = self._load_mcq_metadata()
        # Get the number of images/stimuli from the metadata
        self.num_stimulus_images = len(self.mcqs)
        # Create a root folder for this batch
        self._batches_root_save_dir = self._create_batches_root_save_dir()

        # Initialize the ReaderAgent simulator
        self.reader_agent = ReaderAgent(mcq_metadata=self.mcqs)

    def _load_mcq_metadata(self):
        with open(self.mcq_metadata_path, 'r') as file:
            return json.load(file)

    def _create_batches_root_save_dir(self):
        # Create a folder name related to time
        time_str = datetime.datetime.now().strftime("%m_%d_%H_%M")
        folder_name = f"sim_raw_data_{time_str}_{self._num_episodes}episodes"
        batches_root_dir_dir = os.path.join(self.save_path, folder_name)
        
        # Create this folder if not exists
        if not os.path.exists(batches_root_dir_dir):
            os.makedirs(batches_root_dir_dir)
        return batches_root_dir_dir
    
    # TODO To fix here, maybe not need to save them one by one, but leave it here, fix later
    def _create_individual_trial_dir(self, image_idx, time_constraint):     
        folder_name = f"stimulus_{image_idx}_time_constraint_{time_constraint}s"
        sim_data_dir = os.path.join(self._batches_root_save_dir, folder_name)
        
        # Create this folder if not exists
        if not os.path.exists(sim_data_dir):
            os.makedirs(sim_data_dir)
        return sim_data_dir

    def run_simulation(self, episode_index, stimulus_idx, time_constraint):
        # Set up the directory to save the data
        sim_data_dir = self._create_individual_trial_dir(stimulus_idx, time_constraint)
        
        # Configure reader agent
        self.reader_agent.reset(
            episode_index=episode_index,
            image_env_index=stimulus_idx,
            total_time_limit_in_seconds=time_constraint,
            simulation_data_save_dir=sim_data_dir
        )
        
        # Run the simulation
        self.reader_agent.simulate()

    def batch_run(self):
        # Loop through each episode
        for episode_idx in range(self._num_episodes):
            # Loop through each image index and time constraint
            for image_idx in range(0, self.num_stimulus_images):  # Assuming 0 to 8 based on your description
                for time_constraint in self.time_constraints:
                    stim_idx = image_idx
                    print(f"\n\n*********************************************************************")
                    print(f"(Simulation Run in Batch) Running episode {episode_idx + 1}... out of {self._num_episodes}")
                    print(f"(Simulation Run in Batch) Running simulation image {image_idx}, OR stimulus {stim_idx} with time constraint {time_constraint}s...")
                    print(f"*********************************************************************\n\n")
                    self.run_simulation(episode_idx, stim_idx, time_constraint)


# Example usage:
if __name__ == "__main__":
    # Define constants or load them from a config if needed
    # save_path = "/home/baiy4/reading-model/step5/data/sim_data_in_batches"
    save_path = "/home/baiy4/reading-model/data_analysis/simulation_data"       # The new save path in the data analysis part, outputting final results
    mcq_metadata_path = "/home/baiy4/reading-model/step5/data/assets/MCQ/mcq_metadata.json"
    time_constraints = [30, 60, 90]  # in seconds
    assert all(isinstance(tc, int) for tc in time_constraints), "Time constraints should be integers."
    
    # Initialize the BatchRunSimulator
    simulator = BatchRunSimulator(mcq_metadata_path, save_path, time_constraints)
    
    # Start batch processing
    simulator.batch_run()