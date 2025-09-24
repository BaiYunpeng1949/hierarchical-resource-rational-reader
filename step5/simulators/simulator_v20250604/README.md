# Time-Pressure Reading Simulator

This simulator extends the models from previous sections to incorporate time-perception abilities for reading under time pressure scenarios.

## Overview

The simulator builds upon the base reading models and adds time-pressure specific components:
- Time perception mechanisms
- Reading speed adaptation
- Time constraint handling
- Performance metrics under different time conditions

## Structure

- `simulator.py`: Main simulator implementation
- `sub_models/`: Contains the time-pressure adapted model architectures and environments
  - `text_read_v0604/`: Text-level reading model
  - `sentence_read_v0604/`: Sentence-level reading model
- `utils/`: Utility scripts for testing and validation
  - `_test_code.py`: Scripts for validating simulation results
  - `analyze_data.py`: Scripts for analyzing and plotting simulation results
- `processed_human_data/`: Directory for human data processing
  - `convert_format.py`: Script to convert raw human metrics to analysis format
- `simulated_results/`: Directory for storing simulation outputs
- `config.yaml`: Configuration file for model parameters and simulation settings

## Data Preparation and Analysis Workflow

### 1. Prepare Human Data

The human data needs to be converted to the correct format before comparison with simulation results:

```bash
# Navigate to the processed_human_data directory
cd processed_human_data

# Run the conversion script
python convert_format.py
```

This will:
- Read the raw human metrics from `raw_human_metrics.json`
- Convert the metrics to the analysis format
- Save the processed metrics to `analyzed_human_metrics.json`

### 2. Run Simulations

Run the simulator to generate results:

```python
from simulator import run_batch_simulations

# Run with default parameters
results = run_batch_simulations()
```

This will create a new directory in `simulated_results/` with the format: `YYYYMMDD_HHMM_trials{N}_stims{N}_conds{N}`

### 3. Generate Comparison Plots

To generate comparison plots between human and simulation data:

1. Open `utils/analyze_data.py`
2. Update the `file_name` variable to match your simulation results folder:
   ```python
   file_name = "20250614_2133_trials1_stims9_conds3"  # Replace with your folder name
   ```
3. Run the script:
   ```bash
   cd utils
   python analyze_data.py
   ```

The script will:
- Process the simulation results
- Generate fixation sequences
- Analyze metrics
- Create comparison plots

The final plot will be saved as `metrics_comparison.png` in your simulation results folder.

## Configuration

The simulator can be configured through the following parameters:

1. **Stimulus Selection**:
   - Default range: 0-8 stimulus IDs
   - Can be customized when calling `run_batch_simulations`

2. **Time Conditions**:
   - Available conditions: "30s", "60s", "90s"
   - Can be customized when calling `run_batch_simulations`

3. **Number of Trials**:
   - Default: 1 trial per stimulus-condition combination
   - Can be increased for more robust results

4. **Output Directory**:
   - Default: Creates timestamped directory in `simulated_results/`
   - Format: `YYYYMMDD_HHMM_trials{N}_stims{N}_conds{N}`

## Running Simulations

### Basic Usage

```python
from simulator import run_batch_simulations

# Run with default parameters
results = run_batch_simulations()
```

### Custom Configuration

```python
# Customize simulation parameters
results = run_batch_simulations(
    stimulus_ids=[0, 1, 2, 3, 4],  # Specific stimuli to simulate
    time_conditions=["30s", "60s", "90s"],  # Time conditions to test
    num_trials=5,  # Number of trials per combination
    output_dir="custom_simulation_results"  # Custom output directory
)
```

## Validating Results

The simulator includes validation tools to check the consistency of simulation results.

### Reading Metrics Calculation

The simulator calculates several key reading metrics:

1. **Reading Speed**:
   - Calculated as words per minute (wpm)
   - Formula: `(total_words_read / total_time) * 60`

2. **Skip Rate**:
   - Measures the proportion of saccades that skip words
   - Calculated by counting saccades where words are skipped between fixations
   - Formula: `total_skip_saccades / total_saccades`
   - Range: 0 to 1 (0% to 100%)
   - Example: A skip rate of 0.3 means 30% of saccades skipped words

3. **Regression Rate**:
   - Measures the proportion of fixations that revisit previous words
   - Calculated by tracking the last read word index and counting revisits
   - Formula: `total_revisit_words / total_valid_fixations`
   - Range: 0 to 1 (0% to 100%)
   - Example: A regression rate of 0.2 means 20% of fixations were revisits

Note: Both skip rate and regression rate are calculated excluding unmapped word indices (-1).

### Word Count Validation

Use the `_test_code.py` script to verify that word counts match between text reading logs and sentence reading summaries:

```python
from utils._test_code import check_sentence_word_counts

# Validate results from a specific simulation run
results_file = "simulated_results/20250613_1652_trials5_stims9_conds3/all_simulation_results.json"
check_sentence_word_counts(results_file)
```

The validation script will:
1. Compare word counts between text logs and sentence summaries
2. Report any mismatches with detailed context
3. Show total number of mismatches found
4. Warn about any missing data

### Output Structure

Each simulation run generates:
1. `all_simulation_results.json`: Contains all simulation data
2. `metadata.json`: Configuration and summary information
3. `processed_reading_metrics.json`: Processed reading metrics
4. `processed_fixation_sequences.json`: Processed fixation sequences
5. `analyzed_fixation_metrics.json`: Analyzed fixation metrics
6. `metrics_comparison.png`: Comparison plot with human data

## Dependencies

- Python 3.8+
- Required packages:
  - stable-baselines3
  - gymnasium
  - PyYAML
  - numpy
  - torch
  - matplotlib

## Reproduction
Procedure
1. To run simulation for batches (when fixed with optimized parameters, single trial) `cd step5/simulators/simulator_v20250604/`, then `python simulator.py single` for default. Or `python simulator.py single --stimuli 0-8 --conds 30s,60s,90s --trials 1 \ --rho_inflation_percentage 0.22 --w_skip_degradation_factor 0.78 --coverage_factor 1.2 # Optional: --out simulated_results/custom_run_folder` for specific parameters.
2. Find the simulated results here: `/home/baiy4/reader-agent-zuco/step5/simulators/simulator_v20250604/simulated_results`, copy the folder name, e.g., `20250710_1023_trials1_stims9_conds3`.
3. `cd step5/simulators/simulator_v20250604/utils/`, paste the folder name in `analyze_data.py`, then `python analyze_data.py`, find the plotted figures in the same copied folder.

Parameter Inference Procedure
1. With default parameters `python simulator.py grid`
2. With specified parameters `python simulator.py grid --rho 0.1,0.3,0.02 --w 0.5,1.0,0.02 --cov 0.0,3.0,0.1`
3. Go to folder `parameter_inference` to continue analysis and ploting.

## Troubleshooting

Common issues and solutions:

1. **Word Count Mismatches**:
   - Run the validation script to identify inconsistencies
   - Check the simulation logs for the specific stimulus and time condition
   - Verify the sentence reading environment configuration

2. **Missing Data**:
   - Ensure all required model checkpoints are present
   - Verify the stimulus data format
   - Check the configuration file paths

3. **Performance Issues**:
   - Reduce the number of trials or stimuli
   - Use a smaller subset of time conditions
   - Consider running simulations in parallel (future feature)

4. **Plotting Issues**:
   - Verify that the simulation folder name in `analyze_data.py` matches your results
   - Check that `analyzed_human_metrics.json` exists in the correct location
   - Ensure all required metrics are present in both human and simulation data