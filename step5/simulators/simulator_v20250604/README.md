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
- `simulated_results/`: Directory for storing simulation outputs
- `config.yaml`: Configuration file for model parameters and simulation settings

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

## Dependencies

- Python 3.8+
- Required packages:
  - stable-baselines3
  - gymnasium
  - PyYAML
  - numpy
  - torch

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