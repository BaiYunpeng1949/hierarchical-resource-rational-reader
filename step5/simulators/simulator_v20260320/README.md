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
- `pretrained_rl_models`: Save all the models need to run the simulation. They are all pretrained RL policy weights. Please copy paste all of them under `/home/baiy4/reader-agent-zuco/step5/simulators/simulator_v20250604/sub_models/training/saved_models`.

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

## Reproduction
Procedure
1. To run simulation for batches (when fixed with optimized parameters, single trial) 
```bash 
cd step5/simulators/simulator_v20250604/
```
then 

```bash 
python simulator.py single
```

for default. Or 

```bash
python simulator.py single --stimuli 0-8 --conds 30s,60s,90s --trials 1 \ --rho_inflation_percentage 0.22 --w_skip_degradation_factor 0.78 --coverage_factor 1.2 # Optional: --out simulated_results/custom_run_folder
``` 
for specific parameters.
   - Version 1010 best params: (We recommend running this for best params' simualation results)
      ```bash
      python simulator.py single --stimuli 0-8 --conds 30s,60s,90s --trials 1 --rho_inflation_percentage 0.29 --w_skip_degradation_factor 0.7 --coverage_factor 1.30
      ```
2. Find the simulated results here: `/home/baiy4/reader-agent-zuco/step5/simulators/simulator_v20250604/simulated_results`, copy the folder name, e.g., `20250710_1023_trials1_stims9_conds3`.
3. `cd step5/simulators/simulator_v20250604/utils/`, paste the folder name in `analyze_data.py`, then `python analyze_data.py`, find the plotted figures in the same copied folder.

Parameter Inference Procedure
1. With default parameters `python simulator.py grid`
2. With specified parameters `python simulator.py grid --rho 0.1,0.3,0.02 --w 0.5,1.0,0.02 --cov 0.0,3.0,0.1`
3. Go to folder `step5/simulators/simulator_v20250604/parameter_inference` to continue analysis and ploting.


# Parameter Inference Pipeline

This module implements a reproducible pipeline for **parameter inference** of our reading simulator.  
The workflow takes simulated results generated from a parameter grid, evaluates them against human benchmark data, and visualizes the best-fitting parameter sets.

---

## Project Structure

parameter_inference/
├── human_data/
│ └── analyzed_human_metrics.json # Ground truth metrics from human experiment
├── simulation_data/
│ └── rho_0.100__w_0.500__cov_0.00/ # Example parameter combo run
│ ├── all_simulation_results.json
│ ├── metadata.json
│ ├── processed_fixation_sequences.json (generated)
│ ├── analyzed_fixation_metrics.json (generated)
│ └── comparison_human_vs_sim.png (generated)
├── grid_inference_summary.csv # Summary table of all combos after inference
├── _analyze_data.py # Functions to process & plot metrics
├── infer_parameters.py # Runs inference, compares sims vs humans
├── plot.py # Plots top-k parameter sets vs human
└── README.md


---

## Design

1. **Simulation Data**  
   - Generated by running the reading simulator in **grid mode**.  
   - Each parameter combination (ρ, w, cov) produces a folder under `simulation_data/` with raw outputs (`all_simulation_results.json`, `metadata.json`).

2. **Human Data**  
   - `human_data/analyzed_human_metrics.json` contains aggregated behavioral metrics (reading speed, skip rate, regression rate) for baseline comparison.

3. **Analysis & Inference**  
   - `_analyze_data.py`: converts raw simulation logs into fixation sequences, per-episode metrics, and plotting utilities.  
   - `infer_parameters.py`:  
     - For each parameter-combo folder, ensures metrics are computed (calling `_analyze_data.py`).  
     - Aggregates per-condition means.  
     - Computes discrepancy (SSE or L1) vs human metrics.  
     - Produces `grid_inference_summary.csv`, sorted by loss.  

4. **Visualization**  
   - `plot.py`: reads `grid_inference_summary.csv`, selects the top-k parameter sets, and plots **human vs simulation** using the existing plotting schema.   
   - Figures and params text files are saved directly inside each best-run folder.

---

## Reproduction Commands

### 1. Run Parameter Inference
```bash
python infer_parameters.py --grid_dir simulation_data/ --human human_data/analyzed_human_metrics.json --loss sse --norm zscore --topk 10
```

```bash
python plot.py \
  --grid_dir simulation_data \
  --human human_data/analyzed_human_metrics.json \
  --topk 3
```

### 2. Run the Bayesian Inference
```bash 
python bayesian_inference.py \
  --human human_data/analyzed_human_metrics.json \
  --out_root parameter_inference/bayes_runs \
  --iters 40 --init 8 --cand 512 --xi 0.01 \
  --bounds_rho 0.10 0.30 --bounds_w 0.50 1.00 --bounds_cov 0.00 3.00 \
  --stimuli 0-8 --conds 30s,60s,90s --trials 1 \
  --loss sse \
  --warm_start_from parameter_inference/simulation_data/grid_inference_summary.csv
```
So the best way would be use the grid search to roughly find some good start. Then use the Bayesian inference.

```bash
python plot.py --grid_dir parameter_inference/bayes_runs --human human_data/analyzed_human_metrics.json --topk 3
```

# Plot the final unified figures for the paper
Go to `step5/simulators/simulator_v20250604/plot` to reproduce results reported in Results 'Speed-accuracy trade-off when reading under time pressure'.

# Plot

## Plot for the figure 3d. Unified.

First, generate plotable files from the human data and prior-aggregation simulation data. Get a unified mean, std format.
```bash
cd assets
 
python build_aggregated_panel_metrics.py --human_eye human_data/human_eye_movement_metrics.json --human_mcq human_data/human_mcq_acc_metrics.json --human_fr human_data/human_free_recall_metrics.json --sim_eye simulation_data/simulation_eye_movement_metrics.json --sim_comp simulation_data/comprehension_results_20251006-150327.json --out aggregated_panel_metrics.json
```

Plot.
```bash
cd plot

python plot_eye_comp_from_aggregated_metrics.py 
```

## Plot for the baseline comparisons Figure 5. Unified.

Generate usable data metrics in json.
```bash
cd assets
python build_aggregated_panel_metrics_baseline.py --folder simulation_data_baselines/ 
```

Plot
```bash
python plot_eye_comp_and_baselines_from_aggregated_metrics.py 
```

## Plot for the French corpus effects replication (Extended Data Figure 10)

Generate non-aggregated (by episode) data
```bash
cd assets
python build_french_corpus_effects_metrics.py   --root simulation_data_effects_replication/rho_0.290__w_0.700__cov_1.30   --lang en   --out analyzed_by_episode_fixation_metrics.json
```

Plot
```bash
python plot_french_corpus_effects.py --input assets/analyzed_by_episode_fixation_metrics.json --out french_corpus_effects_panel.pdf
```