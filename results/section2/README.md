# Section 2: Sentence Reader Results Analysis

This section contains the analysis of human reading patterns and comparison with model predictions.

## Data Processing Pipeline

1. **Generate Simulated Results**
   ```bash
   python generate_simulated_results.py
   ```
   - Runs RL model to generate simulated reading patterns
   - Output: `_raw_human_data/raw_simulated_results.json`

2. **Combine Raw Human Data**
   ```bash
   python _combine_human_data.py
   ```
   - Combines individual participant reading pattern files into a single dataset
   - Adds episode_id and participant_id to each trial
   - Output: `_raw_human_data/combined_raw_human_data.json`

3. **Analyze Reading Patterns**
   ```bash
   # Analyze human data
   python analyze_human_data_word_regression_and_skip_probabilities.py
   
   # Analyze simulated data
   python analyze_sim_results_word_regression_and_skip_probabilities.py
   ```
   - Processes both human and simulated reading data
   - Calculates word regression and skip probabilities
   - Outputs:
     - `processed_human_data/all_words_regression_and_skip_probabilities.csv`
     - `processed_simulated_results/all_words_regression_and_skip_probabilities.csv`

4. **Generate Comparison Figures**
   ```bash
   python plot.py
   ```
   - Creates comparison plots between human and model predictions
   - Generates four figures in `figures/`:
     1. `comparison_skip_probability_vs_word_length.png`
     2. `comparison_skip_probability_vs_log_word_frequency.png`
     3. `comparison_regression_probability_vs_word_difficulty.png`
     4. `comparison_regression_probability_vs_skip_probability.png`
   - Each plot shows:
     - Human data (blue) and simulated data (red)
     - Regression lines with confidence bands
     - R² values for both datasets

## Directory Structure
- `_raw_human_data/`: Contains individual participant reading pattern files, combined dataset, and simulated results
- `processed_human_data/`: Contains processed human data analysis results
- `processed_simulated_results/`: Contains processed simulated data analysis results
- `figures/`: Generated comparison plots

## Notes
- Make sure to run the scripts in order as they depend on each other's outputs
- The combined dataset preserves original sentence_ids while adding episode_ids for unique trial identification
- Simulated results use "SIM" as participant_id to distinguish from human data
- If any plot encounters issues (e.g., identical x values), it will be skipped with an informative message