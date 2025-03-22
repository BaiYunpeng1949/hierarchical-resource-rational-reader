# Section 2: Sentence Reader Results Analysis

This section contains the analysis of human reading patterns and comparison with model predictions.

## Data Processing Pipeline

1. **Combine Raw Human Data**
   ```bash
   python _combine_human_data.py
   ```
   - Combines individual participant reading pattern files into a single dataset
   - Adds episode_id and participant_id to each trial
   - Output: `_raw_human_data/combined_raw_human_data.json`

2. **Analyze Reading Patterns**
   ```bash
   python analyze_human_data_word_regression_and_skip_probabilities.py
   ```
   - Processes the combined human reading data
   - Calculates word regression and skip probabilities
   - Output: `processed_human_data/all_words_regression_and_skip_probabilities.json`

3. **Generate Figures**
   ```bash
   python plot.py
   ```
   - Creates comparison plots between human and model predictions
   - Generates figures for the paper

## Directory Structure
- `_raw_human_data/`: Contains individual participant reading pattern files and combined dataset
- `processed_human_data/`: Contains processed analysis results
- `figures/`: Generated comparison plots

## Notes
- Make sure to run the scripts in order as they depend on each other's outputs
- The combined dataset preserves original sentence_ids while adding episode_ids for unique trial identification