# Sentence Reading Under Time Pressure Environment (v0604)

This environment extends the base sentence reading model to incorporate time-pressure specific components for reading under time constraints.

## Overview

The environment builds upon the base sentence reading model and adds time-pressure specific components:
- Time perception mechanisms
- Reading speed adaptation
- Time constraint handling (30s, 60s, 90s)
- Performance metrics under different time conditions

## Structure

- `SentenceReadingEnv.py`: Main environment class for sentence reading under time pressure
- `utils/`: Utility functions and data processing tools
  - `process_sentence_dataset.py`: Scripts for analyzing simulation results and generating metrics
- `simulated_results/`: Directory containing simulation outputs and analysis
  - Contains episode logs and analysis results for different time conditions

## Key Features

1. **Time Pressure Integration**:
   - Three time conditions: 30s, 60s, and 90s
   - Time-aware reading behavior
   - Adaptive reading speed based on time constraints

2. **Reading Metrics**:
   - Regression rate: Percentage of words that are regression targets
   - Skip rate: Percentage of words skipped during first-pass reading
   - Reading speed: Words per minute under different time constraints

3. **Analysis Tools**:
   - Statistical analysis of reading patterns
   - Visualization of metrics across time conditions
   - Comparison of reading behaviors under different time pressures

## Usage

1. **Running Simulations**:
   ```python
   # Example usage of the environment
   env = SentenceReadingEnv()
   obs, info = env.reset()
   ```

2. **Analyzing Results**:
   ```python
   # Process simulation results
   python utils/process_sentence_dataset.py
   ```

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- StatsModels

## Metrics

The environment tracks several key metrics:
- `sentence_wise_regression_rate`: Rate of regressions per sentence
- `sentence_wise_skip_rate`: Rate of word skips per sentence
- Reading speed in words per minute
- Time condition-specific performance measures

## Analysis Output

The analysis generates:
1. Statistical comparisons between time conditions
2. Visualization of reading patterns
3. Performance metrics under different time constraints
4. ANOVA and post-hoc test results for significant differences

## Reproduction

To reproduce the results and generate the dataset, follow these steps:

1. **Setup Environment**:
   ```bash
   cd step5/simulators/simulator_v20250604/sub_models/sentence_read_v0604
   ```

2. **Generate Dataset**:
   ```bash
   # Step 1: Process stimulus data and add word features
   python utils/process_my_stimulus_to_sentences_with_word_features.py
   # This script:
   # - Downloads and processes SUBTLEX-US word frequencies
   # - Downloads required NLTK data
   # - Trains a trigram language model
   # - Processes stimulus data with word-level features
   # Input: assets/processed_my_stimulus_for_text_reading.json
   # Input: assets/metadata_sentence_indeces.json
   # Output: assets/processed_my_stimulus_with_word_features.json

   # Step 2: Add observations and integration features
   python utils/process_my_sentences_with_word_features_add_obs.py
   # This script:
   # - Uses BERT model for word integration difficulty
   # - Computes word prediction and surprisal
   # - Adds observation features to the dataset
   # Input: assets/processed_my_stimulus_with_word_features.json
   # Output: assets/processed_my_stimulus_with_observations.json
   ```

3. **Run Simulations**:
   ```bash
   # Run simulations for different time conditions
   python run_simulations.py --time_condition 30 --output_dir simulated_results/30s/
   python run_simulations.py --time_condition 60 --output_dir simulated_results/60s/
   python run_simulations.py --time_condition 90 --output_dir simulated_results/90s/
   ```

4. **Analyze Results**:
   ```bash
   # Generate analysis and visualizations
   python utils/analyze_results.py --input_dir simulated_results/ --output_dir analysis_results/
   ```

### File Descriptions

- `process_my_stimulus_to_sentences_with_word_features.py`: 
  - Processes raw stimulus data into word-level features
  - Downloads and processes SUBTLEX-US word frequencies
  - Trains a trigram language model for word predictability
  - Calculates word difficulty and frequency features

- `process_my_sentences_with_word_features_add_obs.py`:
  - Uses BERT model to compute word integration difficulty
  - Adds observation features including:
    - Word integration probability
    - Surprisal
    - Integration difficulty
    - Word prediction with preview
    - Ranked word integration probability

- Required Input Files:
  - `assets/processed_my_stimulus_for_text_reading.json`: Raw stimulus data
  - `assets/metadata_sentence_indeces.json`: Sentence metadata
  - `assets/processed_my_stimulus_with_word_features.json`: Intermediate file with word features
  - `assets/processed_my_stimulus_with_observations.json`: Final processed dataset

- Output Directories:
  - `simulated_results/`: Contains simulation outputs for each time condition
  - `analysis_results/`: Contains analysis outputs and visualizations

### Dependencies

- Python 3.8+
- NumPy
- Pandas
- NLTK
- Transformers (BERT)
- Torch
- tqdm
- SciPy
- StatsModels