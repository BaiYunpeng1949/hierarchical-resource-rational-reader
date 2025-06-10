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