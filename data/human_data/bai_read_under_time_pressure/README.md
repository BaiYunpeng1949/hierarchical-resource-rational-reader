# Human Reading Data Analysis

This directory contains the analysis of human reading behavior under different time constraints, using eye-tracking data from the ZUCO dataset.

## Data Source

The analysis uses the integrated corrected human scanpath data from `11_18_17_40_integrated_corrected_human_scanpath.json`, which contains:
- Matched word indices with bounding boxes
- Fixation data across different time conditions
- Participant and stimulus information

## Analysis Procedure

The analysis is performed using `calculate_effects.py`, which:

1. Loads and processes the scanpath data
2. Calculates various reading metrics:
   - Fixation duration
   - Saccade length
   - Number of fixations
   - Word skip rate
   - Regression rate

3. Performs statistical analysis:
   - One-way ANOVA for each metric across time conditions
   - Post-hoc Tukey HSD tests for significant differences
   - Effect size calculations (eta-squared)

4. Generates visualizations:
   - Bar plots with standard deviation error bars
   - Statistical significance indicators
   - ANOVA results and effect sizes

## Metrics Used for Model Simulation

The following metrics are analyzed as potential effects for the model to simulate:

1. **Fixation Duration**
   - Average time spent on each fixation
   - Measured in milliseconds
   - Indicates processing time per word

2. **Saccade Length**
   - Distance between consecutive fixations
   - Measured in pixels
   - Indicates reading strategy and eye movement patterns

3. **Number of Fixations**
   - Total fixations per trial
   - Indicates reading effort and attention distribution

4. **Word Skip Rate**
   - Percentage of words skipped during reading
   - Indicates reading efficiency and strategy
   - Calculated as percentage of words with index difference > 1

5. **Regression Rate**
   - Percentage of backward saccades
   - Indicates re-reading behavior
   - Calculated as percentage of fixations where next word index < current word index

## Output

The analysis generates:
1. `metrics_and_stats.json`: Contains all calculated metrics and statistical results
2. Individual plots for each metric:
   - `fix_duration_with_stats.png`
   - `saccade_length_with_stats.png`
   - `num_fixations_with_stats.png`
   - `skip_rate_with_stats.png`
   - `regression_rate_with_stats.png`

Each plot includes:
- Mean values with standard deviation error bars
- Statistical significance indicators
- ANOVA results (F-statistic, p-value)
- Effect sizes for significant results

## Usage

To run the analysis:
```bash
python calculate_effects.py
```

The script will:
1. Load the integrated corrected scanpath data
2. Process and analyze the data
3. Generate statistical results
4. Create visualizations
5. Save all outputs to the `calculated_effects` directory

## Notes

- The analysis uses the corrected scanpath data to ensure accurate word-fixation mapping
- Statistical significance is determined at p < 0.05
- Effect sizes are calculated using eta-squared
- All metrics are analyzed across different time conditions to understand their variation