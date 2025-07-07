##Documentation
Adaptive reading strategy / reading actions: revisit the sentences with lower appraisal levels. Prioritize the sentences with lower appraisals first.

NOTE: something to improve: 
From the Kintsch's paper, he mentioned something like "The mechanism of only a limited number of propositions could be processed in one cycle due to the STM limit, is akin to having a reading strategy where only the most relevant or recently processed information is actively maintained, while the rest may be dropped if not reinforced by further processing."
Action: so later I could introduce a memory decay over sentences (in a higher level, not micro-propositions), showing the similar trend.
Warning: to make the above-mentioned effect clear, maybe need to make the agent's initial appraisals range from an overall higher region. 

## Why use softmin instead of geometric mean for comprehension aggregation?

Previously, the geometric mean was used to aggregate sentence appraisals:
- The geometric mean hides big gaps: progress on any sentence boosts the mean, so after a few high-score sentences, low outliers hardly move the needle.
- Raising a low score (e.g., 0.3 → 0.6) earns less than raising a high score (0.9 → 1.0), even though the first jump is more useful for comprehension.
- This leads the agent to prefer easy top-ups over patching the biggest gaps, which is not optimal for comprehension.

**Softmin** is now used instead:
- Softmin is a differentiable, risk-averse utility that acts like an "almost-minimum"; it gives much more weight to the lowest scores.
- For typical temperature values (τ in [0.15, 0.30]), one bad sentence dominates the utility enough that the agent is incentivized to repair it first.
- This means the agent receives much stronger reinforcement for closing the biggest gaps, rather than just topping up already high scores.
- Softmin is smooth and differentiable, so it works well with gradient-based RL algorithms.

**Intuition:**
> Softmin acts like a risk-averse utility: the colder the temperature, the more the agent behaves as if "the weakest link sets the value of the whole chain," so the rational move is to close the biggest gap first.

## Metrics Calculation and Visualization

### Regression Detection
- **Detection Method**: Compare `actual_reading_sentence_index` with previous sentence index
- **Regression Event**: When `actual_reading_sentence_index < current_sentence_index`
- **Progression Event**: When `actual_reading_sentence_index > current_sentence_index`

### Comprehension Metrics
- **Ongoing Comprehension**: `on_going_comprehension_log_scalar` from each step
- **Comprehension Change**: `comprehension_after - comprehension_before` for regression events
- **Normalization**: All comprehension scores are between 0 and 1

### Appraisal Metrics
- **Initial Appraisals**: `init_sentence_appraisal_scores_distribution` for each episode
- **Appraisal vs Regression**: Count regressions to sentences with different appraisal scores
- **Low vs High Appraisal**: Threshold at 0.5 (low: <0.5, high: ≥0.5)

### Timing Metrics
- **Normalized Steps**: `step_index / max_steps` (0 = start, 1 = end of longest episode)
- **Episode Length**: Total number of steps excluding termination step
- **Regression Timing**: When regressions occur relative to episode progress

### Visualization Methods

#### 1. Box Plots (Regression Timing Analysis)
- **X-axis**: Reading action types (Read Next vs Regress)
- **Y-axis**: Normalized reading steps (0-1)
- **Box Elements**: 
  - Box: Interquartile range (Q1 to Q3)
  - Line: Median (50th percentile)
  - Whiskers: Full data range
  - Outliers: Individual points beyond whiskers

#### 2. Histograms and Scatter Plots (Appraisal vs Regression)
- **Histogram**: Bins appraisal scores (0-1) and counts regressions per bin
- **Scatter Plot**: Appraisal scores vs regression counts with trend line
- **Correlation**: Pearson correlation coefficient between appraisal and regression frequency

#### 3. Scatter Plot (Comprehension Impact)
- **X-axis**: Number of regressions (1, 2, 3, ...)
- **Y-axis**: Comprehension scores after each regression
- **Trend Line**: Linear regression showing relationship
- **Correlation**: Measures if more regressions improve comprehension

### Statistical Analysis
- **Mean/Median**: Central tendency measures for all metrics
- **Percentiles**: Q1 (25th), Q3 (75th) for distribution analysis
- **Correlation Coefficients**: Relationship strength between variables
- **Percentage Analysis**: Proportion of positive/negative comprehension changes

##Tunable parameters
1. apply stm limit or not [Kintsch's];
2. apply memory gist or not [Kintsch's];
3. stm size s [Kintsch's].
4. prior knowledge classification [Are Good Texts Alawys Better?] (currently by LLM's inputted prompts).
5. (low prority) propositions' integration and inference (currently we do not differentiate this, all done by LLM itself).

##Reproduction
Procedure: 

1. cd step5/
2. python main.py
3. cd step5/modules/rl_envs/text_comprehension_v0516/utils
4. python plot.py