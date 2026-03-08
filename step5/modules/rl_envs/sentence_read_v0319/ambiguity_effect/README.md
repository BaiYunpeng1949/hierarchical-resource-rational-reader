
# Sentence-Level Ambiguity Effect Analysis

This directory contains scripts used to evaluate **sentence-level ambiguity effects on eye-movement behavior**
and compare **human data** with **model simulations**.

The pipeline analyzes whether increased sentence ambiguity leads to changes in:

- Regression probability (rereading behavior)
- Skip probability (forward skipping behavior)

The analysis follows three steps:

1. Compute sentence ambiguity using a language model
2. Aggregate eye-movement behavior per sentence and bin the ambiguity values
3. Plot regression trends and export statistics

---

# Step 1: Calculate Sentence Ambiguity

## Purpose

Estimate how **ambiguous or uncertain** a sentence is using a language model.

Ambiguity is computed as the **average entropy of next-word predictions** while reading the sentence left-to-right.

Higher entropy means the model is less certain about what word should come next, indicating higher ambiguity.

---

## Mathematical formulation

For a sentence:

w₁, w₂, ..., wₙ

the ambiguity score is defined as:

ambiguity = (1 / (n − 1)) * Σ H(P(wₜ₊₁ | w₁ ... wₜ))

where

H(P) = − Σ pᵢ log(pᵢ)

is the entropy of the language model’s predicted distribution.

Thus the score measures the **average uncertainty during incremental sentence processing**.

---

## Command

Run:

```bash
python calculate_sentence_ambiguity.py
```

Output:

```
data/sentence_ambiguity.csv
```

Columns:

| column | description |
|------|-------------|
| index | sentence id |
| sentence | reconstructed sentence |
| ambiguity | raw ambiguity score |
| ambiguity_zscore | standardized ambiguity |
| ambiguity_norm | normalized ambiguity (0–1) |

---

# Step 2: Analyze Ambiguity Effects

## Purpose

Measure how ambiguity influences eye-movement behavior across the sentence.

For each sentence:

avg_regression_probability =
mean(regression_probability(wordᵢ))

avg_skip_probability =
mean(skip_probability(wordᵢ))

This provides a **sentence-level behavioral measure**.

---

## Binning

To reduce noise and match prior work, ambiguity values are partitioned into bins.
Within each bin we compute the **mean ambiguity and mean behavioral measure**.

This produces a smoothed dataset used for regression and plotting.

---

## Command

Run:

```bash
python analyze_ambiguity_effect.py
```

Outputs:

Raw sentence-level data:

```
data/ambiguity_effect_human.csv
data/ambiguity_effect_simulation.csv
```

Binned data for plotting:

```
data/ambiguity_effect_human_binned.csv
data/ambiguity_effect_simulation_binned.csv
```

---

# Step 3: Plot Results

## Purpose

Generate regression plots comparing human and simulation behavior.

Each figure shows:

- binned data points
- linear regression line
- 95% confidence interval

The script also reports regression statistics.

---

## Command

Run:

```bash
python plot.py
```

Outputs:

```
figures/ambiguity_regression_1.pdf
figures/ambiguity_regression_2.pdf
```

Each panel is saved as a **separate PDF** so that they can be arranged later in
Adobe Illustrator or other figure-editing tools.

---

## Regression statistics

The script also exports:

```
figures/ambiguity_effect_regression_stats.txt
```

This file reports:

- β (slope of regression line)
- intercept
- R² (variance explained)
- number of points used in regression

Example reporting style in papers:

Regression probability increases with sentence ambiguity  
(Human: β = 0.03, R² = .12; Simulation: β = 0.05, R² = .18).

---

# Pipeline Overview

```
calculate_sentence_ambiguity.py
        ↓
analyze_ambiguity_effect.py
        ↓
plot.py
```

This workflow enables systematic comparison between **human reading behavior**
and **model predictions** under varying sentence ambiguity levels.
