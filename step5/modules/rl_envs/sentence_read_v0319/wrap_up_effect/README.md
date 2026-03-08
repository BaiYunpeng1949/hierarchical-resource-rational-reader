# Sentence-Level Evaluation Pipeline

This directory contains scripts used to analyze sentence-level behavior of the reader agent and compare it with human eye‑movement data.

The pipeline consists of three main steps:

1. **Compute sentence difficulty**
2. **Analyze wrap‑up related regression behavior**
3. **Visualize the relationship between difficulty and boundary rereading**

---

# Step 1: Process the sentences with difficulty levels

## Job

This step computes a **sentence-level difficulty score** for every sentence used in the evaluation pipeline.

The script reads the simulation dataset:

```
data/simulation/all_words_regression_and_skip_probabilities.csv
```

From this file, it reconstructs each sentence using the word sequence associated with each sentence index.

For every reconstructed sentence, the script computes a **sentence difficulty score** based on the surprisal of the **final word** given its preceding context.

The output file:

```
data/sentence_difficulty.csv
```

contains:

| column | description |
|------|-------------|
| index | sentence index |
| sentence | reconstructed sentence |
| difficulty | raw surprisal score |
| difficulty_zscore | standardized difficulty |
| difficulty_norm | min–max normalized difficulty |

---

## Mechanism (math formulation)

Sentence difficulty is computed using **surprisal**, a standard metric in psycholinguistics.

For a sentence with words:

w₁, w₂, ..., wₙ

difficulty is defined as:

difficulty = − log P(wₙ | w₁, w₂, ..., wₙ₋₁)

Where:

- P(wₙ | context) is estimated using **GPT‑2**
- higher surprisal = lower predictability = higher integration difficulty

Normalized variants:

Z-score:

difficulty_zscore = (difficulty − μ) / σ

Min‑max:

difficulty_norm = (difficulty − min) / (max − min)

---

## Command

```
python calculate_sentence_difficulty.py
```

---

# Step 2: Analyze wrap‑up effect

## Job

This step analyzes whether **sentence difficulty influences rereading behavior near sentence boundaries**, which relates to the classical **wrap‑up effect** in reading research.

The script reads:

```
data/human/all_words_regression_and_skip_probabilities.csv
data/simulation/all_words_regression_and_skip_probabilities.csv
data/sentence_difficulty.csv
```

For each sentence, the script identifies the **last k words** (a boundary region) and computes the average regression probability:

avg_regression_probability =
mean(regression_probability(last k words))

This produces one boundary regression value per sentence.

The script outputs:

```
data/wrap_up_human.csv
data/wrap_up_simulation.csv
```

Each file contains:

| column | description |
|------|-------------|
| sentence_id | sentence index |
| sentence | sentence text |
| difficulty_zscore | sentence difficulty |
| k | number of boundary words used |
| avg_regression_probability | average regression probability in boundary region |

---

## Notes

Replicating the wrap‑up effect using regression probability alone can be challenging because:

- wrap‑up effects are typically measured using **fixation durations**
- regression events are relatively **sparse**
- the ZuCo dataset was not designed specifically to manipulate sentence integration difficulty

Nevertheless, the pipeline provides a **systematic method** for examining boundary rereading behavior in both human data and model simulations.

---

## Command

```
python analyze_wrap_up_effect.py
```

---

# Step 3: Plot results

## Job

This step visualizes the relationship between:

```
sentence difficulty → boundary regression probability
```

Both human and simulation data are plotted together.

A **linear regression line** is fitted for each dataset to summarize the trend.

The output figure illustrates whether harder sentences tend to produce more rereading behavior near the boundary.

---

## Command

```
python plot.py
```

The resulting figure will be saved to:

```
figures/wrap_up_regression_plot.png
```

---

# Summary

Pipeline overview:

```
calculate_sentence_difficulty.py
        ↓
analyze_wrap_up_effect.py
        ↓
plot.py
```

This workflow allows sentence‑level evaluation of the reader agent and enables comparison with human eye‑movement behavior.
