# Rayner (1979) Digitized Eye-Movement Dataset

This folder contains **digitized data extracted from figures in Rayner
(1979)**:

Rayner, K. (1979). *Eye Guidance in Reading: Fixation Locations Within
Words.* Cognitive Psychology.

The data were digitized from **Figure 1 and Figure 3** of the paper to
reproduce classic empirical patterns of eye movements during reading.

These datasets are used in our study to evaluate whether a computational
reading model produces **human-like within-word fixation behavior and
regression behavior**.

------------------------------------------------------------------------

# Purpose of This Dataset

The reviewer asked whether the model reproduces well-known empirical
phenomena in eye movement control during reading, particularly:

1.  **Preferred Viewing Location (PVL)**\
    Readers tend to land slightly left of the center of a word during
    the first fixation.

2.  **Regression behavior depending on landing position**\
    Landing positions near word edges are more likely to trigger
    additional fixations or regressions.

To evaluate these effects, we digitized Rayner's original histogram data
and converted them into structured datasets that can be compared with
model simulations.

------------------------------------------------------------------------

# Raw Digitized Files

The raw digitized CSV files correspond to different word lengths and two
types of measurements.

Example filenames:

length5_fixations_vs_letter_positions.csv\
length5_regressions_vs_letter_positions.csv\
length6_fixations_vs_letter_positions.csv\
...\
length10_regressions_vs_letter_positions.csv

Each CSV contains **two series digitized from the original figure**.

### Fixation files

-   **one_fixation** -- Words that were fixated only once.
-   **more_fixations** -- Words that received more than one forward
    fixation.

### Regression files

-   **interword_only** -- Regressions that move back to a previous word.
-   **both** -- Both interword and intraword regressions.

------------------------------------------------------------------------

# Processed Datasets

The script `generate_dataset.py` converts the raw digitized files into
clean datasets.

Four final datasets are generated:

  ----------------------------------------------------------------------------------------------
  File                                       Description
  ------------------------------------------ ---------------------------------------------------
  rayner_forward_fixations_multiple.csv      Words that received **multiple forward fixations**

  rayner_forward_fixations_single_only.csv   Words fixated **only once**

  rayner_regressions_both.csv                **All regressions** (interword + intraword)

  rayner_regressions_interword_only.csv      **Interword regressions only**
  ----------------------------------------------------------------------------------------------

Each dataset has the following format:

word_length \| letter_number \| proportion_of_fixation

or

word_length \| letter_number \| probability_of_regression

------------------------------------------------------------------------

# Variable Definitions

**word_length**\
Number of letters in the word.

**letter_number**\
Letter position within the word (starting from 0).

**proportion_of_fixation**\
Percentage of fixations landing at that letter position.

**probability_of_regression**\
Probability of a regression occurring from that landing position.

------------------------------------------------------------------------

# Data Cleaning Steps

During preprocessing the script performs the following steps:

1.  Remove non-numeric rows introduced during figure digitization (e.g.,
    rows containing "Y").
2.  Convert percentage values into numeric format.
3.  Assign **letter positions starting from 0**.
4.  Extract the **word length from the filename**.
5.  Merge all word lengths into unified datasets.

------------------------------------------------------------------------

# How to Reproduce the Dataset

Run the dataset generation script:

```bash
python generate_dataset.py
```

This will generate the four processed CSV files used in the analysis.

------------------------------------------------------------------------

# Use in the Paper

These datasets are used to generate the **human benchmark curves** in
the paper's evaluation figure:

-   **Panel A** -- Preferred Viewing Location distribution
-   **Panel B** -- Regression probability vs. landing position

The model's simulated fixation behavior is compared against these human
distributions.