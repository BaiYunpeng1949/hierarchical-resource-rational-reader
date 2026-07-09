# McConkie / Rayner Word Landing Position Dataset

This folder contains digitized data extracted from classic eye‑movement
studies on **fixation landing positions within words during reading**.\
The data are used to reproduce the **Preferred Viewing Location (PVL)**
distribution and compare it with model simulations.

The dataset is used in our paper to evaluate whether the model produces
**human‑like within‑word fixation behavior**.

------------------------------------------------------------------------

# Why This Dataset Is Needed

One of the reviewer questions asks whether the model produces realistic
fixation locations when landing on words.

In human reading research, a well‑established phenomenon is the
**Preferred Viewing Location (PVL)**:

-   When readers fixate a word for the first time, the fixation tends to
    land **slightly left of the word center**.
-   The distribution of first fixation locations forms a **bump‑shaped
    distribution** across letter positions.

This phenomenon is documented in classic studies such as:

-   Rayner (1979) *Eye Guidance in Reading: Fixation Locations Within
    Words*
-   McConkie et al. (1988) *Eye movement control during reading: Initial
    fixation locations on words*

To evaluate the model, we digitized the histogram data from these papers
and compare:

    Human PVL distribution  vs  Model simulated landing positions

This forms **Panel A of Figure 3a** in the paper.

------------------------------------------------------------------------

# Files in This Folder

    digitilized_data/
        proportion_of_fixations_vs_letter_positions.csv
        generate_dataset.py
        mcconkie.csv
    README.md

## 1. proportion_of_fixations_vs_letter_positions.csv

Raw digitized data extracted from the histogram in Rayner (1979).

Each pair of columns corresponds to a **word length condition**.

Example structure:

    length3_x , length3_y , length4_x , length4_y , ...

Where:

-   `x` = landing position within the word
-   `y` = proportion of fixations landing at that position

These x values are slightly noisy due to digitization (e.g., 1.02,
2.97).

------------------------------------------------------------------------

## 2. generate_dataset.py

This script converts the raw digitized CSV into a clean dataset used for
plotting.

The script:

1.  Reads the digitized CSV
2.  Rounds the fixation position values
3.  Converts them to integer letter indices
4.  Outputs a clean long‑format dataset

Output format:

  word_length   letter_number   proportion_of_fixation
  ------------- --------------- ------------------------
  5             1               0.14
  5             2               0.22
  5             3               0.25

This format is easier to use for plotting and analysis.

------------------------------------------------------------------------

# How to Reproduce the Dataset

Step 1 --- Run the conversion script

```bash
python generate_dataset.py
```

Step 2 --- The script outputs

```bash
    mcconkie.csv
```

Step 3 --- Process the dataset to drop out of word fixations.
```bash
python process_mcconkie.py
```

Step 4 --- The outputs
```bash
mcconkie_processed.csv
```

This file contains the cleaned dataset used for plotting.

------------------------------------------------------------------------

# Meaning of the Variables

## word_length

Number of letters in the word.

Example:

    5 → five‑letter words
    7 → seven‑letter words

Different word lengths have different fixation distributions.

------------------------------------------------------------------------

## letter_number

The letter position within the word where the eye landed.

Example for a 5‑letter word:

    1 → first letter
    3 → center of word
    5 → last letter

------------------------------------------------------------------------

## proportion_of_fixation

Probability that the **initial fixation** on a word landed at that
letter position.

Example:

    letter 3 → 0.25

means **25% of fixations landed on letter 3**.
