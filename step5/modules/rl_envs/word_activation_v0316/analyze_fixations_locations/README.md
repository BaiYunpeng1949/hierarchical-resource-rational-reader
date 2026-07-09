# Fixation Location and Intraword Regression Analysis

This module generates and visualizes fixation-location distributions and intraword regression patterns for both human data (Rayner-style datasets) and model simulations.

---

## Folder Structure

analyze_fixations_locations/
├── data/
│   ├── human/
│   └── simulation/
├── figures/
│   ├── previewed_fixation_locations/
│   └── intraword_regressions/
├── generate_simulation_fixation_datasets.py
├── plot.py
└── README.md

---

## Overview

### Fixation Location
Distribution of fixation landing positions within words.

### Intraword Regression
Distribution of regression landing positions within words.

---

## Step 1: Generate Simulation Data

Run:
```bash
    python generate_simulation_fixation_datasets.py
```

Uses executed actions (not intended actions) and outputs:
- sim_forward_fixations_multiple.csv
- sim_forward_fixations_single_only.csv
- sim_intraword_regressions_only.csv

---

## Step 2: Plot Figures

Run:
```bash
    python plot.py
```

Outputs:
- figures/previewed_fixation_locations/
- figures/intraword_regressions/

Word lengths plotted: 5–9

---

## Notes

- No trimming of zero values is applied
- Human data automatically normalized (0–100 → 0–1)
- Distributions normalized per word length
- Missing data is skipped gracefully
