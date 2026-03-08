
# Sentence Reading — Grid Search for `w_regression_cost`

This README explains how to reproduce the sentence-reading **grid search** and the resulting **figures** for the parameter **`w_regression_cost`** using the current plot-matching scorer.

The scorer compares **linear regression lines** (slope + intercept, via `numpy.polyfit`) between human and simulation curves on the *overlap* of x-ranges, exactly as in `plot.py`:

Word-level effects:

- Skip vs **length**
- Skip vs **logit predictability**
- Skip vs **log frequency**
- **Regression** vs **difficulty**

Sentence-level effects:

- **Regression** vs **sentence ambiguity**
- **Skipping** vs **sentence ambiguity**

The objective for a folder is:

F = sum_curves [ (s_sim - s_hum)^2 + (b_sim - b_hum)^2 ]

where **s** is slope and **b** is intercept from `np.polyfit(x, y, deg=1)` on the **overlap** of the human and simulation x-ranges.

---

# 1) Expected layout

```
parameter_inference/
  grid_search_w_regression_cost.py
  plot.py

  human_data/
    all_words_regression_and_skip_probabilities.csv
    ambiguity_effect_human_binned.csv

  simulation_data/
    w_regression_cost_0p00/
      all_words_regression_and_skip_probabilities.csv
      ambiguity_effect_simulation_binned.csv
    w_regression_cost_0p02/
      all_words_regression_and_skip_probabilities.csv
      ambiguity_effect_simulation_binned.csv
    ...

  figures/                # created by the grid search when plotting
```

Each `w_regression_cost_*` folder should contain:

```
all_words_regression_and_skip_probabilities.csv
ambiguity_effect_simulation_binned.csv
```

The human CSVs must include:

Word-level file:

```
length
logit_predictability
log_frequency
difficulty
skip_probability
regression_probability
```

Sentence-level file:

```
ambiguity_zscore
avg_regression_probability
avg_skip_probability
```

---

# 2) Configure the grid (in `config.yaml`)

Set the sentence-reading run to **grid test** and choose the search range.

Example configuration:

```yaml
mode: sentence_grid_test

sentence_grid_test:
  w_regression_cost:
    start: 0.00
    end:   1.00
    step:  0.02

  episodes: 10
  save_raw: true
  analyze:  true
```

The grid will create:

```
simulation_data/w_regression_cost_*/
```

folders and, if `analyze: true`, generate:

```
all_words_regression_and_skip_probabilities.csv
ambiguity_effect_simulation_binned.csv
```

for each parameter setting.

Start the grid run from the repository root:

```bash
python main.py
```

(or your project-specific entry script that triggers `_sentence_reading_grid_test()`).

---

# 3) Run the grid scorer + generate figures

NOTE: if you want to update figures, you need to re-run from parameter inference.

From `parameter_inference/`:

```bash
python grid_search_w_regression_cost.py   --human human_data/all_words_regression_and_skip_probabilities.csv   --sim_root simulation_data   --human_ambiguity human_data/ambiguity_effect_human_binned.csv
```

---

# Outputs

### Ranking CSV

```
simulation_data/grid_search_w_regression_cost_results.csv
```

### Console output

Prints:

```
Best w_regression_cost
per-metric losses
F_total
```

### Figures

Generated in:

```
parameter_inference/figures/
```

Word-level plots:

```
probabilities_1.pdf
probabilities_2.pdf
probabilities_3.pdf
probabilities_4.pdf
```

Sentence-level plots:

```
ambiguity_1.pdf
ambiguity_2.pdf
```

These figures are produced using `plot.py` and saved separately for easy composition in **Adobe Illustrator**.

### Best parameter record

```
parameter_inference/figures/best_param.txt
```

---

# 4) Notes on the loss / weighting

By default, slope and intercept differences are equally weighted.

To emphasize **trend alignment**, edit the constants in  
`grid_search_w_regression_cost.py`:

```python
WEIGHT_SLOPE = 1.0
WEIGHT_INTER = 1.0
```

You can also modify per-curve weighting by adjusting the scoring configuration in the script.

---

# 5) Troubleshooting

### Figures step fails

Ensure `--sim_root` points to the directory containing the  
`w_regression_cost_*` folders.

The script reads simulation CSVs from:

```
<sim_root>/<best_folder>/<file>
```

---

### A curve prints `NA`

Check that:

- both CSVs contain the required columns
- the overlapping x-range contains **at least two points**

---

### Folder naming

Folders should follow:

```
w_regression_cost_XpYY
```

or

```
w_regression_cost_<float>
```

This affects only printed values; data loading uses paths directly.
