# Sentence Reading Environment (v0319)

This environment simulates human‑like sentence reading eye movement behavior. 

It corresponds to the `sentence reader` of the model described in the paper, and is used to generate the results reported in Figure 3b ("Deciding where to fixate in a sentence").



## Overview
The directory `step5/modules/rl_envs/sentence_read_v0319` defines a POMDP-based sentence reading environment, including:
- State dynamics / transition function `TransitionFunction.py`
- Reward specification `RewardFunction.py`
- Lexical and word statistics management `SentencesManager.py`, `Constants.py`
- Environment wrapper `SentenceReadingEnv.py`
- Utilities and helpers `Utilities.py`
These components jointly specify the POMDP tuple underlying the sentence-level reading model.

> ***NOTE:*** Detailed theoretical motivation, model assumptions, and formal definitions are provided in the Methods and Supplementary Information of the paper.



## Data Processing Pipeline

### 1. Integration Difficulty Computation
- **Method**: Uses BERT's masked language modeling with bidirectional context
- **Computation**:
  - Masks target word in full sentence context
  - Computes surprisal: -log P(word | context_left, context_right)
  - Converts surprisal to difficulty score using sigmoid function
- **Rationale**:
  - Bidirectional context captures both preceding and following information
  - Surprisal measures how unexpected a word is in its context
  - Sigmoid scaling provides normalized difficulty scores [0,1]
- **Output Format**:
  ```json
  {
    "word": "example",
    "surprisal": 2.5,
    "integration_difficulty": 0.7
  }
  ```


### 2. Word Prediction System
- **Preview Mechanism**:
  - Uses first 2 letters for clear preview
  - Implements noisy preview matching for subsequent letters
  - Similar-looking letters are considered interchangeable
- **Length Estimation**:
  - Blurry length estimation based on word length
  - Tolerance increases with word length
  - Short words (≤4): ±1 letter
  - Medium words (≤6): ±2 letters
  - Long words (≤8): ±3 letters
  - Very long words (>8): ±(length/3) letters
- **Similar Letter Groups**:
  - Round letters: a/e/o
  - Vertical lines: i/l/1
  - Ascending/descending: h/b, p/q
  - Common confusions: u/n, c/e, v/w, r/n
- **Noise in Preview**:
  - First 2 letters: Exact match required
  - Next 3 letters: Increasing noise threshold (30% per position)
  - Random acceptance based on position
- **Output Format**:
  ```json
  {
    "word": "example",
    "next_word_predicted": "word",
    "predictability": 0.8,
    "prediction_metadata": {
      "preview_letters": 2,
      "clear_preview": "ex",
      "target_length": 7,
      "length_tolerance": {
        "min": 5,
        "max": 9
      }
    },
    "prediction_candidates": [
      {"word": "word", "probability": 0.8},
      {"word": "words", "probability": 0.1},
      {"word": "world", "probability": 0.05},
      {"word": "work", "probability": 0.03},
      {"word": "would", "probability": 0.02}
    ]
  }
  ```



## Quick Start: Sentence-Level Simulation

Prerequisite: Pre-trained RL model and ZuCo corpus (processed by us already). Retraining is not required to reproduce Figure 3b; the provided checkpoint corresponds to the model used in the paper.

This environment requires a pre-trained RL policy. We provide a ready-to-use model checkpoint here: 

```bash
step5/modules/rl_envs/sentence_read_v0319/pretrained_rl_model_weights/
```

Copy the entire folder to: 
```bash
step5/training/saved_models/
```

> ***NOTE:*** The processed ZuCo-related datasets will be prepared soon.

Run the simulation
```bash
conda activate reader_agent
cd step5
python main.py
```
This runs the sentence-level reading simulation using deterministic parameters and the provided pre-trained policy.



## Parameter Inference (Grid Test and Grid Search over $w_{reg}$ -- weighted regression cost)

This section automates **(1) running a $w_{reg}$ sweep (grid test)**, **(2) aggregating per‑$w_{reg}$ episodes**, **(3) producing analyzed CSVs**, and **(4) selecting the best $w_{reg}$** by minimizing a discrepancy to human data.

### 1) Configure the grid test
Edit `step5/config.yaml`:
```yaml
rl:
  mode: grid_test         # <-- enables the sweep
  train:
    checkpoints_folder_name: 0410_sentence_read_v0319_more_plausible_regression_mechanism_v9
  test:
    loaded_model_name: rl_model_150000000_steps
    params:
      w_regression_cost: [0, 1.0, 0.02]   # [start, end, step]  (end is inclusive)
```
Interpretation:
- The sweep runs $w_{reg}$ = 0, 0.02, …, 1.0 (inclusive).
- For **each $w_{reg}$** it runs **self._num_episodes** episodes and aggregates **all episodes** for that $w_{reg}$.


### 2) Run the grid test
From the project root:
```bash
cd step5
python main.py
```
What’s produced (per $w_{reg}$):
```
step5/modules/rl_envs/sentence_read_v0319/parameter_inference/simulation_data/
  w_regression_cost_0p0/
    all_words_regression_and_skip_probabilities.csv                               # all episodes for $w_{reg}=0$
    analysis_summary.txt
    raw_simulated_results.json
    simulated_word_analysis.csv
    word_features.json
    word_regression_analysis.csv
    word_skipping_analysis.csv
  w_regression_cost_0p02/
    ...
  ...
```

**Runtime Warning:** 
A full sweep of $w_{reg}$ may take several hours. For a quick inspection or plotting, we recommend either: 
- Running a reduced sweep, e.g., 
  ```bash
  w_regression_cost: [0.6, 0.8, 0.05]
  ```
- Or directly using our pre-computed best-$w_{reg}$ results:
  ```bash
  step5/modules/rl_envs/sentence_read_v0319/parameter_inference/figures/best_param.txt

  # According to the content in best_param.txt, find this folder:
  step5/modules/rl_envs/sentence_read_v0319/parameter_inference/simulation_data/w_regression_cost_0p8
  ```
The plots reported in Figure 3b could be found in:
  ```bash
  step5/modules/rl_envs/sentence_read_v0319/parameter_inference/figures/probabilities_x.pdf
  ```


### 3) Human reference data
```bash
# Already prepared and processed
step5/modules/rl_envs/sentence_read_v0319/parameter_inference/human_data
```


### 4) Run the grid **search** (pick the best $w_{reg}$)

> ***NOTE:*** Randomness arises from stochastic policy execution and episode sampling. It might result in different simulation results. So we recommend using our searched out best $w_{reg}$. 

The scorer compares **linear regression lines** (slope + intercept, via `numpy.polyfit`) between human and simulation curves on the *overlap* of x‑ranges, exactly as in `plot.py`:

- Skip vs **length**
- Skip vs **logit predictability**
- Skip vs **log frequency**
- **Regression** vs **difficulty**

The objective for a folder is:
$F = \sum_{\text{curves}} \Big[ (s_{\text{sim}} - s_{\text{hum}})^2 + (b_{\text{sim}} - b_{\text{hum}})^2 \Big]$
where $s$ is slope and $b$ is intercept from `np.polyfit(x, y, deg=1)` on the **overlap** of the human and sim x‑ranges. This objective emphasizes matching effect directions and magnitudes rather than absolute probabilities.

---

#### a. Expected layout

```
parameter_inference/
  grid_search_w_regression_cost.py
  plot.py
  human_data/
    all_words_regression_and_skip_probabilities.csv
  simulation_data/
    w_regression_cost_0p00/
      all_words_regression_and_skip_probabilities.csv
    w_regression_cost_0p02/
      all_words_regression_and_skip_probabilities.csv
    ...
  figures/                # created by the grid search when plotting
```

- Each `w_regression_cost_*` folder should contain the **analyzed** CSV
  `all_words_regression_and_skip_probabilities.csv` (produced from the raw logs).
- The human CSV must include columns:
  `length, logit_predictability, log_frequency, difficulty,
   skip_probability, regression_probability`.

---

#### b. Run the grid scorer + **generate figures**

From `parameter_inference/`:

```bash
cd step5/modules/rl_envs/sentence_read_v0319/parameter_inference

# Grid search + plotting together
python grid_search_w_regression_cost.py   --human human_data/all_words_regression_and_skip_probabilities.csv   --sim_root simulation_data
```

> ***NOTE:*** ALL simulated results folders must be put under `\simulation_data` for the grid search or plotting.

**Outputs**
- **Ranking CSV:** `simulation_data/grid_search_w_regression_cost_results.csv`
- **Console:** prints best `w_regression_cost`, per‑curve losses, and `F_total`
- **Figures:** `parameter_inference/figures/`
  - `skip_vs_length_binned_only_regression.png`
  - `skip_vs_logit_predictability_binned_only_regression.png`
  - `skip_vs_log_frequency_binned_only_regression.png`
  - `regression_vs_difficulty_binned_only_regression.png`
- **Best record:** `parameter_inference/figures/best_param.txt`

> Figures are generated by importing your `plot.py` and passing the human vs best‑sim data frames. No plotting happens during simulation—only after the best setting is found.

Figures are written to its configured `figures/` directory (see the script).





## Final Notes for Editors and Reviewers

This README documents all steps required to reproduce the sentence-level simulation and $w_{reg}$-selection results reported in Figure 3b. Higher-level modeling assumptions and derivations are described in the paper’s Methods and Supplementary Information.