# Word Recognition Environment

This environment simulates human‑like word recognition eye movement behavior.

It corresponds to the `word recognizer` of the model described in the paper, and is used to generate the results reported in Figure 3a (“Deciding when and where to fixate in a word”).


## Overview
The directory `step5/modules/rl_envs/word_activation_v0218` defines a POMDP-based word recognition environment, including:
- State dynamics / transition function `TransitionFunction.py`
- Reward specification `RewardFunction.py`
- Lexical and word statistics management `LexiconManager.py`, `Constants.py`
- Environment wrapper `WordActivationEnvV0218.py`
- Utilities and helpers `Utilities.py`
These components jointly specify the POMDP tuple underlying the word-level reading model.

> ***NOTE:*** Detailed theoretical motivation, model assumptions, and formal definitions are provided in the Methods and Supplementary Information of the paper.

## Quick Start: Word-Level Simulation

Prerequisite: Pre-trained RL model

This environment requires a pre-trained RL policy. We provide a ready-to-use model checkpoint here: 

```bash
step5/modules/rl_envs/word_activation_v0218/pretrained_rl_model_weights/
```

Copy the entire foder to: 
```bash
step5/training/saved_models/
```

Run the simulation
```bash
conda activate reader_agent
cd step5
python main.py
```
This runs the word-level recognition simulation using deterministic parameters and the provided pre-trained policy.

---

## Parameter Inference (Grid Test and Grid Search over $\kappa$)

This section automates **(1) running a $\kappa$ sweep (grid test)**, **(2) aggregating per‑$\kappa$ episodes**, **(3) producing analyzed CSVs**
(Length / Log‑Frequency / Logit‑Predictability effects), and **(4) selecting the best $\kappa$** by minimizing a discrepancy to human data.

### 1) Configure the grid test
Edit `step5/config.yaml`:
```yaml
rl:
  mode: grid_test         # <-- enables the sweep
  train:
    checkpoints_folder_name: 0804_word_activation_v0218_00
  test:
    loaded_model_name: rl_model_200000000_steps
    params:
      kappa: [2.0, 4.0, 0.1]   # [start, end, step]  (end is inclusive)
# also set how many repetitions per kappa:
# rl.train/test blocks already define self._num_episodes via your code; set it there.
```
Interpretation:
- The sweep runs $\kappa$ = 2.0, 2.1, …, 4.0 (inclusive).
- For **each κ** it runs **self._num_episodes** episodes and aggregates **all episodes** for that $\kappa$.

### 2) Run the grid test
From the project root:
```bash
cd step5
python main.py
```
What’s produced (per $\kappa$):
```
step5/modules/rl_envs/word_activation_v0218/parameter_inference/simulation_data/
  kappa_2p00/
    logs.json                               # all episodes for κ=2.00
    gaze_duration_vs_word_length.csv
    gaze_duration_vs_word_log_frequency.csv
    gaze_duration_vs_word_log_frequency_binned.csv
    gaze_duration_vs_word_logit_predictability.csv
    gaze_duration_vs_word_logit_predictability_binned.csv
    accuracy/accuracy_results.txt
  kappa_2p10/
    ...
  ...
```

**Notes:**
- During the sweep the env is run with logging enabled (treated like `test` mode) so that analyzers have access to `fixations`.
- Analyzers used: `analyze_priors_effect_on_gaze_duration`, `analyze_word_length_gaze_duration`, `analyze_accuracy`.


**Runtime Warning:** 

A full sweep of $\kappa$ may take several hours. For a quick inspection or plotting, we recommend either: 
- Running a reduced sweep, e.g., 
  ```bash
  kappa: [2.4, 2.5, 0.1]
  ```
- Or directly using our pre-computed best-$\kappa$ results:
  ```bash
  step5/modules/rl_envs/word_activation_v0218/parameter_inference/best_param_simulated_results/
  ```
The plots reported in Figure 3a could be found in:
  ```bash
  step5/modules/rl_envs/word_activation_v0218/parameter_inference/figures/panel_x.pdf
  ```

### 3) Human reference data
Place the human curves here (already present):
```
step5/modules/rl_envs/word_activation_v0218/parameter_inference/human_data/
  gaze_duration_vs_word_length.csv
  gaze_duration_vs_word_log_frequency.csv
  gaze_duration_vs_word_logit_predictability.csv
```

### 4) Run the grid **search** (pick the best $\kappa$)

> ***NOTE:*** randomness might result in different simulation results. So we recommend using our searched out best $\kappa$. 

Use the helper script (place it under `parameter_inference/` and run):
```bash
cd step5/modules/rl_envs/word_activation_v0218/parameter_inference
python grid_search_kappa.py
```
What it does:
- Scans all `simulation_data/kappa_*` folders.
- For each metric (Length / Log‑Freq / Logit‑Pred), loads the **human** curve and the **simulated** curve.
- Computes a **discrepancy score** per metric using Jensen–Shannon divergence on the normalized mean curves, then sums them:
  
$ F(\kappa) \,=\, \sum_{m \in \{L, f, p\}} \omega_m \; JS\big(\,\text{curve}^{\text{sim}}_m \;\|\; \text{curve}^{\text{human}}_m\,\big).
4$
  

  (This approximates your conditional discrepancy by treating each curve as a discrete distribution over its bins.)
- Writes a summary: `simulation_data/grid_search_results.csv`.
- Identifies the **best $\kappa$** (lowest total F).

By default it also copies the **best‑$\kappa$** CSVs into a folder used for plotting:
```
step5/modules/rl_envs/word_activation_v0218/parameter_inference/best_param_simulated_results/
  gaze_duration_vs_word_length.csv
  gaze_duration_vs_word_log_frequency_binned.csv
  gaze_duration_vs_word_logit_predictability_binned.csv
```

### 5) Plot the **best‑$\kappa$** vs human
`plot.py` already expects human curves in `human_data/` and simulated curves in a sibling folder.
Run:
```bash
cd step5/modules/rl_envs/word_activation_v0218/parameter_inference
python plot.py
```
Make sure `plot.py` reads from:
```
human_data/                     # reference curves
best_param_simulated_results/   # best-κ curves (or 'simulated_results/')
```
Figures are written to its configured `figures/` directory (see the script).

---

## Final Notes for Editors and Reviewers

This README documents all steps required to reproduce the word-level simulation and $\kappa$-selection results reported in Figure 3a. Higher-level modeling assumptions and derivations are described in the paper’s Methods and Supplementary Information.