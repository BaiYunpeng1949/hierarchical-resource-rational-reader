# Word Recognition Environment

This environment simulates human‑like word recognition behavior using a Bayesian Reader model.
It focuses on the cognitive aspects of word recognition **without** relying on computer vision.

---

## Key Features
- **Bayesian word recognition** with probabilistic letter sampling, belief updates, and word activation.
- **Human‑like reading characteristics**: configurable foveal span, top‑k lexical activation, stochastic/deterministic activation.
- **No visual perception**: operates on pre‑processed word representations rather than images.

## Technical Details — POMDP formulation
- **State (S)**: external (fixation index, sampled letters, word length, time); internal (lexical belief over candidates).
- **Action (A)**: external (eye movement within word); internal (“continue” vs “activate”).
- **Observation (O)**: external (fixation index, sampled letters, word length, time); internal (lexical belief).
- **Transition (T)**: external static; internal Bayesian update of beliefs.
- **Reward (R)**: reading utility with time cost (r(t) = U + c(t)).

---

## Quick start
```bash
conda activate reader_agent
cd step5
python main.py
```
Figures for the simple single‑run workflow are produced by first copying CSVs into the plotting folder and then:
```bash
cd /home/baiy4/reader-agent-zuco/results/section1
python plot.py
# Figures: /home/baiy4/reader-agent-zuco/results/section1/figures
```

---

# Parameter Inference (Grid Test + Grid Search over κ)

This section automates **(1) running a kappa sweep (“grid test”)**, **(2) aggregating per‑kappa episodes**, **(3) producing analyzed CSVs**
(Length / Log‑Frequency / Logit‑Predictability effects), and **(4) selecting the best κ** by minimizing a discrepancy to human data.

## 1) Configure the grid test
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
- The sweep runs κ = 2.0, 2.1, …, 4.0 (inclusive).
- For **each κ** it runs **self._num_episodes** episodes and aggregates **all episodes** for that κ.

## 2) Run the grid test
From the project root:
```bash
cd step5
python main.py
```
What’s produced (per κ):
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

**Notes**
- During the sweep the env is run with logging enabled (treated like `test` mode) so that analyzers have access to `fixations`.
- Analyzers used: `analyze_priors_effect_on_gaze_duration`, `analyze_word_length_gaze_duration`, `analyze_accuracy`.

## 3) Provide human reference data
Place the human curves here (already present):
```
step5/modules/rl_envs/word_activation_v0218/parameter_inference/human_data/
  gaze_duration_vs_word_length.csv
  gaze_duration_vs_word_log_frequency.csv
  gaze_duration_vs_word_logit_predictability.csv
```

## 4) Run the grid **search** (pick the best κ)
Use the helper script (place it under `parameter_inference/` and run):
```bash
cd step5/modules/rl_envs/word_activation_v0218/parameter_inference
python grid_search_kappa.py
```
What it does:
- Scans all `simulation_data/kappa_*` folders.
- For each metric (Length / Log‑Freq / Logit‑Pred), loads the **human** curve and the **simulated** curve.
- Computes a **discrepancy score** per metric using Jensen–Shannon divergence on the normalized mean curves, then sums them:
  

  \[
  F(\kappa) \,=\, \sum_{m \in \{L, f, p\}} \omega_m \; JS\big(\,\text{curve}^{\text{sim}}_m \;\|\; \text{curve}^{\text{human}}_m\,\big).
  \]
  

  (This approximates your conditional discrepancy by treating each curve as a discrete distribution over its bins.)
- Writes a summary: `simulation_data/grid_search_results.csv`.
- Identifies the **best κ** (lowest total F).

By default it also copies the **best‑κ** CSVs into a folder used for plotting:
```
step5/modules/rl_envs/word_activation_v0218/parameter_inference/best_param_simulated_results/
  gaze_duration_vs_word_length.csv
  gaze_duration_vs_word_log_frequency_binned.csv
  gaze_duration_vs_word_logit_predictability_binned.csv
```

> If you prefer the previous folder name `simulated_results/`, change the constant at the top of `grid_search_kappa.py`:
> ```python
> SIMULATED_RESULTS_FOR_PLOTS = os.path.join(HERE, "simulated_results")
> # or
> SIMULATED_RESULTS_FOR_PLOTS = os.path.join(HERE, "best_param_simulated_results")
> ```

## 5) Plot the **best‑κ** vs human
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

## Tips / Customization
- **Weights**: edit `WEIGHTS = {"length": 1.0, "logfreq": 1.0, "logitpred": 1.0}` in `grid_search_kappa.py` to emphasize certain metrics.
- **Binned vs raw**: the grid search prefers *binned* frequency/predictability CSVs when available; it will fall back to raw files if needed.
- **Inclusive endpoint**: the sweep includes the end value; set per your `config.yaml`.
- **Troubleshooting**:
  - `KeyError: 'fixations'`: ensure the grid run forces env logging (equivalent to `test` mode) before stepping.
  - Missing CSVs: confirm the analyzer functions ran and wrote to each `kappa_*` folder.

---

## Limitations
- Human and simulated curves are compared via **curve‑level JS divergence** because bin‑wise **full distributions** are not available in the CSVs.
- Lexicon is fixed; non‑ASCII not supported yet.

## Future improvements
1. Expose per‑bin distributions to enable the full conditional objective.
2. Multi‑parameter grid/BO (e.g., κ with additional timing parameters).
3. Confidence intervals / bootstrapping on the discrepancy.
