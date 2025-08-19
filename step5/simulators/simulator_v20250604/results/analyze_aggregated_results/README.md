# Analyze Aggregated Results

This folder contains quick utilities to summarize and visualize **reading behavior** from the simulator and compare it with human baselines.

## Folder layout
```
analyze_aggregated_results/
├─ assets/
│  └─ comprehension_results/
│     ├─ human/                     # human comprehension/behavior data (CSV/JSON)
│     └─ simulation/
│        └─ simulation_read_contents.json  # trimmed sim logs (generated)
├─ analyze_data.py                   # compute metrics & plot skip/regression/reading speed
├─ process_simulation_read_contents.py  # trim big sim logs to per-step reading content
└─ README.md
```

---

## 1) Trim simulation logs → reading-content only

**Script:** `process_simulation_read_contents.py`

**Input:** `all_simulation_results.json` (full simulator output for all episodes)  
**Output:** `simulation_read_contents.json` with only the fields we care about:
```json
[{
  "episode_index": 0,
  "stimulus_index": 0,
  "time_condition": "30s",
  "total_time": 30,
  "text_reading_logs": [
    {
      "step": 1,
      "current_sentence_index": 0,
      "num_words_in_sentence": 18,
      "actual_reading_sentence_index": 0,
      "sampled_words_in_sentence": ["..."]
    }
  ]
}]
```

**Run:**
```bash
python process_simulation_read_contents.py  20250819_1700_trials1_stims9_conds3   -o simulation_read_contents.json
```
Use `--indent` (default 2) to change pretty-printing. 
For input, the folder name in `/simulator_v20250604/simulated_results/`.
The content will be automatically stored in `assets/comprehension_results/simulation/`.

---

## 2) Compute metrics & make plots (skip, regression, reading speed)

**Script:** `analyze_data.py`

This script provides functions to:
- extract **global fixation sequences** per episode (`process_simulation_results_to_fixation_sequence`)
- compute episode-level metrics (`process_fixation_sequences_to_metrics`)
- compare to human baselines and **plot** three panels: **reading speed**, **skip rate**, **regression rate** (`plot_metrics_comparison`)

> **Definitions (as implemented):**
> - **Reading speed (wpm)** = `len(global_fixation_sequence) / total_time * 60` (fixation positions per minute).
> - **Skip saccade**: next fixation jumps forward **past at least one word**  
>   `skipped = next_idx - current_idx - 1; skipped > 0` ⇒ counted as a skip saccade.
> - **Regression**: a fixation lands on a word **earlier than the last seen index**  
>   if `word_idx < last_read_word_index` we count a regression fixation.

### Quick start
The bottom of `analyze_data.py` has a small main that expects files in a `simulated_results/<run_name>/` layout.
Edit the line:
```python
file_name = "20250819_0856_trials1_stims9_conds3"  # <- change to your run folder name
```
Then run:
```bash
python analyze_data.py
```

This will write, inside `simulated_results/<file_name>/`:
- `processed_fixation_sequences.json`
- `analyzed_fixation_metrics.json`
- `metrics_comparison.png`  ← bar charts for **30s / 60s / 90s** conditions

It also expects a human metrics JSON at:
```
processed_human_data/analyzed_human_metrics.json
```
Adjust `human_metrics_file` in the script if your path differs.

### Want to call functions manually?
```python
from analyze_data import (
    process_simulation_results_to_fixation_sequence,
    process_fixation_sequences_to_metrics,
    plot_metrics_comparison,
)

# Paths
input_file = "simulated_results/<run>/all_simulation_results.json"
fix_seq_file = "simulated_results/<run>/processed_fixation_sequences.json"
fix_metrics_file = "simulated_results/<run>/analyzed_fixation_metrics.json"
human_metrics_file = "processed_human_data/analyzed_human_metrics.json"
plot_png = "simulated_results/<run>/metrics_comparison.png"

# Pipeline
process_simulation_results_to_fixation_sequence(input_file, fix_seq_file)
process_fixation_sequences_to_metrics(fix_seq_file, fix_metrics_file)

# Load simulation metrics (list of episodes)
import json
sim_data = json.load(open(fix_metrics_file, "r"))
plot_metrics_comparison(human_metrics_file, sim_data, plot_png)
```

---

## Notes & tips
- **Time conditions**: the comparison figure expects exactly `30s`, `60s`, `90s` keys. If your runs differ, adapt the lists in `plot_metrics_comparison`.
- **Unique-words speed**: there is an alternate speed metric in the script (`analyze_fixation_sequences`) that uses **unique words** instead of fixations. Use it if that better suits your analysis.
- **Reproducibility**: keep the trimmed `simulation_read_contents.json` alongside your raw results; it captures the step-wise reading content without the heavy extras.

Questions or changes (e.g., different regression/skip definitions, per-participant plots)? Ping me and I’ll wire it in.
