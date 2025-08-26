# Comprehension Test Runner

Evaluate reading **comprehension** from simulated LTM gists by answering **MCQs** and generating **free recall**, either via a GPT agent or an offline heuristic. This folder contains the runnable code; data artifacts live under `../assets`.

---

## Files in this folder

- `main.py` — entrypoint that calls the pipeline with sensible default paths.
- `comprehension_test_pipeline.py` — end‑to‑end runner:
  - loads simulated LTM gists and MCQs
  - selects top‑K propositions per trial
  - creates a **fresh LLM agent per trial** (LLM mode)
  - answers MCQs, generates free recall, computes free‑recall scores
  - writes JSON + CSV results
  - prints progress logs (configurable with `--verbose`)
- `gpt_comprehension_answer.py` — GPT wrapper (Aalto OpenAI gateway by default), exposes:
  - `get_mcq_answers(ltm_gist, question, options) -> {'A','B','C','D','E'}`
  - `get_free_recall(ltm_gist) -> str`

---

## Data layout (used by defaults)

```
../assets/
└─ comprehension_results/
   ├─ mcq_metadata.json                    # INPUT: MCQs grouped by stimulus_index
   └─ simulation/
      ├─ sim_ltm_gists.json                # INPUT: simulated LTM gists
      ├─ simulation_read_contents.json     # OPTIONAL: original texts per stimulus (for scoring vs stimulus)
      ├─ comprehension_test_results.json   # OUTPUT
      └─ comprehension_test_results.csv    # OUTPUT
```

> You can override any path via CLI flags. The defaults above work when running from `comprehension_tests_runner/`.

---

## Quick start

Run from inside `comprehension_tests_runner/`.

### 1) Heuristic (offline) sanity check
Deterministic run: keyword overlap MCQ + heuristic free‑recall scorer (no API calls).
```bash
python main.py   --mode heuristic   --verbose 2   --fr_score_mode heuristic   --fr_reference gist
```

### 2) Full LLM run (Aalto gateway)
Creates a **new GPT agent per trial** for MCQ + free recall.
```bash
# LLM for QA, score free recall vs gist with heuristic scorer (DO NOT USE THIS, JUST SHOWCASE)
python main.py   --mode llm   --verbose 1   --fr_score_mode heuristic   --fr_reference gist

# LLM for QA and LLM‑judge scoring vs original stimulus text (DO NOT USE THIS, JUST SHOWCASE)
python main.py   --mode llm   --verbose 1   --fr_score_mode llm   --fr_reference stimulus   --stimuli_text_json ../assets/comprehension_results/mcq_metadata.json
```

---

## Outputs

- **JSON** (`comprehension_test_results.json`): list of trials with `episodic_info`, `mcq_logs`, `free_recall_answer`, and `free_recall_score`.
- **CSV** (`comprehension_test_results.csv`): one row per trial with columns:
  ```
  participant_index, stimulus_index, time_constraint, MCQ Accuracy, Free Recall Score
  ```

---

## CLI reference

```
--mode {llm,heuristic,gold}        # llm: call GPT; heuristic: offline; gold: use correct MCQ answers
--verbose {0,1,2}                  # 0=silent, 1=high-level, 2=step-by-step
--ltm PATH                         # path to sim_ltm_gists.json
--mcq PATH                         # path to mcq_metadata.json
--out_json PATH                    # output JSON path
--out_csv PATH                     # output CSV path
--max_props 40                     # max propositions passed to LLM
--sort_by {last_relevance,total_strength,visits}
--participant_id 0                 # value recorded in outputs
--fr_score_mode {heuristic,embedding,llm}
--fr_reference {gist,stimulus}
--stimuli_text_json PATH           # optional mapping stimulus_index -> original text (for scoring vs stimulus)
```

**Logging**
- `--verbose 1`: run summary + per‑trial header + save paths + overall accuracy.
- `--verbose 2`: adds per‑MCQ predictions (✓/✗), free‑recall length and score.

---

## Free‑recall scoring

- **heuristic** (default): blend of difflib ratio and token‑Jaccard, returns `[0,1]`.
- **embedding**: simple token‑Jaccard.
- **llm**: LLM judge returns a single float `[0,1]`; falls back to heuristic if the judge isn’t available.

Reference text for scoring is chosen by `--fr_reference`:
- `gist` — concatenation of top‑K LTM propositions for that trial (prompt content).
- `stimulus` — the original stimulus text (requires `--stimuli_text_json`).

---

## Aalto OpenAI configuration (LLM mode)

`gpt_comprehension_answer.py` expects an Aalto OpenAI key in the environment:
```bash
export AALTO_OPENAI_API_KEY="xxxxx-your-key-xxxxx"
python main.py --mode llm --max_props 100 --verbose 1 --fr_score_mode embedding --fr_reference stimulus
```

---

## Troubleshooting

- **No trials found**: verify you’re running from `comprehension_tests_runner/` and that
  `../assets/comprehension_results/simulation/sim_ltm_gists.json` exists with keys like `episode_0_stim_0_30s__LTM`.
- **Auth/gateway issues**: confirm environment key, or test pipeline end‑to‑end with `--mode heuristic`.
- **Empty CSV**: check `mcq_metadata.json` contains MCQs for those `stimulus_index` values (missing MCQs are skipped).

## Plotting MCQ & Free Recall (30s / 60s / 90s)

This pipeline plots **two grouped bar charts** (Human vs Simulation) for **MCQ accuracy** and **Free Recall** and also produces a **side-by-side combined image**. The color scheme matches our earlier plots: **Human = blue**, **Simulation = green** (error bars in black).

### Inputs
Two CSV files:
- `comprehension_test_results.csv` — simulation/agent results
- `processed_mcq_freerecall_scores_p1_to_p32.csv` — human results

Both CSVs should contain columns:
- `time_constraint` ∈ {`30`, `60`, `90`}
- `MCQ Accuracy` ∈ [0,1]
- `Free Recall Score` ∈ [0,1]

### Script
Run the plotting script (creates individual figures and a stitched, side-by-side image):
```bash
python plot.py   --out_dir figures
```

### Outputs
- `/mnt/data/mcq_by_time.png` – MCQ grouped bar chart (Human vs Simulation)
- `/mnt/data/free_recall_by_time.png` – Free Recall grouped bar chart (Human vs Simulation)
- `/mnt/data/mcq_fr_side_by_side.png` – **combined** image (MCQ left, Free Recall right)
- `/mnt/data/aggregated_mcq_fr_by_time.csv` – per-time-condition means & stds used for plotting

### Dependencies
- `pandas`, `numpy`, `matplotlib`, `Pillow`

### Notes
- Charts show **mean** with **STD** error bars; annotations display `mean (std)` on each bar.
- Times are ordered as `30s`, `60s`, `90s`.
- To adjust colors, edit the constants at the top of `plot_mcq_fr.py`:
  ```python
  HUMAN_COLOR = "blue"
  SIM_COLOR = "green"
  ERR_COLOR = "black"
  ```
