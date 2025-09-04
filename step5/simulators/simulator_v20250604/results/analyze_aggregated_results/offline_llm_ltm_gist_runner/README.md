
# Offline LLM LTM Gist Runner

A tiny, deterministic wrapper that turns noisy, step-wise reading logs into **schema-clustered long‑term memory gists** using an LLM.  
It runs one **processing cycle per step**: extract facets → assign schemas → update a deduped store → export a readable outline and JSON.

## Folder layout

```
offline_llm_ltm_gist_runner/
├── llm_agent.py            # Your LLM wiring + helper calls
├── ltm_gist_pipeline.py    # Pipelines + schema memory
├── main.py                 # CLI entry
└── README.md
```

## Input

Use the JSON like `simulation_read_contents.json` (a list of trials). Each trial carries keys like:

- `episode_index`, `stimulus_index`, `time_condition`, `total_time`
- `text_reading_logs`: list of steps, each with `sampled_words_in_sentence` (tokens with blanks/noise)

> The pipeline reconstructs a sentence hypothesis by joining non-empty tokens and is **robust to incoherence/duplicates** by design.

## What it does

1. **Facet extraction** (LLM): compact, non-redundant facets (≤12 words), one per line.  
2. **Schema assignment** (LLM): each facet is mapped to a short schema label + canonical key.  
3. **Deterministic integration**: facets are inserted under schema buckets with **dedupe** and **evidence tracking** (counts + step indices).  
4. **Export**: per-trial
   - `outline` (nested bullets with counts)
   - `main_schemas` (top schemas by facet frequency)
   - `schemas` (full JSON with evidence)

## Quick start

```bash
# from the parent directory
python -m offline_llm_ltm_gist_runner.main \
  --input_json /path/to/simulation_read_contents.json \
  --output_dir /path/to/outputs
```

This writes:

- `outputs/ltm_gists_<timestamp>.json` – machine-readable
- `outputs/ltm_gists_<timestamp>.md` – quick human skim

## LLM config

`llm_agent.py` already contains the Aalto API routing logic. If you prefer your own key/provider, flip the flag in `LLMAgent.__init__` and configure as needed.

> Tip: The pipeline only uses two agent calls:
> - `get_facet_summaries(role, prompt)` – returns `List[str]`
> - `get_schema_assignments(role, prompt)` – returns `List[{"facet","schema","canonical"}]`

## Design notes

- **Robustness to noise**: very short or punctuation-heavy lines are skipped; the LLM prompt asks to ignore gibberish.  
- **Deduplication**: canonical forms (lowercase, alnum+spaces) index schemas and facets; repeats increment counts and accumulate evidence.  
- **Idempotence**: re-running on overlapping logs merges rather than duplicates.  
- **Extensibility**: add scoring/consistency checks or export CSVs without touching the CLI.

## License

MIT (or your project’s license).

