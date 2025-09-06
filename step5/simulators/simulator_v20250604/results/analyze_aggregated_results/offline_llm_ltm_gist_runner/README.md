
# Offline LLM LTM Gist Runner (memories-native)

This runner **directly calls** your `LLMMemories.py` modules — `LLMLongTermMemory`, `LLMShortTermMemory`, and `LLMWorkingMemory` —
the same way the simulator's `ReaderAgent` does, but with **offline** JSON logs.

## Run

```bash
python -m offline_llm_ltm_gist_runner.main   --input_json assets/comprehension_results/simulation/simulation_read_contents.json   --output_dir assets/comprehension_results/simulation/ltm   --max_episodes 3 --episode_offset 0
```

- `--config` must point to the **same** `config.yaml` your simulator uses (for Aalto/OpenAI credentials & model settings).
- `--max_episodes` and `--episode_offset` let you validate on a small slice quickly.
- `--max_steps` limits steps inside each episode.

And it is worthy noting that the current version takes a long time to fninsh.

## Test comprehension performance

```bash
python -m offline_llm_ltm_gist_runner.comprehension_test --ltm_gists_json assets/comprehension_results/simulation/ltm/ltm_gists_20250904-035909.json --output_dir assets/comprehension_results/simulation/comprehension_performance --max_episodes 3 --mcq_metadata assets/comprehension_results/mcq_metadata.json --input_json assets/comprehension_results/simulation/simulation_read_contents.json --ltm_md_path assets/comprehension_results/simulation/ltm/ltm_gists_20250904-035909.md --stimuli_json assets/comprehension_results/stimuli_texts.json
```

## Input format

A list of episodes (trials). For each, `text_reading_logs` has steps with `sampled_words_in_sentence`. The pipeline joins tokens
into a sentence hypothesis every cycle, activates schemas, extracts the STM micro-gist, updates the LTM macrostructure, then finalizes.
