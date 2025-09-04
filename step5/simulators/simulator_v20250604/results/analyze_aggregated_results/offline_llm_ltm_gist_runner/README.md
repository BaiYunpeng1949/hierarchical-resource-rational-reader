
# Offline LLM LTM Gist Runner (memories-native)

This runner **directly calls** your `_LLMMemories.py` modules — `LLMLongTermMemory`, `LLMShortTermMemory`, and `LLMWorkingMemory` —
the same way the simulator's `ReaderAgent` does, but with **offline** JSON logs.

## Run

```bash
python -m offline_llm_ltm_gist_runner.main   --input_json /mnt/data/simulation_read_contents.json   --output_dir /mnt/data/outputs   --config /home/baiy4/reading-model/step5/config.yaml   --max_episodes 3 --episode_offset 0 --max_steps 40
```

- `--config` must point to the **same** `config.yaml` your simulator uses (for Aalto/OpenAI credentials & model settings).
- `--max_episodes` and `--episode_offset` let you validate on a small slice quickly.
- `--max_steps` limits steps inside each episode.

## Input format

A list of episodes (trials). For each, `text_reading_logs` has steps with `sampled_words_in_sentence`. The pipeline joins tokens
into a sentence hypothesis every cycle, activates schemas, extracts the STM micro-gist, updates the LTM macrostructure, then finalizes.
