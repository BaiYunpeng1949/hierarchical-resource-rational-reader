# Reproduction 

Generating comprehension

```bash
python text_comprehension_pipeline_v1.py --input ../assets/comprehension_results/simulation/simulation_read_contents.json --output ../assets/comprehension_results/simulation/ltm_gists_v1.json --max_facets 5 --model gpt-4o --max_episodes 3
```

# Reproduction

Running comprehension tests

```bash
python -m offline_kintsch_text_comprehension_runner.comprehension_test --ltm_gists_json assets/comprehension_results/simulation/ltm_gists_v1.json --output_dir assets/comprehension_results/simulation/comprehension_performance --max_episodes 9 --mcq_metadata assets/comprehension_results/mcq_metadata.json --input_json assets/comprehension_results/simulation/simulation_read_contents.json --stimuli_json assets/comprehension_results/stimuli_texts.json
```