# Reproduction 

Generating comprehension

### Version 1 (v1): only stacking facets as ltm gist (comparison condition 1)

```bash
python text_comprehension_pipeline_v1.py --input ../assets/comprehension_results/simulation/simulation_read_contents.json --output ../assets/comprehension_results/simulation/ltm_gists_v1.json --max_facets 5 --model gpt-4o --max_episodes 3
```

### Version 2a (v2a): locally selecting the relevant facets (top k, k as an argument), then stacking together (comparison condition 2)
```bash
python text_comprehension_pipeline_v2a.py --input ../assets/comprehension_results/simulation/ltm_gists_v1.json --output ../assets/comprehension_results/simulation/ltm_gists_v2a.json --k_per_step 2
```

### Version 2b (v2b): locally selecting the relevant facets (threshold), then stacking together (comparison condition 2)
```bash
python text_comprehension_pipeline_v2b.py --input ../assets/comprehension_results/simulation/ltm_gists_v1.json --output ../assets/comprehension_results/simulation/ltm_gists_v2b_tau0.4.json --tau_gist 0.4
```

# Reproduction

Running comprehension tests

```bash
python -m offline_kintsch_text_comprehension_runner.comprehension_test --ltm_gists_json assets/comprehension_results/simulation/ltm_gists_v1.json --output_dir assets/comprehension_results/simulation/comprehension_performance --max_episodes 9 --mcq_metadata assets/comprehension_results/mcq_metadata.json --input_json assets/comprehension_results/simulation/simulation_read_contents.json --stimuli_json assets/comprehension_results/stimuli_texts.json
```


# Simulation Results (Documentation)
- Version 1002:
    - v1 (full stacks of facets): /comprehension_performance_v1/comprehension_metrics_20251002-071341.json
    - v2a
        - v2a (top 2 facets in each sentence): /comprehension_performance_v2a/comprehension_metrics_20251002-081130.json
        - v2a (top 1 facet in each sentence): /comprehension_performance_v2a/comprehension_metrics_20251006-054303.json
    - v2b (hard threshold of relevance in local sentence): /comprehension_performance_v2a/comprehension_metrics_20251006-061212.json