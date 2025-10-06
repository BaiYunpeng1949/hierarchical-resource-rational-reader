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

### Version 3 (v3): globally selecting the relevant facets (threshold-based), then stacking together.
```bash
python text_comprehension_pipeline_v3.py --input ../assets/comprehension_results/simulation/ltm_gists_v1.json --output ../assets/comprehension_results/simulation/ltm_gists_v3.json --tau_gist 0.3 --context_window 5 --half_life 3 --ctx_boost 1.0
```

# Reproduction

Running comprehension tests

```bash
python -m offline_kintsch_text_comprehension_runner.comprehension_test --ltm_gists_json assets/comprehension_results/simulation/ltm_gists_v3_tau_0.35.json --output_dir assets/comprehension_results/simulation/comprehension_performance_v3 --max_episodes 27 --mcq_metadata assets/comprehension_results/mcq_metadata.json --input_json assets/comprehension_results/simulation/simulation_read_contents.json --stimuli_json assets/comprehension_results/stimuli_texts.json
```


# Simulation Results (Documentation)
- Version 1002:
    - v1 (full stacks of facets): /comprehension_performance_v1/comprehension_metrics_20251002-071341.json
    - v2a
        - v2a (top 2 facets in each sentence): /comprehension_performance_v2a/comprehension_metrics_20251002-081130.json
        - v2a (top 1 facet in each sentence): /comprehension_performance_v2a/comprehension_metrics_20251006-054303.json
    - v2b (hard threshold of relevance in local sentence): 
        - v2b (tau=0.4): /comprehension_performance_v2a/comprehension_metrics_20251006-061212.json
    - v3 (hard threshold of relevance in gloabal disclosure): 
        - v3 (tau=0.35): /comprehension_performance_v3/comprehension_metrics_20251006-134536.json
        - v3 (tau=0.34): /comprehension_performance_v3/comprehension_metrics_20251006-142230.json
        - v3 (tau=0.33): /comprehension_performance_v3/comprehension_metrics_20251006-144557.json
        - v3 (tau=0.32): /comprehension_performance_v3/comprehension_metrics_20251006-150327.json
        - v3 (tau=0.31): /comprehension_performance_v3/comprehension_metrics_20251006-160113.json
        - v3 (tau=0.3): /comprehension_performance_v3/comprehension_metrics_20251006-071108.json (re-summarise ltm gists for free recall generation) --> /comprehension_performance_v3/comprehension_metrics_20251006-092044.json (pure ltm gists as free recall)
        - v3 (tau=0.29): /comprehension_performance_v3/comprehension_metrics_20251006-170412.json
        - v3 (tau=0.28): /comprehension_performance_v3/comprehension_metrics_20251006-180058.json
        - v3 (tau=0.27):
        - v3 (tau=0.26):
        - v3 (tau=0.25): /comprehension_performance_v3/comprehension_metrics_20251006-082610.json
        - v3 (tau=0.2): /comprehension_performance_v3/comprehension_metrics_20251006-080633.json
    - v3a (**based on v3**, hard-threshold for removing duplicates in the ltm gists by groupping)
        - v3a (tau=0.1 new): (input from v3=0.3) /comprehension_performance_v3a/comprehension_metrics_20251006-114130.json
        - v3a (tau=0.5 new): (input from v3=0.3) /comprehension_performance_v3a/comprehension_metrics_20251006-112557.json

# Technical Report (Internal)
The global (local-based) relevance / coherence analysis, then use the threshold to filter important things, is a systematic way to do research. We build from the simplest cases (v1), all the way step-by-step to v3. While the v1 and v2a, v2b could serve as a the baselines, they also tease out why our v3 is good and working.

Local relevance analysis naturally captures the reading time's effect: for repeatedly read sentences, it has more chances to have more propositions with higher relevance scores, thus later with richer ltm gists. And the global relevance analysis greatly inherited this trait.