# Text Reading Environment (v0523)

This environment simulates human‑like text reading eye movement behavior and comprehension (mostly proposition-based, following Kiontsch's theoretical framework).

It corresponds to the `text reader` of the model described in the paper, and is used to generate the results reported in Figure 3c ("Text comprehension and deciding where to read in text").




## Overview
The directory `step5/modules/rl_envs/text_comprehension_v0523` mainly analyze how reader agent comprehend text by decomposing texts into propositions, and analyze their relevance to determine which to remember and which to forget (Kintsch, 1978).




## Quick Start: Text-Level Simulation

We assume you already run the simulation in `step5/modules/rl_envs/text_comprehension_v0516` and have all necessary files.




## Reproduction: Parameter Inference and Plotting

> ***NOTE:***: you need to infer the parameters, then plots used in the paper (Figure 3c) will be automatically generated.

```bash
cd step5/modules/rl_envs/text_comprehension_v0523/utils/

python infer_parameter.py --input_json ../assets/organized_example_propositions_v0527.json --calc_path . --out_dir parameter_inference/ltm_threshold_grid/ --high_range 0.8 1.0 0.001 --low_range 0.8 1.0 0.001 --sim_json ../../text_comprehension_v0516/temp_sim_data/0708_text_comprehension_v0516_no_time_decay_softmin_reward_function_hierarchical_discrete_actions_limited_episodes_03_rl_model_40000000_steps/1000ep/raw_sim_results.json
```

All results will be generated and stored in `step5/modules/rl_envs/text_comprehension_v0523/utils/parameter_inference/ltm_threshold_grid`. Plots used in Figure 3c are `panel_best_params_and_regression_*.pdf`.


## Final Notes for Editors and Reviewers

This README documents all steps required to reproduce the text-level simulation and $\theta_{L}$ and $\theta_{H}$-selection results reported in Figure 3c. Higher-level modeling assumptions and derivations are described in the paper’s Methods and Supplementary Information.



## Supplementary Information (read if you want)
### Workflow and File Generation

#### File Structure
```
text_comprehension_v0523/
├── assets/
│   ├── example_texts_v0526_chatgpt_generated.json      # Input texts with coherence levels
│   ├── organized_example_propositions_v0527.json        # Organized propositions with coherence scores
│   └── proportional_recall_results/                     # Results directory
│       └── recall_results_YYYYMMDD_HHMMSS.txt          # Generated recall results
└── utils/
    ├── organize_propositions.py                        # Organizes propositions and calculates coherence
    └── calculate_proportional_recall.py                # Calculates proportional recall metrics
```

### Workflow Steps

1. **Generate Example Texts** (if needed)
   - Input: Texts
   - Output: `example_texts_v0526_chatgpt_generated.json`
   - Structure: List of texts with `text_title` and `coherence` levels

2. **Organize Propositions**
   ```bash
   conda activate text_comprehension
   python organize_propositions.py
   ```
   - Input: `example_texts_v0526_chatgpt_generated.json`
   - Output: `organized_example_propositions_v0527.json`
   - Function: Extracts propositions and calculates coherence scores
   - Tunable Parameters:
     - `window_size`: Size of sliding window for local coherence (default: 5)
     - Input/Output file paths can be modified in the script

3. **Calculate Proportional Recall**
   ```bash
   python calculate_proportional_recall.py [--high_threshold HIGH] [--low_threshold LOW]
   ```
   - Input: `organized_example_propositions_v0527.json`
   - Output: `proportional_recall_results/recall_results_*.txt`
   - Tunable Parameters:
     - `--high_threshold`: Global coherence threshold for high knowledge (default: 0.85)
     - `--low_threshold`: Global coherence threshold for low knowledge (default: 0.70)
     - `--input_file`: Path to input JSON file (default: ../assets/organized_example_propositions_v0527.json)

### Tunable Parameters Summary

```bash
python infer_parameter.py --high_range 0 1 0.001 --low_range 0 1 0.001 --input_json ../assets/organized_example_propositions_v0527.json  
```

1. **Proposition Organization** (`organize_propositions.py`)
   - `window_size`: Controls local coherence calculation window
     - Default: 5 (based on short-term memory capacity)
     - Range: 2-7 recommended
     - Effect: Larger windows capture more context but may dilute local coherence

2. **Proportional Recall** (`calculate_proportional_recall.py`)
   - `high_threshold`: Global coherence threshold for high knowledge
     - Default: 0.85
     - Range: 0.70-0.95 recommended
     - Effect: Higher values make high knowledge criteria more strict
   
   - `low_threshold`: Global coherence threshold for low knowledge
     - Default: 0.70
     - Range: 0.50-0.80 recommended
     - Effect: Higher values make low knowledge criteria more strict
