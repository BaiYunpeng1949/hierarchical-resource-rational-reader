Description:

    This is a testing version for propositions listing of memory representations.

    If this version works, we will close the project; if not, move to Kintsch's more complicated tree graph; or the knowledge graph.

Objective: 

    1. validate whether the agent, with only stacks of propositions, could capture the high- vs. low-knowledge's difference / effects on the text comprehension performance (text-base level evaluation, like proportional evaluation).

    2. validate whether the agent, with only stacks of propositions, could capture the high- vs. low-coherence's difference / effects on the text comprehension performance.

Evaluations:

    1. Proportional recall
        Calculation of Proportional Recall:
            1. The text was propositionalized using van Dijk & Kintsch's (1983) framework, breaking it into:
                a. Micropropositions (detailed sentence-level content)
                b. Macropropositions (global or paragraph-level main ideas)
            2. Participants' recalls were scored for the presence of any part of a proposition.
            3. Calculation (normalised): # propositions recalled / total propositions in the text version

    2. Text-based questions
        Definition: These questions can be answered using a single sentence from the text. They require literal recall of explicitly stated information.
    
    3. Bridging Inference Questions [out of scope]
        Require connecting information across multiple sentences in the text.
        Example:
            "Which disease causes scarring of the heart valves?"
            You need to infer from different text segments that link rheumatic fever with valve scarring.
    
    4. Elaborative inference questions [out of scope]
        Require integrating textual information with prior knowledge.
        Example:
            "Explain why arrhythmia would be a threat to life."
            Not directly stated; requires applying knowledge of heart function.
    
    5. Problem-Solving Questions [out of scope]
        Require applying multiple pieces of information to novel scenarios.
        Example:
            "What treatment would help a 28-year-old experiencing heart failure from a muscle disease?"
            Involves diagnosing from text info and reasoning about treatments.

**A basic testing method**: use the LLM to first construct a memory text, then redistill the propositions out of this memory text. Compare.

Estimated results: 

    1. by prompting the llm agent with high- or low-prior knowledge on the text, not much difference for the text comprehension tests, since all propositions are observable.

    2. by qa based on proposition stacks from high- or low-coherent texts, not much difference for the text comprehension tests, because there is no need for integration between propositions, all internal processings (e.g., inference) are done by the llm.

Reference paper: Are Good Texts Always Better? Interactions of Text Coherence, Background Knowledge, and Levels of Understanding in Learning from Text

Results

    1. Proportional recall: (human data vs. simulated results) -- All using GPT to parse propositions, and then integrate to texts, then parse to propositions.

                                    High Knowledge              Low Knowledge
        High Coherence              0.484 / 0.963               0.381 / 0.552

        Low Coherence               0.417 / 0.815               0.291 / 0.529
    
    2. Surface-Level Similarity

                                    High Knowledge              Low Knowledge
        High Coherence              0.944                       0.849 

        Low Coherence               0.925                       0.905

    3. Semantic Similairy (Ref: https://www.editpad.org/tool/text-similarity-checker, python has env compatibility issues)

                                    High Knowledge              Low Knowledge
        High Coherence              0.738                       0.592

        Low Coherence               0.762                       0.677


## Updated Modeling Method V0526

1. Objective:
    a. Make the simulation results better align with the human data.
    b. More rigorous implementation of the Kintsch's model.

2. Technical details:
    a. Original version [Kintsch, 1978]: 

    Operation	                            Brain Activity (Model-wise)                                     Notes
    Parsing	                                Build semantic propositions	                                    From syntactic structure to GIVE(JOHN, BOOK)
    Coherence checking	                    Try connecting to STM	                                        Based on argument overlap
    Inference	                            Fill coherence gaps	                                            Based on world knowledge
    Search	                                LTM retrieval	                                                Fallback if STM fails
    Selection	                            Keep important info in STM	                                    Size-limited (s propositions)
    Graph building	                        Create structured memory trace	                                Nodes = propositions, edges = overlap
    Forgetting / pruning	                Drop unconnected or irrelevant propositions	                    Or store in LTM if processed enough


    b. My simplified version [Bai, 2025]: 
        Parsing: sentences -> propositions -> one sentence / clause into one cycle, contains n propositions;
        STM selection (with capacity s): up to s propositions are selected to enter the STM buffer, depends on: (1) overlap with STM (local coherence), or prior propositions, or LTM? (2) If not (coherent), try infer or LTM search (global coherence); 
        Track processing frequency: Each time a proposition is retained across cycles, its "activation count" increases.;
        LTM Storage: (1) Propositions that are frequently processed (high activation) are more likely to be stored in LTM. (2) Less-relevant, unintegrated propositions are pruned or left out.
    
    NOTE: for the simiplicity, I will not consider the local inference in the STM and global search and retrieval in the LTM.

3. Implementation:
    a. parse text into sentences;
    b. parse sentence into propositions; 
    c. hold a STM buffer to store the most relevant propositions;
    d. count each propositions' activation;
    e. select the most activated ones into the LTM in the end as gists;
    f. (optional) forge the propositions into the LTM as some new propositions;
    g. (alternative) directly use the remaining propositions in the LTM for calculating the proportaional recall; --> then text coherence's difference mainly comes from b and c, knowledge level's comes from b; 
    h. reconstruct the text based on the gist;
    i. regenerate into propositions to calculate the proportional recall.

4. Execution
    a. parse text into sentences
    b. parse sentences into propositions (including grouped ones) using either SpaCy (relible, reproducible, but low quality and low adaptivity) and ChatGPT (laborious but of high quality).
    c. [Simple approach] rate all propositions' coherence and importance score all together, do not differentiate high or low knowledge background. Later, parameterize knowledge threshold: for high knowledge reader, a lower threshold for activating these content into the LTM. For low knowledge reader, a higher threshold for the activation.
    d. [Medium approach] use high- and low-knowledge reader to rate propositions differently.
    e. [Sophisticated approach] use cycles to dynamically update the stm and propositions, get rid of some in the cycles; rate propositions more easily if the reader is of high-knowledge, and vice versa. Use the number of involving processing to include in the LTM.
    f. [Extremely sophisticated approach] explicitly add inference and LTM search / retrieval to the [Sophisticated approach].
    g. [Reliable approach] use STM as a sliding window buffer, each cycle counts, only the most relevant and important information could be staying. I will rank these propositions later then deciding which one could go to the LTM for calculating the proportional recall. We could calculate the p_reproduction_or_store_in_ltm = (1 - p)^k, where k is the chance of being stored into the LTM. We define this as an empirically tunable parameter to account for knowledge level's effect.

5. Tunable parameters
    a. STM buffer size: s, default as 4, could be 2-7 or over 10.
    b. [Optional] Chances of being stored in the LTM k. Have k_high and k_low. Should be an integer. Or just a simple threshold value of ranking.
    c. [High Priority] Compute all the propositions' coherence with each other, and select the average highest ones to store in the ltm. But there will be two 


Results

    1. Proportional recall: (human data vs. simulated results)  -- to be parameter inferenced: --high_threshold and --low_threshold

                                    High Knowledge              Low Knowledge
        High Coherence              0.484 / 0.461               0.381 / 0.425

        Low Coherence               0.417 / 0.339               0.291 / 0.241

** Reproduction **

## Workflow and File Generation

### File Structure
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
   - Input: Manually created or generated texts
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

```
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

### Plot
1. Fill the data from /home/baiy4/reader-agent-zuco/step5/modules/rl_envs/text_comprehension_v0523/assets/proportional_recall_results to plot.py
    ```bash
    python plot.py
    ```

### Example Usage

1. Basic run with default parameters:
   ```bash
   python calculate_proportional_recall.py
   ```

2. Custom thresholds:
   ```bash
   python calculate_proportional_recall.py --high_threshold 0.90 --low_threshold 0.75
   ```

3. Custom input file:
   ```bash
   python calculate_proportional_recall.py --input_file ../assets/custom_propositions.json
   ```

### Results Interpretation

The output file in `proportional_recall_results/` contains:
- Timestamp of generation
- Used threshold values
- Proportional recall results for:
  - Fully Coherent texts (high and low knowledge)
  - Minimally Coherent texts (high and low knowledge)

Results are formatted as percentages and can be compared with human data from the reference paper.



# Parameter inference and Plotting

NOTE: you have to infer the parameters to plot haha.

```bash
python infer_parameter.py --input_json ../assets/organized_example_propositions_v0527.json --calc_path . --out_dir parameter_inference/ltm_threshold_grid/ --high_range 0.8 1.0 0.001 --low_range 0.8 1.0 0.001 --sim_json ../../text_comprehension_v0516/temp_sim_data/0708_text_comprehension_v0516_no_time_decay_softmin_reward_function_hierarchical_discrete_actions_limited_episodes_03_rl_model_40000000_steps/1000ep/raw_sim_results.json
```

