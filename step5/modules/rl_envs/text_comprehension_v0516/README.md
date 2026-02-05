# Text Reading Environment (v0516)

This environment simulates human‑like text reading eye movement behavior and comprehension (mostly proposition-based, following Kiontsch's theoretical framework).

It corresponds to the `text reader` of the model described in the paper, and is used to generate the results reported in Figure 3c ("Text comprehension and deciding where to read in text").




## Overview
The directory `step5/modules/rl_envs/text_comprehension_v0516` defines a POMDP-based text reading environment, including:
- State dynamics / transition function `TransitionFunction.py`
- Reward specification `RewardFunction.py`
- Lexical and word statistics management `TextManager.py`, `Constants.py`
- Environment wrapper `TextComprehensionEnv.py`
- Utilities and helpers `Utilities.py`
These components jointly specify the POMDP tuple underlying the text-level reading model.

> ***NOTE:*** Detailed theoretical motivation, model assumptions, and formal definitions are provided in the Methods and Supplementary Information of the paper.




## Quick Start: Text-Level Simulation

Prerequisite: Pre-trained RL model. Retraining is not required to reproduce Figure 3c; the provided checkpoint corresponds to the model used in the paper, and we provide it (a ready-to-use model checkpoint) here: 

```bash
step5/modules/rl_envs/text_comprehension_v0516/pretrained_rl_model_weights/
```

Copy the entire folder to: 
```bash
step5/training/saved_models/
```

Run the simulation
```bash
conda activate reader_agent
cd step5
python main.py
```
This runs the sentence-level reading simulation using deterministic parameters and the provided pre-trained policy.

> ***NOTE:*** All required data for eye movement analysis are ready here. Text comprehension analysis were done by parsing propositions and using language models to calculate relevance score, further determining the memory storage threshold. How to reproduce results, further see: `step5/modules/rl_envs/text_comprehension_v0523/README.md`.