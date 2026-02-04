# Hierarchical Resource-Rational Reader
This repository contains the code and example data associated with the paper:

> [**Hierarchical Resource Rationality Explains Human Reading Behavior**](https://osf.io/preprints/psyarxiv/26hb8)  
> Yunpeng Bai, Xiaofu Jin, Shengdong Zhao, Antti Oulasvirta

The code implements a hierarchical, resource-rational computational model of human reading that integrates perceptual, memory, and decision-making constraints to simulate eye-movement behavior and comprehension under varying task demands. WThe repository also includes scripts for running experiments and reproducing key results reported in the paper.

The repository is work in progress. We expect to have everything completed by Feb 15, 2026.
---

## 1. Overview

### What does the code do?
The codes simulate human reading as a sequential decision-making process under bounded cognitive resources. It generates predicted eye-movement patterns (e.g., fixation durations, skipping, regressions) and comprehension-related measures under different experimental conditions (e.g., time pressure) at different cognitive levels (word, sentence, and text levels).

### Key components
- Hierarchical reader model (word-, sentence-, and text-level control)
- Environment interface for reading tasks and stimuli
- Simulation and inference code for fitting model parameters
- Analysis scripts for generating figures and metrics reported in the paper

A detailed description of the modeling framework and POMDP formulations is provided in Supplementary Information, Sections 1.3 and 1.4 of the manuscript.


---

## 2. System Requirements (For both training and testing/simulating)

### Operating systems tested
- Ubuntu 24.04.3 LTS

### Programming language
- Python ≥ 3.10 (we used 3.10)

### Hardware we used
- GPU: 
   -  NVIDIA Tesla V100 (PCIe)
   -  VRAM: 16 GB
   -  Driver: 560.35.05
   -  CUDA runtime: 12.6
- CPU:
   -  CPU model: Intel Xeon Gold 6130 @ 2.10 GHz
   -  Sockets: 2
   -  Cores per socket: 16
   -  Threads per core: 2
   -  Total logical cores: 64
   -  NUMA nodes: 2

## 3. Software (packages) Dependencies (For both training and testing/simulating)

All required dependencies with **exact version numbers** are listed in `reqirements.txt` in the root directory. The code has been tested using the versions specified in `requirements.txt`. Results reported in the paper were generated using these versions.

Key dependencies preview:
-  numpy
-  gym
-  PyTorch
-  Stable-Baselines3

---

## 4. Installation Guide

### Typical installation time
Approximately **20 minutes** on a standard Linux workstation or server. We strongly recommend set up your environment in a Ubuntu server that has Nvidia GPU. GPU is needed for both training and testing/simulation.

### Installation steps
We strongly recommend using `conda` to manage your virtual environments.
```bash
git clone https://github.com/BaiYunpeng1949/hierarchical-resource-rational-reader.git

cd hierarchical-resource-rational-reader

conda create -n reader_model python=3.10

conda activate reader_model

pip install -r requirements.txt
```

--- 
## 5. Demo: Running experiments

### Datasets
Human eye-tracking datasets are used only for comparison with simulation results, not for training the model. Due to their size, these datasets are not hosted directly on GitHub. Publicly available datasets and newly collected data can be accessed via OSF:
`https://osf.io/q2dm6/`

The simulations themselves do not require human data as input. Once the environment, configuration files, and reinforcement learning models are set correctly, simulations can be run independently. Pre-trained policy network weights are provided in relevant branches.


### Activate the environment
```bash
conda activate reader_agent
```

### Word-level recognition simulation
Results refer to the Section Results "Deciding when and where to fixate in a word". Figure 3(a).

> ***NOTE:*** Detailed descriptions and instructions could be found in `step5/modules/rl_envs/word_activation_v0218/README.md`.

```bash
# Change to the correct branch (must do this)
git checkout word_recognition/gaze-duration-effects

cd step5/

python main.py
```
If you just want to run a single batch simulation, change to `mode: test` in `\step5\config.yaml`.


### Sentence-level reading simulation
Results refer to the Section Results "Deciding where to fixate in a sentence". Figure 3(b).
```bash
git checkout sentence_reading/skip-regression-effects

cd step5/

python main.py
```

### Text-level reading simulation
Results refer to the Section Results "Text comprehension and deciding where to read in text". Figure 3(c).
```bash
git checkout text_comprehension/effects

cd step5/

python main.py
```

### Read under time pressure
Results refer to the Section Results "Speed-accuracy trade-off when reading under time pressure" and "Validating the necessity of hierarchical resource rationality". Figure 3(d). Figure 5, and Extended Data Figure 10.
```bash
git checkout read_under_time_pressure

cd step5/simulators/simulator_v20250604

python simulator.py
```

Please find the detailed instructions, description and explanation of the output (usually eye movement data in `json` files) in each branch's folder we listed above. Simulation outputs are typically stored as JSON files containing eye-movement trajectories (fixation positions within words and sentences) and associated behavioral metrics.

### Typical runtime
-  For eyemovement inference: 10-20 mins.
-  For comprehension inference: 2 hours.
-  For model training: 24 hours.


---
## 6. Reproduction 
Details could be found in specific branches. We will merge them to here by 15 Feb 2026.