# Hierarchical Resource-Rational Reader
This repository contains the code and example data associated with the paper:

> [**Hierarchical Resource Rationality Explains Human Reading Behavior**](https://osf.io/preprints/psyarxiv/26hb8)  
> Yunpeng Bai, Xiaofu Jin, Shengdong Zhao, Antti Oulasvirta

The code implements a hierarchical, resource-rational computational model of human reading that integrates perceptual, memory, and decision-making constraints to simulate eye-movement behavior and comprehension under varying task demands. We also include results reproduction and experiment running in the repository.


---

## 1. Overview

### What does the codes do?
The codes simulate human reading as a sequential decision-making process under bounded cognitive resources. It generates predicted eye-movement patterns (e.g., fixation durations, skipping, regressions) and comprehension-related measures under different experimental conditions (e.g., time pressure) at different cognitive levels (word, sentence, and text levels).

### Key components
- Hierarchical reader model (word-, sentence-, and text-level control)
- Environment interface for reading tasks and stimuli
- Simulation and inference code for fitting model parameters
- Analysis scripts for generating figures and metrics reported in the paper

A detailed methodology description and POMDP tuple designs are provided in **Supplementary Information, Section 1.3 and 1.4** of the manuscript.


---

## 2. System Requirements

### Operating systems tested
- Ubuntu 24.04.3 LTS

### Programming language
- Python ≥ 3.10

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

## 3. Software (packages) Dependencies

All required dependencies with exact version numbers are listed in `reqirements.txt` in the root folder.

Key dependencies preview:
-  numpy
-  gym
-  PyTorch
-  Stable-Baselines3

---

## 4. Installation Guide

### Typical installation time
Approximately **20 minutes** on a normal server. I strongly recommend implement your codes in a Ubuntu server that has Nvidia GPU because there's a lot of packages better-maintained.

### Installation steps
I strongly recommend using `conda` to manage your virtual environments.
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
We only use human datasets for comparing against simulation results, but not training. Due to their big size, github does not hold them. I will provide them in a google folder or you may download them here: `https://osf.io/q2dm6/overview`.

### Word-level recognition simulation (grid_test)
```bash
git checkout word_recognition/gaze-duration-effects

cd step5/

python main.py
```
If you just want to run a single batch simulation, change to `mode: test` in `\step5\config.yaml`.

### Sentence-level reading simulation
```bash
git checkout sentence_reading/skip-regression-effects

cd step5/

python main.py
```

### Text-level reading simulation
```bash
git checkout text_comprehension/effects

cd step5/

python main.py
```

### Read under time pressure
```bash
git checkout read_under_time_pressure

cd step5/simulators/simulator_v20250604

python simulator.py
```

Please find the detailed instructions, description and explanation of the output (usually eye movement data in `json` files) in each branch's folder I listed above.

### Typical runtime
-  For eyemovement inference: 10-20 mins.
-  For comprehension inference: 2 hours.
-  For model training: 24 hours.


---
## 6. Reproduction 
Details could be found in specific branches. I will merge them to here soon.