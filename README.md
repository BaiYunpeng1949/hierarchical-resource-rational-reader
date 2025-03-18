# CR-ReadAgent
Official codes for the computationally rational model on reading simulation. "Computationally Rational Read Agent".

## Publications
[publication_name](publication_link), VENUE'XX

## Word Skipping Mechanism

### Current Implementation
The model implements a sophisticated word skipping mechanism that reflects human reading behavior:

1. **Uncertainty-based Comprehension**
   - Compares predicted word meaning with actual word embedding
   - Uncertainty affects comprehension at multiple levels:
     * Reduces the input (predicted word state)
     * Reduces the previous comprehension state
     * Reduces the new hidden state
   - Higher uncertainty leads to lower comprehension, reflecting reduced understanding when skipping difficult words

2. **State Propagation**
   - Reduced comprehension from skipped words affects subsequent word processing
   - Maintains cognitive plausibility where poor understanding of skipped words impacts overall sentence comprehension

### Future Enhancements: Preview Effects
The model will be enhanced with preview effects to better simulate human reading:

1. **Parafoveal Preview**
   - First few letters of upcoming words (e.g., "com" for "comprehension")
   - Word length information
   - Word shape features (ascenders/descenders)

2. **Prediction Mechanism**
   - Context-based prediction combined with preview information
   - Higher likelihood of skipping when:
     * Word is predictable from context
     * Preview information matches prediction
     * Word is frequent/familiar

3. **Uncertainty Calculation**
   - Will incorporate both contextual and visual information
   - Lower uncertainty when preview confirms predictions
   - More realistic skipping behavior for common/predictable words

This implementation aims to capture the cognitive processes involved in word skipping during natural reading, where readers make decisions based on both contextual understanding and visual preview information.

## Contact person
[Bai Yunpeng](https://baiyunpeng1949.github.io/bai.yunpeng/)


## Project links
- Project folder: [here](https://drive.google.com/drive/folders/1nIytcHTvDBfrBHHSch3Z0VPFiFVQEY3m?ths=true)
- Documentation: [here](guide_link)
- [Version info](VERSION.md)


[//]: # (## Requirements)

[//]: # (See requirements.txt)


## Installation
Option 1: Use requirement.txt to install the dependencies
```bash
pip install -r requirements.txt
```
Option 2: Use setup.py to install the package
```bash
pip install -e .
```
**NOTE**: Some env setup issues:

1. When installing **gym==0.21.0**, you may face wheel errors, please run this command on your terminal
    ```bash
    pip install setuptools==65.5.0 "wheel<0.40.0"
    ```
    [Reference to solving this issue.](https://stackoverflow.com/questions/76129688/why-is-pip-install-gym-failing-with-python-setup-py-egg-info-did-not-run-succ)
    
    One could install other versions of gym, but it may not be compatible with the code.
    For example, they require gym envs to have seed. You may need to modify our source code to make it work.

2. When installing **annoy**, if you are using windows, Microsoft Visual C++ 14.0 or greater is required. Need to get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/

## Application Execution 
Run main.py on different folders.
```bash
python /path/to/main.py
```


## References


## Contributors
Bai Yunpeng

Antti Oulasvirta

Shengdong Zhao

David Hsu

