# CR-ReadAgent
Official codes for the computationally rational model on reading simulation. "Computationally Rational Read Agent".

## Publications
[publication_name](publication_link), VENUE'XX
```
<Bibtext>

```

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

