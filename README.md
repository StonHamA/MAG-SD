# Implementation of MAG-SD

Implementation of article ''

MAG-SD is an image classification model focusing on pneumonia (including COVID-19) using CXR images.


---

## Architecture 
![architecture of MAGSD](./pics/AllPipeline.jpg)


## Get Started
You can run and check this code by yourself.
1. `cd` to folder where you want to download this repo
2. Run `git clone https://github.com/JasonLeeGHub/MAG-SD.git`
3. Install dependencies:
    - pytorch>=0.4
    - torchvision
    - numpy
    - matplotlib
4. Prepare dataset
    
    Create directory to store datasets, set the path of datasets in `main.py`
    Default path is `./datasets`. 
    By default setting, the code uses `COVID-19 xray dataset` from
    `https://github.com/v7labs/covid-19-xray-dataset`.
    Dataset definitions and utilities could be found in
    `dataset/covid19.py` 
 
## Train

configurations are listed in `main.py`. Once prepared, you can run the project by
```python
python main.py
```
Results will be saved into `./out`, including a log file and saved model.

## Test 
set configuration `--mode` in `main.py` to `test`, then run `python main.py`.



    




  

