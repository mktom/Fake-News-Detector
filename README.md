# Fake News Challenge (FNC)
 COMP-9417 - 2019 - T2 - Machine Learning Project

**Submitted By:**

Jianan Yan **(z5168722)**, Mark Thomas **(z5194597)**,
Qiqi Zhang **(z5185698)**, Sudhan Maharjan **(z5196539)**





### Required libraries 
list of required packages are in `requirements.txt`

Install the packages: `pip install -r requirements.txt`

Install Spacy language module
`en_core_web_lg` package is needed for spacy

`py -m spacy download en-core-web-lg`



### Preprocessing
- Update `F_STANCES, F_BODIES, O_H5, O_CSV` values in `pre_process_spacy.py`
- Run the file

- Update `F_BODIES` to the value from `F_BODIES` above

### Feature Extraction
- Update `F_H5` with the value we set above for `O_H5`
- Update `F_PKL` and `T_PKL` based on train and generated test features. This is just name of file generated
- Run the file

### Classification
- download `feature_new.pkl` file and put it in the folder where the `classification.py` is.
- Update `F_H5` value with the value we set above for `O_H5` above
- Update `F_PKL` value with the value we set above for `F_PKL`
- Run the file for the classification


* `agree_classifier.pkl`, `discuss_classifier.pkl`, `relatedness_classifier.pkl` should be in the same location as the program.

#### Required Files are in google drive
https://drive.google.com/drive/folders/1NiFtlfnxgLteDjlUsVKPJfGjQJD3dKq4?usp=sharing


