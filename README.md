# clinicalbert_replication
In this project, the main purpose is to replicate the result of clinicalBERT from paper ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission by Kexin Huang, Jaan Altosaar, Rajesh Ranganath. The original repo for clinicalBERT can be found https://github.com/kexinhuang12345/clinicalBERT.

In addition to that, this repo also tries to replicate the baseline models including Bag-of-Words and BI-LSTM mentioned from the paper What’s in a Note? Unpacking Predictive Value in Clinical Note Representations by Willie Boag, B.S.,1,* Dustin Doss, S.B.,1,* Tristan Naumann, M.S.,1,* and Peter Szolovits, Ph.D.1. The original repo for baselines can be found https://github.com/wboag/wian/tree/13bc79ebf7724aa007dbb436b4daa1ddd40e4734.

Modifications are made in order to carry out the experiments.



## clinicalBERT

### Datasets

The paper uses MIMIC-III(https://mimic.mit.edu/) dataset, which requires the CITI training program in order to use it. Dataset_Split.ipynb is used to preprocess and split the dataset.


### Data split for 5-folder cross-validation:
notebooks/Dataset_Split.ipynb
File system expected:
```
-data
  -good_datasets
    -fold1
      -discharge
        -train.csv
        -val.csv
        -test.csv
      -3days
        -train.csv
        -val.csv
        -test.csv
      -2days
        -test.csv
```
Data file is expected to have column "TEXT", "ID" and "Label" (Note chunks, Admission ID, Label of readmission) as in data/good_datasets/fold1/. TEXT field is blanked out.
