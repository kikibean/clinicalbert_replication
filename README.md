# clinicalbert_replication
In this project, the main purpose is to replicate the result of clinicalBERT from paper [ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission](https://arxiv.org/pdf/1904.05342.pdf) by Kexin Huang, Jaan Altosaar, Rajesh Ranganath. The original repo for clinicalBERT can be found https://github.com/kexinhuang12345/clinicalBERT.

In addition to that, this repo also tries to replicate the baseline models including Bag-of-Words and BI-LSTM mentioned from the paper [What’s in a Note? Unpacking Predictive Value in Clinical Note Representations](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5961801/pdf/2840866.pdf) by Willie Boag, B.S.,1,* Dustin Doss, S.B.,1,* Tristan Naumann, M.S.,1,* and Peter Szolovits, Ph.D.1. The original repo for baselines can be found https://github.com/wboag/wian/tree/13bc79ebf7724aa007dbb436b4daa1ddd40e4734.

Modifications are made in order to carry out the experiments.



# clinicalBERT

### Datasets

The paper uses [MIMIC-III](https://mimic.mit.edu/) dataset, which requires the CITI training program in order to use it. preprocess.ipynb is used to preprocess and merge data from admission information and clinical notes, Dataset_Split.ipynb is used to split the dataset for 5-folder cross-validation.


### Data split for 5-folder cross-validation:
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

### Pre-training from BERT checkpoints
Mainly used modified code from [BERT repo]{https://github.com/google-research/bert}
File system expected:
```
-INITIAL_DATA_PATH (for BERT config file, initial checkpoints, etc)
-INITIAL_MODEL_PATH (for BERT vocab.txt)
-PRETRAIN_DATA_PATH (to store pre-training tensorflow records)
-PRETRAINED_MODEL_PATH (to save pre-trained model checkpoints)
```
```
#convert data to tensorflow record
create_pretraining_data.ipynb

#
 %tensorflow_version 1.x
!python ./run_pretraining.py \
--input_file ./PRETRAIN_DATA_PATH/tf_examples_128_fold12.tfrecord \
--output_dir ./PRETRAINED_MODEL_PATH/pretraining_output_discharge_fold12 \
--do_train \
--do_eval \
--bert_config_file ./INITIAL_DATA_PATH/bert_config.json \
--init_checkpoint ./PRETRAINED_MODEL_PATH/pretraining_output_discharge_fold12/model.ckpt-50000  \
--train_batch_size 64 \
--max_seq_length 128 \
--max_predictions_per_seq 20 \
--num_train_steps 10000 \
--num_warmup_steps 10 \
--learning_rate 2e-5 \


```


