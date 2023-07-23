# Type-enhanced Inductive Knowledge Completion

The repository provides the code and data used in our experiments.

## Requirements

python==3.7.13

torch==1.5.0

dgl==0.4.2

scikit-learn

tqdm

lmdb

## Directory

`data`: The inductive datasets split by GraIL

`types`: The raw types of entities we obtained and the types of entities after preprocessing.

`expri_save_models`: The trained models to generate experimental results in the paper.

## Run
We provide the commands to train and test our model, and the illustration of their parameters. Take `nell_v1` for example.
+ training
  `python train.py -d nell_v1 -e nell_v1 -ne 20 --ont`
  + `-d`: the name of training dataset
  + `-e`: the directory of saved models
  + `-ne`: the number of epoches
  + `--ont`: type-enhanced model
+ test on `AUC-PR`
  `python test_auc.py -d nell_v1_ind -e nell_v1 --ont --runs 5`
  + `-d`: the name of test dataset
  + `-e`: the directory of saved models
  + `--ont`: type-enhanced model
  + `--runs`: run times
  
+ test on `Hits@10`
  `python test_ranking.py -d nell_v1_ind -e nell_v1 --ont`
  + `-d`: the name of test dataset
  + `-e`: the directory of saved models
  + `--ont`: type-enhanced model

