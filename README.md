# ABS
# Code and data for the paper "A Logical Learning Framework with Adaptive Dual Encoding and Bidirectional State Flow for Document-Level Relation Extraction"
## Prerequisites
Create environment using the provided YAML file:
```bash
conda env create -f ATLOP_ABS.yml
conda activate ATLOP_ABS
```

## Use examples
### ABS-ATLOP

Path for code: `./ABS-ATLOP`

The script for both training and evaluation on the DWIE dataset is:
```bash
python -u train.py --dataset dwie --transformer_type bert --model_name_or_path ../PLM/bert-base-uncased --train_file train_annotated.json --dev_file dev.json --test_file test.json --save_path ../trained_model/ABS_ALTOP_DWIE.pth --num_train_epochs 100.0 --train_batch_size 2 --test_batch_size 4 --seed 66 --num_class 66 --tau 1.0
```
The script for both training and evaluation on the Re-DocRED dataset is:
```bash
python -u train.py --dataset ReDocRE --transformer_type bert --model_name_or_path ../PLM/bert-base-uncased --train_file train_revised.json --dev_file dev_revised.json --test_file test_revised.json --save_path ../trained_model/ABS_ALTOP_REDOCRED.pth --num_train_epochs 30.0 --train_batch_size 4 --test_batch_size 4 --seed 66 --num_class 97 --tau 0.2
```
The script for both training and evaluation on the HacRED dataset is:
```bash
python -u train.py --dataset hacred --transformer_type bert --model_name_or_path ../PLM/bert-base-chinese --train_file train.json --dev_file dev.json --test_file test.json --save_path ../trained_model/model_ALTOP_HACRED_123L_1.pth --num_train_epochs 20.0 --train_batch_size 4 --test_batch_size 4 --seed 66 --num_class 27 --tau 1.0
```
