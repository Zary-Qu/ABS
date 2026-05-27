# ABS
# Code and data for the paper "A Logical Learning Framework with Adaptive Dual Encoder and Bidirectional State Flow for Document-Level Relation Extraction"
## Datasets

| Dataset | Language | Source |
|---------|----------|--------|
| DWIE | English | (https://github.com/klimzaporojets/DWIE) |
| Re-DocRED | English | (https://github.com/tonytan48/Re-DocRED) |
| DocRED | English | (https://github.com/thunlp/DocRED) |
| HacRED | Chinese | (https://github.com/qiaojiim/HacRED) |

### Prerequisites
Create environment using the provided YAML file:
```bash
conda env create -f ATLOP_ABS.yml
conda activate ATLOP_ABS
```

### Use examples
#### ABS-ATLOP

Path for code: `./ABS-ATLOP`

The script for both training and evaluation on the DWIE dataset is:
```bash
python -u train.py --dataset dwie --transformer_type bert --model_name_or_path ../PLM/bert-base-uncased --train_file train_annotated.json --dev_file dev.json --test_file test.json --save_path ../trained_model/ABS_ALTOP_DWIE.pth --num_train_epochs 300.0 --train_batch_size 4 --test_batch_size 4 --seed 66 --num_class 66 --tau 1.0 --lambda_sym 0.1
```
The script for both training and evaluation on the Re-DocRED dataset is:
```bash
python -u train.py --dataset ReDocRE --transformer_type bert --model_name_or_path ../PLM/bert-base-uncased --train_file train_revised.json --dev_file dev_revised.json --test_file test_revised.json --save_path ../trained_model/ABS_ALTOP_REDOCRED.pth --num_train_epochs 100.0 --train_batch_size 4 --test_batch_size 4 --seed 66 --num_class 97 --tau 0.2 --lambda_sym 0.1
```
For running experiments on the HacRED dataset, please replace the original `evaluation.py` with the provided `evaluation_HacRED.py`. Please also use the Chinese versions of BERT and Longformer when specifying the encoder.

The script for both training and evaluation on the HacRED dataset is:
```bash
python -u train.py --dataset hacred --transformer_type bert --model_name_or_path ../PLM/bert-base-chinese --train_file train.json --dev_file dev.json --test_file test.json --save_path ../trained_model/model_ALTOP_HACRED.pth --num_train_epochs 100.0 --train_batch_size 4 --test_batch_size 4 --seed 66 --num_class 27 --tau 1.0 --lambda_sym 0.1
```
