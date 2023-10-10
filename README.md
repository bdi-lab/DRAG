# Dynamic Relation-Attentive Graph Neural Networks for Fraud Detection
This code is the official implementation of the following [paper](http://arxiv.org/abs/2310.04171):

> Heehyeon Kim, Jinhyeok Choi, and Joyce Jiyoung Whang, Dynamic Relation-Attentive Graph Neural Networks for Fraud Detection, Machine Learning on Graphs (MLoG) Workshop at the 23rd IEEE International Conference on Data Mining (ICDM), 2023

All codes are written by Heehyeon Kim (heehyeon@kaist.ac.kr) and Jinhyeok Choi (cjh0507@kaist.ac.kr). When you use this code, please cite our paper.

```bibtex
@article{drag,
  author={Heehyeon Kim, Jinhyeok Choi, and Joyce Jiyoung Whang},
  title={Dynamic Relation-Attentive Graph Neural Networks for Fraud Detection},
  year={2023},
  journal={arXiv preprint arXiv.2310.04171},
  doi = {10.48550/arXiv.2310.04171}
}
```

## Requirments
We used Python 3.8, Pytorch 1.12.1, and DGL 1.0.2 with cudatoolkit 11.3.

## Usage
### DRAG
We used NVIDIA RTX A6000 and NVIDIA GeForce RTX 3090 for all our experiments. We provide the template configuration file (`template.json`) for the YelpChi and Amazon_new datasets.

To train DRAG, use the `run.py` file as follows:

```python
python run.py --exp_config_path=./template.json
```
Results will be printed in the terminal and saved in the directory designated by the configuration file.

Each run corresponds to an experiment ID `f"{dataset_name}-{train_ratio}-{seed}-{time}"`.

You can find log files and pandas DataFrame pickle files associated with experiment IDs in the designated directory.

There are some useful functions to handle experiment results in `utils.py`.

You can find an example in `performance_check.ipynb`.

### Training from Scratch
To train DRAG from scratch, run `run.py` with the configuration file. Please refer to `model_handler.py`, `data_handler.py`, and `model.py` for examples of the arguments in the configuration file.

The list of arguments of the configuration file:
- `--seed`: seed
- `--data_name`: name of the fraud detection dataset (available datasets are YelpChi(`yelp`) and Amazon_new(`amazon_new`))
- `--n_head`: a list consisting of the number of heads for each DRAGConv layer $N_{\alpha}$
- `--n_head_agg`: a list consisting of the number of heads for aggregation from different relations $N_{\gamma}$ and layers $N_{\beta}$
- `--train_ratio`: train ratio
- `--test_ratio`: test ratio
- `--emb_size`: a list consisting of the embedding size $d'$ for each DRAGConv layer
- `--lr`: learning rate
- `--weight_decay`: weight decay
- `--feat_drop`: feature dropout rate for DRAGConv layer
- `--attn_drop`: attention dropout rate for DRAGConv layer
- `--epochs`: total number of training epochs 
- `--valid_epochs`: the duration of validation
- `--batch_size`: the batch size
- `--patience`: early stopping patience
- `--save_dir`: directory path for saving train, validation, test logs, and the best model

## Hyperparameters
We tuned DRAG with the following tuning ranges:
- `lr`: {0.01, 0.001}
- `weight_decay`: {0.001, 0.0001}
- `feat_drop`: 0.0
- `attn_drop`: 0.0
- `len(emb_size)`: $L$ = {1, 2, 3}
- `n_head`: $N_{\alpha}$ = {2, 8}
- `n_head_agg`: $N_{\beta}$ , $N_{\gamma}$ = {2, 8}
- `train_ratio`: {0.01, 0.1, 0.4}
- `test_ratio`: 0.67
- `batch_size`: 1024
- `embedding size`: $d'$ = 64
- `epochs`: 1000
- `patience`: 100

## Description for each file
- `datasets.py`: A file for loading the YelpChi and Amazon_new datasets
- `data_handler.py`: A file for processing the given dataset according to the arguments
- `layers.py`: A file for defining the DRAGConv layer
- `model_handler.py`: A file for training DRAG
- `models.py`: A file for defining DRAG architecture
- `performance_check.ipynb`: A file for checking the fraud detection performance of DRAG
- `run.py`: A file for running DRAG on the YelpChi and Amazon_new datasets
- `result_manager.py`: A file for managing train, validation, and test logs
- `template.json`: A template file consisting of arguments
- `utils.py`: A file for defining utility functions
