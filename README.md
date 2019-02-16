# DualRE

Source code for WWW'19 paper [Learning Dual Retrieval Module for Semi-supervised Relation Extraction](a).

DualRE proposes to construct a retrieval module as the dual problem of relation prediction. Two modules are trained in an iterative way and retrieve new instances with higher quality.

## Dependencies

We use ubuntu 18.0 and python 3.6.5 to conduct the experimented documented in paper. The required python packages are documented in `requirements.txt`. Note that we use an early version of pytorch, which may throw several warnings in latest version.

## Data Preparation

All data should be put into `dataset/$data_name` folder in a similar format as `dataset/sample`, with a naming convention such that (1) `train-$ratio.json` indicates that certain percentage of training data are used. (2) `raw-$ratio.json` is a part of original training data, in which we assume the labels are unknown to model.

To replicate the experiments, first prepare the required dataset as below:

- SemEval: SemEval 2010 Task 8 data (included in `dataset/semeval`)
- TACRED: The TAC Relation Extraction Dataset ([download](https://catalog.ldc.upenn.edu/LDC2018T24))
  - Put the official dataset (in JSON format) under folder `dataset/tacred` in a similar format like [here](https://github.com/yuhaozhang/tacred-relation/tree/master/dataset/tacred).

Then use the scripts from `utils/data_utils.py` to further preprocess the data. For SemEval, the script split the original training data into two sets (labeled and unlabeled) and then separate them into multiple ratios. For TACRED, the script first perform some preprocessing to ensure the same format as SemEval.

```bash
python utils/data_utils.py --data_name semeval --in_dir dataset/semeval --out_dir dataset/semeval  # update the original repository
```

## Model Running

To run DualRE on specific dataset, check out `run-once.sh` for required parameters. For example, to run DualRE model on SemEval with 10% of labeled data and 50% of unlabeled data, and save the result logs in a specific format, use the following command:

```bash
./run-once.sh semeval 0.1 0.5 DualRE >> results/semeval/DualRE/dr0.1_0.5-log.txt
```

`run-once.sh` script also support some baseline models such as Self-Training and RE-Ensemble.

To further fine tune specific model parameters (e.g. dropout rate, \alpha, \beta as confidence weights, check out argument declarations in `train.py`)

## Evaluation

The output log stores the precision, recall and F-1 in each iteration at the end of file. To automate the process of parsing and calculating results when running on different random seeds, run the following command:

```bash
python utils/scorer.py semeval DualRE 0.1 0.5
```

The scripts suppose the output logs are stored with the format `results/semeval/DualRE/dr0.1_0.5.*\.txt`, with different logs indicating different runs of experiments.

## Code Overview

The main entry for all models is in `train.py`. In the code, the prediction and retrieval modules are aliased as 'predictor' and 'selector', with some abbreviated names such as 'p_selection_method' indicating selection method for predictor.

For all selection-related logics, given trained prediction and retrieval module, check out `selection.py`.

For modification of predictor/selector, check out `models/predictor.py` and `models/selector.py`. These two types of models are wrapped up by a unified Trainer object in `models/trainer.py`, which provides interface for training, predicting and evaluating modules. Functionality to retrieve of new instances is also included.
