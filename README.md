# DualRE

Source code for WWW'19 paper [Learning Dual Retrieval Module for Semi-supervised Relation Extraction](https://arxiv.org/abs/1902.07814).

[Presentation](https://docs.google.com/presentation/d/15TLIQOWhMa3MCkDrCiBUz2r2HBJx9P79R-EmouXsJB4/edit?usp=sharing) for WWW

DualRE is a **semi-supervised relation extraction** model, that proposes to construct a retrieval module as the dual problem of relation prediction. Two modules are trained in an iterative way and retrieve new instances with higher quality.

## Benchmark

Performance comparison with several relation extraction systems in **SemEval** dataset (sentence-level extraction). We compared with (1) supervised models (PCNN, PRNN) training on 10% labeled data; (2) semi-supervised models (Mean-Teacher, Self-Training) training on 10% labeled data and 50% unlabeled data.


Method | Precision | Recall | F1 
-------|:-----------:|:--------:|:----:
PCNN ([Zeng et al., 2015](https://aclanthology.info/papers/D15-1203/d15-1203)) | 53.78 ± 1.51 | 49.11 ± 2.22 | 51.32 ± 1.74
PRNN ([Zhang et al., 2017](https://nlp.stanford.edu/pubs/zhang2017tacred.pdf)) | 61.70 ± 1.16 | 63.61 ± 2.07 | 62.63 ± 1.42
Mean-Teacher ([Tarvainen and Valpola, 2017](https://arxiv.org/abs/1703.01780)) | 62.43 ± 1.28 | 60.34 ± 0.62 | 61.36 ± 0.75
Self-Training ([Rosenberg et al., 2005](https://dl.acm.org/citation.cfm?id=1042449.1043907)) | 64.27 ± 2.37 | 63.48 ± 2.02 | 63.79 ± 0.28
**DualRE** ([Lin et al., 2019](https://arxiv.org/abs/1902.07814)) | 64.50 ± 1.14 | 67.67 ± 1.66 | **66.03 ± 1.00**

Performance comparison in **TACRED** dataset with the same setting.

Method | Precision | Recall | F1 
-------|:-----------:|:--------:|:----:
PCNN ([Zeng et al., 2015](https://aclanthology.info/papers/D15-1203/d15-1203)) | 64.32 ± 7.78 | 42.06 ± 4.94 | 50.16 ± 1.15
PRNN ([Zhang et al., 2017](https://nlp.stanford.edu/pubs/zhang2017tacred.pdf)) | 53.44 ± 2.82 | 51.77 ± 1.88 | 52.49 ± 0.64
Mean-Teacher ([Tarvainen and Valpola, 2017](https://arxiv.org/abs/1703.01780)) | 58.53 ± 2.56 | 50.08 ± 1.14 | 53.94 ± 0.91
Self-Training ([Rosenberg et al., 2005](https://dl.acm.org/citation.cfm?id=1042449.1043907)) | 56.54 ± 0.72 | 53.00 ± 0.49 | 54.71 ± 0.09
**DualRE** ([Lin et al., 2019](https://arxiv.org/abs/1902.07814)) | 61.61 ± 1.30 | 52.30 ± 0.89 | **56.56 ± 0.42**



## Get Started

We use ubuntu 18.0 and python 3.6.5 to conduct the experimented documented in paper. The required python packages are documented in `requirements.txt`. Note that we use an early version of pytorch, which may throw several warnings in latest version.

### Data Preparation

All data should be put into `dataset/$data_name` folder in a similar format as `dataset/sample`, with a naming convention such that (1) `train-$ratio.json` indicates that certain percentage of training data are used. (2) `raw-$ratio.json` is a part of original training data, in which we assume the labels are unknown to model.

To replicate the experiments, first prepare the required dataset as below:

- SemEval: SemEval 2010 Task 8 data (included in `dataset/semeval`)
- TACRED: The TAC Relation Extraction Dataset ([download](https://catalog.ldc.upenn.edu/LDC2018T24))
  - Put the official dataset (in JSON format) under folder `dataset/tacred` in a similar format like [here](https://github.com/yuhaozhang/tacred-relation/tree/master/dataset/tacred).

Then use the scripts from `utils/data_utils.py` to further preprocess the data. For SemEval, the script split the original training data into two sets (labeled and unlabeled) and then separate them into multiple ratios. For TACRED, the script first perform some preprocessing to ensure the same format as SemEval.

```bash
python utils/data_utils.py --data_name semeval --in_dir dataset/semeval --out_dir dataset/semeval  # update the original repository
```

### Model Running

To run DualRE on specific dataset, check out `run-once.sh` for required parameters. For example, to run DualRE model on SemEval with 10% of labeled data and 50% of unlabeled data, and save the result logs in a specific format, use the following command:

```bash
./run-once.sh semeval 0.1 0.5 DualRE >> results/semeval/DualRE/dr0.1_0.5-log.txt
```

`run-once.sh` script also support some baseline models such as Self-Training and RE-Ensemble.

To further fine tune specific model parameters (e.g. dropout rate, \alpha, \beta as confidence weights, check out argument declarations in `train.py`)

### Evaluation

The output log stores the precision, recall and F-1 in each iteration at the end of file. To automate the process of parsing and calculating results when running on different random seeds, run the following command:

```bash
python utils/scorer.py semeval DualRE 0.1 0.5
```

The scripts suppose the output logs are stored with the format `results/semeval/DualRE/dr0.1_0.5.*\.txt`, with different logs indicating different runs of experiments.

## Code Overview

The main entry for all models is in `train.py`. In the code, the prediction and retrieval modules are aliased as 'predictor' and 'selector', with some abbreviated names such as 'p_selection_method' indicating selection method for predictor.

For all selection-related logics, given trained prediction and retrieval module, check out `selection.py`.

For modification of predictor/selector, check out `models/predictor.py` and `models/selector.py`. These two types of models are wrapped up by a unified Trainer object in `models/trainer.py`, which provides interface for training, predicting and evaluating modules. Functionality to retrieve of new instances is also included.


## Citation

```latex
@inproceedings{lin2019dualre,
  title={Learning Dual Retrieval Module for Semi-supervised Relation Extraction},
  author={Lin, Hongtao and Yan, Jun and Qu, Meng and Ren, Xiang},
  booktitle={The Web Conference},
  year={2019}
}
```
