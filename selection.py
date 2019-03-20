"""Select new instances given prediction and retrieval modules"""
import math
import collections
import torch
from torchtext import data

from utils import torch_utils, scorer
from utils.torch_utils import example_to_dict

TOKEN = data.Field(sequential=True, batch_first=True, lower=True, include_lengths=True)
RELATION = data.Field(sequential=False, unk_token=None, pad_token=None)
POS = data.Field(sequential=True, batch_first=True)
NER = data.Field(sequential=True, batch_first=True)
PST = data.Field(sequential=True, batch_first=True)
PR_CONFIDENCE = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
SL_CONFIDENCE = data.Field(sequential=False, use_vocab=False, dtype=torch.float)

FIELDS = {
    'tokens': ('token', TOKEN),
    'stanford_pos': ('pos', POS),
    'stanford_ner': ('ner', NER),
    'relation': ('relation', RELATION),
    'subj_pst': ('subj_pst', PST),
    'obj_pst': ('obj_pst', PST),
    'pr_confidence': ('pr_confidence', PR_CONFIDENCE),
    'sl_confidence': ('sl_confidence', SL_CONFIDENCE)
}


def get_relation_distribution(dataset):
    """Get relation distribution of a dataset

    Args:
      dataset (data.Dataset or list): The dataset to consider
    """
    if isinstance(dataset, data.Dataset):
        counter = collections.Counter([ex.relation for ex in dataset.examples])
    else:
        counter = collections.Counter([pred for eid, pred, actual in dataset])
    return {k: v / len(dataset) for k, v in counter.items()}


def split_samples(dataset, meta_idxs, batch_size=50, conf_p=None, conf_s=None):
    """Split dataset using idxs

    Args:
        dataset (data.Dataset): Dataset instance
        meta_idxs (list): List of indexes with the form (idx, predict_label, gold_label)
        batch_size (int, optional): Defaults to 50
        conf_p (dict, optional): An optional attribute for confidence of samples for predictor
        conf_s (dict, optional): An optional attribute for confidence of samples for selector
    """
    iterator_unlabeled = data.Iterator(
        dataset=dataset,
        batch_size=batch_size,
        device=-1,
        repeat=False,
        train=False,
        shuffle=False,
        sort=True,
        sort_key=lambda x: -len(x.token),
        sort_within_batch=False)
    examples = iterator_unlabeled.data()
    new_examples, rest_examples, example_ids = [], [], set(idx for idx, pred, actual in meta_idxs)
    if conf_p is not None and conf_s is not None:
        meta_idxs = [(idx, pred, actual, conf_p[idx], conf_s[idx])
                     for idx, pred, actual in meta_idxs]
    elif conf_p is None and conf_s is None:
        meta_idxs = [(idx, pred, actual, 1.0, 1.0) for idx, pred, actual in meta_idxs]
    else:
        raise NotImplementedError('Can not split_samples.')
    for idx, pred, _, pr_confidence, sl_confidence in meta_idxs:
        output = example_to_dict(examples[idx], pr_confidence, sl_confidence, pred)
        new_examples.append(data.Example.fromdict(output, FIELDS))
        rest_examples = [example for k, example in enumerate(examples) if k not in example_ids]
    return new_examples, rest_examples


def intersect_samples(meta_idxs1, s_retrieve_fn, k_samples, prior_distribution):
    upperbound, meta_idxs, confidence_idxs_s = k_samples, [], []
    while len(meta_idxs) < min(k_samples, len(meta_idxs1)):
        upperbound = math.ceil(1.25 * upperbound)
        ori_meta_idxs_s, confidence_idxs_s = s_retrieve_fn(upperbound, prior_distribution)
        meta_idxs = sorted(set(meta_idxs1[:upperbound]).intersection(
            set(ori_meta_idxs_s)))[:k_samples]
        if upperbound > k_samples * 30:  # set a limit for growing upperbound
            break
    print('Infer on combination...')
    scorer.score([actual for _, _, actual in meta_idxs], [pred for _, pred, _ in meta_idxs],
                 verbose=False)
    scorer.score([actual for _, _, actual in meta_idxs], [pred for _, pred, _ in meta_idxs],
                 verbose=False,
                 NO_RELATION='-1')
    return meta_idxs, confidence_idxs_s


def select_samples(model_p, model_s, dataset_infer, k_samples, args, default_distribution):
    max_upperbound = int(math.ceil(k_samples * args.selector_upperbound))
    # predictor selection
    meta_idxs_p, confidence_idxs_p = model_p.retrieve(
        dataset_infer, len(dataset_infer))  # retrieve all the samples
    print('Infer on predictor: ')  # Track performance of predictor alone
    gold, guess = [t[2] for t in meta_idxs_p[:k_samples]], [t[1] for t in meta_idxs_p[:k_samples]]
    scorer.score(gold, guess, verbose=False)
    scorer.score(gold, guess, verbose=False, NO_RELATION='-1')

    # for self-training
    if args.integrate_method == 'p_only':
        return split_samples(dataset_infer, meta_idxs_p[:k_samples], args.batch_size)

    # selector selection
    label_distribution = None
    if args.integrate_method == 's_only' or max_upperbound == 0:
        label_distribution = default_distribution
    else:
        label_distribution = get_relation_distribution(meta_idxs_p[:max_upperbound])

    def s_retrieve_fn(k_samples, label_distribution):
        return model_s.retrieve(dataset_infer, k_samples, label_distribution=label_distribution)

    ori_meta_idxs_s, _ = s_retrieve_fn(k_samples, label_distribution)
    print('Infer on selector: ')
    gold, guess = [t[2] for t in ori_meta_idxs_s], [t[1] for t in ori_meta_idxs_s]
    scorer.score(gold, guess, verbose=False)
    scorer.score(gold, guess, verbose=False, NO_RELATION='-1')

    # If we only care about performance of selector
    if args.integrate_method == 's_only':
        return split_samples(dataset_infer, ori_meta_idxs_s)

    # integrate method
    if args.integrate_method == 'intersection':
        meta_idxs, confidence_idxs_s = intersect_samples(meta_idxs_p, s_retrieve_fn, k_samples,
                                                         label_distribution)
    else:
        raise NotImplementedError('integrate_method {} not implemented'.format(
            args.integrate_method))
    confidence_dict_p = dict((id, confidence) for id, confidence in confidence_idxs_p)
    confidence_dict_s = dict((id, confidence) for id, confidence in confidence_idxs_s)
    return split_samples(
        dataset_infer, meta_idxs, conf_p=confidence_dict_p, conf_s=confidence_dict_s)
