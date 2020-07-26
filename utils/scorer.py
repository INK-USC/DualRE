#!/usr/bin/env python
"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

import argparse
import sys
from collections import Counter
import numpy as np

from pathlib import Path

NO_RELATION = "no_relation"


def parse_arguments():
    parser = argparse.ArgumentParser(description='Score a prediction file using the gold labels.')
    parser.add_argument('gold_file', help='The gold relation file; one relation per line')
    parser.add_argument(
        'pred_file',
        help='A prediction file; one relation per line, in the same order as the gold file.')
    args = parser.parse_args()
    return args


def score(key, prediction, verbose=False, NO_RELATION=NO_RELATION):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1:
                sys.stdout.write(' ')
            if prec < 1.0:
                sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1:
                sys.stdout.write(' ')
            if recall < 1.0:
                sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1:
                sys.stdout.write(' ')
            if f1 < 1.0:
                sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(
            sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(
            sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("SET NO_RELATION ID: ", NO_RELATION)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro


def AUC(logits, labels):
    num_right = sum(labels)
    num_total = len(labels)
    num_total_pairs = (num_total - num_right) * num_right

    if num_total_pairs == 0:
        return 0.5

    num_right_pairs = 0
    hit_count = 0
    for label in labels:
        if label == 0:
            num_right_pairs += hit_count
        else:
            hit_count += 1

    return float(num_right_pairs) / num_total_pairs


def print_table(*args, header=''):
    print(header)
    for tup in zip(*args):
        print('\t'.join(['%.3f' % t for t in tup]))


def result_summary(result_dir, dr=None, write_to_file=True):
    prefix = 'dr' + str(dr[0]) + '_' + str(dr[1])
    base_test_p = []
    base_test_r = []
    base_test_f1 = []
    dev_p = []
    dev_r = []
    dev_f1 = []
    test_p = []
    test_r = []
    test_f1 = []

    for file_name in Path(result_dir).glob(prefix + '*.txt'):  # each seed
        with open(file_name) as f:
            lines = f.readlines()
            for j in range(len(lines) - 1, 0, -1):
                if lines[j].startswith('Best dev and test F1'):
                    break
            if j == 1:
                print('Miss result in: ' + str(file_name))
                continue
            dev_f1_total = []
            test_f1_total = []

            for k in range(j + 1, len(lines)):
                line = lines[k]
                dev_f1_total.append(float(line.split('\t')[0].strip()))
                test_f1_total.append(float(line.split('\t')[1].strip()))
            i = dev_f1_total.index(max(dev_f1_total))
            # get i: the best round
            for j in range(len(lines) - 1, 0, -1):
                if lines[j].startswith('Final evaluation #0 on test set'):
                    break
            for k in range(j, len(lines)):
                if lines[k].startswith('Precision'):
                    break
            p = float(lines[k].split()[-1][:-1])
            r = float(lines[k + 1].split()[-1][:-1])
            f1 = float(lines[k + 2].split()[-1][:-1])
            base_test_p += [p]
            base_test_r += [r]
            base_test_f1 += [f1]

            for j in range(len(lines) - 1, 0, -1):
                if lines[j].startswith('Final evaluation #' + str(i) + ' on dev set'):
                    break
            for k in range(j, len(lines)):
                if lines[k].startswith('Precision'):
                    break
            p = float(lines[k].split()[-1][:-1])
            r = float(lines[k + 1].split()[-1][:-1])
            f1 = float(lines[k + 2].split()[-1][:-1])
            dev_p += [p]
            dev_r += [r]
            dev_f1 += [f1]

            for j in range(len(lines) - 1, 0, -1):
                if lines[j].startswith('Final evaluation #' + str(i) + ' on test set'):
                    break
            for k in range(j, len(lines)):
                if lines[k].startswith('Precision'):
                    break
            p = float(lines[k].split()[-1][:-1])
            r = float(lines[k + 1].split()[-1][:-1])
            f1 = float(lines[k + 2].split()[-1][:-1])
            test_p += [p]
            test_r += [r]
            test_f1 += [f1]
    if len(base_test_p) == 0:
        return
    base_mean_p, base_std_p = float(np.mean(base_test_p)), float(np.std(base_test_p))
    base_mean_r, base_std_r = float(np.mean(base_test_r)), float(np.std(base_test_r))
    base_mean_f1, base_std_f1 = float(np.mean(base_test_f1)), float(np.std(base_test_f1))

    dev_mean_p, dev_std_p = float(np.mean(dev_p)), float(np.std(dev_p))
    dev_mean_r, dev_std_r = float(np.mean(dev_r)), float(np.std(dev_r))
    dev_mean_f1, dev_std_f1 = float(np.mean(dev_f1)), float(np.std(dev_f1))

    test_mean_p, test_std_p = float(np.mean(test_p)), float(np.std(test_p))
    test_mean_r, test_std_r = float(np.mean(test_r)), float(np.std(test_r))
    test_mean_f1, test_std_f1 = float(np.mean(test_f1)), float(np.std(test_f1))

    print('\n\n#####\t%s\t#####' % prefix)
    print(len(base_test_p), 'seeds')
    print('base: %.2f $\pm$ %.2f\t%.2f $\pm$ %.2f\t%.2f $\pm$ %.2f' %
          (base_mean_p, base_std_p, base_mean_r, base_std_r, base_mean_f1, base_std_f1))
    print('dev: %.2f $\pm$ %.2f\t%.2f $\pm$ %.2f\t%.2f $\pm$ %.2f' %
          (dev_mean_p, dev_std_p, dev_mean_r, dev_std_r, dev_mean_f1, dev_std_f1))
    print('test: %.2f $\pm$ %.2f\t%.2f $\pm$ %.2f\t%.2f $\pm$ %.2f' %
          (test_mean_p, test_std_p, test_mean_r, test_std_r, test_mean_f1, test_std_f1))


if __name__ == "__main__":
    data_name = sys.argv[1]
    model_name = sys.argv[2]
    data_ratio = (float(sys.argv[3]), float(sys.argv[4]))
    result_summary(
        './results/' + data_name + '/' + model_name + '/', dr=data_ratio, write_to_file=False)
