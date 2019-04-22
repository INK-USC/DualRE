"""
A rnn model for relation extraction, written in pytorch.
"""
import math
import time
import os
from datetime import datetime
from shutil import copyfile
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchtext import data

from utils import torch_utils, scorer
from utils.torch_utils import batch_to_input, arg_max


def idx_to_onehot(target, opt, confidence=None):
    sample_size, class_size = target.size(0), opt['num_class']
    if confidence is None:
        y = torch.zeros(sample_size, class_size)
        y = y.scatter_(1, torch.unsqueeze(target.data, dim=1), 1)
    else:
        y = torch.ones(sample_size, class_size)
        y = y * (1 - confidence.data).unsqueeze(1).expand(-1, class_size)
        y[torch.arange(sample_size).long(), target.data] = confidence.data

    y = Variable(y)

    return y


def evaluate(model, dataset, evaluate_type='prf', verbose=False):
    rel_stoi, rel_itos = model.opt['rel_stoi'], model.opt['rel_itos']
    iterator_test = data.Iterator(
        dataset=dataset,
        batch_size=model.opt['batch_size'],
        device=-1,
        repeat=False,
        train=True,
        shuffle=False,
        sort=True,
        sort_key=lambda x: -len(x.token),
        sort_within_batch=False)

    if evaluate_type == 'prf':
        predictions = []
        all_probs = []
        golds = []
        all_loss = 0
        for batch in iterator_test:
            inputs, target = batch_to_input(batch, model.opt['vocab_pad_id'])
            preds, probs, loss = model.predict(inputs, target)
            predictions += preds
            all_probs += probs
            all_loss += loss
            golds += target.data.tolist()
        predictions = [rel_itos[p] for p in predictions]
        golds = [rel_itos[p] for p in golds]
        p, r, f1 = scorer.score(golds, predictions, verbose=verbose)
        return p, r, f1, all_loss
    elif evaluate_type == 'auc':
        logits, labels = [], []
        for batch in iterator_test:
            inputs, target = batch_to_input(batch, model.opt['vocab_pad_id'])
            logits += model.predict(inputs)[0]
            labels += batch.relation.data.numpy().tolist()
        p, q = 0, 0
        for rel in range(len(rel_itos)):
            if rel == rel_stoi['no_relation']:
                continue
            logits_rel = [logit[rel] for logit in logits]
            labels_rel = [1 if label == rel else 0 for label in labels]
            ranking = list(zip(logits_rel, labels_rel))
            ranking = sorted(ranking, key=lambda x: x[0], reverse=True)
            logits_rel, labels_rel = zip(*ranking)
            p += scorer.AUC(logits_rel, labels_rel)
            q += 1

        dev_auc = p / q * 100
        return dev_auc, None, None, None


def calc_confidence(probs, exp):
    '''Calculate confidence score from raw probabilities'''
    return max(probs)**exp


class Trainer(object):
    """ A wrapper class for the training and evaluation of models. """

    def __init__(self, opt, model, model_type='predictor'):
        self.opt = opt
        self.model_type = model_type
        self.model = model
        if model_type == 'predictor':
            self.criterion = nn.CrossEntropyLoss(reduce=False)
        elif model_type == 'pointwise':
            self.criterion = nn.BCEWithLogitsLoss()
        elif model_type == 'pairwise':
            self.criterion = nn.BCEWithLogitsLoss(
            )  # Only a placeholder, will NOT use this criterion
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]

        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()

        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def train(self, dataset_train, dataset_dev):
        opt = self.opt.copy()
        iterator_train = data.Iterator(
            dataset=dataset_train,
            batch_size=opt['batch_size'],
            device=-1,
            repeat=False,
            train=True,
            shuffle=True,
            sort_key=lambda x: len(x.token),
            sort_within_batch=True)
        iterator_dev = data.Iterator(
            dataset=dataset_dev,
            batch_size=opt['batch_size'],
            device=-1,
            repeat=False,
            train=True,
            sort_key=lambda x: len(x.token),
            sort_within_batch=True)
        dev_score_history = []
        current_lr = opt['lr']

        global_step = 0
        format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
        max_steps = len(iterator_train) * opt['num_epoch']

        # start training
        epoch = 0
        patience = 0
        while True:
            epoch = epoch + 1
            train_loss = 0

            for batch in iterator_train:
                start_time = time.time()
                global_step += 1

                inputs, target = batch_to_input(batch, opt['vocab_pad_id'])
                loss = self.update(inputs, target)
                train_loss += loss
                if global_step % opt['log_step'] == 0:
                    duration = time.time() - start_time
                    print(
                        format_str.format(datetime.now(), global_step, max_steps, epoch,
                                          opt['num_epoch'], loss, duration, current_lr))

            # eval on dev
            print("Evaluating on dev set...")
            if self.model_type == 'predictor':
                dev_p, dev_r, dev_score, dev_loss = evaluate(self, dataset_dev)
            else:
                dev_score = evaluate(self, dataset_dev, evaluate_type='auc')[0]
                dev_loss = dev_score

            # print training information
            train_loss = train_loss / len(iterator_train) * opt['batch_size']  # avg loss per batch
            dev_loss = dev_loss / len(iterator_dev) * opt['batch_size']
            print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_score = {:.4f}".format(
                epoch, train_loss, dev_loss, dev_score))

            # save the current model
            model_file = opt['model_save_dir'] + '/checkpoint_epoch_{}.pt'.format(epoch)
            self.save(model_file, epoch)
            if epoch == 1 or dev_score > max(dev_score_history):  # new best
                copyfile(model_file, opt['model_save_dir'] + '/best_model.pt')
                print("new best model saved.")
                patience = 0
            else:
                patience = patience + 1
            if epoch % opt['save_epoch'] != 0:
                os.remove(model_file)

            # change learning rate
            if len(dev_score_history) > 10 and dev_score <= dev_score_history[-1] and \
                    opt['optim'] in ['sgd', 'adagrad']:
                current_lr *= opt['lr_decay']
                self.update_lr(current_lr)

            dev_score_history += [dev_score]
            print("")
            if opt['patience'] != 0:
                if patience == opt['patience'] and epoch > opt['num_epoch']:
                    break
            else:
                if epoch == opt['num_epoch']:
                    break
        print("Training ended with {} epochs.".format(epoch))

    def retrieve(self, dataset, k_samples, label_distribution=None):
        if self.model_type != 'predictor' and label_distribution is None:
            raise ValueError('Retrival from selector cannot be done without label_distribution')

        iterator_unlabeled = data.Iterator(
            dataset=dataset,
            batch_size=self.opt['batch_size'],
            device=-1,
            repeat=False,
            train=False,
            shuffle=False,
            sort=True,
            sort_key=lambda x: -len(x.token),
            sort_within_batch=False)

        preds = []

        for batch in iterator_unlabeled:
            inputs, _ = batch_to_input(batch, self.opt['vocab_pad_id'])
            # print(inputs)
            preds += self.predict(inputs)[1]

        meta_idxs = []
        confidence_idxs = []
        examples = iterator_unlabeled.data()
        num_instance = len(examples)

        if label_distribution:
            label_distribution = {
                k: math.ceil(v * k_samples)
                for k, v in label_distribution.items()
            }
        if self.model_type == 'predictor':
            # ranking
            ranking = list(zip(range(num_instance), preds))
            ranking = sorted(
                ranking, key=lambda x: calc_confidence(x[1], self.opt['alpha']), reverse=True)
            # selection
            for eid, pred in ranking:
                if len(meta_idxs) == k_samples:
                    break
                rid, _ = arg_max(pred)
                val = calc_confidence(pred, self.opt['alpha'])
                rel = self.opt['rel_itos'][rid]
                if label_distribution:
                    if not label_distribution[rel]:
                        continue
                    label_distribution[rel] -= 1
                meta_idxs.append((eid, rel, examples[eid].relation))
                confidence_idxs.append((eid, val))
            return meta_idxs, confidence_idxs
        else:
            for rid in range(self.opt['num_class']):
                # ranking
                ranking = list(
                    zip(range(num_instance), [preds[k][rid] for k in range(num_instance)]))
                ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
                rel = self.opt['rel_itos'][rid]
                # selection
                cnt = min(len(ranking), label_distribution.get(rel, 0))
                for k in range(cnt):
                    eid, val = ranking[k]
                    meta_idxs.append((eid, rel, examples[eid].relation))
                    confidence_idxs.append((eid, val**self.opt['beta']))
                meta_idxs.sort(key=lambda t: preds[t[0]][self.opt['rel_stoi'][t[1]]], reverse=True)
            return meta_idxs, confidence_idxs

        return meta_idxs

    # train the model with a batch
    def update(self, inputs, target):
        """ Run a step of forward and backward model update. """
        self.model.train()
        self.optimizer.zero_grad()

        sl_confidence = inputs['sl_confidence']

        if self.model_type == 'pointwise':
            target = idx_to_onehot(target, self.opt)
        if self.opt['cuda']:
            target = target.cuda()
            inputs = dict([(k, v.cuda()) for k, v in inputs.items()])
            pr_confidence = inputs['pr_confidence']

        logits, _ = self.model(inputs)

        if self.model_type == 'pointwise':
            confidence = sl_confidence.unsqueeze(1).expand(-1, logits.size(1))
            if self.opt['cuda']:
                confidence = confidence.cuda()
            loss = F.binary_cross_entropy_with_logits(
                logits, target, weight=confidence, size_average=True)
            loss *= self.opt['num_class']
        elif self.model_type == 'pairwise':
            # Form a matrix with row_i indicate which samples are its negative samples (0, 1)
            matrix = torch.stack(
                [target.ne(rid) for rid in range(self.opt['num_class'])])  # R * B matrix
            matrix = matrix.index_select(0, target)  # B * B matrix
            confidence = sl_confidence.unsqueeze(1).expand_as(matrix)
            if self.opt['cuda']:
                confidence = confidence.cuda()
            pos_logits = logits.gather(1, target.view(-1, 1))  # B * 1 logits
            # B * B logits out[i][j] = j-th sample's score on class y[i]
            neg_logits = logits.t().index_select(0, target)
            # calculate pairwise loss
            loss = F.binary_cross_entropy_with_logits(
                pos_logits - neg_logits, (matrix.float() * 1 / 2 + 1 / 2) * confidence,
                size_average=True)
            loss *= self.opt['num_class']
        else:
            loss = self.criterion(logits, target)
            loss = torch.mean(loss * pr_confidence)

        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val

    def predict(self, inputs, target=None):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        if self.opt['cuda']:
            inputs = dict([(k, v.cuda()) for k, v in inputs.items()])
            target = None if target is None else target.cuda()

        self.model.eval()
        logits, _ = self.model(inputs)
        loss = None if target is None else self.criterion(logits, target).mean().data.item()

        if self.model_type == 'predictor':
            probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
            predictions = np.argmax(probs, axis=1).tolist()
        elif self.model_type == 'pointwise':
            probs = F.sigmoid(logits).data.cpu().numpy().tolist()
            predictions = logits.data.cpu().numpy().tolist()
        elif self.model_type == 'pairwise':
            probs = F.sigmoid(logits).data.cpu().numpy().tolist()
            predictions = logits.data.cpu().numpy().tolist()

        return predictions, probs, loss

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    # save the model
    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),  # model parameters
            'encoder': self.model.encoder.state_dict(),
            'classifier': self.model.classifier.state_dict(),
            'config': self.opt,  # options
            'epoch': epoch,  # current epoch
            'model_type': self.model_type  # current epoch
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    # load the model
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.encoder.load_state_dict(checkpoint['encoder'])
        self.model.classifier.load_state_dict(checkpoint['classifier'])
        self.opt = checkpoint['config']
        self.model_type = checkpoint['model_type']
        if self.model_type == 'predictor':
            self.criterion = nn.CrossEntropyLoss()
        elif self.model_type == 'pointwise':
            self.criterion = nn.BCEWithLogitsLoss()
