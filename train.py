'''Main file for training DualRE'''
# pylint: disable=invalid-name, missing-docstring
import math
import random
import argparse
import torch
from torch.autograd import Variable
from torchtext import data
import functools

from model.predictor import Predictor
from model.selector import Selector
from model.trainer import Trainer, evaluate
from utils import torch_utils, scorer, helper
from utils.torch_utils import batch_to_input, example_to_dict, arg_max
from selection import get_relation_distribution, select_samples

parser = argparse.ArgumentParser()

# Begin DualRE specific arguments
parser.add_argument(
    '--selector_model',
    type=str,
    default='pointwise',
    choices=['pointwise', 'pairwise', 'none'],
    help='Method for selector. \'none\' indicates using self-training model')
parser.add_argument(
    '--integrate_method',
    type=str,
    default='intersection',
    choices=['intersection', 'p_only', 's_only'],
    help='Method to combine results from prediction and retrieval module.')
parser.add_argument(
    '--selector_upperbound',
    type=float,
    default=3,
    help='# of samples / k taken before intersection.')
parser.add_argument(
    '--num_iters',
    type=int,
    default=-1,
    help='# of iterations. -1 indicates it\'s determined by data_ratio.')
parser.add_argument(
    '--alpha', type=float, default=0.5, help='confidence hyperparameter for predictor.')
parser.add_argument('--beta', type=float, default=2, help='confidence hyperparameter for selector')

# Begin original TACRED arguments
parser.add_argument('--p_dir', type=str, help='Directory of the predictor.')
parser.add_argument('--s_dir', type=str, help='Directory of the selector.')
parser.add_argument('--data_dir', type=str, default='dataset/dataname')
parser.add_argument('--labeled_ratio', type=float)
parser.add_argument('--unlabeled_ratio', type=float)
# ratio of instances to promote each round
parser.add_argument('--data_ratio', type=float, default=0.1)

parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
parser.add_argument('--p_dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
parser.add_argument(
    '--s_dropout', type=float, default=0.5, help='Input and RNN dropout rate for selector.')

parser.add_argument('--attn', dest='attn', action='store_true', help='Use attention layer.')
parser.add_argument('--no-attn', dest='attn', action='store_false')
parser.set_defaults(attn=True)
parser.add_argument('--attn_dim', type=int, default=200, help='Attention size.')
parser.add_argument('--pe_dim', type=int, default=30, help='Position encoding dimension.')

parser.add_argument('--lr', type=float, default=1.0, help='Applies to SGD and Adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--optim', type=str, default='sgd', help='sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=30)
parser.add_argument('--patience', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument(
    '--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument(
    '--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=1)
# torch.cuda.is_available())
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

args, _ = parser.parse_known_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# make opt
opt = vars(args)

# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
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
dataset_vocab = data.TabularDataset(
    path=opt['data_dir'] + '/train.json', format='json', fields=FIELDS)
dataset_train = data.TabularDataset(
    path=opt['data_dir'] + '/train-' + str(opt['labeled_ratio']) + '.json',
    format='json',
    fields=FIELDS)
dataset_infer = data.TabularDataset(
    path=opt['data_dir'] + '/raw-' + str(opt['unlabeled_ratio']) + '.json',
    format='json',
    fields=FIELDS)
dataset_dev = data.TabularDataset(path=opt['data_dir'] + '/dev.json', format='json', fields=FIELDS)
dataset_test = data.TabularDataset(
    path=opt['data_dir'] + '/test.json', format='json', fields=FIELDS)

print('=' * 100)
print('Labeled data path: ' + opt['data_dir'] + '/train-' + str(opt['labeled_ratio']) + '.json')
print('Unlabeled data path: ' + opt['data_dir'] + '/raw-' + str(opt['unlabeled_ratio']) + '.json')
print('Labeled instances #: %d, Unlabeled instances #: %d' % (len(dataset_train.examples),
                                                              len(dataset_infer.examples)))
print('=' * 100)

TOKEN.build_vocab(dataset_vocab)
RELATION.build_vocab(dataset_vocab)
POS.build_vocab(dataset_vocab)
NER.build_vocab(dataset_vocab)
PST.build_vocab(dataset_vocab)

opt['num_class'] = len(RELATION.vocab)
opt['vocab_pad_id'] = TOKEN.vocab.stoi['<pad>']
opt['pos_pad_id'] = POS.vocab.stoi['<pad>']
opt['ner_pad_id'] = NER.vocab.stoi['<pad>']
opt['pe_pad_id'] = PST.vocab.stoi['<pad>']
opt['vocab_size'] = len(TOKEN.vocab)
opt['pos_size'] = len(POS.vocab)
opt['ner_size'] = len(NER.vocab)
opt['pe_size'] = len(PST.vocab)
opt['rel_stoi'] = RELATION.vocab.stoi
opt['rel_itos'] = RELATION.vocab.itos

helper.ensure_dir(opt['p_dir'], verbose=True)
helper.ensure_dir(opt['s_dir'], verbose=True)

TOKEN.vocab.load_vectors('glove.840B.300d', cache='./dataset/.vectors_cache')
# TOKEN.vocab.load_vectors(
#     'glove.840B.300d',
#     cache='./dataset/.vectors_cache',
#     unk_init=functools.partial(torch.nn.init.uniform_, a=-1, b=1))  # randomly 
if TOKEN.vocab.vectors is not None:
    opt['emb_dim'] = TOKEN.vocab.vectors.size(1)


def load_best_model(model_dir, model_type='predictor'):
    model_file = model_dir + '/best_model.pt'
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    if model_type == 'predictor':
        predictor = Predictor(model_opt)
        model = Trainer(model_opt, predictor, model_type=model_type)
    else:
        selector = Selector(model_opt)
        model = Trainer(model_opt, selector, model_type=model_type)
    model.load(model_file)
    helper.print_config(model_opt)
    return model


num_iters = math.ceil(1.0 / opt['data_ratio'])
if args.num_iters >= 0:
    num_iters = min(num_iters, args.num_iters)
k_samples = math.ceil(len(dataset_infer.examples) * opt['data_ratio'])
train_label_distribution = get_relation_distribution(dataset_train)
dev_f1_iter, test_f1_iter = [], []

for num_iter in range(num_iters + 1):
    print('')
    print('=' * 100)
    print(
        'Training #: %d, Infer #: %d' % (len(dataset_train.examples), len(dataset_infer.examples)))

    # ====================== #
    # Begin Train on Predictor
    # ====================== #
    print('Training on iteration #%d for dualRE Predictor...' % num_iter)
    opt['model_save_dir'] = opt['p_dir']
    opt['dropout'] = opt['p_dropout']

    # save config
    helper.save_config(opt, opt['model_save_dir'] + '/config.json', verbose=True)
    helper.print_config(opt)

    # prediction module
    predictor = Predictor(opt, emb_matrix=TOKEN.vocab.vectors)
    model = Trainer(opt, predictor, model_type='predictor')
    model.train(dataset_train, dataset_dev)

    # Evaluate
    best_model_p = load_best_model(opt['model_save_dir'], model_type='predictor')
    print('Final evaluation #%d on train set...' % num_iter)
    evaluate(best_model_p, dataset_train, verbose=True)
    print('Final evaluation #%d on dev set...' % num_iter)
    dev_f1 = evaluate(best_model_p, dataset_dev, verbose=True)[2]
    print('Final evaluation #%d on test set...' % num_iter)
    test_f1 = evaluate(best_model_p, dataset_test, verbose=True)[2]
    dev_f1_iter.append(dev_f1)
    test_f1_iter.append(test_f1)
    best_model_p = load_best_model(opt['p_dir'], model_type='predictor')

    # ====================== #
    # Begin Train on Selector
    # ====================== #
    best_model_s = None
    if args.selector_model != 'none':
        print('Training on iteration #%d for dualRE Selector...' % num_iter)
        opt['model_save_dir'] = opt['s_dir']
        opt['dropout'] = opt['s_dropout']

        # save config
        helper.save_config(opt, opt['model_save_dir'] + '/config.json', verbose=True)
        helper.print_config(opt)

        # model
        selector = Selector(opt, emb_matrix=TOKEN.vocab.vectors)
        if args.selector_model == 'predictor':
            selector = Predictor(opt, emb_matrix=TOKEN.vocab.vectors)
        model = Trainer(opt, selector, model_type=args.selector_model)
        model.train(dataset_train, dataset_dev)

        # Sample from cur_model
        best_model_s = load_best_model(opt['s_dir'], model_type=args.selector_model)

    # ====================== #
    # Select New Instances
    # ====================== #
    new_examples, rest_examples = select_samples(best_model_p, best_model_s, dataset_infer,
                                                 k_samples, args, train_label_distribution)

    # update dataset
    dataset_train.examples = dataset_train.examples + new_examples
    dataset_infer.examples = rest_examples

scorer.print_table(
    dev_f1_iter, test_f1_iter, header='Best dev and test F1 with seed=%s:' % args.seed)
