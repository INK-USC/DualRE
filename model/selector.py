import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

from .layers import Discriminator, Classifier
from .encoder import RNNEncoder, CNNEncoder

class Selector(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(Selector, self).__init__()
        self.encoder = RNNEncoder(opt, emb_matrix)
        self.classifier = Classifier(opt)
    
    def forward(self, inputs):
        encoding = self.encoder(inputs)
        logits = self.classifier(encoding)
        return logits, encoding

    def predict(self, inputs):
        encoding = self.encoder(inputs)
        logits = self.classifier(encoding)
        return logits
    

