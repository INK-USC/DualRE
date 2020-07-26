"""
A rnn model for relation extraction, written in pytorch.
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class PositionAwareAttention(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + Wf)
    where x is the input, q is the query, and f is additional position features.
    """

    def __init__(self, input_size, query_size, feature_size, attn_size):
        super(PositionAwareAttention, self).__init__()
        self.input_size = input_size
        self.query_size = query_size
        self.feature_size = feature_size
        self.attn_size = attn_size
        self.ulinear = nn.Linear(input_size, attn_size)
        self.vlinear = nn.Linear(query_size, attn_size, bias=False)
        if feature_size > 0:
            self.wlinear = nn.Linear(feature_size, attn_size, bias=False)
        else:
            self.wlinear = None
        self.tlinear = nn.Linear(attn_size, 1)
        self.init_weights()

    def init_weights(self):
        self.ulinear.weight.data.normal_(std=0.001)
        self.vlinear.weight.data.normal_(std=0.001)
        if self.wlinear is not None:
            self.wlinear.weight.data.normal_(std=0.001)
        self.tlinear.weight.data.zero_()  # use zero to give uniform attention at the beginning

    def forward(self, x, x_mask, q, f):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size

        x is the sequence of word embeddings
        q is the last hidden state
        f is the position embeddings
        """
        batch_size, seq_len, _ = x.size()

        x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
            batch_size, seq_len, self.attn_size
        )
        q_proj = (
            self.vlinear(q.view(-1, self.query_size))
            .contiguous()
            .view(batch_size, self.attn_size)
            .unsqueeze(1)
            .expand(batch_size, seq_len, self.attn_size)
        )
        if self.wlinear is not None:
            f_proj = (
                self.wlinear(f.view(-1, self.feature_size))
                .contiguous()
                .view(batch_size, seq_len, self.attn_size)
            )
            projs = [x_proj, q_proj, f_proj]
        else:
            projs = [x_proj, q_proj]
        scores = self.tlinear(F.tanh(sum(projs)).view(-1, self.attn_size)).view(
            batch_size, seq_len
        )

        # mask padding
        scores.data.masked_fill_(x_mask.data, -float("inf"))
        weights = F.softmax(scores, dim=-1)
        # weighted average input vectors
        outputs = weights.unsqueeze(1).bmm(x).squeeze(1)
        return outputs


class RNNEncoder(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(RNNEncoder, self).__init__()
        self.drop = nn.Dropout(opt["dropout"])
        self.emb = nn.Embedding(
            opt["vocab_size"], opt["emb_dim"], padding_idx=opt["vocab_pad_id"]
        )
        if opt["pos_dim"] > 0:
            self.pos_emb = nn.Embedding(
                opt["pos_size"], opt["pos_dim"], padding_idx=opt["pos_pad_id"]
            )
        if opt["ner_dim"] > 0:
            self.ner_emb = nn.Embedding(
                opt["ner_size"], opt["ner_dim"], padding_idx=opt["ner_pad_id"]
            )

        input_size = opt["emb_dim"] + opt["pos_dim"] + opt["ner_dim"]
        self.rnn = nn.LSTM(
            input_size,
            opt["hidden_dim"],
            opt["num_layers"],
            batch_first=True,
            dropout=opt["dropout"],
        )

        # attention layer
        if opt["attn"]:
            self.attn_layer = PositionAwareAttention(
                opt["hidden_dim"], opt["hidden_dim"], 2 * opt["pe_dim"], opt["attn_dim"]
            )
            self.pe_emb = nn.Embedding(
                opt["pe_size"], opt["pe_dim"], padding_idx=opt["pe_pad_id"]
            )

        self.opt = opt
        self.use_cuda = opt["cuda"]
        self.emb_matrix = emb_matrix
        if emb_matrix is not None:
            self.emb.weight.data.copy_(emb_matrix)

    def zero_state(self, batch_size):
        state_shape = (self.opt["num_layers"], batch_size, self.opt["hidden_dim"])
        h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0

    def forward(self, inputs):
        # words: [batch size, seq length]
        words, masks = inputs["words"], inputs["masks"]
        pos, ner = inputs["pos"], inputs["ner"]
        subj_pst, obj_pst = inputs["subj_pst"], inputs["obj_pst"]
        seq_lens = inputs["length"]

        batch_size = words.size()[0]

        # embedding lookup
        # word_inputs: [batch size, seq length, embedding size]
        # inputs: [batch size, seq length, embedding size * 3]
        word_inputs = self.emb(words)
        inputs = [word_inputs]
        if self.opt["pos_dim"] > 0:
            inputs += [self.pos_emb(pos)]
        if self.opt["ner_dim"] > 0:
            inputs += [self.ner_emb(ner)]
        inputs = self.drop(torch.cat(inputs, dim=2))  # add dropout to input
        input_size = inputs.size(2)

        # rnn
        h0, c0 = self.zero_state(batch_size)
        inputs = nn.utils.rnn.pack_padded_sequence(
            inputs, seq_lens.tolist(), batch_first=True
        )
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        outputs, output_lens = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True
        )
        hidden = self.drop(ht[-1, :, :])  # get the outmost layer h_n
        outputs = self.drop(outputs)

        # attention
        if self.opt["attn"]:
            # convert all negative PE numbers to positive indices
            # e.g., -2 -1 0 1 will be mapped to 98 99 100 101
            subj_pe_inputs = self.pe_emb(subj_pst)
            obj_pe_inputs = self.pe_emb(obj_pst)
            pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)
            final_hidden = self.attn_layer(outputs, masks, hidden, pe_features)
        else:
            final_hidden = hidden

        return final_hidden


class CNNEncoder(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(CNNEncoder, self).__init__()
        # initialize drop out rate
        self.drop = nn.Dropout(opt["dropout"])
        # initialize embedding layer
        self.emb = nn.Embedding(
            opt["vocab_size"], opt["emb_dim"], padding_idx=opt["vocab_pad_id"]
        )
        if opt["pos_dim"] > 0:
            self.pos_emb = nn.Embedding(
                opt["pos_size"], opt["pos_dim"], padding_idx=opt["pos_pad_id"]
            )
        if opt["ner_dim"] > 0:
            self.ner_emb = nn.Embedding(
                opt["ner_size"], opt["ner_dim"], padding_idx=opt["ner_pad_id"]
            )
        if opt["pe_dim"] > 0:
            self.pe_emb = nn.Embedding(
                opt["pe_size"], opt["pe_dim"], padding_idx=opt["pe_pad_id"]
            )

        # input layer
        input_size = (
            opt["emb_dim"] + opt["pos_dim"] + opt["ner_dim"] + 2 * opt["pe_dim"]
        )

        # encoding layer
        self.convs = nn.ModuleList(
            [
                torch.nn.Conv1d(input_size, opt["hidden_dim"], ksize, padding=2)
                for ksize in opt["kernels"]
            ]
        )

        # prediction layer
        self.linear = nn.Linear(
            opt["hidden_dim"] * len(opt["kernels"]), opt["hidden_dim"]
        )

        # save other parameters
        self.opt = opt
        self.use_cuda = opt["cuda"]
        if emb_matrix is not None:
            self.emb.weight.data.copy_(emb_matrix)

    def forward(self, inputs):
        # words: [batch size, seq length]
        words, masks = inputs["words"], inputs["masks"]
        pos, ner = inputs["pos"], inputs["ner"]
        subj_pst, obj_pst = inputs["subj_pst"], inputs["obj_pst"]

        word_inputs = self.emb(words)
        inputs = [word_inputs]
        if self.opt["pos_dim"] > 0:
            inputs += [self.pos_emb(pos)]
        if self.opt["ner_dim"] > 0:
            inputs += [self.ner_emb(ner)]
        if self.opt["pe_dim"] > 0:
            inputs += [self.pe_emb(subj_pst)]
            inputs += [self.pe_emb(obj_pst)]
        inputs = self.drop(torch.cat(inputs, dim=2))  # add dropout to input

        embedded = torch.transpose(inputs, 1, 2)
        hiddens = [F.relu(conv(embedded)) for conv in self.convs]  # b *
        hiddens = [
            torch.squeeze(F.max_pool1d(hidden, hidden.size(2)), dim=2)
            for hidden in hiddens
        ]
        hidden = self.drop(torch.cat(hiddens, dim=1))
        encoding = F.tanh(self.linear(hidden))
        return encoding

