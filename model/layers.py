"""
My Layers.
"""
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from model.lstm import MyLSTM
from utils import constant, torch_utils


class PositionAwareAttention(nn.Module):
    """
    A position-augmented attetnion layer where the attention weight is 
    a = T' . Tanh(Ux + Vq + Wf)
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
        self.init_weight()
    
    def init_weight(self):
        init.normal_(self.ulinear.weight, std=0.001)
        init.normal_(self.vlinear.weight, std=0.001)
        if self.wlinear is not None:
            init.normal_(self.wlinear.weight, std=0.001)
        init.zeros_(self.tlinear.weight)
    
    def forward(self, x, x_mask, q, f):
        """
        B batch size
        T seq length
        I input size
        Q query size
        F feature size
        A attn size

        x: B, T, I
        q: B, Q
        f: B, T, F
        """
        batch_size, seq_len, _ = x.size()

        # x_proj = self.ulinear(x.reshape(-1, self.input_size)).reshape(batch_size, seq_len, self.attn_size)
        x_proj = self.ulinear(x) # ==> B, T, A
        q_proj = self.vlinear(q) # ==> B, A
        q_proj = q_proj.unsqueeze(1).expand(batch_size, seq_len, self.attn_size) # B, A ==> B, 1, A ==> B, T, A
        if self.wlinear is not None:
            f_proj = self.wlinear(f) # ==> B, T, A
            projs = [x_proj, q_proj, f_proj] 
        else:
            projs = [x_proj, q_proj]

        scores = self.tlinear(torch.tanh(sum(projs))).reshape(batch_size, seq_len)

        # mask padding
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        weights = F.softmax(scores, dim=1) # ==> B, T
        # weighted average input vectors
        out = weights.unsqueeze(1).bmm(x).squeeze(1) # B, 1, T x B, T, I ==> B, 1, I ==> B, I
        return out


class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()

        self.opt = opt
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']

        self.emb, self.pos_emb, self.ner_emb = embeddings

        # rnn layer if contextualized
        if self.opt.get('rnn', False):
            input_size = self.in_dim
            self.rnn = MyLSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                dropout=opt['rnn_dropout'], bidirectional=True, use_cuda=opt['cuda'])
            self.in_dim = opt['rnn_hidden'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout'])
        
        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))
    
    def conv_l2(self):
        # l2 regularization
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def forward(self, adj, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        word_embs = self.emb(words)
        embs = [word_embs]
        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        if self.opt.get('rnn', False):
            gcn_inputs = self.rnn_drop(self.rnn(embs, masks, words.size()[0])[0])
        else:
            gcn_inputs = embs
        
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        # zero out adj for ablation
        if self.opt.get('no_adj', False):
            adj = torch.zeros_like(adj)

        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs)
            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
        
        return gcn_inputs, mask

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)