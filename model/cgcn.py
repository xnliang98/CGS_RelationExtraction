"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.layers import GCN, pool
from model.tree import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils

class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    def forward(self, inputs):
        outputs, pooling_output = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, pooling_output

class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super(GCNRelationModel, self).__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None

        embeddings = (self.emb, self.pos_emb, self.ner_emb)
        self.init_embeddings()

        self.gcn = GCN(opt, embeddings, opt['hidden_dim'], opt['num_layers'])

        # output mlp layers
        in_dim = opt['hidden_dim'] * 3

        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers'] - 1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)
    
    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].unifrom_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")
    
    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        seq_length = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(seq_length)

        def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos):
            head, words, subj_pos, obj_pos = head.cpu().numpy(), words.cpu().numpy(), subj_pos.cpu().numpy(), obj_pos.cpu().numpy()
            trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)
        
        adj = inputs_to_tree_reps(head.data, words.data, seq_length, self.opt['prune_k'], subj_pos.data, obj_pos.data)
        h, pool_mask = self.gcn(adj, inputs)

        # pooling
        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2) # invert mask
        pool_type = self.opt['pooling']
        h_out = pool(h, pool_mask, type=pool_type)
        subj_out = pool(h, subj_mask, type=pool_type)
        obj_out = pool(h, obj_mask, type=pool_type)
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)
        return outputs, h_out
