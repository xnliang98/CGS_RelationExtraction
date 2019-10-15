"""
A rnn model for relation extraction, written in pytorch.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from utils import constant, torch_utils
from model.lstm_re import LSTMRelation


class RelationModel(object):

    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.model = LSTMRelation(opt, emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
    
    def update(self, batch):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = [b.cuda() for b in batch[:7]]
            labels = batch[7].cuda()
        else:
            inputs = [b for b in batch[:7]]
            labels = batch[7]

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, _ = self.model(inputs)
        loss = self.criterion(logits, labels)
        
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val

    def predict(self, batch, unsort=True):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        if self.opt['cuda']:
            inputs = [b.cuda() for b in batch[:7]]
            labels = batch[7].cuda()
        else:
            inputs = [b for b in batch[:7]]
            labels = batch[7]

        orig_idx = batch[8]

        # forward
        self.model.eval()
        logits, _ = self.model(inputs)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        return predictions, probs, loss.data.item()

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                'epoch': epoch
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']