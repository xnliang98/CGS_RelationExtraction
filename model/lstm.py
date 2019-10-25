"""
Wapper of LSTM layer.
"""
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from utils import constant, torch_utils

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, batch_first=True, \
                dropout=0, bidirectional=True, use_cuda=True):
        super(MyLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.num_layers = num_layers
        self.direction = 2 if bidirectional else 1
        self.use_cuda = use_cuda
        self.batch_first = batch_first

        self.rnn = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=batch_first, \
                dropout=dropout, bidirectional=bidirectional)

    def forward(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = self.rnn_zero_state(batch_size)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=self.batch_first)
        rnn_outputs , (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=self.batch_first)
        return rnn_outputs, ht

    def rnn_zero_state(self, batch_size):
        total_layers = self.num_layers * self.direction
        state_shape = (total_layers, batch_size, self.hidden_dim)
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0