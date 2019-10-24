"""
My Layers.
"""
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

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
