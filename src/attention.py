import torch
import torch.nn as nn

class SelfAttention(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.w_q = nn.Linear(dim_in, dim_out, bias=False)
        self.w_k = nn.Linear(dim_in, dim_out, bias=False)
        self.w_v = nn.Linear(dim_in, dim_out, bias=False)


    def forward(self, X):
        Q = self.w_q(X)
        K = self.w_k(X)
        V = self.w_v(X)

        scores = Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        context = weights @ V
        
        return context