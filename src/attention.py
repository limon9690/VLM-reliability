import torch
import torch.nn as nn

class SelfAttention(nn.Module):

    def __init__(self, dim_in, dim_out, causal = False):
        super().__init__()
        self.causal = causal
        self.w_q = nn.Linear(dim_in, dim_out, bias=False)
        self.w_k = nn.Linear(dim_in, dim_out, bias=False)
        self.w_v = nn.Linear(dim_in, dim_out, bias=False)


    def forward(self, X):
        Q = self.w_q(X)
        K = self.w_k(X)
        V = self.w_v(X)

        scores = Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5)

        if self.causal:
            seq_len = scores.shape[-1]
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        context = weights @ V
        
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, causal=False):
        super().__init__()
        assert dim_out % num_heads == 0, "dim_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.w_q = nn.Linear(dim_in, dim_out, bias=False)
        self.w_k = nn.Linear(dim_in, dim_out, bias=False)
        self.w_v = nn.Linear(dim_in, dim_out, bias=False)
        self.w_o = nn.Linear(dim_out, dim_out)
        self.causal = causal


    def forward(self, X):
        B, T, _ = X.shape
        # 1. project to Q, K, V   (B, T, dim_out)
        Q = self.w_q(X)
        K = self.w_k(X)
        V = self.w_v(X)

        # 2. reshape each -> (B, T, num_heads, head_dim) -> transpose to (B, num_heads, T, head_dim)
        Q = Q.reshape(B, T, self.num_heads, self.head_dim)
        K = K.reshape(B, T, self.num_heads, self.head_dim)
        V = V.reshape(B, T, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 3. scores, mask (if causal), softmax, @ V   — your existing attention, now 4D
        scores = Q @ K.transpose(-2, -1) / self.head_dim**0.5 

        if self.causal:
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))

        weights = torch.softmax(scores, dim = -1)
        context = weights @ V

        # 4. transpose back, reshape to (B, T, dim_out), apply w_o
        context = context.transpose(1, 2)
        context = context.reshape(B, T, self.num_heads * self.head_dim)
        out = self.w_o(context)

        return out