import torch
import torchvision
import torch.nn as nn
from src.attention import MultiHeadAttention

# Define the PatchEmbedding class
class PatchEmbedding(nn.Module):
    def __init__(self, num_channel, patch_size, embedding_dim):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels = num_channel, out_channels = embedding_dim, kernel_size = patch_size, stride = patch_size)

    def forward(self, X):
        out = self.patch_embed(X)
        out = out.flatten(2)
        out = out.transpose(-2, -1)

        return out


# Define the TransformerEncoder class by using the handcoded MultiHeadAttention module
class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, mlp_hidden_nodes, attn_heads):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.multihead_attention = MultiHeadAttention(embedding_dim, embedding_dim, num_heads=attn_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_nodes),
            nn.GELU(),
            nn.Linear(mlp_hidden_nodes, embedding_dim)            
        )

    def forward(self, X):
        residual1 = X
        out = self.layer_norm1(X)
        out = self.multihead_attention(out)
        out = out + residual1

        residual2 = out
        out = self.layer_norm2(out)
        out = self.mlp(out)
        out = out + residual2

        return out
    

# Define the MLPHead class
class MLPHead(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mlp_layer = nn.Linear(embedding_dim, num_classes)

    def forward(self, X):
        out = self.layer_norm(X)
        out = self.mlp_layer(out)

        return out
    

# Define the VisionTransformer class
class VisionTransformer(nn.Module):
    def __init__(self, num_channel, patch_size, embedding_dim, mlp_hidden_nodes, attn_heads, transformer_blocks, num_patches, num_classes):
        super().__init__()
        self.patch_embedding = PatchEmbedding(num_channel, patch_size, embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embedding_dim))
        self.transformer_block = nn.Sequential(*[TransformerEncoder(embedding_dim, mlp_hidden_nodes, attn_heads) for _ in range(transformer_blocks)])
        self.mlp_head = MLPHead(embedding_dim, num_classes)

    def forward(self, X):
        out = self.patch_embedding(X)
        B = out.size()[0]
        class_tokens = self.cls_token.expand(B, -1, -1)
        out = torch.cat((class_tokens, out), dim = 1)
        out = out + self.position_embedding
        out = self.transformer_block(out)
        out = out[:, 0]
        out = self.mlp_head(out)

        return out
