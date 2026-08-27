import torch
import torch.nn as nn
import torch.nn.functional as F

# Image Encoder: A simple CNN to extract features from the input images
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.convolutions = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU()
        )

        self.projection = nn.Linear(256, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)


    def forward(self, X):
        out = self.convolutions(X)
        out = out.mean(dim=[2,3])          # avg pooling
        out = self.projection(out)
        out = F.normalize(self.layer_norm(out), dim=-1)

        return out


# Text Encoder
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, context_window):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_window, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.projection_layer = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)


    def forward(self, tokens):
        N, L = tokens.shape
        pos_emb_ids = torch.arange(L, device=tokens.device).unsqueeze(0).expand(N, L)
        pos_emb_vectors = self.position_embedding(pos_emb_ids)
        token_emb_vectors = self.token_embedding(tokens)
        final_embedding = token_emb_vectors + pos_emb_vectors
        context_vector = self.mha(final_embedding, final_embedding, final_embedding)[0]
        final_tokens = context_vector[:, 0]
        projection = self.projection_layer(final_tokens)
        output = F.normalize(self.layer_norm(projection), dim=-1)

        return output


# CLIP Loss
def clip_loss(img_embeddings, txt_embeddings, temperature):
    logits = img_embeddings @ txt_embeddings.T / temperature
    targets = torch.arange(img_embeddings.size(0), device=img_embeddings.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.T, targets)

    return (loss_i + loss_t) / 2.0