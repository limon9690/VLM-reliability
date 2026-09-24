import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip.transformer import text_global_pool
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

CTX_DIM = 512  # ViT-B/16 text width


class PromptLearner(nn.Module):
    def __init__(self, clip_model, device, n_ctx, tokenizer, ctx_dim, class_names):
        super().__init__()

        placeholder = "X " * n_ctx
        prompts = [f"{placeholder}{name}." for name in class_names]
        tokenized_prompts = tokenizer(prompts).to(device)
        self.num_classes = len(class_names)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)

        prefix = embedding[:, :1, :]
        suffix = embedding[:, 1 + n_ctx :, :]
        self.register_buffer("prefix", prefix)
        self.register_buffer("suffix", suffix)
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        # ctx initialized from "a photo of a" (shared by CoOp and TPT; CoOp paper default is random init)
        ctx_init = "a photo of a"
        init_tokens = tokenizer(ctx_init).to(device)
        with torch.no_grad():
            init_embedding = clip_model.token_embedding(init_tokens)

        ctx_vectors = init_embedding[0, 1 : 1 + n_ctx, :]
        self.ctx = nn.Parameter(ctx_vectors.clone())

        self.register_buffer("ctx_init_state", self.ctx.detach().clone())

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.num_classes, -1, -1)
        prompts = torch.cat([self.prefix, ctx, self.suffix], dim=1)
        return prompts, self.tokenized_prompts

    def reset_context(self):
        self.ctx.data.copy_(self.ctx_init_state)


class TextEncoderWrapper(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.attn_mask = clip_model.attn_mask
        self.text_pool_type = clip_model.text_pool_type
        self.text_eos_id = getattr(clip_model, "text_eos_id", None)

    def forward(self, prompt_embeddings, tokenized_prompts):
        cast_dtype = self.transformer.get_cast_dtype()
        x = prompt_embeddings.to(cast_dtype) + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
        x = text_global_pool(
            x, tokenized_prompts, self.text_pool_type, eos_token_id=self.text_eos_id
        )
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection
        return x


def train_coop(
    model,
    tokenizer,
    class_names,
    train_features,
    train_labels,
    seed,
    device,
    n_ctx=4,
    lr=0.002,
    epochs=10,
    batch_size=32,
    verbose=False,
    progress=True,
):
    """Trains a CoOp context on cached image features. Returns the context on CPU."""
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)

    prompt_learner = PromptLearner(
        model, device, n_ctx, tokenizer, CTX_DIM, class_names
    ).to(device)
    text_encoder = TextEncoderWrapper(model)
    optimizer = torch.optim.Adam([prompt_learner.ctx], lr=lr)
    logit_scale = model.logit_scale.exp()

    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    t0 = time.time()
    bar = tqdm(
        range(epochs), desc=f"CoOp seed {seed}", disable=not progress, leave=False
    )
    for epoch in bar:
        total_loss = 0.0
        for img_feat, labels in loader:
            img_feat, labels = img_feat.to(device), labels.to(device)
            prompts, tok = prompt_learner()
            text_features = text_encoder(prompts, tok)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            loss = F.cross_entropy(logit_scale * img_feat @ text_features.t(), labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = total_loss / len(loader)
        bar.set_postfix(loss=f"{epoch_loss:.4f}")
        if verbose:
            print(f"epoch {epoch + 1}: loss {epoch_loss:.4f}")

    if progress:
        print(
            f"CoOp seed {seed}: {epochs} epochs, final loss {epoch_loss:.4f}, {time.time() - t0:.0f}s"
        )
    return prompt_learner.ctx.detach().cpu()


@torch.no_grad()
def coop_text_features(model, tokenizer, class_names, ctx, device, n_ctx=4):
    """Turns a trained context into normalized class text features."""
    pl = PromptLearner(model, device, n_ctx, tokenizer, CTX_DIM, class_names).to(device)
    pl.ctx.data.copy_(ctx.to(device))
    prompts, tok = pl()
    t = TextEncoderWrapper(model)(prompts, tok)
    return t / t.norm(dim=-1, keepdim=True)


def save_coop_ctx(path, ctx, class_names, **meta):
    """Saves the context together with the class order it was trained on."""
    torch.save({"ctx": ctx, "class_names": list(class_names), **meta}, path)
