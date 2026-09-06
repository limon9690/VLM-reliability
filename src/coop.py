import torch
import torch.nn as nn
from open_clip.transformer import text_global_pool


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
    suffix = embedding[:, 1 + n_ctx:, :]
    self.register_buffer("prefix", prefix)
    self.register_buffer("suffix", suffix)
    self.register_buffer("tokenized_prompts", tokenized_prompts)

    # --- ctx initialized from "a photo of a" (for TPT) ---
    ctx_init = "a photo of a"
    init_tokens = tokenizer(ctx_init).to(device)
    with torch.no_grad():
      init_embedding = clip_model.token_embedding(init_tokens)

    ctx_vectors = init_embedding[0, 1:1 + n_ctx, :]
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
        x = text_global_pool(x, tokenized_prompts, self.text_pool_type, eos_token_id=self.text_eos_id)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection
        return x