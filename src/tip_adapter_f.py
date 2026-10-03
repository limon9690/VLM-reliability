"""Tip-Adapter-F: Tip-Adapter with the cache keys trained by cross-entropy.

Training settings follow the official repo (gaopengcuhk/Tip-Adapter, configs/imagenet.yaml
and run_tip_adapter_F): keys initialised from the cache features, AdamW (lr 1e-3, eps 1e-4),
cosine schedule stepped per batch, 20 epochs, batch 256.

Stated deviations:
- trained on cached features, no augmentation;
- leave-one-out: a training shot never retrieves its own cache entry (without augmentation it
  would match itself at similarity 1 and get its label for free);
- fixed alpha and beta, no search;
- the final epoch is kept (the official code keeps the epoch with the best test accuracy).
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _loo_logits(
    adapter, feats, idx, cache_values, text_features, logit_scale, alpha, beta
):
    affinity = adapter(feats)  # (B, N_cache)
    A = torch.exp(-beta * (1 - affinity))
    own = F.one_hot(idx, num_classes=A.shape[-1]).bool()
    A = A.masked_fill(own, 0.0)  # leave-one-out
    return logit_scale * (feats @ text_features) + alpha * (A @ cache_values)


def train_tip_adapter_f(
    cache_keys,
    cache_values,
    text_features,
    logit_scale,
    alpha,
    beta,
    seed,
    device,
    epochs=20,
    lr=1e-3,
    eps=1e-4,
    batch_size=256,
    **cfg,
):
    """Returns (trained keys on device, stats). cache_keys: (N, D); cache_values: (N, C) one-hot."""
    torch.manual_seed(seed)
    t0 = time.time()
    feats = cache_keys.detach().float().to(device)
    values = cache_values.float().to(device)
    labels = values.argmax(dim=-1)
    idx = torch.arange(len(feats), device=device)

    adapter = nn.Linear(feats.shape[1], feats.shape[0], bias=False).to(device)
    adapter.weight = nn.Parameter(
        feats.clone()
    )  # weight (N, D): adapter(x) = x @ keys.T

    def full_loss():
        with torch.no_grad():
            logits = _loo_logits(
                adapter, feats, idx, values, text_features, logit_scale, alpha, beta
            )
            return F.cross_entropy(logits, labels).item()

    loss_start = full_loss()
    loader = DataLoader(
        TensorDataset(feats, labels, idx),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, eps=eps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs * len(loader)
    )
    with torch.enable_grad():
        for _ in range(epochs):
            for x, y, i in loader:
                logits = _loo_logits(
                    adapter, x, i, values, text_features, logit_scale, alpha, beta
                )
                loss = F.cross_entropy(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

    keys = adapter.weight.detach()
    stats = {
        "loss_start": loss_start,
        "loss_end": full_loss(),
        "key_shift": ((keys - feats).norm() / feats.norm()).item(),  # relative change
        "seconds": time.time() - t0,
    }
    return keys, stats
