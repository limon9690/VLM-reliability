"""CLAP (Silva-Rodriguez et al., CVPR 2024): a linear probe started at the zero-shot
classifier, with a class-wise L2 penalty towards it. Settings follow the official repo
(jusiro/CLAP, configs/trainers/SGD_lr1e-1_B256_ep300.yaml, constraint "l2").

Stated deviations:
- trained on cached features, no augmentation;
- prototypes start at the cached 80-template classifier, scaled per class to the norm of
  the mean raw template embedding (the official code starts from that raw mean).
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def raw_text_norms(model, tokenizer, class_names, templates, device):
    norms, directions = [], []
    for name in class_names:
        tokens = tokenizer([t.format(name) for t in templates]).to(device)
        raw = model.encode_text(tokens).float()
        norms.append(raw.mean(dim=0).norm())
        directions.append(F.normalize(F.normalize(raw, dim=-1).mean(dim=0), dim=-1))
    return torch.stack(norms), torch.stack(directions, dim=1)


def class_weights(shot_logits, shot_labels, n_classes):
    probs = shot_logits.softmax(dim=-1)
    true_prob = probs.gather(-1, shot_labels[:, None]).squeeze(-1)
    counts = torch.bincount(shot_labels, minlength=n_classes)
    assert (counts > 0).all(), "every class needs at least one shot"
    total = torch.zeros(n_classes, device=probs.device).index_add_(0, shot_labels, true_prob)
    return total / counts


def train_clap(
    shot_features,
    shot_labels,
    shot_logits,
    text_features,
    norms,
    logit_scale,
    seed,
    device,
    epochs=300,
    lr=0.1,
    momentum=0.9,
    batch_size=256,
    warmup_lr=1e-5,
    **cfg,
):
    torch.manual_seed(seed)
    t0 = time.time()
    feats = shot_features.detach().float().to(device)
    labels = shot_labels.to(device)
    n_classes = text_features.shape[-1]
    target = (text_features.float().transpose(-2, -1) * norms[:, None]).to(device)
    lam = class_weights(shot_logits.float().to(device), labels, n_classes)

    head = nn.Linear(feats.shape[1], n_classes, bias=False).to(device)
    head.weight = nn.Parameter(target.clone())

    def losses(x, y):
        logits = logit_scale * (x @ F.normalize(head.weight, dim=-1).transpose(-2, -1))
        penalty = (lam * (head.weight - target).pow(2).sum(dim=-1)).mean()
        return F.cross_entropy(logits, y), penalty

    def full_loss():
        with torch.no_grad():
            ce, penalty = losses(feats, labels)
            return ce.item(), penalty.item()

    ce_start, _ = full_loss()
    optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=momentum, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    n = len(feats)
    n_batches = n // batch_size if n >= batch_size else 1

    with torch.enable_grad():
        for epoch in range(epochs):
            epoch_lr = warmup_lr if epoch == 0 else 0.5 * lr * (1 + math.cos(math.pi * epoch / epochs))
            for group in optimizer.param_groups:
                group["lr"] = epoch_lr
            order = torch.randperm(n, generator=generator).to(device)
            for b in range(n_batches):
                idx = order[b * batch_size:(b + 1) * batch_size]
                ce, penalty = losses(feats[idx], labels[idx])
                loss = ce + penalty
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    ce_end, penalty_end = full_loss()
    weight = head.weight.detach()
    stats = {
        "ce_start": ce_start,
        "ce_end": ce_end,
        "penalty_end": penalty_end,
        "drift": ((weight - target).norm(dim=-1) / target.norm(dim=-1)).mean().item(),
        "lambda_mean": lam.mean().item(),
        "seconds": time.time() - t0,
    }
    return F.normalize(weight, dim=-1).transpose(-2, -1), stats
