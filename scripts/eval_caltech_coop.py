"""Reproduces the "CoOp on Caltech101 (16-shot)" table in the README:
zero-shot CLIP (single template "a photo of a {}."), CoOp trained on all 101
classes, and CoOp's base-to-new generalization split. Top-1 accuracy only --
this table has no ECE column.

Usage:
    python scripts/eval_caltech_coop.py [--data-root data]

First run downloads Caltech101 via torchvision (needs `gdown`, same as
torchvision's own Caltech101(download=True)), extracts image/text features
(needs a GPU), trains two CoOp context vectors, and caches everything under
features/. Later runs load from cache and skip straight to training/eval.

Two things this script does NOT literally lift from notebooks/05_coop.ipynb,
because that notebook doesn't contain them:

- "zero-shot (1 template)" and "CoOp, all 101 classes" have no corresponding
  cells there -- the notebook only runs the base/new half of this table
  (train on the 51 base classes, zero-shot-transfer eval on the 50 new
  classes, held-out eval on base classes). This script extends the same
  training loop (same architecture/optimizer/lr/epochs) to all 101 classes
  for those two rows, evaluating zero-shot and CoOp on the same held-out
  split so the comparison is apples-to-apples.
- 05_coop.ipynb never seeds Python's `random` module before drawing the
  16-shot split, so the exact images sampled for the logged numbers can't be
  recovered. This script seeds everything (42) for its own determinism, but
  expect this run to land close to -- not bit-for-bit at -- the logged numbers.
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from clip_zeroshot import (
    build_and_cache_image_features,
    build_and_cache_text_features,
    load_cached_image_features,
    load_cached_text_features,
)
from coop import PromptLearner, TextEncoderWrapper
from harness import accuracy, zero_shot_logits

import open_clip

FEATURES_DIR = REPO_ROOT / "features"
K_SHOT = 16
N_CTX = 4
CTX_DIM = 512
EPOCHS = 10
LR = 0.002
BASE_SPLIT = 51  # first 51 of 101 classes are "base", the rest are "new"
ZS_TEMPLATE = ["a photo of a {}."]


def load_or_build_image_features(cache_name, loader, model, device):
    cache_path = FEATURES_DIR / f"{cache_name}.pt"
    if cache_path.exists():
        print(f"loading cached image features from {cache_path}")
        return load_cached_image_features(str(cache_path))
    print(f"{cache_path} not found, extracting image features (this needs a GPU)")
    return build_and_cache_image_features(model, device, loader, str(FEATURES_DIR), cache_name)


def load_or_build_text_features(cache_name, class_names, templates, model, tokenizer, device):
    cache_path = FEATURES_DIR / f"{cache_name}.pt"
    if cache_path.exists():
        print(f"loading cached text features from {cache_path}")
        return load_cached_text_features(str(cache_path))["text_features"]
    print(f"{cache_path} not found, building text features")
    return build_and_cache_text_features(model, tokenizer, class_names, templates, device, str(FEATURES_DIR), cache_name)


def train_coop(class_names, features, labels, model, tokenizer, text_encoder, device):
    prompt_learner = PromptLearner(
        clip_model=model, device=device, n_ctx=N_CTX, tokenizer=tokenizer, ctx_dim=CTX_DIM, class_names=class_names
    ).to(device)
    optimizer = torch.optim.Adam(prompt_learner.parameters(), lr=LR)
    logit_scale = model.logit_scale.exp()
    loader = DataLoader(TensorDataset(features, labels), batch_size=32, shuffle=True)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for img_feat, batch_labels in loader:
            img_feat = img_feat.to(device)
            batch_labels = batch_labels.to(device)

            prompts, tok_prompts = prompt_learner()
            text_features = text_encoder(prompts, tok_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logits = logit_scale * img_feat @ text_features.t()
            loss = F.cross_entropy(logits, batch_labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"  epoch {epoch + 1}: loss {total_loss / len(loader):.4f}")

    return prompt_learner


def eval_prompt_learner(prompt_learner, text_encoder, features, labels, device, label_offset=0):
    prompt_learner.eval()
    with torch.no_grad():
        prompts, tok = prompt_learner()
        text_features = text_encoder(prompts, tok)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = features.to(device) @ text_features.t()
    return accuracy(logits, (labels - label_offset).to(device))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root", type=Path, default=REPO_ROOT / "data",
        help="Where the Caltech101 torchvision download lives",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    args.data_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    model.eval()
    model.to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    for p in model.parameters():
        p.requires_grad_(False)
    text_encoder = TextEncoderWrapper(model)

    print("loading Caltech101")
    caltech_dataset = datasets.Caltech101(root=str(args.data_root), download=True, transform=preprocess)
    caltech_classes = [c.replace("_", " ") for c in caltech_dataset.categories]
    base_classes = caltech_classes[:BASE_SPLIT]
    new_classes = caltech_classes[BASE_SPLIT:]

    # caltech_dataset.y holds every image's label directly; avoids decoding
    # all ~9k images just to bucket indices by class.
    classes_to_indices = defaultdict(list)
    for idx, label in enumerate(caltech_dataset.y):
        classes_to_indices[label].append(idx)
    base_classes_to_indices = {c: idxs for c, idxs in classes_to_indices.items() if c < BASE_SPLIT}
    new_classes_to_indices = {c: idxs for c, idxs in classes_to_indices.items() if c >= BASE_SPLIT}

    all_few_shot_indices = []
    for indices in classes_to_indices.values():
        all_few_shot_indices.extend(random.sample(indices, K_SHOT))

    base_few_shot_indices = []
    for indices in base_classes_to_indices.values():
        base_few_shot_indices.extend(random.sample(indices, K_SHOT))

    all_indices = list(range(len(caltech_dataset)))
    all101_eval_indices = list(set(all_indices) - set(all_few_shot_indices))

    new_eval_indices = []
    for indices in new_classes_to_indices.values():
        new_eval_indices.extend(indices)

    base_all_indices = []
    for indices in base_classes_to_indices.values():
        base_all_indices.extend(indices)
    base_eval_indices = list(set(base_all_indices) - set(base_few_shot_indices))

    all101_train_cache = load_or_build_image_features(
        "caltech_img_train_all",
        DataLoader(Subset(caltech_dataset, all_few_shot_indices), batch_size=32, shuffle=True),
        model, device,
    )
    all101_eval_cache = load_or_build_image_features(
        "caltech_img_eval_all",
        DataLoader(Subset(caltech_dataset, all101_eval_indices), batch_size=32, shuffle=False),
        model, device,
    )
    base_train_cache = load_or_build_image_features(
        "caltech_img_train_base",
        DataLoader(Subset(caltech_dataset, base_few_shot_indices), batch_size=32, shuffle=True),
        model, device,
    )
    new_eval_cache = load_or_build_image_features(
        "caltech_eval_new",
        DataLoader(Subset(caltech_dataset, new_eval_indices), batch_size=32, shuffle=False),
        model, device,
    )
    base_eval_cache = load_or_build_image_features(
        "caltech_img_eval_base",
        DataLoader(Subset(caltech_dataset, base_eval_indices), batch_size=32, shuffle=True),
        model, device,
    )

    zs_text_features = load_or_build_text_features(
        "caltech_zs_text_features", caltech_classes, ZS_TEMPLATE, model, tokenizer, device
    ).to(device)

    print("evaluating zero-shot (1 template)")
    zs_logits = zero_shot_logits(
        all101_eval_cache["image_features"].to(device), zs_text_features, model.logit_scale.exp()
    )
    zero_shot_acc = accuracy(zs_logits, all101_eval_cache["labels"].to(device))

    print("training CoOp on all 101 classes")
    all101_prompt_learner = train_coop(
        caltech_classes, all101_train_cache["image_features"], all101_train_cache["labels"], model, tokenizer, text_encoder, device
    )
    coop_all101_acc = eval_prompt_learner(
        all101_prompt_learner, text_encoder, all101_eval_cache["image_features"], all101_eval_cache["labels"], device
    )

    print("training CoOp on the 51 base classes")
    base_prompt_learner = train_coop(
        base_classes, base_train_cache["image_features"], base_train_cache["labels"], model, tokenizer, text_encoder, device
    )
    coop_base_acc = eval_prompt_learner(
        base_prompt_learner, text_encoder, base_eval_cache["image_features"], base_eval_cache["labels"], device
    )

    print("evaluating base-trained context on the 50 new (unseen) classes")
    new_prompt_learner = PromptLearner(
        clip_model=model, device=device, n_ctx=N_CTX, tokenizer=tokenizer, ctx_dim=CTX_DIM, class_names=new_classes
    ).to(device)
    new_prompt_learner.ctx.data = base_prompt_learner.ctx.data.clone()
    coop_new_acc = eval_prompt_learner(
        new_prompt_learner, text_encoder, new_eval_cache["image_features"], new_eval_cache["labels"], device, label_offset=BASE_SPLIT
    )

    print()
    print(f"{'zero-shot (1 template)':28s}: {zero_shot_acc:.2f}%")
    print(f"{'CoOp, all 101 classes':28s}: {coop_all101_acc:.2f}%")
    print(f"{'CoOp, base classes':28s}: {coop_base_acc:.2f}%")
    print(f"{'CoOp, new (unseen) classes':28s}: {coop_new_acc:.2f}%")


if __name__ == "__main__":
    main()
