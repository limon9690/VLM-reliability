"""Reproduces the "Adaptation methods on ImageNet-R" table in the README:
zero-shot CLIP, Tip-Adapter, CoOp, and TPT, all reporting top-1 accuracy and ECE.

Usage:
    python scripts/eval_imagenet_r.py [--data-root data]

First run extracts image/text features (needs a GPU, several minutes) and
caches them under features/. Later runs load straight from that cache.

CoOp is evaluated only: it loads the context vector trained in
notebooks/05_coop.ipynb (features/coop_ctx_200_cls.pt) rather than retraining it.
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import open_clip
from datasets import load_dataset

from clip_zeroshot import (
    build_and_cache_image_features,
    build_and_cache_text_features,
    load_cached_image_features,
    load_cached_text_features,
)
from coop import PromptLearner, TextEncoderWrapper
from harness import (
    accuracy,
    coop_logits,
    ece,
    run_comparison,
    run_tpt,
    tip_adapter_logits,
    zero_shot_logits,
    signed_gap,
)
from imagenet_classes import IMAGENET_TEMPLATES

FEATURES_DIR = REPO_ROOT / "features"
K_SHOT = 16
TIP_ADAPTER_ALPHA = 1.5
TIP_ADAPTER_BETA = 5.0
DISPLAY_NAMES = {"zero_shot": "zero-shot", "tip_adapter": "Tip-Adapter", "coop": "CoOp", "tpt": "TPT"}


class HFImageDataset(Dataset):
    def __init__(self, hf_dataset, preprocess, wnid_to_index):
        self.hf_dataset = hf_dataset
        self.preprocess = preprocess
        self.wnid_to_index = wnid_to_index

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        image = self.preprocess(example["image"].convert("RGB"))
        label = self.wnid_to_index[example["wnid"]]
        return image, label


def build_r_classes(ds):
    wnids_col = ds["test"]["wnid"]
    class_names_col = ds["test"]["class_name"]

    unique_pairs = sorted(set(zip(wnids_col, class_names_col)))
    r_wnids = [wnid for wnid, _ in unique_pairs]
    r_class_names = [name.replace("_", " ") for _, name in unique_pairs]
    wnid_to_r_index = {wnid: i for i, wnid in enumerate(r_wnids)}
    return r_class_names, wnid_to_r_index, class_names_col


def build_few_shot_split(class_names_col, wnid_to_r_index):
    labels = np.array([c.replace("_", " ") for c in class_names_col])
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    classes_to_indices = defaultdict(list)
    for idx, label_idx in enumerate(inverse):
        classes_to_indices[unique_labels[label_idx]].append(idx)

    few_shot_indices = []
    for indices in classes_to_indices.values():
        few_shot_indices.extend(random.sample(indices, K_SHOT))

    all_indices = list(range(len(class_names_col)))
    eval_indices = list(set(all_indices) - set(few_shot_indices))
    return few_shot_indices, eval_indices


def load_or_build_image_features(cache_name, loader, model, device):
    cache_path = FEATURES_DIR / f"{cache_name}.pt"
    if cache_path.exists():
        print(f"loading cached image features from {cache_path}")
        return load_cached_image_features(str(cache_path))
    print(f"{cache_path} not found, extracting image features (this needs a GPU)")
    return build_and_cache_image_features(model, device, loader, str(FEATURES_DIR), cache_name)


def load_or_build_text_features(cache_name, class_names, model, tokenizer, device):
    cache_path = FEATURES_DIR / f"{cache_name}.pt"
    if cache_path.exists():
        print(f"loading cached text features from {cache_path}")
        return load_cached_text_features(str(cache_path))["text_features"]
    print(f"{cache_path} not found, building text features")
    return build_and_cache_text_features(
        model, tokenizer, class_names, IMAGENET_TEMPLATES, device, str(FEATURES_DIR), cache_name
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=REPO_ROOT / "data",
        help="Where the ImageNet-R HuggingFace dataset is downloaded/cached",
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

    print("loading axiong/imagenet-r")
    ds = load_dataset("axiong/imagenet-r", cache_dir=str(args.data_root))

    r_class_names, wnid_to_r_index, class_names_col = build_r_classes(ds)
    few_shot_indices, eval_indices = build_few_shot_split(class_names_col, wnid_to_r_index)

    full_wrapped = HFImageDataset(ds["test"], preprocess, wnid_to_r_index)
    raw_few_shot_loader = DataLoader(Subset(full_wrapped, few_shot_indices), batch_size=32, shuffle=True)
    raw_eval_loader = DataLoader(Subset(full_wrapped, eval_indices), batch_size=32)

    few_shot_cache = load_or_build_image_features("r_few_shot_image_features", raw_few_shot_loader, model, device)
    few_shot_image_features = few_shot_cache["image_features"]
    few_shot_image_labels = few_shot_cache["labels"]

    text_features = load_or_build_text_features("r_text-features", r_class_names, model, tokenizer, device)

    eval_cache = load_or_build_image_features("r_eval_features", raw_eval_loader, model, device)
    eval_features = eval_cache["image_features"]
    eval_labels = eval_cache["labels"]

    one_hot = F.one_hot(few_shot_image_labels, num_classes=len(r_class_names))
    cache_keys = few_shot_image_features
    cache_values = one_hot.float()

    coop_ctx_path = FEATURES_DIR / "coop_ctx_200_cls.pt"
    if not coop_ctx_path.exists():
        raise FileNotFoundError(
            f"{coop_ctx_path} not found. CoOp is evaluated from a pre-trained context "
            "vector here; train one with notebooks/05_coop.ipynb first."
        )
    text_encoder = TextEncoderWrapper(model)
    coop_prompt_learner = PromptLearner(
        clip_model=model, device=device, n_ctx=4, tokenizer=tokenizer, ctx_dim=512, class_names=r_class_names
    ).to(device)
    coop_prompt_learner.ctx.data.copy_(torch.load(coop_ctx_path, map_location=device))
    with torch.no_grad():
        prompts, tok = coop_prompt_learner()
        coop_text_features = text_encoder(prompts, tok)
        coop_text_features = coop_text_features / coop_text_features.norm(dim=-1, keepdim=True)

    augment_transform = transforms.Compose(
        [
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ]
    )
    tpt_images = [ds["test"][i]["image"] for i in range(200)]
    tpt_labels = [wnid_to_r_index[ds["test"][i]["wnid"]] for i in range(200)]
    tpt_prompt_learner = PromptLearner(
        clip_model=model, device=device, n_ctx=4, tokenizer=tokenizer, ctx_dim=512, class_names=r_class_names
    ).to(device)

    for p in model.parameters():
        p.requires_grad_(False)

    shared = {
        "test_features": eval_features.to(device),
        "labels": eval_labels.to(device),
        "text_features": text_features.to(device),
        "cache_keys": cache_keys.to(device),
        "cache_values": cache_values.to(device),
        "logit_scale": model.logit_scale.exp(),
        "coop_text_features": coop_text_features.to(device),
    }
    metrics = {"accuracy": accuracy, "ece": ece, "signed_gap": signed_gap}
    methods = {
        "zero_shot": {"fn": zero_shot_logits, "params": {}},
        "tip_adapter": {"fn": tip_adapter_logits, "params": {"alpha": TIP_ADAPTER_ALPHA, "beta": TIP_ADAPTER_BETA}},
        "coop": {"fn": coop_logits, "params": {}},
    }
    results = run_comparison(shared, methods, metrics)

    print("running TPT (per-image gradient steps, this is slow)")
    results["tpt"] = run_tpt(
        model, tpt_prompt_learner, text_encoder, preprocess, tpt_images, tpt_labels, device, augment_transform, metrics
    )

    print()
    for method, r in results.items():
        n = r.get("n", len(eval_labels))
        name = DISPLAY_NAMES.get(method, method)
        print(f"{name:12s}: top-1 {r['accuracy']:.2f}%  ECE {r['ece']:.2f}% Signed-Gap {r['signed_gap']:.2f}% (n={n})")


if __name__ == "__main__":
    main()
