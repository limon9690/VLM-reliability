"""Reproduces the "ImageNet-Sketch" table in the README: zero-shot CLIP and
Tip-Adapter, reporting top-1 accuracy and ECE.

Usage:
    python scripts/eval_imagenet_sketch.py [--data-root data]

First run extracts image/text features (needs a GPU, several minutes) and
caches them under features/. Later runs load straight from that cache.

CoOp and TPT are not run here: both need 1000-class prompt training/tuning,
which exceeds free-tier GPU memory (see README "Known limits"). The R-trained
CoOp context is deliberately not reused on Sketch's label space -- that would
be a cross-domain transfer experiment, not "CoOp on Sketch".

If the raw dataset isn't already downloaded, this needs the `gdown` package
(`pip install gdown`) to fetch the same ImageNet-Sketch mirror the notebooks
used.
"""

import argparse
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from clip_zeroshot import (
    build_and_cache_image_features,
    build_and_cache_text_features,
    load_cached_image_features,
    load_cached_text_features,
)
from datasets_sketch import load_imagenet_sketch
from harness import accuracy, ece, run_comparison, tip_adapter_logits, zero_shot_logits, signed_gap
from imagenet_classes import IMAGENET_CLASS_NAMES, IMAGENET_TEMPLATES

import open_clip

FEATURES_DIR = REPO_ROOT / "features"
K_SHOT = 16
TIP_ADAPTER_ALPHA = 1.5
TIP_ADAPTER_BETA = 5.0
SKETCH_GDRIVE_ID = "1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA"  # same mirror used to build the existing sk_*.pt cache
DISPLAY_NAMES = {"zero_shot": "zero-shot", "tip_adapter": "Tip-Adapter"}
MODEL_NAME = "ViT-B-16-quickgelu"
PRETRAINED = "openai"


def ensure_sketch_dataset(data_root):
    sketch_dir = data_root / "sketch"
    if sketch_dir.exists() and any(sketch_dir.iterdir()):
        return sketch_dir

    try:
        import gdown
    except ImportError as e:
        raise ImportError(
            "ImageNet-Sketch isn't downloaded and `gdown` isn't installed. "
            "Run `pip install gdown` or download/extract the dataset manually to "
            f"{sketch_dir}."
        ) from e

    zip_path = data_root / "ImageNet-Sketch.zip"
    if not zip_path.exists():
        print(f"downloading ImageNet-Sketch to {zip_path}")
        gdown.download(id=SKETCH_GDRIVE_ID, output=str(zip_path), quiet=False)

    print(f"extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_root)

    if not sketch_dir.exists():
        raise RuntimeError(
            f"expected a '{sketch_dir.name}' folder under {data_root} after extracting "
            f"{zip_path}, but it's not there -- the mirror's zip layout may have changed."
        )
    return sketch_dir


def check_class_ordering(raw_dataset, class_names):
    """ImageFolder assigns label indices by sorting folder names alphabetically. That
    only lines up with IMAGENET_CLASS_NAMES's standard ImageNet-1k order if the folders
    are named by WordNet synset ID (n01440764, ...), which is how this mirror ships."""
    if len(raw_dataset.classes) != len(class_names):
        raise RuntimeError(
            f"ImageFolder found {len(raw_dataset.classes)} class folders but "
            f"IMAGENET_CLASS_NAMES has {len(class_names)}. Label indices would be "
            "misaligned with the text classifier."
        )
    bad = [c for c in raw_dataset.classes if not (c.startswith("n") and c[1:].isdigit())]
    if bad:
        raise RuntimeError(
            f"sketch class folders don't look like WordNet synset IDs (e.g. {bad[:3]}), "
            "so ImageFolder's alphabetical class order can't be trusted to match "
            "IMAGENET_CLASS_NAMES."
        )


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
        help="Where the raw ImageNet-Sketch download is extracted",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    args.data_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.eval()
    model.to(device)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    few_shot_cache_path = FEATURES_DIR / "sk_few_shot_image_features.pt"
    eval_cache_path = FEATURES_DIR / "sk_eval_features.pt"

    if few_shot_cache_path.exists():
        few_shot_cache = load_cached_image_features(str(few_shot_cache_path))
    if eval_cache_path.exists():
        eval_cache = load_cached_image_features(str(eval_cache_path))

    if not few_shot_cache_path.exists() or not eval_cache_path.exists():
        sketch_dir = ensure_sketch_dataset(args.data_root)
        split = load_imagenet_sketch(str(sketch_dir), preprocess, IMAGENET_CLASS_NAMES, k_shot=K_SHOT, seed=42)
        check_class_ordering(split["raw_dataset"], IMAGENET_CLASS_NAMES)

        if not few_shot_cache_path.exists():
            few_shot_loader = DataLoader(split["few_shot_ds"], batch_size=32, shuffle=True)
            few_shot_cache = load_or_build_image_features("sk_few_shot_image_features", few_shot_loader, model, device)
        if not eval_cache_path.exists():
            eval_loader = DataLoader(split["eval_ds"], batch_size=32)
            eval_cache = load_or_build_image_features("sk_eval_features", eval_loader, model, device)

    few_shot_image_features = few_shot_cache["image_features"]
    few_shot_image_labels = few_shot_cache["labels"]
    eval_features = eval_cache["image_features"]
    eval_labels = eval_cache["labels"]

    text_features = load_or_build_text_features("sk_text-features", IMAGENET_CLASS_NAMES, model, tokenizer, device)

    one_hot = F.one_hot(few_shot_image_labels, num_classes=len(IMAGENET_CLASS_NAMES))
    cache_keys = few_shot_image_features
    cache_values = one_hot.float()

    shared = {
        "test_features": eval_features.to(device),
        "labels": eval_labels.to(device),
        "text_features": text_features.to(device),
        "cache_keys": cache_keys.to(device),
        "cache_values": cache_values.to(device),
        "logit_scale": model.logit_scale.exp(),
    }
    metrics = {"accuracy": accuracy, "ece": ece, "signed_gap": signed_gap}
    methods = {
        "zero_shot": {"fn": zero_shot_logits, "params": {}},
        "tip_adapter": {"fn": tip_adapter_logits, "params": {"alpha": TIP_ADAPTER_ALPHA, "beta": TIP_ADAPTER_BETA}},
    }
    results = run_comparison(shared, methods, metrics)

    print()
    for method, r in results.items():
        name = DISPLAY_NAMES.get(method, method)
        print(f"{name:12s}: top-1 {r['accuracy']:.2f}%  ECE {r['ece']:.2f}% Signed-Gap {r['signed_gap']:.2f}% (n={len(eval_labels)})")


if __name__ == "__main__":
    main()
