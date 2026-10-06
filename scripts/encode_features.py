"""Encodes images in manifest order into the feature caches run_grid.py reads.

Usage:
    python scripts/encode_features.py                     # all 8 datasets
    python scripts/encode_features.py --datasets pacs_photo

Writes features/<cache>.pt (image features + labels) and the 80-template text
feature for each dataset. A dataset with no manifest yet (a new dataset) gets one
first, in image_refs order, so cache index i is image i. If a cache already exists it is NOT overwritten:
the new encoding is compared against it instead (labels must match exactly,
cosine should be ~1.0), which makes this a check of an existing cache too.
--overwrite replaces existing caches.
"""

import argparse, sys, time
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from clip_zeroshot import MODEL_NAME, build_and_cache_text_features
from features_registry import CLASS_NAMES, DATASETS, FEATURES_DIR
from image_sources import (
    MANIFEST_DIR,
    encode_images,
    image_refs,
    load_manifest,
    write_manifest,
)
from imagenet_classes import IMAGENET_TEMPLATES

MIN_COS = 0.99  # two known images decode at 0.996-0.997; everything else is ~1.0


def image_cache(name, model, preprocess, device, args):
    path = FEATURES_DIR / f"{DATASETS[name]['images']}.pt"
    if not (MANIFEST_DIR / f"{name}.csv").exists():
        refs = image_refs(name, args.data_root)
        write_manifest(name, refs)
        print(f"  wrote manifests/{name}.csv ({len(refs)} images, image_refs order)")
    refs, labels = load_manifest(name)
    labels = torch.tensor(labels)
    feats = encode_images(
        model, preprocess, refs, args.data_root, device, args.batch_size, args.workers
    )

    if path.exists() and not args.overwrite:
        old = torch.load(path)
        cos = (F.normalize(old["image_features"].float(), dim=-1) * feats).sum(-1)
        same = torch.equal(old["labels"].long(), labels)
        ok = bool(same and cos.min() >= MIN_COS)
        print(
            f"  {path.name} exists, compared: labels {'match' if same else 'DIFFER'}, "
            f"cosine min {cos.min():.6f} median {cos.median():.6f}  "
            f"{'OK' if ok else 'MISMATCH'}"
        )
        return ok

    torch.save({"image_features": feats, "labels": labels}, path)
    print(f"  wrote {path.name} ({len(feats)} images)")
    return True


def text_cache(name, model, tokenizer, device, args):
    spec = DATASETS[name]
    path = FEATURES_DIR / f"{spec['text']}.pt"
    class_names = CLASS_NAMES[spec["class_names"]]
    if path.exists() and not args.overwrite:
        old = torch.load(path)
        new = build_and_cache_text_features(
            model,
            tokenizer,
            class_names,
            IMAGENET_TEMPLATES,
            device,
            "/tmp",
            f"check_{spec['text']}",
        )
        ok = torch.allclose(old.float(), new.float(), atol=1e-4)
        print(f"  {path.name} exists, compared: {'OK' if ok else 'MISMATCH'}")
        return ok
    build_and_cache_text_features(
        model,
        tokenizer,
        class_names,
        IMAGENET_TEMPLATES,
        device,
        str(FEATURES_DIR),
        spec["text"],
    )
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS)
    )
    p.add_argument("--data-root", type=Path, default=REPO_ROOT.parent / "data")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained="openai"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    FEATURES_DIR.mkdir(exist_ok=True)

    ok = True
    for name in args.datasets:
        t0 = time.time()
        print(f"{name}:")
        ok &= image_cache(name, model, preprocess, device, args)
        ok &= text_cache(name, model, tokenizer, device, args)
        print(f"  ({time.time() - t0:.0f}s)")
    sys.exit(0 if ok else "some caches did not match")


if __name__ == "__main__":
    main()
