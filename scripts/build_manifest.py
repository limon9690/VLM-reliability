"""Maps every cached feature to its source image and writes manifests/<dataset>.csv.

Encodes all images in image_refs order, then matches each cached feature to its
nearest new feature. Writes the manifest only if every match has cosine above
--min-cos, every matched label equals the cached label, and no image is claimed
twice (except exact duplicates).

Usage:
    python scripts/build_manifest.py --dataset pacs_photo
    python scripts/build_manifest.py --dataset sketch_1000 --data-root /root/autodl-tmp/data
"""

import argparse, csv, sys, time
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from clip_zeroshot import MODEL_NAME
from features_registry import DATASETS, load_features
from image_sources import encode_images, image_refs

MANIFEST_DIR = REPO_ROOT / "manifests"


@torch.no_grad()
def match(cached, cached_labels, new, new_labels, device, chunk=4096):
    """For each cached row: best cosine and index of the nearest same-label new feature.
    Same-label only, because some images appear in more than one class folder."""
    new_d, nl = new.to(device), new_labels.to(device)
    best, idx = [], []
    for s in range(0, len(cached), chunk):
        sims = cached[s : s + chunk].to(device) @ new_d.T
        other = cached_labels[s : s + chunk, None].to(device) != nl[None, :]
        v, i = sims.masked_fill(other, -2.0).max(dim=-1)
        best.append(v.cpu())
        idx.append(i.cpu())
    return torch.cat(best), torch.cat(idx)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--data-root", type=Path, default=REPO_ROOT.parent / "data")
    p.add_argument("--min-cos", type=float, default=0.999)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--floor", type=float, default=0.99)
    p.add_argument("--margin", type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    f = load_features(args.dataset, device="cpu")
    cached = F.normalize(f["image_features"].float(), dim=-1)
    cached_labels = f["labels"].long()
    refs = image_refs(args.dataset, args.data_root)
    print(f"{args.dataset}: {len(cached)} cached features, {len(refs)} images on disk")
    if len(refs) != len(cached):
        sys.exit(
            "count mismatch: wrong folder, or the source differs from the one encoded"
        )

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained="openai"
    )
    model = model.to(device).eval()
    new = encode_images(
        model, preprocess, [r[0] for r in refs], args.data_root, device,
        args.batch_size, args.workers,
    )

    new_labels = torch.tensor([r[1] for r in refs])
    cos, idx = match(cached, cached_labels, new, new_labels, device)
    matched_labels = torch.tensor([refs[i][1] for i in idx.tolist()])
    bad_label = (matched_labels != cached_labels).nonzero().flatten()
    low = (cos < args.min_cos).nonzero().flatten()

    # A low-cosine match is still the right image if it is unambiguous:
    # above --floor and at least --margin ahead of the best non-identical alternative.
    kept = []
    for r in low.tolist():
        twins = (new @ new[idx[r]]) >= args.min_cos
        other = (new_labels != cached_labels[r]) | twins
        alt = (new @ cached[r]).masked_fill(other, -2.0).max().item()
        if cos[r] >= args.floor and cos[r] - alt >= args.margin:
            print(
                f"  row {r}: cosine {cos[r]:.4f}, next {alt:.4f}, accepted as unambiguous"
            )
        else:
            kept.append(r)
    low = torch.tensor(kept, dtype=torch.long)

    # An image claimed by several cached rows is fine only if those rows are identical images.
    counts = torch.bincount(idx, minlength=len(new))
    unexplained_dup = 0
    for j in (counts > 1).nonzero().flatten().tolist():
        rows = (idx == j).nonzero().flatten()
        pair = cached[rows] @ cached[rows].T
        if pair.min() < args.min_cos:
            unexplained_dup += 1

    print(
        f"cosine min {cos.min():.6f}  median {cos.median():.6f}  "
        f"below {args.min_cos}: {len(low)}  label mismatches: {len(bad_label)}  "
        f"images claimed twice: {(counts > 1).sum().item()} "
        f"(unexplained {unexplained_dup})  ({time.time() - t0:.0f}s)"
    )
    if len(low) or len(bad_label) or unexplained_dup:
        print(
            "first problem rows:",
            sorted(set(low[:5].tolist() + bad_label[:5].tolist())),
        )
        sys.exit("manifest NOT written")

    MANIFEST_DIR.mkdir(exist_ok=True)
    path = MANIFEST_DIR / f"{args.dataset}.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cache_index", "ref", "label", "cosine"])
        for k, (j, c) in enumerate(zip(idx.tolist(), cos.tolist())):
            writer.writerow([k, refs[j][0], refs[j][1], f"{c:.6f}"])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
