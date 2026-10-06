"""Ordered image lists per dataset: the one place that decides image order.

image_refs(name, data_root) returns [(ref, label), ...] in a fixed order:
sorted relative paths for folder datasets, row order for the HF dataset.
A ref is a path relative to data_root, or "hf:<repo>:<split>:<row>".
Manifests map cached feature indices to these refs; TPT opens images through them.

Named image_sources, not datasets, so it doesn't shadow HuggingFace `datasets`.
"""

import collections
import csv
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from imagenet_a_classes import A_KEPT_WNIDS, A_WNIDS, MIN_IMAGES, wnid_to_a_index
from imagenet_r_classes import wnid_to_r_index

IMG_EXT = {".jpg", ".jpeg", ".png"}
PACS_DIRS = {
    "pacs_photo": "photo",
    "pacs_art": "art_painting",
    "pacs_cartoon": "cartoon",
    "pacs_sketch": "sketch",
}
SKETCH_DIR = "sketch"
V2_DIR = "imagenetv2-matched-frequency-format-val"
PACS_ROOT = "pacs_data/pacs_data"
A_REPO, A_SPLIT = "barkermrl/imagenet-a", "train"
R_REPO, R_SPLIT = "axiong/imagenet-r", "test"
MANIFEST_DIR = Path(__file__).resolve().parent.parent / "manifests"


@lru_cache(maxsize=None)
def _hf(repo, split, cache_dir):
    from datasets import load_dataset

    return load_dataset(repo, split=split, cache_dir=cache_dir)


def _folder_refs(data_root, sub, label_of):
    """Every image under data_root/sub, sorted by path; label_of(class folder) -> label or None."""
    base = Path(data_root)
    root = base / sub
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found")
    refs = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        label = label_of(p.parent.name)
        if label is not None:
            refs.append((p.relative_to(base).as_posix(), label))
    return refs


def _sorted_dirs(path):
    return sorted(d.name for d in Path(path).iterdir() if d.is_dir())


def image_refs(name, data_root):
    data_root = Path(data_root)

    if name == "imagenet_r":
        ds = _hf(R_REPO, R_SPLIT, str(data_root / "hf"))
        return [
            (f"hf:{R_REPO}:{R_SPLIT}:{i}", wnid_to_r_index[w])
            for i, w in enumerate(ds["wnid"])
        ]

    if name == "imagenet_a":
        ds = _hf(A_REPO, A_SPLIT, str(data_root / "hf"))
        assert ds.features["label"].names == A_WNIDS, "imagenet_a: label order differs"
        wnids = [A_WNIDS[l] for l in ds["label"]]
        counts = collections.Counter(wnids)
        kept = sorted(w for w, c in counts.items() if c >= MIN_IMAGES)
        assert kept == A_KEPT_WNIDS, "imagenet_a: kept classes differ from A_KEPT_WNIDS"
        return [
            (f"hf:{A_REPO}:{A_SPLIT}:{i}", wnid_to_a_index[w])
            for i, w in enumerate(wnids)
            if w in wnid_to_a_index
        ]

    if name == "sketch_1000":
        wnids = _sorted_dirs(data_root / SKETCH_DIR)
        index = {w: i for i, w in enumerate(wnids)}
        return _folder_refs(data_root, SKETCH_DIR, index.get)

    if name == "sketch_200":
        return _folder_refs(data_root, SKETCH_DIR, wnid_to_r_index.get)

    if name == "imagenet_v2":
        return _folder_refs(data_root, V2_DIR, int)

    if name in PACS_DIRS:
        sub = f"{PACS_ROOT}/{PACS_DIRS[name]}"
        classes = _sorted_dirs(data_root / sub)
        index = {c: i for i, c in enumerate(classes)}
        return _folder_refs(data_root, sub, index.get)

    raise KeyError(f"unknown dataset {name}")


def open_image(ref, data_root):
    data_root = Path(data_root)
    if ref.startswith("hf:"):
        _, repo, split, row = ref.split(":")
        image = _hf(repo, split, str(data_root / "hf"))[int(row)]["image"]
    else:
        image = Image.open(data_root / ref)
    return image.convert("RGB")


def load_manifest(name):
    """Refs and labels indexed by cache index, from manifests/<name>.csv."""
    with open(MANIFEST_DIR / f"{name}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [int(r["cache_index"]) for r in rows] == list(range(len(rows)))
    return [r["ref"] for r in rows], [int(r["label"]) for r in rows]


def write_manifest(name, refs):
    """Manifest for a dataset encoded fresh in image_refs order: cache index i = refs[i]."""
    MANIFEST_DIR.mkdir(exist_ok=True)
    with open(MANIFEST_DIR / f"{name}.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cache_index", "ref", "label", "cosine"])
        for i, (ref, label) in enumerate(refs):
            writer.writerow([i, ref, label, "1.000000"])


class RefDataset(Dataset):
    def __init__(self, refs, data_root, preprocess):
        self.refs = refs
        self.data_root = data_root
        self.preprocess = preprocess

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, i):
        return self.preprocess(open_image(self.refs[i], self.data_root))


@torch.no_grad()
def encode_images(
    model, preprocess, refs, data_root, device, batch_size=256, workers=8
):
    """Normalized image features for refs, in the given order, on CPU."""
    loader = DataLoader(
        RefDataset(refs, data_root, preprocess),
        batch_size=batch_size,
        num_workers=workers,
    )
    out = []
    for i, x in enumerate(loader):
        out.append(F.normalize(model.encode_image(x.to(device)).float(), dim=-1).cpu())
        if i % 50 == 0:
            print(f"  encoded {i * batch_size}/{len(refs)}")
    return torch.cat(out)
