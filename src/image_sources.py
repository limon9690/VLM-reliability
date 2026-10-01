"""Ordered image lists per dataset: the one place that decides image order.

image_refs(name, data_root) returns [(ref, label), ...] in a fixed order:
sorted relative paths for folder datasets, row order for the HF dataset.
A ref is a path relative to data_root, or "hf:<repo>:<split>:<row>".
Manifests map cached feature indices to these refs; TPT opens images through them.

Named image_sources, not datasets, so it doesn't shadow HuggingFace `datasets`.
"""

from functools import lru_cache
from pathlib import Path

from PIL import Image

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
R_REPO, R_SPLIT = "axiong/imagenet-r", "test"


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
