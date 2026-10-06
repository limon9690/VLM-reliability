"""Downloads the raw datasets into data_root (default: ../data next to the repo).

Usage:
    python scripts/download_data.py                       # all 8 datasets
    python scripts/download_data.py --datasets pacs_photo imagenet_v2

Sources (all reachable from mainland China through an HF mirror):
    imagenet_r            HuggingFace axiong/imagenet-r (datasets library)
    imagenet_a            HuggingFace barkermrl/imagenet-a (datasets library)
    sketch_1000/200       songweig/imagenet_sketch, data/ImageNet-Sketch.zip
    imagenet_v2           vaishaal/ImageNetV2, imagenetv2-matched-frequency.tar.gz
    pacs_*                Kaggle nickfratto/pacs-dataset

HuggingFace files come from $HF_ENDPOINT if set (e.g. https://hf-mirror.com),
otherwise huggingface.co. Anything already on disk is skipped. Each dataset is
then checked against its manifest: every image the manifest names must exist.
A dataset with no manifest yet gets one from encode_features.py.
"""

import argparse, os, sys, tarfile, zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from features_registry import DATASETS
from image_sources import (
    A_REPO,
    A_SPLIT,
    MANIFEST_DIR,
    PACS_ROOT,
    R_REPO,
    R_SPLIT,
    SKETCH_DIR,
    V2_DIR,
    image_refs,
    load_manifest,
)

SKETCH_FILE = "datasets/songweig/imagenet_sketch/resolve/main/data/ImageNet-Sketch.zip"
V2_FILE = (
    "datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz"
)
PACS_URL = "https://www.kaggle.com/api/v1/datasets/download/nickfratto/pacs-dataset"


def hf_url(path):
    return (
        f"{os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')}/{path}"
    )


def fetch(url, dest):
    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))


def get_imagenet_r(root):
    from datasets import load_dataset  # cached after the first call

    load_dataset(R_REPO, split=R_SPLIT, cache_dir=str(root / "hf"))


def get_imagenet_a(root):
    from datasets import load_dataset

    load_dataset(A_REPO, split=A_SPLIT, cache_dir=str(root / "hf"))


def get_sketch(root):
    if (root / SKETCH_DIR).is_dir():
        return print("  sketch/ already present")
    archive = root / "ImageNet-Sketch.zip"
    fetch(hf_url(SKETCH_FILE), archive)
    print("  extracting (several minutes)")
    with zipfile.ZipFile(archive) as z:
        z.extractall(root)
    archive.unlink()


def get_v2(root):
    if (root / V2_DIR).is_dir():
        return print(f"  {V2_DIR}/ already present")
    archive = root / "imagenetv2-matched-frequency.tar"
    fetch(hf_url(V2_FILE), archive)
    with tarfile.open(
        archive, "r:*"
    ) as t:  # named .tar.gz but actually plain tar; r:* detects
        t.extractall(root, filter="data")
    archive.unlink()


def get_pacs(root):
    if (root / PACS_ROOT).is_dir():
        return print(f"  {PACS_ROOT}/ already present")
    archive = root / "pacs-dataset.zip"
    fetch(PACS_URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(root)
    archive.unlink()


SOURCES = {
    "imagenet_r": get_imagenet_r,
    "imagenet_a": get_imagenet_a,
    "sketch": get_sketch,
    "v2": get_v2,
    "pacs": get_pacs,
}


def source_of(name):
    if name in ("imagenet_r", "imagenet_a"):
        return name
    if name.startswith("sketch"):
        return "sketch"
    if name == "imagenet_v2":
        return "v2"
    return "pacs"


def check(name, root):
    """Every image the manifest names must be on disk."""
    if not (MANIFEST_DIR / f"{name}.csv").exists():
        print(f"  {name}: no manifest yet (encode_features.py writes it)")
        return True
    on_disk = {ref for ref, _ in image_refs(name, root)}
    refs, _ = load_manifest(name)
    missing = [r for r in refs if r not in on_disk]
    status = "OK" if not missing else f"{len(missing)} MISSING, e.g. {missing[:3]}"
    print(
        f"  {name}: {len(set(refs))} images in manifest, {len(on_disk)} on disk, {status}"
    )
    return not missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS)
    )
    p.add_argument("--data-root", type=Path, default=REPO_ROOT.parent / "data")
    args = p.parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)

    for src in dict.fromkeys(source_of(n) for n in args.datasets):
        print(f"{src}:")
        SOURCES[src](args.data_root)

    print("checking against manifests:")
    ok = all([check(n, args.data_root) for n in args.datasets])
    sys.exit(0 if ok else "some images are missing")


if __name__ == "__main__":
    main()
