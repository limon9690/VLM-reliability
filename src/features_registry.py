from pathlib import Path
from clip_zeroshot import load_cached_image_features, load_cached_text_features
from imagenet_classes import IMAGENET_CLASS_NAMES
from imagenet_r_classes import r_class_names

FEATURES_DIR = Path(__file__).resolve().parent.parent / "features"
PACS_CLASS_NAMES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
CLASS_NAMES = {
    "imagenet_r": r_class_names,
    "imagenet_1000": IMAGENET_CLASS_NAMES,
    "pacs": PACS_CLASS_NAMES,
}

DATASETS = {
    "imagenet_r": {
        "images": "r_all_features",
        "text": "r_text_features",
        "class_names": "imagenet_r",
        "n_classes": 200,
        "n_shot": 16,
        "n_val": 10,
    },
    "sketch_200": {
        "images": "sk200_all_features",
        "text": "r_text_features",
        "class_names": "imagenet_r",
        "n_classes": 200,
        "n_shot": 16,
        "n_val": 10,
    },
    "sketch_1000": {
        "images": "sk_all_features",
        "text": "sk_text_features",
        "class_names": "imagenet_1000",
        "n_classes": 1000,
        "n_shot": 16,
        "n_val": 10,
    },
    "imagenet_v2": {
        "images": "v2_all_features",
        "text": "v2_text_features",
        "class_names": "imagenet_1000",
        "n_classes": 1000,
        "n_shot": 4,
        "n_val": 0,
    },
    "pacs_photo": {
        "images": "pacs_photo",
        "text": "pacs_text_features",
        "class_names": "pacs",
        "n_classes": 7,
        "n_shot": 16,
        "n_val": 10,
    },
    "pacs_art": {
        "images": "pacs_art_painting",
        "text": "pacs_text_features",
        "class_names": "pacs",
        "n_classes": 7,
        "n_shot": 16,
        "n_val": 10,
    },
    "pacs_cartoon": {
        "images": "pacs_cartoon",
        "text": "pacs_text_features",
        "class_names": "pacs",
        "n_classes": 7,
        "n_shot": 16,
        "n_val": 10,
    },
    "pacs_sketch": {
        "images": "pacs_sketch",
        "text": "pacs_text_features",
        "class_names": "pacs",
        "n_classes": 7,
        "n_shot": 16,
        "n_val": 10,
    },
}


def load_features(name, device="cpu"):
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    spec = DATASETS[name]

    images = load_cached_image_features(str(FEATURES_DIR / f"{spec['images']}.pt"))
    text = load_cached_text_features(str(FEATURES_DIR / f"{spec['text']}.pt"))[
        "text_features"
    ]

    image_features, labels = images["image_features"], images["labels"]
    n_classes = spec["n_classes"]
    class_names = CLASS_NAMES[spec["class_names"]]

    assert (
        len(class_names) == n_classes
    ), f"{name}: {len(class_names)} class names, expected {n_classes}"
    assert (
        text.shape[-1] == n_classes
    ), f"{name}: text features have {text.shape[-1]} classes, expected {n_classes}"
    assert (
        image_features.shape[0] == labels.shape[0]
    ), f"{name}: features and labels disagree"
    assert (
        int(labels.max()) < n_classes
    ), f"{name}: label {int(labels.max())} out of range"

    return {
        "image_features": image_features.to(device),
        "labels": labels,
        "text_features": text.to(device),
        "n_classes": n_classes,
        "n_shot": spec["n_shot"],
        "n_val": spec["n_val"],
        "class_names": class_names,
    }
