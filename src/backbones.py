import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BACKBONES = {
    "vit_b16": {
        "model": "ViT-B-16-quickgelu",
        "features": REPO_ROOT / "features",
        "grid": REPO_ROOT / "results" / "grid.csv",
    },
    "rn50": {
        "model": "RN50-quickgelu",
        "features": REPO_ROOT / "features" / "rn50",
        "grid": REPO_ROOT / "results" / "grid_rn50.csv",
    },
}

BACKBONE = os.environ.get("VLM_BACKBONE", "vit_b16")
if BACKBONE not in BACKBONES:
    raise KeyError(f"VLM_BACKBONE={BACKBONE!r}; known: {sorted(BACKBONES)}")

MODEL_NAME = BACKBONES[BACKBONE]["model"]
FEATURES_DIR = BACKBONES[BACKBONE]["features"]
GRID = BACKBONES[BACKBONE]["grid"]
