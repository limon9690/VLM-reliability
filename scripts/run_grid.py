"""Runs every (dataset, seed) cell and writes results/grid.csv."""

import csv, hashlib, json, sys
from pathlib import Path
import torch
import open_clip
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from features_registry import DATASETS, load_features
from harness import (
    accuracy,
    ece,
    run_comparison,
    signed_gap,
    zero_shot_logits,
    tip_adapter_logits,
)
from splits import split_indices

SEEDS = [42, 43, 44]
ALPHA = 1.5
RESULTS = REPO_ROOT / "results" / "grid.csv"
METRICS = {"accuracy": accuracy, "ece": ece, "signed_gap": signed_gap}
METHODS = {
    "zero_shot": {"method": "zero_shot", "fn": zero_shot_logits, "params": {}},
    "tip_adapter_b5": {
        "method": "tip_adapter",
        "fn": tip_adapter_logits,
        "params": {"alpha": ALPHA, "beta": 5.0},
    },
    "tip_adapter_b1": {
        "method": "tip_adapter",
        "fn": tip_adapter_logits,
        "params": {"alpha": ALPHA, "beta": 1.0},
    },
}
MODEL_NAME = "ViT-B-16-quickgelu"


def config_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained="openai")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    logit_scale = model.logit_scale.exp().item()

    rows = []
    for name in DATASETS:
        f = load_features(name, device=device)
        for seed in SEEDS:
            cache_idx, _, test_idx = split_indices(
                f["labels"].tolist(), seed=seed, n_cache=f["n_shot"], n_val=f["n_val"]
            )
            cache_labels = f["labels"][cache_idx]

            shared = {
                "test_features": f["image_features"][test_idx],
                "labels": f["labels"][test_idx].to(device),
                "text_features": f["text_features"],
                "logit_scale": logit_scale,
                "cache_keys": f["image_features"][cache_idx],
                "cache_values": F.one_hot(cache_labels, num_classes=f["n_classes"])
                .float()
                .to(device),
            }

            results = run_comparison(shared, METHODS, METRICS)
            for key, r in results.items():
                spec = METHODS[key]
                cfg = {
                    "model": MODEL_NAME,
                    "dataset": name,
                    "method": spec["method"],
                    "params": spec["params"],
                    "seed": seed,
                    "n_shot": f["n_shot"],
                    "n_val": f["n_val"],
                }

                rows.append(
                    {
                        "method": spec["method"],
                        "dataset": name,
                        "n_classes": f["n_classes"],
                        "n_shot": f["n_shot"],
                        "seed": seed,
                        "alpha": spec["params"].get("alpha", ""),
                        "beta": spec["params"].get("beta", ""),
                        "n_test": len(test_idx),
                        **{k: round(v, 4) for k, v in r.items()},
                        "model_name": MODEL_NAME,
                        "gpu": gpu,
                        "config_hash": config_hash(cfg),
                    }
                )

            zs = results["zero_shot"]["signed_gap"]
            print(
                f"{name:13s} seed {seed}  ZS gap {zs:+.2f}  "
                f"Δβ5 {results['tip_adapter_b5']['signed_gap'] - zs:+.2f}  "
                f"Δβ1 {results['tip_adapter_b1']['signed_gap'] - zs:+.2f}"
            )

    RESULTS.parent.mkdir(exist_ok=True)
    with open(RESULTS, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {RESULTS}")


if __name__ == "__main__":
    main()
