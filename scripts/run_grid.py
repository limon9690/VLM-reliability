"""Runs every (dataset, seed) cell and writes results/grid.csv."""
import csv, hashlib, json, sys
from pathlib import Path
import torch
import open_clip

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from features_registry import DATASETS, load_features
from harness import accuracy, ece, run_comparison, signed_gap, zero_shot_logits
from splits import split_indices

SEEDS = [42, 43, 44]
RESULTS = REPO_ROOT / "results" / "grid.csv"
METRICS = {"accuracy": accuracy, "ece": ece, "signed_gap": signed_gap}
METHODS = {"zero_shot": {"fn": zero_shot_logits, "params": {}}}
MODEL_NAME = "ViT-B-16-quickgelu"


def config_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained="openai")
    logit_scale = model.logit_scale.exp().item()   # only the scale is needed; features are cached

    rows = []
    for name in DATASETS:
        f = load_features(name, device=device)
        for seed in SEEDS:
            _, _, test_idx = split_indices(f["labels"].tolist(), seed=seed, n_cache=f["n_shot"], n_val=f["n_val"])
            shared = {
                "test_features": f["image_features"][test_idx],
                "labels": f["labels"][test_idx].to(device),
                "text_features": f["text_features"],
                "logit_scale": logit_scale,
            }
            results = run_comparison(shared, METHODS, METRICS)
            for method, r in results.items():
                cfg = {"model": MODEL_NAME, "dataset": name, "method": method,
                       "params": METHODS[method]["params"], "seed": seed,
                       "n_shot": f["n_shot"], "n_val": f["n_val"]}
                rows.append({
                    "method": method, "dataset": name, "n_classes": f["n_classes"],
                    "n_shot": f["n_shot"], "seed": seed,
                    "alpha": "", "beta": "",
                    "n_test": len(test_idx),
                    **{k: round(v, 4) for k, v in r.items()},
                    "model_name": MODEL_NAME, "gpu": gpu, "config_hash": config_hash(cfg),
                })
            print(f"{name:13s} seed {seed}  " + "  ".join(f"{k} {v:.2f}" for k, v in results["zero_shot"].items()))

    RESULTS.parent.mkdir(exist_ok=True)
    with open(RESULTS, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {RESULTS}")


if __name__ == "__main__":
    main()