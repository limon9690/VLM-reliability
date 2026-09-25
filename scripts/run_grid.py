"""Runs every (dataset, seed) cell and writes results/grid.csv."""

import csv, hashlib, json, sys, time
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from clip_zeroshot import MODEL_NAME
from coop import coop_text_features, save_coop_ctx, train_coop
from features_registry import DATASETS, load_features
from harness import (
    accuracy,
    coop_logits,
    ece,
    run_comparison,
    signed_gap,
    tip_adapter_logits,
    zero_shot_logits,
)
from splits import split_indices

DATASETS_TO_RUN = list(DATASETS)
SEEDS = [42, 43, 44]
RESULTS = REPO_ROOT / "results" / "grid.csv"
CTX_DIR = REPO_ROOT / "features" / "coop"

ALPHA = 1.5
COOP_CFG = {"n_ctx": 4, "lr": 0.002, "epochs": 10, "batch_size": 32}

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
    "coop": {"method": "coop", "fn": coop_logits, "params": COOP_CFG},
}


def config_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def load_or_train_ctx(name, seed, f, cache_idx, model, tokenizer, device):
    """Reuses a saved context only if it was trained under the same settings."""
    path = CTX_DIR / f"coop_ctx_{name}_seed{seed}.pt"
    expected = {"model": MODEL_NAME, "seed": seed, "n_shot": f["n_shot"], **COOP_CFG}
    if path.exists():
        saved = torch.load(path)
        if saved["class_names"] == list(f["class_names"]) and all(
            saved.get(k) == v for k, v in expected.items()
        ):
            print(f"  loaded CoOp context from {path.name}")
            return saved["ctx"]
        print(f"  {path.name} was trained under different settings, retraining")

    ctx = train_coop(
        model,
        tokenizer,
        f["class_names"],
        f["image_features"][cache_idx],
        f["labels"][cache_idx],
        seed=seed,
        device=device,
        **COOP_CFG,
    )
    CTX_DIR.mkdir(parents=True, exist_ok=True)
    save_coop_ctx(path, ctx, f["class_names"], **expected)
    return ctx


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained="openai")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    logit_scale = model.logit_scale.exp().item()

    rows = []
    for name in DATASETS_TO_RUN:
        f = load_features(name, device=device)
        for seed in SEEDS:
            t0 = time.time()
            cache_idx, _, test_idx = split_indices(
                f["labels"].tolist(), seed=seed, n_cache=f["n_shot"], n_val=f["n_val"]
            )
            cache_labels = f["labels"][cache_idx]

            ctx = load_or_train_ctx(name, seed, f, cache_idx, model, tokenizer, device)

            shared = {
                "test_features": f["image_features"][test_idx],
                "labels": f["labels"][test_idx].to(device),
                "text_features": f["text_features"],
                "logit_scale": logit_scale,
                "cache_keys": f["image_features"][cache_idx],
                "cache_values": F.one_hot(cache_labels, num_classes=f["n_classes"])
                .float()
                .to(device),
                "coop_text_features": coop_text_features(
                    model,
                    tokenizer,
                    f["class_names"],
                    ctx,
                    device,
                    n_ctx=COOP_CFG["n_ctx"],
                ),
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
                f"Δβ1 {results['tip_adapter_b1']['signed_gap'] - zs:+.2f}  "
                f"ΔCoOp {results['coop']['signed_gap'] - zs:+.2f}  ({time.time() - t0:.0f}s)"
            )

    RESULTS.parent.mkdir(exist_ok=True)
    with open(RESULTS, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {RESULTS}")


if __name__ == "__main__":
    main()
