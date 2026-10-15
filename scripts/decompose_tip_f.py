"""Splits Tip-Adapter and Tip-Adapter-F logits into the zero-shot part and the alpha * cache part.

Usage:
    python scripts/decompose_tip_f.py
"""

import argparse, sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_grid as rg
from features_registry import DATASETS, load_features
from splits import split_indices
from tip_adapter_f import train_tip_adapter_f

LOGIT_SCALE = 100.0


def cache_term(test_features, keys, values, alpha, beta):
    affinity = torch.exp(-beta * (1 - test_features @ keys.T))
    return alpha * (affinity @ values)


def describe(logits, labels):
    probs = logits.softmax(dim=-1)
    top2 = logits.topk(2, dim=-1).values
    conf, pred = probs.max(dim=-1)
    acc = (pred == labels).float().mean().item()
    return {
        "acc": 100 * acc,
        "conf": 100 * conf.mean().item(),
        "gap": 100 * (conf.mean().item() - acc),
        "range": (logits.max(dim=-1).values - logits.min(dim=-1).values).mean().item(),
        "margin": (top2[:, 0] - top2[:, 1]).mean().item(),
    }


def decompose(test_features, labels, text_features, keys, values, alpha, beta):
    zs = LOGIT_SCALE * (test_features @ text_features)
    cache = cache_term(test_features, keys, values, alpha, beta)
    full = zs + cache
    z, c, a = describe(zs, labels), describe(cache, labels), describe(full, labels)
    zs_pred = zs.argmax(dim=-1)
    winner = full.argmax(dim=-1)
    return {
        "acc_zs": z["acc"], "acc": a["acc"],
        "conf_zs": z["conf"], "conf": a["conf"],
        "gap_zs": z["gap"], "gap": a["gap"],
        "range_zs": z["range"], "range": a["range"], "range_cache": c["range"],
        "margin_zs": z["margin"], "margin": a["margin"],
        "cache_on_winner": cache.gather(-1, winner[:, None]).mean().item(),
        "cache_mean": cache.mean().item(),
        "cache_agrees_zs": 100 * (cache.argmax(dim=-1) == zs_pred).float().mean().item(),
        "winner_changed": 100 * (winner != zs_pred).float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "decompose_tip_f.csv")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha, beta = rg.TIP_F_CFG["alpha"], rg.TIP_F_CFG["beta"]
    grid = pd.read_csv(rg.RESULTS)
    rows = []
    for name in args.datasets:
        f = load_features(name, device=device)
        for seed in rg.SEEDS.get(name, rg.DEFAULT_SEEDS):
            cache_idx, _, test_idx = split_indices(
                f["labels"].tolist(), seed=seed, n_cache=f["n_shot"], n_val=f["n_val"]
            )
            values = F.one_hot(f["labels"][cache_idx], num_classes=f["n_classes"]).float().to(device)
            keys = f["image_features"][cache_idx].float()
            f_keys, _ = train_tip_adapter_f(keys, values, f["text_features"], LOGIT_SCALE,
                                            seed=seed, device=device, **rg.TIP_F_CFG)
            test = f["image_features"][test_idx].float()
            labels = f["labels"][test_idx].to(device)
            for method, k in [("tip_adapter", keys), ("tip_adapter_f", f_keys)]:
                r = decompose(test, labels, f["text_features"].float(), k, values, alpha, beta)
                ref = grid[(grid.method == method) & (grid.dataset == name)
                           & (grid.seed == seed) & (grid.beta == beta)]
                assert len(ref) == 1, (method, name, seed, len(ref))
                assert abs(r["acc"] - ref.accuracy.item()) <= 100 / len(test_idx) + 1e-6, (method, name, seed)
                assert abs(r["gap"] - ref.signed_gap.item()) < 0.05, (method, name, seed)
                rows.append({"method": method, "dataset": name, "seed": seed, **r})
            print(name, seed, "reproduces grid")
    pd.DataFrame(rows).round(4).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
