"""Reruns the grid's TPT cells (seed 42, run 0) and compares logit range with top-class margins.

Usage:
    python scripts/tpt_margins.py
    python scripts/tpt_margins.py --datasets imagenet_r imagenet_a
"""

import argparse, random, sys
from pathlib import Path

import open_clip
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_grid as rg
from features_registry import load_features
from harness import run_tpt, zero_shot_logits
from image_sources import load_manifest
from splits import split_indices

N = 2000
SEED = 42
RUN = 0


def describe(logits, labels):
    logits = logits.float()
    probs = logits.softmax(dim=-1)
    conf, pred = probs.max(dim=-1)
    top5 = logits.topk(5, dim=-1).values
    acc = (pred == labels).float().mean().item()
    return {
        "acc": 100 * acc,
        "conf": 100 * conf.mean().item(),
        "gap": 100 * (conf.mean().item() - acc),
        "range": (logits.max(dim=-1).values - logits.min(dim=-1).values).mean().item(),
        "margin_top2": (top5[:, 0] - top5[:, 1]).mean().item(),
        "margin_top5": (top5[:, 0] - top5[:, 1:].mean(dim=-1)).mean().item(),
        "top1_minus_median": (top5[:, 0] - logits.median(dim=-1).values).mean().item(),
        "std": logits.std(dim=-1).mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["imagenet_r", "imagenet_a"])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "tpt_margins.csv")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent / "data")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(rg.MODEL_NAME, pretrained="openai")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(rg.MODEL_NAME)
    logit_scale = model.logit_scale.exp().item()
    for p in model.parameters():
        p.requires_grad_(False)
    text_encoder = rg.TextEncoderWrapper(model)
    grid = pd.read_csv(rg.RESULTS)

    rows = []
    for name in args.datasets:
        f = load_features(name, device=device)
        refs, _ = load_manifest(name)
        single_text = rg.single_template_text(model, tokenizer, f["class_names"], device)
        _, _, test_idx = split_indices(
            f["labels"].tolist(), seed=SEED, n_cache=f["n_shot"], n_val=f["n_val"])
        sample = random.Random(SEED).sample(test_idx, min(N, len(test_idx)))
        labels = f["labels"][sample]

        base_logits = zero_shot_logits(f["image_features"][sample], single_text, logit_scale).cpu()
        torch.manual_seed(RUN)
        prompt_learner = rg.PromptLearner(model, device, 4, tokenizer, rg.CTX_DIM,
                                          f["class_names"]).to(device)
        _, tpt_logits = run_tpt(model, prompt_learner, text_encoder, preprocess,
                                rg.ManifestImages([refs[i] for i in sample], args.data_root),
                                labels.tolist(), device, rg.TPT_AUGMENT, rg.METRICS,
                                subset=len(sample), return_logits=True, **rg.TPT_CFG)

        for method, logits in [("zero_shot_single", base_logits), ("tpt", tpt_logits.cpu())]:
            r = describe(logits, labels)
            ref = grid[(grid.method == method) & (grid.dataset == name) & (grid.seed == SEED)
                       & (grid.n_test == len(sample))]
            if method == "tpt":
                ref = ref[ref.run == RUN]
            assert len(ref) == 1, (method, name, len(ref))
            d_acc = abs(r["acc"] - ref.accuracy.item())
            d_gap = abs(r["gap"] - ref.signed_gap.item())
            print(f"{name} {method}: acc {r['acc']:.2f} (grid {ref.accuracy.item():.2f}), "
                  f"gap {r['gap']:+.2f} (grid {ref.signed_gap.item():+.2f})")
            assert d_acc <= 0.5 and d_gap <= 0.5, f"{name} {method} does not reproduce the grid"
            rows.append({"dataset": name, "method": method, "n": len(sample),
                         "acc_diff_vs_grid": d_acc, "gap_diff_vs_grid": d_gap, **r})

    pd.DataFrame(rows).round(4).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
