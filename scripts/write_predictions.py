"""Writes the forecast for a backbone's grid before any adaptation runs.

Usage:
    VLM_BACKBONE=rn50 python scripts/write_predictions.py
    python scripts/write_predictions.py --out /tmp/predictions_check.csv
"""

import argparse, csv, sys
from pathlib import Path

import open_clip
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")]

import run_grid as rg
from backbones import BACKBONE
from score_forecast import KIND, forecast_better

METHODS = {m: k for m, k in KIND.items() if m != "tpt"}


def shot_estimates(name, f, logit_scale):
    rows = []
    for seed in rg.SEEDS.get(name, rg.DEFAULT_SEEDS):
        cache_idx, _, _ = rg.split_indices(
            f["labels"].tolist(), seed=seed, n_cache=f["n_shot"], n_val=f["n_val"]
        )
        logits = rg.zero_shot_logits(f["image_features"][cache_idx], f["text_features"], logit_scale)
        labels = f["labels"][cache_idx]
        rows.append({
            "dataset": name,
            "seed": seed,
            "g_shots": round(rg.signed_gap(logits, labels), 4),
            "n_shot_errors": int((logits.argmax(dim=-1) != labels).sum().item()),
        })
    return rows


def check_against_grid(estimates):
    if not rg.RESULTS.exists():
        return 0
    with open(rg.RESULTS, newline="") as fh:
        grid = {
            (r["dataset"], int(r["seed"])): r
            for r in csv.DictReader(fh)
            if r["method"] == "zero_shot"
        }
    for e in estimates:
        r = grid[(e["dataset"], e["seed"])]
        assert abs(float(r["g_shots"]) - e["g_shots"]) < 1e-3, (e, r["g_shots"])
        assert int(r["n_shot_errors"]) == e["n_shot_errors"], (e, r["n_shot_errors"])
    return len(estimates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / f"predictions_{BACKBONE}.csv")
    args = parser.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} exists; predictions are written once")
    commit = rg.git_commit()
    if commit.endswith("-dirty"):
        sys.exit(f"commit {commit}: src/ or scripts/ differ from HEAD; commit first")

    torch.manual_seed(42)
    model, _, _ = open_clip.create_model_and_transforms(rg.MODEL_NAME, pretrained="openai")
    logit_scale = model.logit_scale.exp().item()

    estimates = []
    for name in rg.DATASETS:
        estimates += shot_estimates(name, rg.load_features(name), logit_scale)
    checked = check_against_grid(estimates)

    rows = []
    for e in estimates:
        for method, kind in METHODS.items():
            if method == "zero_shot_ts" and e["n_shot_errors"] == 0:
                continue
            better = forecast_better(kind, e["g_shots"])
            rows.append({
                "method": method,
                "kind": kind,
                **e,
                "predicted_ece": "better" if better else "worse",
                "model_name": rg.MODEL_NAME,
                "git_commit": commit,
            })

    args.out.parent.mkdir(exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{rg.MODEL_NAME}  commit {commit}  logit scale {logit_scale:.2f}")
    for e in estimates:
        print(f"  {e['dataset']:13s} seed {e['seed']}  g_shots {e['g_shots']:+6.2f}  shot errors {e['n_shot_errors']}")
    worse = sum(r["predicted_ece"] == "worse" for r in rows)
    print(f"{len(rows)} predictions ({worse} worse), g_shots checked against {checked} grid rows")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
