"""Adds brier, nll and t_star to the existing rows of results/grid.csv.

Rebuilds each row's logits the way run_grid.py made them, checks the stored metrics,
and fills only the new columns. Every other value is kept as written.
Default: every cached-feature row and the single-template baselines.
--tpt: reruns TPT (run 0) on the cells with n_test != 500 and fills tpt and tpt_sals.

Usage:
    python scripts/backfill_metrics.py
    python scripts/backfill_metrics.py --tpt
"""

import argparse, copy, csv, io, math, random, sys
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_grid as rg
from clap import raw_text_norms, train_clap
from features_registry import DATASETS, load_features
from harness import evaluate, fit_temperature, run_comparison, run_tpt, sals, zero_shot_logits
from image_sources import load_manifest
from imagenet_classes import IMAGENET_TEMPLATES
from splits import split_indices
from tip_adapter_f import train_tip_adapter_f

NEW = ["brier", "nll", "t_star"]
CHECK = ["accuracy", "ece", "signed_gap", "logit_range"]


def read_grid():
    with open(rg.RESULTS, newline="") as fh:
        text = fh.read()
    reader = csv.DictReader(io.StringIO(text, newline=""))
    return text, reader.fieldnames, list(reader)


def serialize(fields, rows):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields, restval="")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def key(method, dataset, seed, n_test, beta="", run=""):
    beta = "" if beta in ("", None) else float(beta)
    return (method, dataset, int(seed), int(n_test), beta, str(run))


def index(rows):
    idx = {}
    for r in rows:
        k = key(r["method"], r["dataset"], r["seed"], r["n_test"], r["beta"], r["run"])
        assert k not in idx, f"duplicate row {k}"
        idx[k] = r
    return idx


def fill(row, metrics, tol, filled):
    n_test = int(row["n_test"])
    for k in CHECK:
        limit = 100 / n_test + 1e-6 if k == "accuracy" else tol
        diff = abs(metrics[k] - float(row[k]))
        assert diff <= limit, f"{row['method']} {row['dataset']} seed {row['seed']}: {k} differs by {diff:.4f}"
    for k in NEW:
        v = metrics[k]
        row[k] = "" if math.isnan(v) else round(v, 4)
    filled.append(id(row))


def coop_ctx(name, seed, f):
    path = rg.CTX_DIR / f"coop_ctx_{name}_seed{seed}.pt"
    assert path.exists(), f"missing {path}"
    saved = torch.load(path)
    assert saved["class_names"] == list(f["class_names"]), f"{path.name}: class order differs"
    return saved["ctx"]


def backfill_cached(name, f, idx, model, tokenizer, logit_scale, device, filled):
    norms, _ = raw_text_norms(model, tokenizer, f["class_names"], IMAGENET_TEMPLATES, device)
    seeds = sorted(k[2] for k in idx if k[0] == "zero_shot" and k[1] == name)
    for seed in seeds:
        cache_idx, _, test_idx = split_indices(
            f["labels"].tolist(), seed=seed, n_cache=f["n_shot"], n_val=f["n_val"]
        )
        n_test = len(test_idx)
        cache_labels = f["labels"][cache_idx]
        shot_feats = f["image_features"][cache_idx]
        shot_labels = cache_labels.to(device)
        shot_logits = zero_shot_logits(shot_feats, f["text_features"], logit_scale)
        n_shot_errors = (shot_logits.argmax(dim=-1) != shot_labels).sum().item()

        cache_values = F.one_hot(cache_labels, num_classes=f["n_classes"]).float().to(device)
        f_keys, _ = train_tip_adapter_f(
            shot_feats, cache_values, f["text_features"], logit_scale,
            seed=seed, device=device, **rg.TIP_F_CFG,
        )
        shared = {
            "test_features": f["image_features"][test_idx],
            "labels": f["labels"][test_idx].to(device),
            "text_features": f["text_features"],
            "logit_scale": logit_scale,
            "cache_keys": shot_feats,
            "cache_values": cache_values,
            "f_cache_keys": f_keys,
            "coop_text_features": rg.coop_text_features(
                model, tokenizer, f["class_names"], coop_ctx(name, seed, f), device,
                n_ctx=rg.COOP_CFG["n_ctx"],
            ),
        }
        results = run_comparison(shared, rg.METHODS, rg.METRICS)
        for k, r in results.items():
            spec = rg.METHODS[k]
            row = idx[key(spec["method"], name, seed, n_test, spec["params"].get("beta", ""))]
            fill(row, r, 1e-3, filled)

        ts_row = idx[key("zero_shot_ts", name, seed, n_test)]
        if n_shot_errors > 0:
            T = fit_temperature(shot_logits, shot_labels)
            assert abs(T - float(ts_row["temperature"])) < 1e-3, f"{name} seed {seed}: T differs"
            ts = evaluate(zero_shot_logits(**shared) / T, shared["labels"], rg.METRICS)
            fill(ts_row, ts, 1e-3, filled)

        protos, _ = train_clap(
            shot_feats, shot_labels, shot_logits, f["text_features"], norms,
            logit_scale, seed=seed, device=device, **rg.CLAP_CFG,
        )
        clap = evaluate(zero_shot_logits(shared["test_features"], protos, logit_scale),
                        shared["labels"], rg.METRICS)
        fill(idx[key("clap", name, seed, n_test)], clap, 1e-3, filled)
        print(f"{name:13s} seed {seed}  cached rows reproduce")

    single_text = rg.single_template_text(model, tokenizer, f["class_names"], device)
    for k, row in idx.items():
        if k[0] != "zero_shot_single" or k[1] != name:
            continue
        _, _, test_idx = split_indices(
            f["labels"].tolist(), seed=k[2], n_cache=f["n_shot"], n_val=f["n_val"]
        )
        sample = random.Random(k[2]).sample(test_idx, k[3])
        logits = zero_shot_logits(f["image_features"][sample], single_text, logit_scale)
        fill(row, evaluate(logits, f["labels"][sample].to(device), rg.METRICS), 1e-3, filled)
    print(f"{name:13s} single-template baselines reproduce")


def backfill_tpt(name, f, idx, model, tokenizer, preprocess, logit_scale, device, data_root, filled):
    refs, _ = load_manifest(name)
    for p in model.parameters():
        p.requires_grad_(False)
    single_text = rg.single_template_text(model, tokenizer, f["class_names"], device)
    text_encoder = rg.TextEncoderWrapper(model)
    cells = [k for k in idx if k[0] == "tpt" and k[1] == name and k[3] != 500 and k[5] == "0"]
    for k in cells:
        seed, n_test = k[2], k[3]
        _, _, test_idx = split_indices(
            f["labels"].tolist(), seed=seed, n_cache=f["n_shot"], n_val=f["n_val"]
        )
        sample = random.Random(seed).sample(test_idx, n_test)
        labels = f["labels"][sample]
        base_logits = zero_shot_logits(f["image_features"][sample], single_text, logit_scale)
        torch.manual_seed(0)
        prompt_learner = rg.PromptLearner(model, device, 4, tokenizer, rg.CTX_DIM,
                                          f["class_names"]).to(device)
        r, tpt_logits = run_tpt(
            model, prompt_learner, text_encoder, preprocess,
            rg.ManifestImages([refs[i] for i in sample], data_root), labels.tolist(),
            device, rg.TPT_AUGMENT, rg.METRICS, subset=n_test, return_logits=True, **rg.TPT_CFG,
        )
        fill(idx[k], r, 0.05, filled)
        r_sals = evaluate(sals(tpt_logits, base_logits.cpu()), labels.cpu(), rg.METRICS)
        fill(idx[("tpt_sals",) + k[1:]], r_sals, 0.05, filled)
        print(f"{name:13s} seed {seed}  TPT reproduces")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpt", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent / "data")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(rg.MODEL_NAME, pretrained="openai")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(rg.MODEL_NAME)
    logit_scale = model.logit_scale.exp().item()

    text, fields, rows = read_grid()
    assert serialize(fields, rows) == text, "grid.csv does not round-trip; refusing to write"
    before = copy.deepcopy(rows)
    out_fields = list(fields)
    if "brier" not in out_fields:
        at = out_fields.index("logit_range") + 1
        out_fields[at:at] = NEW
    idx = index(rows)

    filled = []
    for name in args.datasets:
        f = load_features(name, device=device)
        if args.tpt:
            backfill_tpt(name, f, idx, model, tokenizer, preprocess, logit_scale, device,
                         args.data_root, filled)
        else:
            backfill_cached(name, f, idx, model, tokenizer, logit_scale, device, filled)

    targeted = set(filled)
    for old, new in zip(before, rows):
        for k in fields:
            if k in NEW and id(new) in targeted:
                continue
            assert new.get(k, "") == old.get(k, ""), f"{k} changed in {new['method']} {new['dataset']}"

    with open(rg.RESULTS, "w", newline="") as fh:
        fh.write(serialize(out_fields, rows))
    blank = {}
    for r in rows:
        if r.get("brier", "") == "":
            blank[r["method"]] = blank.get(r["method"], 0) + 1
    print(f"filled {len(targeted)} rows; still blank: {blank or 'none'}")


if __name__ == "__main__":
    main()
