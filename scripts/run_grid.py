"""Runs every (dataset, seed) cell and merges the rows into results/grid.csv.

Default: zero-shot, Tip-Adapter (β=5, β=1) and CoOp from cached features.
--tpt: single-template zero-shot baseline and TPT (2 augmentation runs) on
500 test images per seed, opened through manifests/<dataset>.csv.
"""

import argparse, csv, hashlib, json, random, subprocess, sys, time
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from clip_zeroshot import MODEL_NAME
from coop import (
    CTX_DIM,
    PromptLearner,
    TextEncoderWrapper,
    coop_text_features,
    save_coop_ctx,
    train_coop,
)
from features_registry import DATASETS, load_features
from harness import (
    accuracy,
    coop_logits,
    ece,
    evaluate,
    run_comparison,
    run_tpt,
    signed_gap,
    tip_adapter_logits,
    zero_shot_logits,
)
from image_sources import load_manifest, open_image
from splits import split_indices

SEEDS = {"imagenet_r": [42, 43, 44, 45, 46], "sketch_200": [42, 43, 44, 45, 46]}
DEFAULT_SEEDS = [42, 43, 44]
RESULTS = REPO_ROOT / "results" / "grid.csv"
CTX_DIR = REPO_ROOT / "features" / "coop"

TPT_SEEDS = [42]
TPT_N = 500
TPT_RUNS = [0, 1]  # augmentation seeds; everything else identical
TPT_CFG = {"n_views": 63, "lr": 0.005, "top_k": 0.1}
SINGLE_TEMPLATE = "a photo of a {}."  # TPT's starting prompt, so the baseline is TPT before any update
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
TPT_AUGMENT = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ]
)

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


def git_commit():
    """Short commit hash, with -dirty if src/ or scripts/ differ from HEAD."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = (
            subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--", "src", "scripts"],
                cwd=REPO_ROOT,
            ).returncode
            != 0
        )
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_rows(new_rows):
    """Replaces rows for the same (dataset, seed, method); keeps the rest."""
    keys = {(r["dataset"], int(r["seed"]), r["method"]) for r in new_rows}
    old = []
    if RESULTS.exists():
        with open(RESULTS, newline="") as fh:
            old = [
                r
                for r in csv.DictReader(fh)
                if (r["dataset"], int(r["seed"]), r["method"]) not in keys
            ]
    fields = list(new_rows[0])
    for r in old:
        fields += [k for k in r if k not in fields]
    RESULTS.parent.mkdir(exist_ok=True)
    with open(RESULTS, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(old + new_rows)
    print(f"wrote {len(new_rows)} rows ({len(old)} kept) to {RESULTS}")


def make_row(method, name, f, seed, n_test, metrics, params, gpu, commit, run=""):
    cfg = {
        "model": MODEL_NAME,
        "dataset": name,
        "method": method,
        "params": params,
        "seed": seed,
        "n_shot": f["n_shot"],
        "n_val": f["n_val"],
    }
    return {
        "method": method,
        "dataset": name,
        "n_classes": f["n_classes"],
        "n_shot": f["n_shot"],
        "seed": seed,
        "run": run,
        "alpha": params.get("alpha", ""),
        "beta": params.get("beta", ""),
        "n_test": n_test,
        **{k: round(v, 4) for k, v in metrics.items()},
        "model_name": MODEL_NAME,
        "gpu": gpu,
        "config_hash": config_hash(cfg),
        "git_commit": commit,
    }


class ManifestImages:
    """Opens images lazily, so run_tpt never holds 500 full-size images in memory."""

    def __init__(self, refs, data_root):
        self.refs = refs
        self.data_root = data_root

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, i):
        return open_image(self.refs[i], self.data_root)


@torch.no_grad()
def single_template_text(model, tokenizer, class_names, device):
    """(512, n_classes), the same layout as the cached ensembled text features."""
    tokens = tokenizer([SINGLE_TEMPLATE.format(c) for c in class_names]).to(device)
    return F.normalize(model.encode_text(tokens), dim=-1).T


def run_tpt_cells(
    name,
    f,
    seeds,
    model,
    tokenizer,
    preprocess,
    logit_scale,
    device,
    gpu,
    commit,
    data_root,
):
    refs, manifest_labels = load_manifest(name)
    assert (
        manifest_labels == f["labels"].tolist()
    ), f"{name}: manifest labels differ from cache"
    for p in model.parameters():
        p.requires_grad_(False)
    single_text = single_template_text(model, tokenizer, f["class_names"], device)
    text_encoder = TextEncoderWrapper(model)

    rows = []
    for seed in seeds:
        _, _, test_idx = split_indices(
            f["labels"].tolist(), seed=seed, n_cache=f["n_shot"], n_val=f["n_val"]
        )
        sample = random.Random(seed).sample(test_idx, min(TPT_N, len(test_idx)))
        labels = f["labels"][sample]

        base = evaluate(
            zero_shot_logits(f["image_features"][sample], single_text, logit_scale),
            labels.to(device),
            METRICS,
        )
        rows.append(
            make_row(
                "zero_shot_single",
                name,
                f,
                seed,
                len(sample),
                base,
                {"template": SINGLE_TEMPLATE},
                gpu,
                commit,
            )
        )

        images = ManifestImages([refs[i] for i in sample], data_root)
        for run in TPT_RUNS:
            t0 = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(run)
            prompt_learner = PromptLearner(
                model, device, 4, tokenizer, CTX_DIM, f["class_names"]
            ).to(device)
            r = run_tpt(
                model,
                prompt_learner,
                text_encoder,
                preprocess,
                images,
                labels.tolist(),
                device,
                TPT_AUGMENT,
                METRICS,
                subset=len(sample),
                **TPT_CFG,
            )
            r.pop("n")
            rows.append(
                make_row(
                    "tpt",
                    name,
                    f,
                    seed,
                    len(sample),
                    r,
                    {**TPT_CFG, "aug_seed": run},
                    gpu,
                    commit,
                    run=run,
                )
            )
            mem = (
                f"  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
                if device.type == "cuda"
                else ""
            )
            print(
                f"{name:13s} seed {seed} run {run}  base gap {base['signed_gap']:+.2f}  "
                f"TPT gap {r['signed_gap']:+.2f}  Δ {r['signed_gap'] - base['signed_gap']:+.2f}  "
                f"ECE {base['ece']:.2f} → {r['ece']:.2f}  "
                f"acc {base['accuracy']:.2f} → {r['accuracy']:.2f}  ({time.time() - t0:.0f}s){mem}"
            )
    return rows


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS)
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="override the per-dataset seeds (spot checks)",
    )
    p.add_argument(
        "--tpt", action="store_true", help="run the TPT cells instead of the grid"
    )
    p.add_argument("--data-root", type=Path, default=REPO_ROOT.parent / "data")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    commit = git_commit()
    print(f"commit {commit}  device {gpu}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained="openai"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    logit_scale = model.logit_scale.exp().item()

    for name in args.datasets:
        f = load_features(name, device=device)
        if args.tpt:
            seeds = args.seeds or TPT_SEEDS
            write_rows(
                run_tpt_cells(
                    name,
                    f,
                    seeds,
                    model,
                    tokenizer,
                    preprocess,
                    logit_scale,
                    device,
                    gpu,
                    commit,
                    args.data_root,
                )
            )
            continue

        rows = []
        for seed in args.seeds or SEEDS.get(name, DEFAULT_SEEDS):
            t0 = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

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
                rows.append(
                    make_row(
                        spec["method"],
                        name,
                        f,
                        seed,
                        len(test_idx),
                        r,
                        spec["params"],
                        gpu,
                        commit,
                    )
                )

            zs = results["zero_shot"]["signed_gap"]
            mem = (
                f"  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
                if device.type == "cuda"
                else ""
            )
            print(
                f"{name:13s} seed {seed}  ZS gap {zs:+.2f}  "
                f"Δβ5 {results['tip_adapter_b5']['signed_gap'] - zs:+.2f}  "
                f"Δβ1 {results['tip_adapter_b1']['signed_gap'] - zs:+.2f}  "
                f"ΔCoOp {results['coop']['signed_gap'] - zs:+.2f}  "
                f"({time.time() - t0:.0f}s){mem}"
            )

        write_rows(rows)


if __name__ == "__main__":
    main()
