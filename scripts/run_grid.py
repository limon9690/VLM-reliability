"""Runs every (dataset, seed) cell and merges the rows into results/grid.csv.

Default: zero-shot, temperature scaling, Tip-Adapter (β=5, β=1), CoOp,
Tip-Adapter-F, and SaLS on Tip-Adapter, CoOp and Tip-Adapter-F, from cached features.
--tpt: single-template zero-shot baseline, TPT and TPT+SaLS on --tpt-n test images per
seed (default 2,000, or the whole test split if smaller), one row per
augmentation run in --tpt-runs, images opened through manifests/<dataset>.csv.
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
    fit_temperature,
    logit_range,
    run_comparison,
    sals,
    run_tpt,
    tip_adapter_f_logits,
    signed_gap,
    tip_adapter_logits,
    with_sals,
    zero_shot_logits,
)
from image_sources import load_manifest, open_image
from splits import split_indices
from tip_adapter_f import train_tip_adapter_f

SEEDS = {"imagenet_r": [42, 43, 44, 45, 46], "sketch_200": [42, 43, 44, 45, 46]}
DEFAULT_SEEDS = [42, 43, 44]
RESULTS = REPO_ROOT / "results" / "grid.csv"
CTX_DIR = REPO_ROOT / "features" / "coop"

TPT_SEEDS = [42]
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
# official Tip-Adapter-F training settings; fixed alpha/beta; leave-one-out (see tip_adapter_f.py)
TIP_F_CFG = {
    "alpha": ALPHA,
    "beta": 5.0,
    "epochs": 20,
    "lr": 1e-3,
    "eps": 1e-4,
    "batch_size": 256,
    "leave_one_out": True,
}

METRICS = {
    "accuracy": accuracy,
    "ece": ece,
    "signed_gap": signed_gap,
    "logit_range": logit_range,
}
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
    "tip_adapter_f": {
        "method": "tip_adapter_f",
        "fn": tip_adapter_f_logits,
        "params": TIP_F_CFG,
    },
    # SaLS: each method's logits rescaled to zero-shot's per-image range
    "tip_adapter_sals": {
        "method": "tip_adapter_sals",
        "fn": with_sals(tip_adapter_logits),
        "params": {"alpha": ALPHA, "beta": 5.0},
    },
    "coop_sals": {
        "method": "coop_sals",
        "fn": with_sals(coop_logits),
        "params": COOP_CFG,
    },
    "tip_adapter_f_sals": {
        "method": "tip_adapter_f_sals",
        "fn": with_sals(tip_adapter_f_logits),
        "params": TIP_F_CFG,
    },
}
SALS_OF = {  # SaLS row -> the row it rescales
    "tip_adapter_sals": "tip_adapter_b5",
    "coop_sals": "coop",
    "tip_adapter_f_sals": "tip_adapter_f",
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
    """Replaces rows for the same (dataset, seed, method, n_test); keeps the rest.
    n_test is in the key so TPT rows at different sample sizes coexist.
    """

    def key(r):
        return (r["dataset"], int(r["seed"]), r["method"], int(r["n_test"]))

    keys = {key(r) for r in new_rows}
    old = []
    if RESULTS.exists():
        with open(RESULTS, newline="") as fh:
            old = [r for r in csv.DictReader(fh) if key(r) not in keys]
    fields = list(new_rows[0])
    for r in old:
        fields += [k for k in r if k not in fields]
    RESULTS.parent.mkdir(exist_ok=True)
    with open(RESULTS, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(old + new_rows)
    print(f"wrote {len(new_rows)} rows ({len(old)} kept) to {RESULTS}")


def make_row(
    method,
    name,
    f,
    seed,
    n_test,
    metrics,
    params,
    gpu,
    commit,
    run="",
    g_shots="",
    n_shot_errors="",
    temperature="",
):
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
        "g_shots": round(g_shots, 4) if g_shots != "" else "",
        "n_shot_errors": n_shot_errors,
        "temperature": round(temperature, 4) if temperature != "" else "",
        "model_name": MODEL_NAME,
        "gpu": gpu,
        "config_hash": config_hash(cfg),
        "git_commit": commit,
    }


class ManifestImages:
    """Opens images lazily, so run_tpt never holds the sample's full-size images in memory."""

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
    n,
    runs,
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
        cache_idx, _, test_idx = split_indices(
            f["labels"].tolist(), seed=seed, n_cache=f["n_shot"], n_val=f["n_val"]
        )

        # starting gap estimated on the labeled shots, with TPT's own starting prompt
        shot_logits = zero_shot_logits(
            f["image_features"][cache_idx], single_text, logit_scale
        )
        shot_labels = f["labels"][cache_idx].to(device)
        g_shots = signed_gap(shot_logits, shot_labels)
        n_shot_errors = (shot_logits.argmax(dim=-1) != shot_labels).sum().item()
        sample = random.Random(seed).sample(test_idx, min(n, len(test_idx)))
        labels = f["labels"][sample]

        base_logits = zero_shot_logits(
            f["image_features"][sample], single_text, logit_scale
        )
        base = evaluate(base_logits, labels.to(device), METRICS)

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
                g_shots=g_shots,
                n_shot_errors=n_shot_errors,
            )
        )

        images = ManifestImages([refs[i] for i in sample], data_root)
        for run in runs:
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
                return_logits=True,
                **TPT_CFG,
            )
            r, tpt_logits = r
            r.pop("n")
            # SaLS on TPT: rescaled to the single-template zero-shot range (TPT's own start)
            r_sals = evaluate(
                sals(tpt_logits, base_logits.cpu()), labels.cpu(), METRICS
            )

            # an affine rescale can't change the argmax; float ties may flip one image
            assert (
                abs(r_sals["accuracy"] - r["accuracy"]) <= 100 / len(sample) + 1e-6
            ), "SaLS changed TPT accuracy"

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
                    g_shots=g_shots,
                    n_shot_errors=n_shot_errors,
                )
            )

            rows.append(
                make_row(
                    "tpt_sals",
                    name,
                    f,
                    seed,
                    len(sample),
                    r_sals,
                    {**TPT_CFG, "aug_seed": run},
                    gpu,
                    commit,
                    run=run,
                    g_shots=g_shots,
                    n_shot_errors=n_shot_errors,
                )
            )

            mem = (
                f"  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
                if device.type == "cuda"
                else ""
            )
            print(
                f"{name:13s} seed {seed} run {run}  g_shots {g_shots:+.2f}  "
                f"base gap {base['signed_gap']:+.2f}  "
                f"TPT gap {r['signed_gap']:+.2f}  Δ {r['signed_gap'] - base['signed_gap']:+.2f}  "
                f"SaLS gap {r_sals['signed_gap']:+.2f}  "
                f"ECE {base['ece']:.2f} → {r['ece']:.2f} (SaLS {r_sals['ece']:.2f})  "
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
    p.add_argument(
        "--tpt-n",
        type=int,
        default=2000,
        help="TPT test images per seed (capped at the test split size)",
    )
    p.add_argument(
        "--tpt-runs",
        nargs="+",
        type=int,
        default=[0],
        help="TPT augmentation seeds, one row each",
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
                    args.tpt_n,
                    args.tpt_runs,
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

            # starting gap estimated on the labeled shots (zero-shot never trains on them)
            shot_logits = zero_shot_logits(
                f["image_features"][cache_idx], f["text_features"], logit_scale
            )
            shot_labels = cache_labels.to(device)
            g_shots = signed_gap(shot_logits, shot_labels)
            n_shot_errors = (shot_logits.argmax(dim=-1) != shot_labels).sum().item()

            ctx = load_or_train_ctx(name, seed, f, cache_idx, model, tokenizer, device)

            cache_values = (
                F.one_hot(cache_labels, num_classes=f["n_classes"]).float().to(device)
            )
            f_keys, f_stats = train_tip_adapter_f(
                f["image_features"][cache_idx],
                cache_values,
                f["text_features"],
                logit_scale,
                seed=seed,
                device=device,
                **TIP_F_CFG,
            )

            shared = {
                "test_features": f["image_features"][test_idx],
                "labels": f["labels"][test_idx].to(device),
                "text_features": f["text_features"],
                "logit_scale": logit_scale,
                "cache_keys": f["image_features"][cache_idx],
                "cache_values": cache_values,
                "f_cache_keys": f_keys,
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
            for s_key, base_key in SALS_OF.items():
                assert (
                    abs(results[s_key]["accuracy"] - results[base_key]["accuracy"])
                    <= 100 / len(test_idx) + 1e-6
                ), f"{s_key} changed accuracy"
                assert (
                    abs(
                        results[s_key]["logit_range"]
                        - results["zero_shot"]["logit_range"]
                    )
                    < 1e-3
                ), f"{s_key} range differs from zero-shot"

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
                        g_shots=g_shots,
                        n_shot_errors=n_shot_errors,
                    )
                )

            # temperature scaling, fitted on the same labeled shots CoOp trains on.
            # No finite fit when every shot is correct: the row is written with
            # n_shot_errors 0 and blank metrics.
            if n_shot_errors > 0:
                T = fit_temperature(shot_logits, shot_labels)
                ts_logits = zero_shot_logits(**shared) / T
                ts = evaluate(ts_logits, shared["labels"], METRICS)
                assert (
                    abs(ts["accuracy"] - results["zero_shot"]["accuracy"])
                    <= 100 / len(test_idx) + 1e-6
                ), "TS changed accuracy"
            else:
                T, ts = "", {}
            rows.append(
                make_row(
                    "zero_shot_ts",
                    name,
                    f,
                    seed,
                    len(test_idx),
                    ts,
                    {},
                    gpu,
                    commit,
                    g_shots=g_shots,
                    n_shot_errors=n_shot_errors,
                    temperature=T,
                )
            )

            zs = results["zero_shot"]["signed_gap"]

            ts_msg = (
                f"T {T:.3f} ΔTS {ts['signed_gap'] - zs:+.2f}"
                if ts
                else "TS: no shot errors"
            )

            mem = (
                f"  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
                if device.type == "cuda"
                else ""
            )
            print(
                f"{name:13s} seed {seed}  g_shots {g_shots:+.2f}  ZS gap {zs:+.2f}  "
                f"Δβ5 {results['tip_adapter_b5']['signed_gap'] - zs:+.2f}  "
                f"Δβ1 {results['tip_adapter_b1']['signed_gap'] - zs:+.2f}  "
                f"ΔCoOp {results['coop']['signed_gap'] - zs:+.2f}  "
                f"ΔTA-F {results['tip_adapter_f']['signed_gap'] - zs:+.2f}  "
                f"{ts_msg}  ({time.time() - t0:.0f}s){mem}"
            )

            print(
                "  SaLS Δ "
                + "  ".join(
                    f"{s_key} {results[s_key]['signed_gap'] - zs:+.2f}"
                    for s_key in SALS_OF
                )
            )

            print(
                f"  TA-F loss {f_stats['loss_start']:.4f} → {f_stats['loss_end']:.4f}  "
                f"key shift {f_stats['key_shift']:.4f}  ({f_stats['seconds']:.0f}s)"
            )

        write_rows(rows)


if __name__ == "__main__":
    main()
