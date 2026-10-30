# vlm-reliability

This repo analyzes what happens to CLIP's confidence when we adapt it to a shifted domain
with a few labeled images.

I ran seven adaptation methods on nine distribution-shift test sets, on two CLIP backbones
(ViT-B/16 and ResNet-50), through one evaluation harness. Every run is one row in a results grid
with accuracy, ECE, the signed calibration gap, Brier score and NLL.

## What the grids show

Signed gap = mean confidence − accuracy, in percentage points. Positive means overconfident,
negative means underconfident. I use it alongside ECE because ECE has no sign: it can't tell us
whether a method made the model too confident or not confident enough.

**1. Zero-shot CLIP is not always overconfident under shift.** It starts underconfident on some
test sets and overconfident on others, and the sign moves with the class count and the backbone.
Sketch-200 and Sketch-1000 are the same images scored against 200 and 1,000 classes: ViT-B/16 is
underconfident on the first (−2.27) and overconfident on the second (+4.58). PACS sketch flips
between backbones (ViT +3.17, RN50 −0.72).

**2. Methods fall into two kinds, and we have to measure which.**

- _Boosters_ raise confidence faster than accuracy, from any start: Tip-Adapter (gap rose in all
  31 cells on both backbones), Tip-Adapter-F, and TPT (all 9 test sets).
- _Correctors_ push the gap toward zero from either side: CoOp, and temperature scaling fitted on
  the few-shot images.

**3. The starting sign decides whether a booster helps.** From an overconfident start,
Tip-Adapter made ECE worse every time (12/12 cells on ViT, 9/9 on RN50), even while accuracy went
up. The worst case is Sketch-1000 on ViT: accuracy 48.3 → 55.8, ECE 4.6 → 12.1. From an
underconfident start a booster helps until it overshoots, roughly once the boost passes twice the
starting gap.

**4. The rule holds as a forecast.** Estimate the starting gap on the labeled shots (one zero-shot
pass, no training), look up the method's kind, and predict whether ECE improves:

| Backbone | Cells called correctly | How                                                              |
| -------- | ---------------------- | ---------------------------------------------------------------- |
| ViT-B/16 | 119 / 128              | scored after the grid was run                                    |
| RN50     | 104 / 123              | predictions committed (`6bf6817`) before any RN50 adaptation ran |

It predicts the direction of the change, not its size, and it needs a few wrong predictions among
the shots to estimate the gap.

### ECE by test set (ViT-B/16, mean over seeds)

| Test set     | Zero-shot gap | Zero-shot | Temp. scaling | Tip-Adapter | Tip-Adapter-F | CoOp |
| ------------ | ------------: | --------: | ------------: | ----------: | ------------: | ---: |
| ImageNet-R   |         −3.81 |      3.82 |          1.03 |        0.84 |          0.94 | 0.93 |
| PACS art     |         −2.36 |      2.62 |          0.77 |        1.11 |          0.82 | 0.59 |
| Sketch-200   |         −2.27 |      2.55 |          1.16 |        2.67 |          1.03 | 1.14 |
| PACS cartoon |         −2.11 |      2.11 |          0.27 |        0.86 |          0.53 | 0.53 |
| PACS photo   |         −1.22 |      1.22 |             – |        0.12 |          0.13 | 0.12 |
| ImageNet-V2  |         +1.97 |      2.94 |          2.73 |        4.61 |          4.20 | 2.64 |
| PACS sketch  |         +3.17 |      3.19 |          3.52 |        5.89 |          3.28 | 1.87 |
| Sketch-1000  |         +4.58 |      4.58 |          1.35 |       12.13 |          7.93 | 0.87 |
| ImageNet-A   |         +6.85 |      6.92 |          3.39 |        9.34 |          7.83 | 4.69 |

Temperature scaling is blank on PACS photo because zero-shot gets every shot right, so there is
nothing to fit. Accuracy, the RN50 numbers, CLAP, TPT and SaLS are all in the grids.

## Setup

Both backbones use OpenAI weights through open_clip, frozen, with QuickGELU
(`ViT-B-16-quickgelu`, `RN50-quickgelu`). The plain open_clip configs use standard GELU, which
doesn't match those weights; `results/gelu/` keeps the earlier runs that had the mismatch.

| Method              | What it adapts                                                                                                                | Labels   |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------- | -------- |
| Zero-shot           | nothing; 80-template prompt ensemble                                                                                          | none     |
| Temperature scaling | one temperature, fitted on zero-shot logits of the shots                                                                      | few-shot |
| Tip-Adapter         | training-free cache of shot features; α = 1.5, β = 5 (and β = 1)                                                              | few-shot |
| Tip-Adapter-F       | the same cache, keys trained; AdamW, 20 epochs                                                                                | few-shot |
| CoOp                | 4 context tokens from "a photo of a"; Adam, lr 0.002, 10 epochs                                                               | few-shot |
| CLAP                | linear probe started at the zero-shot classifier, L2 pull back to it                                                          | few-shot |
| TPT                 | the 4 tokens, reset per image; one step on 63 augmented views (ViT only)                                                      | none     |
| SaLS                | test-time rescaling of each image's logits to its zero-shot range; applied on top of Tip-Adapter, Tip-Adapter-F, CoOp and TPT | none     |

| Test set                                      | Classes | Shots | Test images | Seeds |
| --------------------------------------------- | ------: | ----: | ----------: | ----: |
| ImageNet-R                                    |     200 |    16 |      24,800 |     5 |
| Sketch-200 (ImageNet-Sketch, R's classes)     |     200 |    16 |       4,952 |     5 |
| Sketch-1000 (ImageNet-Sketch, all classes)    |    1000 |    16 |      24,889 |     3 |
| ImageNet-V2 (matched frequency)               |    1000 |     4 |       6,000 |     3 |
| ImageNet-A (the 101 classes with ≥ 26 images) |     101 |    16 |       3,403 |     3 |
| PACS photo / art / cartoon / sketch           |       7 |    16 | 1,488–3,747 |     3 |

Shots come from the target domain itself. Each seed draws its own
per-class split: shots, 10 validation images (none for V2, which only has 10 per class), the rest
test. The same seed gives the same split on both backbones. TPT runs on up to 2,000 test images of
seed 42, against a single-prompt zero-shot baseline on the same images; ImageNet-R, Sketch-200 and
ImageNet-A have two extra runs that change only the augmentation seed.

## Running it

You need a GPU for encoding and for the grid. CoOp and TPT on the 1000-class sets peak at about
28 GB (I used a 32 GB RTX 4080 SUPER); everything else fits in about 6 GB. The analysis scripts
run on CPU from the CSVs alone.

**Install** (Python 3.12). Install torch and torchvision for your CUDA version first, then:

```bash
pip install open_clip_torch datasets requests tqdm pillow numpy pandas matplotlib
```

**Data and features**

```bash
python scripts/download_data.py                        # images into ../data (--data-root to change)
python scripts/encode_features.py                      # ViT-B/16 features into features/
VLM_BACKBONE=rn50 python scripts/encode_features.py    # RN50 features into features/rn50/
```

Datasets come from Hugging Face (set `HF_ENDPOINT` to use a mirror) and PACS from Kaggle.
`download_data.py` checks that every image the manifests name is on disk. If a feature cache
already exists, `encode_features.py` re-encodes and compares instead of overwriting, so it doubles
as a check.

**Grid**

```bash
python scripts/run_grid.py            # zero-shot, TS, Tip-Adapter, Tip-Adapter-F, CoOp, SaLS
python scripts/run_grid.py --tpt      # TPT and its single-prompt baseline
python scripts/run_grid.py --clap
VLM_BACKBONE=rn50 python scripts/run_grid.py
VLM_BACKBONE=rn50 python scripts/run_grid.py --clap
python scripts/backfill_metrics.py    # Brier, NLL, T* (add --tpt for the TPT rows)
```

`VLM_BACKBONE` picks the model, the feature folder and the grid file (`src/backbones.py`). Rows
merge into the grid by (dataset, seed, method, n_test, run, beta), so you can rerun a piece without
touching the rest:

```bash
python scripts/run_grid.py --datasets sketch_200 --seeds 42
```

**Analysis (CPU)**

```bash
python scripts/score_forecast.py                       # ViT forecast, asserts 119/128, two figures
VLM_BACKBONE=rn50 python scripts/score_predictions.py  # scores the committed RN50 predictions
python scripts/compare_ts_coop.py                      # temperature scaling vs CoOp
python scripts/seed_intervals.py                       # seed-to-seed spread of every change
python scripts/decompose_tip_f.py                      # zero-shot part vs cache part of Tip-Adapter(-F)
```

`write_predictions.py` is the script that wrote the RN50 forecast before the RN50 grid existed. It
refuses to overwrite an existing prediction file or to run from a dirty tree.

## Layout

```
src/
  harness.py            methods as logits; metrics as (logits, labels) -> number; TPT loop
  backbones.py          VLM_BACKBONE -> model, feature folder, grid file
  coop.py, tip_adapter.py, tip_adapter_f.py, tpt.py, clap.py
  clip_zeroshot.py      text and image feature builders
  features_registry.py  the 9 test sets: caches, class names, shots
  image_sources.py      image order per dataset, manifests, image encoding
  splits.py             seeded per-class shot / validation / test split
  attention.py, vit.py, nanovlm.py   from-scratch learning code, not used by the grid
scripts/                download, encode, run_grid, backfill, analysis (above)
manifests/              cached feature index -> source image, one CSV per test set
results/
  grid.csv              ViT-B/16, every reported number
  grid_rn50.csv         RN50
  predictions_rn50.csv  the RN50 forecast, committed before the RN50 runs
  gelu/                 old runs from before the QuickGELU fix
notebooks/              one per step of building this; numbers before notebook 13 predate the fix
```

Every method returns logits of shape `(N, num_classes)` and every metric takes logits and labels,
so adding a metric never touches a method. The manifests exist because some early caches were built
through a shuffled loader; `build_manifest.py` recovered the order by re-encoding every image and
matching it to its cached feature, so encoding in manifest order reproduces the caches and every
split.
