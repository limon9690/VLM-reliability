# vlm-reliability

How adapting CLIP to a new domain changes its calibration, not just its accuracy.

The repo runs four adaptation methods (zero-shot CLIP, Tip-Adapter, CoOp and TPT) on eight
distribution-shift test sets through one evaluation harness, and reports top-1 accuracy, ECE and
the signed calibration gap for every run. All results live in one file, `results/grid.csv`, and
every row records the settings and the commit that produced it.

I work on the reliability and robustness of vision-language models under distribution shift:
adapting models like CLIP to new domains with little or no labeled data. This repo is the working
code behind a paper in progress, so the findings below are current readings rather than final
claims.

## What is measured

**Signed gap** is mean confidence minus accuracy, in percentage points. Positive means the model is
overconfident, negative means underconfident. It is the primary metric here because ECE has no
sign: an underconfident model and an overconfident one can have the same ECE, and an adaptation
method can raise ECE either by overshooting or by moving the wrong way.

**Δ** is the signed gap after adapting minus the signed gap before. Δ > 0 means confidence rose
faster than accuracy (or fell more slowly).

ECE (10 equal-width bins) and top-1 accuracy are reported alongside, from the same logits.

## Setup

Backbone: CLIP ViT-B/16 with OpenAI weights, frozen throughout, loaded through open_clip as
`ViT-B-16-quickgelu`. The plain ViT-B-16 config uses standard GELU, which does not match the OpenAI weights; earlier runs in this repo had that mismatch, and every number here comes from the corrected model.

| Method      | What it adapts                                                                                                              | Labels used             |
| ----------- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| Zero-shot   | nothing; 80-template prompt ensemble                                                                                        | none                    |
| Tip-Adapter | training-free cache of few-shot image features; α = 1.5, β = 5 (headline) and β = 1                                         | few-shot, target domain |
| CoOp        | 4 learned context tokens, initialised from "a photo of a"; Adam, lr 0.002, 10 epochs                                        | few-shot, target domain |
| TPT         | the same 4 tokens, reset for every image; one AdamW step per test image on 63 augmented views, top 10% lowest-entropy views | none                    |

The few-shot methods use labeled images from the target domain itself (16 per class, 4 for
ImageNet-V2), following those methods' benchmark convention. TPT starts from the single prompt
"a photo of a {class}.", so its Δ is measured against a zero-shot baseline with that same prompt
on the same images, not against the 80-template ensemble.

| Test set                                        | Classes | Shots | Test images | Seeds |
| ----------------------------------------------- | ------- | ----- | ----------- | ----- |
| ImageNet-R                                      | 200     | 16    | 24,800      | 5     |
| ImageNet-Sketch, R's 200 classes ("Sketch-200") | 200     | 16    | 4,952       | 5     |
| ImageNet-Sketch, all classes ("Sketch-1000")    | 1000    | 16    | 24,889      | 3     |
| ImageNet-V2 (matched frequency)                 | 1000    | 4     | 6,000       | 3     |
| PACS photo / art painting / cartoon / sketch    | 7       | 16    | 1,488–3,747 | 3     |

Each seed draws its own split per class: the shots, 10 validation images (none for V2, which has
only 10 images per class), and the rest as test. Sketch-200 and Sketch-1000 are the same images
scored against different label sets, which makes them a direct test of what class count alone does.
TPT runs on 500 test images from seed 42, twice, with only the augmentation seed changed.

## Results

Means over seeds. Full per-seed rows are in `results/grid.csv`.

**Change in signed gap after adapting (Δ).** The first column is where zero-shot CLIP starts. TPT
has its own starting point (single prompt, 500 images):

| Test set     | Zero-shot gap | Δ Tip-Adapter (β=5) | Δ CoOp | TPT start | Δ TPT |
| ------------ | ------------- | ------------------- | ------ | --------- | ----- |
| ImageNet-R   | −3.81         | +4.37               | +4.05  | −4.25     | +1.17 |
| PACS art     | −2.36         | +1.36               | +2.65  | −2.42     | −2.47 |
| Sketch-200   | −2.27         | +4.93               | +1.77  | −2.20     | −0.37 |
| PACS cartoon | −2.11         | +1.35               | +1.93  | −2.49     | −0.85 |
| PACS photo   | −1.22         | +1.21               | +1.15  | −1.13     | +0.11 |
| ImageNet-V2  | +1.97         | +2.33               | −0.06  | −1.35     | −0.71 |
| PACS sketch  | +3.17         | +2.72               | −1.45  | +4.98     | −1.95 |
| Sketch-1000  | +4.58         | +7.55               | −4.42  | +4.11     | +3.41 |

**After adapting: accuracy and ECE.**

| Test set     | Accuracy ZS / TA / CoOp | ECE ZS / TA / CoOp  |
| ------------ | ----------------------- | ------------------- |
| ImageNet-R   | 77.79 / 78.69 / 78.97   | 3.82 / 0.84 / 0.93  |
| PACS art     | 97.53 / 97.62 / 97.82   | 2.62 / 1.11 / 0.59  |
| Sketch-200   | 81.72 / 84.28 / 86.05   | 2.55 / 2.67 / 1.14  |
| PACS cartoon | 99.24 / 99.23 / 99.31   | 2.11 / 0.86 / 0.53  |
| PACS photo   | 99.93 / 99.82 / 99.93   | 1.22 / 0.12 / 0.12  |
| ImageNet-V2  | 62.06 / 63.08 / 63.65   | 2.94 / 4.61 / 2.64  |
| PACS sketch  | 90.01 / 90.10 / 92.92   | 3.19 / 5.89 / 1.87  |
| Sketch-1000  | 48.29 / 55.82 / 53.40   | 4.58 / 12.13 / 0.87 |

### What the grid shows so far

1. **The starting point varies in sign, and class count moves it.** Sketch-200 and Sketch-1000
   contain the same images; scored against 200 classes the model is underconfident (−2.27), against
   1000 it is overconfident (+4.58).

2. **Tip-Adapter raises confidence relative to accuracy from every starting point.** Δ > 0 in all
   56 Tip-Adapter rows (8 test sets, both β values, every seed). From an overconfident start this
   always makes calibration worse: on Sketch-1000 ECE goes from 4.58 to 12.13 while accuracy improves
   by 7.5 points. From an underconfident start g it helps only while the boost is smaller than 2|g|;
   past that it overshoots into overconfidence, as on Sketch-200 (start −2.27, Δ +4.93, ECE 2.55 to
   2.67). So the starting gap predicts the outcome, but only together with an estimate of Δ.

3. **CoOp roughly removes the starting gap, from either side.** Its Δ has the opposite sign to the
   start in 26 of 28 rows, and its final gap is within ±0.5 on 6 of the 8 test sets. ImageNet-V2
   (the only 4-shot set) and PACS sketch are the exceptions. The evidence from overconfident starts
   is thin, and CoOp fits on the same labeled data temperature scaling would use; both are under
   Known limits.

4. **TPT has no consistent direction.** Its Δ is positive on 3 test sets and negative on 5, it
   lowers mean confidence on 6 of 8, and its Δ mostly follows its accuracy change: where accuracy
   drops sharply (ImageNet-R, Sketch-1000) the gap rises. At n = 500, ECE also disagreed between the
   two augmentation runs on the same images (Sketch-200: 3.34 vs 6.08) while the signed gap did not.

## Reproducing the results

You need a GPU. You need a GPU. CoOp and TPT on the 1000-class sets peak at about 28 GB (I used a 32 GB RTX 4080 SUPER); everything else fits in about 6 GB.

**Install.** Python 3.12. Install PyTorch and torchvision for your CUDA version first, then:

```bash
pip install open_clip_torch datasets requests tqdm pillow
```

(`requirements.txt` is a lockfile of my development environment and pins a specific CUDA build of
torch, so on a new machine install as above instead.)

**Run.** Four commands, from the repo root:

```bash
python scripts/download_data.py      # raw images into ../data
python scripts/encode_features.py    # CLIP features, in manifest order, into features/
python scripts/run_grid.py           # zero-shot, Tip-Adapter, CoOp  -> results/grid.csv
python scripts/run_grid.py --tpt     # TPT and its single-template baseline
```

The data goes to a `data/` folder next to the repo; pass `--data-root` to put it elsewhere.

Both setup scripts check their own output. `download_data.py` confirms that every image the
manifests name is on disk. `encode_features.py` writes the feature caches on a fresh machine; if
caches already exist it re-encodes and compares instead (labels must match exactly, cosine must be
at least 0.99), so it also works as a check on existing caches.

`run_grid.py` merges its rows into `results/grid.csv` keyed by (dataset, seed, method), so a subset
can be rerun without touching the rest:

```bash
python scripts/run_grid.py --datasets sketch_200 --seeds 42
```

Each row records the model, GPU, a hash of its settings and the git commit it ran from (marked
`-dirty` if `src/` or `scripts/` had uncommitted changes).

## How it fits together

**Manifests.** `manifests/<dataset>.csv` maps every cached feature to the image it came from. Some
of the original caches were built through a shuffled data loader or a filesystem-ordered file list,
so their order could not be reconstructed from code. `scripts/build_manifest.py` recovered it by
re-encoding every image and matching it to its cached feature (same label, cosine ≥ 0.996, with the
two lowest matches checked by hand for ambiguity). Encoding in manifest order reproduces the caches,
and with them every split and every number in `grid.csv`. TPT, which needs the raw image behind each
test feature, opens images through the manifests.

**Harness.** Every method returns logits of shape `(N, num_classes)`, and every metric is a function
`(logits, labels) -> number`, so adding a metric never touches a method.

```
src/
  harness.py            methods as logits, pluggable metrics, TPT loop
  coop.py               PromptLearner, text encoder wrapper, CoOp training
  tip_adapter.py        cache model
  tpt.py                entropy objective with confidence selection
  clip_zeroshot.py      model name, text and image feature builders
  features_registry.py  the 8 datasets: cache files, class names, shots
  image_sources.py      image order per dataset, manifest loading, image encoding
  splits.py             seeded per-class cache / validation / test split
  attention.py, vit.py, nanovlm.py   from-scratch attention, ViT and a tiny VLM (learning work, not used by the grid)
scripts/
  download_data.py, encode_features.py, run_grid.py, build_manifest.py
manifests/              cached feature index -> image, one CSV per dataset
results/grid.csv        every reported number
results/gelu/           earlier runs from before the QuickGELU fix, kept as a record
notebooks/              exploratory work, one per step; numbers before notebook 13 predate the QuickGELU fix
```

## Known limits

- CoOp uses Adam instead of the paper's SGD with cosine schedule, and initialises its context from
  "a photo of a" instead of randomly. Internal comparisons are consistent; absolute numbers will
  differ slightly from published ones.
- ImageNet-V2 has 10 images per class, so it runs 4-shot with no validation split. Its Δ is not
  comparable in size to the 16-shot sets.
- Tip-Adapter's α and β are fixed, not tuned per test set. They are reported at two β values
  because the size of Δ depends on β.
- TPT runs on 500 images from one seed, with random-crop and flip augmentation rather than the
  AugMix used in the TPT paper. Its two runs measure augmentation noise, not sampling noise.
- Only three test sets start overconfident, and they share sketch style or low shot count, so the
  CoOp result rests on thin evidence for overconfident starts.
- There is no temperature-scaling baseline yet. It is the obvious competing explanation for CoOp.
- ImageNet-Sketch contains duplicate images, some filed under more than one class. Those test
  images cannot all be classified correctly by any method.
- The current `grid.csv` comes from two commits (the main grid and the TPT rows). The numbers in the
  paper will come from a single clean run.
