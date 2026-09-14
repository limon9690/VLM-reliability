# vlm-reliability

Reproductions of CLIP adaptation methods under distribution shift, with
calibration measured alongside accuracy.

Four methods run through one harness: zero-shot CLIP, CoOp, Tip-Adapter, and
TPT. Each is evaluated on shifted versions of ImageNet and on PACS, reporting
top-1 accuracy and Expected Calibration Error from the same code path.

Backbone is CLIP ViT-B/16 (OpenAI weights), frozen throughout.

---

## About this work

I work on the reliability of vision-language models under distribution shift —
adapting frozen models like CLIP to new domains with little or no labeled data.
This repo is the working artifact: it reproduces four adaptation methods and
measures calibration alongside accuracy, which most work in this area does not
report. The main finding so far is that adaptation shifts a model's confidence
in a consistent direction, so whether it improves or damages calibration
depends on where the model started, not on the method alone.

## Results

All numbers are my own runs, not copied from papers. Published numbers are shown
where a direct comparison exists.

### Zero-shot CLIP (80-template prompt ensembling)

| Dataset              | Top-1  | Top-5  |
| -------------------- | ------ | ------ |
| ImageNet-V2          | 53.21% | 79.50% |
| ImageNet-R (200 cls) | 72.80% | 90.51% |
| ImageNet-Sketch      | 44.24% | 72.10% |
| PACS — photo         | 99.94% | 100.0% |
| PACS — art_painting  | 96.73% | 100.0% |
| PACS — cartoon       | 98.72% | 100.0% |
| PACS — sketch        | 87.99% | 99.95% |

ImageNet-V2 sits below the published ~62%. Labels were verified correct, so the
gap is prompt/methodology, not a pipeline bug.

### Adaptation methods on ImageNet-R

| Method      | Top-1  | ECE   |
| ----------- | ------ | ----- |
| zero-shot   | 73.54% | 6.45% |
| Tip-Adapter | 74.86% | 1.09% |
| CoOp        | 75.07% | 0.76% |
| TPT         | 65.00% | 8.65% |

TPT ran on a 200-image subset against a single-template baseline, so its numbers
are not directly comparable to the full-set ensembled rows. Tip-Adapter used
α=1.5, β=5.0.

### ImageNet-Sketch

| Method      | Top-1  | ECE   |
| ----------- | ------ | ----- |
| zero-shot   | 44.10% | 0.95% |
| Tip-Adapter | 52.85% | 8.77% |

CoOp and TPT are not run on Sketch — 1000-class prompt training exceeds
free-tier GPU memory.

### CoOp on Caltech101 (16-shot)

| Setup                      | Top-1  |
| -------------------------- | ------ |
| zero-shot (1 template)     | 84.58% |
| CoOp, all 101 classes      | 91.42% |
| CoOp, base classes         | 95.52% |
| CoOp, new (unseen) classes | 93.14% |

The ~2.4 point base-to-new gap reproduces CoOp's known generalization weakness.

### Calibration direction

ECE is unsigned, so it does not show whether a model is over- or
underconfident. Signed gap (mean confidence − accuracy) does:

| Dataset         | zero-shot | Tip-Adapter |
| --------------- | --------- | ----------- |
| ImageNet-R      | −6.45     | −0.40       |
| ImageNet-Sketch | −0.13     | +8.76       |

Zero-shot CLIP is underconfident on ImageNet-R and close to calibrated on
Sketch. Tip-Adapter raises confidence faster than accuracy on both. On R that
corrects the underconfidence; on Sketch the same shift overshoots into
overconfidence.

---

## What's here

```
src/
  attention.py        self-attention and multi-head attention, written from scratch
  vit.py              Vision Transformer (patch embedding, encoder, head)
  nanovlm.py          minimal VLM: ViT encoder → projection → causal LM
  clip_zeroshot.py    feature caching, text/image feature builders, top-k accuracy
  imagenet_classes.py CLIP class names + 80 prompt templates
  coop.py             PromptLearner and TextEncoderWrapper
  tip_adapter.py      training-free cache model (affinity + residual)
  tpt.py              test-time prompt tuning entropy loss
  harness.py          unified evaluation: shared configs, pluggable metrics

notebooks/            one per roadmap week
features/             cached features (gitignored)
data/                 datasets (gitignored)
```

The harness runs every method through one interface. Each method returns logits
of shape `(N, num_classes)`, and metrics are pluggable functions of
`(logits, labels) → number`, so adding a metric does not touch the methods.

---

## Running it

### 1. Clone and set up a Python environment

```
git clone https://github.com/limon9690/VLM-reliability.git
cd VLM-reliability

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Tested on Python 3.12.

### 2. Install dependencies

```
pip install -r requirements.txt
pip install open_clip_torch datasets torchvision gdown scikit-learn
```

`requirements.txt` covers the notebook/dev tooling (`torch`, `jupyter`, etc).
The second line installs what `src/` and `scripts/` actually import — CLIP
backbone, HuggingFace `datasets`, `torchvision`, `gdown`, `scikit-learn` for
the separability probes in `notebooks/09`. Not folded into the lockfile yet.

### 3. Folder structure

`data/` (raw datasets) and `features/` (cached tensors) are both gitignored
and empty on a fresh clone; each script creates them and fills in only the
files it needs. You need a GPU for the first run — feature extraction and
CoOp training are too slow on CPU to be usable.

### 4. Reproduce a results table

Each script checks `features/` for its cache first and only recomputes what's
missing, so the first run per dataset is the slow one (feature extraction,
and for Caltech101, CoOp training) and later runs load straight from cache.

```
# "Adaptation methods on ImageNet-R" (zero-shot, Tip-Adapter, CoOp, TPT)
# downloads axiong/imagenet-r from HuggingFace on first run
python scripts/eval_imagenet_r.py

# "ImageNet-Sketch" (zero-shot, Tip-Adapter)
# downloads the ImageNet-Sketch mirror via gdown on first run
python scripts/eval_imagenet_sketch.py

# "CoOp on Caltech101 (16-shot)" (zero-shot, CoOp all-101, CoOp base/new split)
# downloads Caltech101 via torchvision on first run
python scripts/eval_caltech_coop.py
```

All three accept `--data-root <dir>` to change where raw datasets are
downloaded (default `data/`).

`eval_imagenet_r.py`'s CoOp row is eval-only: it loads a pre-trained context
vector from `features/coop_ctx_200_cls.pt` rather than training one. That file
is part of the author's local cache, gitignored like the rest of `features/`,
and there is currently no script in this repo that reproduces it — a fresh
clone can run the zero-shot/Tip-Adapter/TPT rows immediately, but needs that
checkpoint supplied separately to run the CoOp row too.

---

## Known limits

- TPT ran on n=200–500 subsets, not the full set. Per-image gradient steps
  plus 63 augmented views make the full set impractical on free-tier GPU.
- TPT is compared against a single-template zero-shot baseline, while the
  other methods use 80-template ensembling. The +1.5 gain over its own
  baseline is real; the absolute number isn't comparable across rows.
- Tip-Adapter's α and β were tuned on ImageNet-R and carried over to
  ImageNet-Sketch without retuning. The Sketch numbers may reflect a poor α
  for a 1000-class softmax more than a property of that shift.
- ImageNet-R has 200 classes, ImageNet-Sketch has 1000. Softmax confidence
  distributions aren't directly comparable across them, which matters for any
  cross-dataset calibration claim.
- CoOp used Adam, not the SGD + cosine schedule from the paper. Internal
  comparisons are consistent; absolute numbers differ slightly from published.

---

## Notes

CLIP is frozen in every method here — the pattern across the field is a frozen
backbone with something small attached: a learned prompt (CoOp), a feature
cache (Tip-Adapter), or a per-image prompt update with no labels (TPT).
