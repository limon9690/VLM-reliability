# vlm-reliability

Reproductions of CLIP adaptation methods under distribution shift, with
calibration measured alongside accuracy.

Four methods run through one harness: zero-shot CLIP, CoOp, Tip-Adapter, and
TPT. Each is evaluated on shifted versions of ImageNet and on PACS, reporting
top-1 accuracy and Expected Calibration Error from the same code path.

Backbone is CLIP ViT-B/16 (OpenAI weights), frozen throughout. Everything runs
on free-tier Colab/Kaggle GPU.

---

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
| CoOp, all 101 classes      | 90.21% |
| CoOp, base classes         | 96.79% |
| CoOp, new (unseen) classes | 91.87% |

The ~5 point base-to-new gap reproduces CoOp's known generalization weakness.

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

<!-- TODO:
     - environment setup (python version, pip install line)
     - how to download/point at each dataset
     - the command that builds the feature cache
     - the command that reproduces each results table above
     One command per table row is the goal. -->

```
# setup

# build feature cache

# reproduce ImageNet-R comparison

# reproduce ImageNet-Sketch comparison

# reproduce CoOp on Caltech101
```

---

## Known limits

- **TPT runs on subsets.** Per-image gradient steps plus 63 augmented views make
  full-set TPT impractical on free-tier GPU. Reported runs use n=200–500.
- **TPT's baseline differs.** TPT is compared against a single-template
  zero-shot baseline, while the other methods use 80-template ensembling. The
  gain (+1.5 over its own baseline) is real; the absolute number is not
  comparable across rows.
- **Tip-Adapter's α and β were tuned on ImageNet-R** and carried over to
  ImageNet-Sketch without retuning. The Sketch numbers may reflect a poor α for
  a 1000-class softmax rather than a property of that shift.
- **Class counts differ across datasets.** ImageNet-R has 200 classes,
  ImageNet-Sketch 1000. Softmax confidence distributions are not directly
  comparable across them, which matters for any cross-dataset calibration claim.
- **ECE numbers come from two implementations** with different bin-edge
  handling. They are being consolidated into the single `harness.ece`.
- **CoOp used Adam**, not the SGD + cosine schedule from the paper. Internal
  comparisons are consistent; absolute numbers differ slightly from published.

---

## Notes

CLIP is frozen in every method here. The pattern across the field — and across
this repo — is a frozen backbone with a small trainable or training-free piece
attached: learned prompt context (CoOp), a feature cache (Tip-Adapter), or a
per-image prompt update with no labels (TPT).

Feature caching is what makes this run on free-tier hardware. Image features are
extracted once and reused; only the trained side (learned prompts) is recomputed.
