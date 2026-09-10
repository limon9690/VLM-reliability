## Q1 — What does pseudo-labeling / confidence-filtering become when the model is zero-shot CLIP?

Pseudo-labeling assumes confidence tracks correctness: keep only the model's
high-confidence guesses, treat them as trustworthy labels. That assumption only
holds if the model's self-reported confidence is calibrated.

A confidence-binning probe on zero-shot CLIP's ImageNet-R predictions (full
set, ~30k samples) shows a clear pattern:

| conf bin | n     | acc   |
| -------- | ----- | ----- |
| 0.0–0.1  | 620   | 0.106 |
| 0.1–0.2  | 2072  | 0.208 |
| 0.2–0.3  | 2283  | 0.328 |
| 0.3–0.4  | 2225  | 0.464 |
| 0.4–0.5  | 2268  | 0.545 |
| 0.5–0.6  | 2229  | 0.651 |
| 0.6–0.7  | 2165  | 0.763 |
| 0.7–0.8  | 2385  | 0.851 |
| 0.8–0.9  | 3321  | 0.922 |
| 0.9–1.0  | 10432 | 0.987 |

In every bin, actual accuracy exceeds the confidence range itself — e.g. the
0.6–0.7 "confident" bin is 76.3% accurate, and the 0.9–1.0 bin is 98.7%
accurate. Zero-shot CLIP is **systematically underconfident** on ImageNet-R,
not overconfident — the opposite of the failure mode pseudo-labeling is
usually built to guard against.

This changes the answer to Q1: naive high-confidence pseudo-labeling
(e.g. threshold ≥0.9) is _safe_ for zero-shot CLIP — that bin is 98.7% clean —
but overly conservative. Because confidence underestimates true accuracy, a
fixed high threshold discards a large pool of correct predictions sitting in
the 0.6–0.9 confidence range (76–92% accurate) that a properly calibrated
filter would keep. The interesting version of pseudo-labeling here isn't
"filter by raw confidence" — it's "recalibrate confidence first, then filter,"
since the raw signal already contains more correct information than its face
value suggests.

---

## Q2 — Does DANN-style adversarial alignment do anything on frozen CLIP features?

DANN removes domain-discriminative structure from features so a classifier
can't tell which domain a sample came from — the idea being that domain signal
doesn't transfer. Its precondition: there has to _be_ linearly-accessible domain
structure in the features to remove. Since CLIP is frozen (backbone can't be
retrained), the first question is just whether that structure exists at all.

Probe: train a balanced logistic regression to predict _domain identity_ (not
class) from cached image features, held-out split, with a shuffled-label null
control on each pair. Chance = 0.50; the **gap** (real − null) is the domain
signal and is the only cross-pair-comparable number.

| domain pair (source vs shift) | real  | null  | gap    |
| ----------------------------- | ----- | ----- | ------ |
| v2 vs Sketch                  | 0.991 | 0.496 | +0.495 |
| v2 vs R                       | 0.957 | 0.500 | +0.457 |
| R vs Sketch                   | 0.894 | 0.498 | +0.395 |
| PACS photo vs Sketch          | 1.000 | 0.492 | +0.508 |

The domain gap is **large and consistent across every shift tested** — the DANN
precondition is met everywhere, not just for one shift type. Frozen CLIP
features are _not_ domain-invariant; a trivial linear classifier separates
source from shifted domain near-perfectly. (v2 is a near-source photo domain,
mildly shifted itself — not a pure origin — so read it as "photo-ish source.")

So the interesting question isn't "is there a gap for alignment to grab?" (yes,
plainly). It's: **if the domain gap is this linearly obvious, why does the
field's toolkit — CoOp, Tip-Adapter, TPT — do prompt-tuning and test-time
entropy minimization, and not feature alignment at all?** That absence, given
how accessible the gap is, is the real thread for the map.

**Caveat 1 (the likely answer to that question):** separable ≠
removable-without-damage. A domain signal this strong is probably entangled
with class-discriminative signal; stripping it DANN-style may destroy what makes
classification work. That entanglement — not absence of a gap — is the more
plausible reason feature alignment fell out of favor when CLIP arrived.

**Caveat 2:** this measures _linear_ separability. A non-result would not rule
out non-linear structure — but every result here is strongly positive, so that
doesn't bite.

**Secondary note:** sketch pairs (0.99–1.00) sit slightly above the rendition
pair (0.96), directionally consistent with the Month-2 sketch-is-hardest
finding — but all are near ceiling, so the ordering is weak evidence, not a
clean result.

---

## Q4 — Where do test-time methods go unstable with CLIP, and what stabilizes them?

TPT's ECE (8.65%) is worse than zero-shot's (6.45%) and worse than the few-shot
methods (CoOp 0.76%, Tip-Adapter 1.09%) — entropy-minimization TTA measurably
degrades calibration. The natural hypothesis: entropy minimization pushes
predictions toward higher confidence by design, so TPT should look
_overconfident_ relative to zero-shot.

A confidence-binning probe (n=500, ImageNet-R subset — compute-limited, vs. the
full set used for zero-shot) doesn't support that hypothesis. TPT remains
underconfident in every bin, same direction as zero-shot:

| conf bin  | acc   |
| --------- | ----- |
| 0.00–0.50 | 0.289 |
| 0.50–0.70 | 0.740 |
| 0.70–0.85 | 0.810 |
| 0.85–1.00 | 0.947 |

But the gap between confidence and accuracy isn't uniform — it's ~3x larger in
the 0.5–0.7 bin (+0.14) than in the bins on either side (+0.04 each). So TPT's
calibration failure isn't a clean confidence inflation; it looks more like
**localized instability concentrated in the mid-confidence range**, which the
aggregate ECE number hides. What stabilizes it is still open — worth testing
whether restricting TPT's confidence-selection threshold (top 5% vs 10% vs 20%
of views) changes where this instability sits.

**Caveat:** this binning uses coarser bins and a smaller sample than the ECE
calculation, so it's a directional decomposition, not an exact one.
