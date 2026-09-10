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
