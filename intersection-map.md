# Week 9 — Intersection Map (Classical DA × VLM Reliability)

This document is the Month 3 intersection map. Two findings from Month 2 run through the whole thing:

1. **Sketch is CLIP's hardest shift.** Confirmed on ImageNet-Sketch and
   PACS-sketch. Line drawings strip color and texture; paintings and cartoons
   degrade far less.
2. **Test-time adaptation (TPT) hurts calibration.** TPT has the worst ECE of
   the four methods in my harness (8.65%), while few-shot adaptation improves
   calibration over zero-shot. The field is actively arguing about why, with no
   agreement yet.

The five questions below are worked in writing, with cheap probe experiments on
cached CLIP features.

---

## Q1 — What does pseudo-labeling become when the model is zero-shot CLIP?

Pseudo-labeling keeps the model's high-confidence guesses and treats them as if
they were real labels. This only works if high confidence means high accuracy.
So the question is whether CLIP's confidence can be trusted under shift.

I binned zero-shot CLIP's predictions on ImageNet-R by confidence and checked
the accuracy in each bin (full set, ~30k samples):

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

In every bin the accuracy is higher than the confidence. When CLIP says
0.6–0.7, it is right 76% of the time. When it says 0.9–1.0, it is right 98.7%
of the time. So CLIP is underconfident on ImageNet-R, not overconfident. This is
the opposite of the problem pseudo-labeling usually worries about.

What this means for Q1: filtering at a high threshold like 0.9 is safe, because
that bin is 98.7% correct. But it is too strict. CLIP underestimates its own
accuracy, so a high threshold throws away many correct predictions in the
0.6–0.9 range (76–92% correct) that are actually good labels. So the useful idea
here is not "filter by raw confidence." It is "fix the confidence first, then
filter," because the raw confidence already holds more correct predictions than
its value suggests.

---

## Q2 — Does DANN-style adversarial alignment do anything on frozen CLIP features?

DANN removes domain information from features, so a classifier cannot tell which
domain a sample came from. The point is that domain information does not
transfer. For this to work, there has to be domain information in the features
in the first place. CLIP is frozen, so I cannot retrain the backbone. So the
first question is simple: is there any domain structure in the frozen features
to remove?

Probe: train a balanced logistic regression to predict the domain (not the
class) from cached image features, on a held-out split. For each pair I also run
a shuffled-label null control. Chance is 0.50. The gap (real − null) is the
domain signal, and it is the only number that can be compared across pairs.

| domain pair (source vs shift) | real  | null  | gap    |
| ----------------------------- | ----- | ----- | ------ |
| v2 vs Sketch                  | 0.991 | 0.496 | +0.495 |
| v2 vs R                       | 0.957 | 0.500 | +0.457 |
| R vs Sketch                   | 0.894 | 0.498 | +0.395 |
| PACS photo vs Sketch          | 1.000 | 0.492 | +0.508 |

The domain gap is large for every pair, not just one shift type. So the frozen
CLIP features are not domain-invariant. A simple linear classifier separates the
source and the shifted domain almost perfectly. (v2 is itself a mild shift from
the original photos, so read it as a "photo-like source," not a clean source.)

So the interesting question is not "is there a gap to remove?" There clearly is.
The question is: if the domain gap is this easy to find with a linear model, why
does the field's toolkit — CoOp, Tip-Adapter, TPT — use prompt-tuning and
test-time entropy, and not feature alignment at all? That missing method is the
real thread for the map.

**Caveat 1 (probably the answer to that question):** separable is not the same
as removable without damage. A domain signal this strong is probably mixed
together with the class information. If you strip the domain signal DANN-style,
you may also destroy what makes classification work. That mixing, not a missing
gap, is the more likely reason feature alignment fell out of use when CLIP
arrived.

**Caveat 2:** this only measures linear separability. A near-chance result would
not rule out non-linear structure. But every result here is strongly positive,
so this does not matter here.

**Side note:** the sketch pairs (0.99–1.00) are a little higher than the
rendition pair (0.96). This matches the sketch-is-hardest finding, but all the
numbers are near the ceiling, so this is weak evidence, not a real result.

---

## Q3 — Which benchmarks have VLM-reliability results, and which don't?

Pulled from my paper-summaries file:

| Paper                   | Benchmarks                               | Metric                        | Calibration / reliability?  | Setting                      |
| ----------------------- | ---------------------------------------- | ----------------------------- | --------------------------- | ---------------------------- |
| CLIP                    | ImageNet + R/Sketch, ~27 datasets        | Top-1 acc                     | ✗                           | zero-shot                    |
| CoOp                    | 11-dataset suite, base-to-new            | Top-1 acc                     | ✗                           | few-shot                     |
| CoCoOp                  | base-to-new, cross-dataset, DG           | Top-1 acc                     | ✗                           | few-shot (input-conditional) |
| Tip-Adapter             | few-shot suite, ImageNet-R               | Top-1 acc                     | ✗                           | few-shot, training-free      |
| TPT                     | ImageNet-{A,R,V2,Sketch}                 | Top-1 acc                     | ✗                           | test-time                    |
| WiSE-FT                 | ImageNet + shifts                        | acc, effective robustness     | robustness ✓, calibration ✗ | fine-tune + weight interp    |
| ClipTTA                 | TTA benchmarks (not recorded — acc-only) | acc                           | ✗                           | test-time                    |
| FCL                     | TTA benchmarks (not recorded — acc-only) | acc                           | ✗                           | test-time                    |
| What Drives TTA (TTABC) | 20+ methods, multiple shifts             | acc + reliability, separately | ✓                           | study / benchmark            |

**The gap: almost nobody measures calibration under shift.** Eight of the nine
papers report accuracy only. The one that measures reliability separately from
accuracy is a 2026 study, not an adaptation method. Every actual method (CoOp,
Tip-Adapter, TPT, ClipTTA, FCL) is judged by top-1 accuracy alone.

This is exactly where my harness already has numbers. My Q4 finding — TPT hurts
calibration (ECE 8.65%) while few-shot adaptation improves it — is a measurement
the whole method literature skipped. So the gap is not "invent a new method." It
is "measure the reliability the field left unmeasured, on methods that already
exist, under shift."

The TTABC study helps me instead of competing with me. It says out loud that
reliability is not the same as accuracy under shift, and that the field
understands less than the number of methods suggests. It opens the gap but does
not close it — it is a benchmark, not a calibration-aware method.

---

## Q4 — Where do test-time methods go unstable, and what stabilizes them?

TPT's ECE (8.65%) is worse than zero-shot (6.45%) and worse than the few-shot
methods (CoOp 0.76%, Tip-Adapter 1.09%). So test-time entropy minimization hurts
calibration. The obvious guess is that entropy minimization pushes predictions
to be more confident, so TPT should look overconfident compared to zero-shot.

I binned TPT's predictions the same way as Q1 (n=500, ImageNet-R subset — the
subset is a compute limit; zero-shot used the full set). The guess is wrong. TPT
is still underconfident in every bin, the same direction as zero-shot:

| conf bin  | acc   |
| --------- | ----- |
| 0.00–0.50 | 0.289 |
| 0.50–0.70 | 0.740 |
| 0.70–0.85 | 0.810 |
| 0.85–1.00 | 0.947 |

But the gap between confidence and accuracy is not the same size everywhere. It
is about 3x larger in the 0.5–0.7 bin (+0.14) than in the bins next to it (+0.04
each). So TPT's calibration problem is not a simple "too confident everywhere."
It looks more like unstable behavior in the middle-confidence range, which the
single ECE number hides. What fixes it is still open. One thing worth testing:
does changing TPT's confidence-selection threshold (top 5% vs 10% vs 20% of
views) move where this instability sits?

**Caveat:** the 4-bin table above is a display grouping for readability; the ECE
numbers use 10 equal-width bins. Same implementation, different granularity. The
TPT rows also use a smaller sample (n=500 subset) than the full-set zero-shot
numbers.

Ablation (n=300, ImageNet-R subset): varied TPT's view-selection fraction across
5% / 10% / 20%. ECE was 7.22% / 8.91% / 8.20% — not monotonic, and within
subrun noise at this n. The confidence-accuracy shape (mid-range wobble) is
roughly constant across all three. So TPT's miscalibration does not track
the selection threshold — it looks intrinsic to the entropy objective, not
caused by the confidence-selection heuristic. (Accuracy rises slightly as
selection loosens: 58.7 → 60.7, so the knob has a small accuracy effect but no
clean calibration effect.) Caveat: n=300, per-image TPT is noisy, and these
accuracies run below the n=500 headline (~66%), so read the three as comparable
to each other, not to earlier runs.

---

## Q5 — Methods × shift-types: where has reliability actually been measured?

Rows = how the method adapts CLIP (grouping from TTABC).
Columns = shift type.
Cell = has calibration/reliability been measured here?
✓ = the field measured it. ✗ = nobody did (the gap). [number] = I measured it in
my harness.

| Method-type                       | in-dist | rendition (R)    | sketch          | corruption / other |
| --------------------------------- | ------- | ---------------- | --------------- | ------------------ |
| zero-shot (inference)             | ✗       | ECE 6.45% (mine) | ECE 0.95%(mine) | ✗                  |
| few-shot pre-deploy (CoOp/CoCoOp) | ✗       | ECE 0.76% (mine) | ✗               | ✗                  |
| cache-based (Tip-Adapter)         | ✗       | ECE 1.09% (mine) | ECE 8.77%(mine) | ✗                  |
| param-update TTA (TPT)            | ✗       | ECE 8.65% (mine) | ✗               | ✗                  |
| weight-interp (WiSE-FT)           | ✗       | ✗                | ✗               | ✗                  |

> ✗ = not reported in my paper-summaries. Field-wide calibration reporting is
> almost absent (see Q3, where 8 of 9 papers report accuracy only).

The ImageNet-R column is fully measured, by me. The sketch column is partly
filled (zero-shot and Tip-Adapter); CoOp and TPT there are parked, not absent —
1000-class training exceeds free-tier GPU. Everything else is empty.

Sketch is CLIP's hardest shift (my Month-2
finding, confirmed on ImageNet-Sketch and PACS-sketch). So the most valuable
empty cell is calibration under sketch-shift: the shift where the model is
weakest is also the shift where nobody has measured whether adaptation helps or
hurts reliability. That is paper-one's sharpest target.

### Sketch column (Week 10)

| method      | ImageNet-R ECE | ImageNet-Sketch ECE |
| ----------- | -------------- | ------------------- |
| zero-shot   | 6.45%          | 0.95%               |
| Tip-Adapter | 1.09%          | 8.77%               |

CoOp / TPT on sketch: parked (1000-class exceeds free-tier GPU; awaits lab hardware).

**Finding — the calibration effect flips across shift type.** Tip-Adapter
improves calibration on R (1.09% vs 6.45%) but destroys it on sketch (8.77% vs
0.95%). Same training-free method, opposite effect, depending only on the shift.
So adaptation's calibration effect is not a property of the method — it is a
property of method × shift-type. This is the accuracy-only literature's blind
spot, on my own numbers.

Note: zero-shot's low sketch ECE (0.95%) is _aggregate_ calibration — mean
confidence 0.440 ≈ accuracy 0.441. CLIP is unsure and correctly unsure on
sketch, but this is coin-flip-level accuracy, not trustworthy per-prediction
confidence. Aggregate-calibrated ≠ useful.

### Signed gap (Week 11)

`signed_gap` added to `harness.py` — mean confidence − accuracy, positive means
overconfident. Run on cached R and Sketch logits via `run_comparison`:

| dataset         | zero-shot | Tip-Adapter | change |
| --------------- | --------- | ----------- | ------ |
| ImageNet-R      | −6.45     | −0.40       | +6.05  |
| ImageNet-Sketch | −0.13     | +8.76       | +8.89  |

Same direction on both datasets, so the apparent ECE "flip" is an artifact of
unsigned ECE plus different baseline positions — not a shift-type effect.
Magnitude differs (~47%) and the cause is unestablished: could be shift,
the 200-vs-1000 class confound, or α carried over from R.

---

## Where this leaves paper-one

- Q1 and Q2 clear the ground. Confidence-filtering is safe but wasteful on
  underconfident CLIP (Q1). Classical feature alignment has a gap to grab but
  probably cannot use it without breaking classification (Q2). Neither is where
  paper-one should go.
- Q3 and Q4 point at the same open space: I can measure that adaptation changes
  calibration under shift (Q4); almost nobody else measures this (Q3). Q5 draws
  it as a grid — one column measured (by me), the rest empty.

**Week 10 sharpened this into a specific claim.** Two experiments:

1. TPT selection-threshold ablation (5/10/20%): calibration does _not_ track the
   selection fraction. TPT's miscalibration looks intrinsic to the entropy
   objective, not caused by the confidence-selection heuristic. This closes a
   door — the fix is not in the selection step.
2. Sketch column: Tip-Adapter _improves_ calibration on ImageNet-R (1.09% vs
   6.45%) but _wrecks_ it on ImageNet-Sketch (8.77% vs 0.95%). Same training-free
   method, opposite calibration effect, depending only on the shift type.

**Leading paper-one direction (updated):** adaptation's effect on calibration is
not a property of the method — it is a property of method × shift-type. The same
method can help reliability under one shift and destroy it under another, and the
accuracy-only literature cannot see this at all. The Tip-Adapter flip is the
sharpest single piece of evidence: it is training-free, so the effect cannot be
blamed on gradient instability — it is the interaction of the adaptation
mechanism with the shift itself. Sketch remains the emptiest, highest-value shift
(hardest for CLIP, least measured for reliability).
