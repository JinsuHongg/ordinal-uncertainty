# Phase 3.10C — RetinaMNIST Direction-Only Head Adaptation Falsification

## Question and frozen causal protocol

Can balanced classifier-direction adaptation retain rare class-4 localization
while avoiding the class-dependent norm inflation that damages class 0? This is
a single causal falsification following Phase 3.10B's mixed parameter
mechanism audit, not a final method.

The experiment reused only the canonical seed-0 RPS checkpoint, the Phase
3.10A training-only `(1080,512)` frozen feature archive, and the exact saved
five-fold assignments. B is the saved Phase 3.10A balanced-head OOF result. A
is the untouched original RPS head. No validation/test values, backbone,
representation, ROP, logit adjustment, class weighting, or new dataset was
used.

For C, class `k` is parameterized as
\(w_k=\|w_k^{(0)}\|v_k/\|v_k\|\), with original RPS bias fixed. Only `v` is
trainable. Initialization reproduces original weights exactly. AdamW uses the
same LR `.001`, balanced batch size 64, 100 fixed epochs, folds, sampler, and
seeds as B. Weight decay is `0`: radial decay on `v` is removed by row
normalization, so retaining it would be mathematically inert rather than a
matched meaningful regularizer.

The initial maximum logit replay error was `9.54e-7`; the maximum row-norm
error across all training epochs/folds was `1.19e-7`, below the `1e-6` limit.

## Pooled training-only OOF results

| Condition | L1 accuracy / MAE / QWK / severe % | NLL / Brier / RPS / ECE | Spearman / AUROC / AUPRC / selective MAE |
|---|---|---|---|
| A original RPS | .518 / .696 / .604 / 19.1 | 1.141 / .554 / .122 / .035 | .467 / .674 / .269 / .393 |
| B balanced | .507 / .744 / .612 / 19.4 | 1.236 / .603 / .136 / .072 | .471 / .700 / .354 / .427 |
| C direction-only | .503 / .721 / .621 / 18.0 | 1.229 / .602 / .133 / .076 | .447 / .701 / .287 / .422 |

C improves global L1 MAE/QWK/severe prevalence over B, but has lower Spearman
and AUPRC than B. This remains a causal mechanism comparison, not a method
selection claim.

| Condition | C4 routing 0/1/2/3/4 | C4 MAE / exact / severe % | C4 p4 / mean / shrinkage | C0 MAE / severe % / p0 / p1 |
|---|---|---|---|---|
| A | 1/1/41/23/0 | 1.697 / 0 / 65.2 | .116 / 2.203 / 1.797 | .570 / 25.1 / .673 / .052 |
| B | 0/2/15/38/11 | 1.121 / 11 / 25.8 | .338 / 2.727 / 1.273 | .733 / 24.3 / .565 / .158 |
| C | 0/3/19/42/2 | 1.348 / 2 / 33.3 | .306 / 2.616 / 1.384 | .691 / 22.0 / .543 / .183 |

Thus C retains meaningful improvement over A but loses most B exact recovery;
it partially restores class-0/global behavior relative to B.

## Direction and margin mechanism

C preserves every original norm exactly and fixes all original biases, so its
margin changes are direction-only. C mean margins are `z4-z3=.030` (median
`.108`, 57.6% positive), `z4-z2=.272` (median `.362`, 62.1% positive), and
`z0-z1=.934` (median `1.241`, 69.5% positive). B values were `.078`, `.431`,
and `1.246`; A values were `-1.068`, `-1.087`, and `5.153`.

C directions are substantially rotated from A but generally close to their
fold-balanced counterparts for C0–C3; C4 stays materially less aligned with B
(roughly `.56–.61` cosine in the first folds), consistent with reduced exact
recovery. Full fold-level direction tables are saved in the artifact.

There are 2 exact C4 recoveries, of which 1 overlaps B's 11 exact recoveries:
exact retention `1/11`. C has 24 inward-improved C4 cases, 21 overlapping B's
37 inward-improved cases: `21/37` retention. Of 84 class-0 samples damaged by
B, 34 are closer to zero under C than B.

## Decision

\[
\boxed{\text{PARTIAL SUPPORT}}
\]

Direction adaptation is useful: C improves rare localization over A and
partially reduces B's global/class-0 cost. But fixed original scale removes
most exact C4 recovery, demonstrating that direction-only adaptation is not
sufficient. The evidence supports a future separately authorized
controlled-scale design question; it does not authorize that design, a norm
multiplier, a new loss, test evaluation, seed expansion, or other dataset.

No final method was designed.

Artifacts: `outputs/retinamnist/phase3_10c_direction_only_head/`.
