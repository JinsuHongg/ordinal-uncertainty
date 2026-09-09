# Phase 3.18A — RetinaMNIST CE Representation Robustness

## Question and scope

This frozen-representation robustness gate asks whether the rare class-4
localization response to direction-only head adaptation, previously shown on
the RetinaMNIST RPS representation, also occurs on the canonical CE
representation. It is not a CE-versus-RPS comparison and does not propose a
new classifier method.

Only canonical RetinaMNIST **training** evidence was used: deterministic
five-fold OOF predictions for all 1,080 training examples. Historical
RetinaMNIST validation and test arrays were not loaded or used for fitting,
selection, debugging, or interpretation. No backbone was retrained or
fine-tuned.

## Frozen protocol and provenance

| Item | Frozen choice |
| --- | --- |
| CE feature archive | `outputs/retinamnist/native28/phase3_3_representation_audit_replay_verified/ce/seed_0/features.npz` |
| CE checkpoint | `outputs/retinamnist/resolution_sanity_check/seed_0/size_28/best_checkpoint.pt` |
| Features read | training keys only; `(1080, 512)` float32 |
| Folds | exact Phase 3.10A deterministic training-only five-fold assignments, seed 0 |
| B/C initialization | original CE `fc.weight` and `fc.bias` |
| Optimizer / LR / batch / epochs | AdamW / `.001` / `64` / fixed `100` |
| B objective | balanced-sampler CE, full linear weight and bias trainable, weight decay `1e-4` |
| C objective | identical balanced-sampler CE; directions trainable only, weight decay `0` |

A is the untouched original CE head. B is the established full balanced linear
head control. C uses

\[
w_k^C=\|w_k^A\|_2\frac{v_k}{\|v_k\|_2},\qquad b_k^C=b_k^A,
\]

so only row directions can change. B and C used identical fold seeds and
identical replacement balanced batches at every optimization step. The
five-fold execution was mechanically dispatched one fold at a time because of
the local command time limit; the parameters, sampler, seed, batch size,
objective, and 100-epoch budget were unchanged. Final OOF metrics were then
assembled only from the saved five B/C fold states. Two interrupted attempts
ended before any pooled predictions, metrics, or scientific result were
created and are preserved separately as incomplete infrastructure artifacts.

## Integrity checks

- CE feature shape/count/labels: `(1080, 512)` / 1,080, with no NaN or Inf.
- The CE cached logits replayed from frozen features and the checkpoint head
  with maximum absolute error `2.86e-6`.
- The original-head replay at C initialization had maximum error `9.54e-7`.
- C's maximum row-norm error was `5.96e-8`; its maximum fixed-bias error was
  `0`.
- Fold assignments align exactly by sample ID and label. Held-out class-4
  support is `13, 14, 13, 13, 13` and all 1,080 samples appear once OOF.
- Only the established training cache keys were read; no validation/test split
  was constructed or loaded by the runner.

## Parameter diagnostics

Balanced fitting strongly increased CE row norms, especially the rare class.
C preserved every original norm and bias by construction.

| Class | A norm | B norm (fold mean) | B Δnorm | B Δbias | A→B cosine | A→C cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | .548 | 1.017 | +.469 | -.006 | .686 | .678 |
| 1 | .540 | 1.292 | +.751 | +.008 | .657 | .648 |
| 2 | .539 | 1.666 | +1.127 | -.003 | .724 | .675 |
| 3 | .517 | 1.514 | +.997 | -.005 | .632 | .620 |
| 4 | .604 | 2.694 | +2.089 | +.003 | .426 | .616 |

Thus the C4 direction changes materially under both B and C, while C excludes
the large B norm increase and all bias movement.

## Pooled OOF results

All decisions below are exact discrete L1 Bayes decisions.

| Condition | Accuracy | MAE | QWK | Severe | NLL | Brier | RPS | ECE | Spearman | AUROC | AUPRC | Selective MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A-CE original | .534 | .693 | .616 | .188 | 1.028 | .528 | .118 | .059 | .542 | .769 | .410 | .343 |
| B-CE balanced | .537 | .716 | .621 | .179 | 1.098 | .556 | .125 | .069 | .498 | .768 | .433 | .373 |
| C-CE direction-only | .519 | .709 | .627 | .172 | 1.124 | .562 | .127 | .055 | .507 | .767 | .393 | .376 |

C modestly improves QWK and severe prevalence relative to A, but worsens
global MAE and most probability/risk-quality quantities. These secondary
trade-offs do not determine the direction-mechanism gate.

### Rare class 4

| Condition | L1 routing `4→0/1/2/3/4` | Exact | C4 MAE | Severe | Mean / median p4 | Mean / median p3 | Mean / median p3+p4 | Predictive mean | Shrinkage | Mean / median L1 risk |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A-CE | `0/0/29/37/0` | 0 | 1.439 | 43.9% | .157 / .166 | .360 / .379 | .518 / .537 | 2.322 | 1.678 | .892 / .874 |
| B-CE | `0/0/11/31/24` | 24 | .803 | 16.7% | .417 / .368 | .269 / .244 | .685 / .694 | 2.888 | 1.112 | .786 / .786 |
| C-CE | `0/0/13/42/11` | 11 | 1.030 | 19.7% | .363 / .350 | .269 / .262 | .631 / .650 | 2.756 | 1.244 | .900 / .895 |

The primary A→C result is outward and substantial: C4 MAE improves by `.409`,
exact decisions change `0→11`, mean `p4` rises `.157→.363`, predictive mean
rises `.434`, and inward shrinkage falls `.434`. The C4 `z4-z3` margin moves
from `-.903` (0% positive) to `.261` (71.2% positive); `z4-z2` moves from
`-.618` to `.604` (15.2% to 80.3% positive).

Fold-level C4 MAE for C is `1.039 ± .303`; its exact count is `2.2 ± 3.3` per
fold. This is OOF fold variability, not a multi-seed estimate.

### Class-0 collateral control

| Condition | L1 routing `0→0/1/2/3/4` | C0 MAE | Severe | Mean / median p0 | Mean / median p1 | Mean / median p0+p1 | Predictive mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A-CE | `334/23/97/32/0` | .644 | 26.5% | .650 / .769 | .071 / .044 | .721 / .833 | .778 |
| B-CE | `305/68/54/50/9` | .745 | 23.3% | .600 / .747 | .135 / .052 | .734 / .857 | .865 |
| C-CE | `305/62/65/50/4` | .737 | 24.5% | .576 / .687 | .141 / .087 | .717 / .812 | .927 |

CE C carries a class-0/global MAE cost relative to A, although its severe
prevalence falls. The C0 MAE increase (`+.093`) is smaller than the saved RPS
C increase (`+.121`), but both directions are adverse on this OOF control.

## Representation/head decomposition

Raw-Euclidean training-centroid assignment was recomputed from the saved CE
training features using the Phase 3.3-compatible definition. Of 66 true C4
OOF cases, 48 are feature-nearest to C4 and 18 are representation-inward.

| CE subgroup | Support | A routing | C routing | Exact A→C | Inward-improved A→C |
| --- | ---: | --- | --- | ---: | ---: |
| Feature-nearest-4 | 48 | `0/0/12/36/0` | `0/0/4/33/11` | 0→11 | 17 |
| Representation-inward | 18 | `0/0/17/1/0` | `0/0/9/9/0` | 0→0 | 8 |

Exact recovery is concentrated in the feature-nearest-4 subgroup. The
representation-inward subgroup nevertheless moves toward class 3, so
nearest-centroid assignment is a useful descriptive stratification rather than
proof of irrecoverability.

## Saved RPS comparison

| Representation | A C4 MAE | C C4 MAE | Exact A→C | Shrinkage A→C | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| RPS (saved Phase 3.10C) | 1.697 | 1.348 | 0→2 | 1.797→1.384 | saved direction effect |
| CE (new Phase 3.18A) | 1.439 | 1.030 | 0→11 | 1.678→1.244 | cross-objective Retina robustness |

Both representations show the same A→C direction: class-4 MAE and shrinkage
fall while exact and adjacent routing move outward. The magnitude differs and
is not a CE/RPS ranking. The CE result extends the RetinaMNIST finding beyond
the RPS representation, but it does not establish an objective-wide or
cross-dataset CE mechanism.

## Questions answered

1. **Is CE head-actionable under B?** Yes. B recovers 24 exact C4 decisions,
   improves C4 MAE to `.803`, and carries global/C0 trade-offs.
2. **Does CE direction-only C improve C4 localization over A?** Yes, on MAE,
   exact routing, probability mass, predictive mean, shrinkage, severe burden,
   and C4 decision margins.
3. **Does CE A→C match RPS A→C in direction?** Yes. Both move C4 outward;
   exact-recovery magnitude differs.
4. **Where is CE recovery concentrated?** Exact recovery is entirely among
   feature-nearest-4 examples; representation-inward cases receive adjacent
   but no exact recovery.
5. **Are C0/global trade-offs different from RPS?** Their magnitudes differ,
   but both CE and saved RPS C conditions increase C0/global L1 MAE over A.
6. **Must the paper retain the frozen-RPS-only limitation?** The exact wording
   must change: direct RetinaMNIST direction evidence now covers both frozen CE
   and RPS representations. The cross-dataset mechanism claim remains limited
   to frozen RPS until a separately authorized non-RPS confirmation is run.

## Decision

\[
\boxed{\text{A — RETINAMNIST CROSS-OBJECTIVE DIRECTION ROBUSTNESS CONFIRMED}}
\]

CE direction-only adaptation clearly improves rare C4 outward localization and
is qualitatively consistent with the saved RPS direction effect. This permits
a later, separately authorized Solar CE confirmation, but does not authorize
it here. No controlled scale, bias intervention, ROP, new loss, additional
seed, validation/test evaluation, or further dataset was run.

Artifacts: `outputs/retinamnist/phase3_18a_ce_direction_robustness/`.
