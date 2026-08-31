# Phase 3.2 — Endpoint-Neighborhood RPS, Seed-0 GO/NO-GO

## Scope and scientific target

This is the authorized seed-0 diagnostic for **RPS + Endpoint-Neighborhood
Mass Correction** only.  It tests rare upper-extreme ordinal location shrinkage:
true class-4 examples are high-risk but their probability mass and mode/L1/L2
actions remain too central.  The desired movement is from central severe errors
to class 3 or 4, without discarding the useful RPS risk geometry.

Only seed 0 was run.  Candidates 2 and 3 were not implemented, no seeds 1--4
were launched, no test result selected a checkpoint or lambda, and no novelty
claim is made.

An interrupted launcher attempt left an empty lambda-0.1 directory, and a
same-configuration retry was subsequently finalized before the canonical
`lambda_0p1/seed_0` run.  Both are preserved under ignored `outputs/` per the
workspace safety policy; only the explicitly named canonical `seed_0` artifacts
are used below, and the retry is not an additional scientific replicate.

## Objective and implementation

For Softmax probabilities \(p=\operatorname{softmax}(z)\), the training loss
was

\[
\ell(z,y)=\ell_{\mathrm{RPS}}(z,y)+
\lambda\,\mathbf 1\{y\in\{0,K-1\}\}w_y
\left[-\log\left(\sum_{k:|k-y|\le1}p_k\right)\right].
\]

The correction is zero for interior labels.  It uses `{0,1}` for lower-endpoint
examples and `{K-2,K-1}` for upper-endpoint examples, so no RetinaMNIST class
identity is hard-coded.  Safe clamping at the floating-point dtype floor is
used before the logarithm.

The endpoint weights are computed from training counts only:

\[
w_y=\sqrt{\frac{\min_{e\in\{0,K-1\}}n_e}{n_y}}.
\]

With counts `[486, 128, 206, 194, 66]`, this gives `w0=0.3686`,
`w4=1.0000`, and zero correction weights for interior labels.  This is relative
endpoint balancing (majority-endpoint attenuation), not generic rare-class
up-weighting.

Implementation is in the shared Softmax/RPS loss module and one configurable
seed-0 trainer.  Unit tests verify interior equivalence, endpoint symmetry,
near-neighborhood preference, far-mass penalty, lambda-zero RPS equivalence,
and finite gradients.

## Protocol

* Canonical native 28x28 RetinaMNIST, official split, unpretrained small-image
  ResNet18, batch size 64, AdamW (`1e-3`, weight decay `1e-4`), 20 epochs,
  seed 0, CPU.
* Predeclared sensitivity values: `lambda in {0.1, 0.3, 1.0}`.
* Every run selected its checkpoint by **minimum validation RPS only**.
  Class-4 validation values (six examples) and test values did not select a
  checkpoint or a lambda.
* The two-epoch lambda-0.3 smoke run was finite, selected epoch 2 by validation
  RPS (`0.155814`), and finalized all probability artifacts successfully.  It
  is infrastructure evidence only.

| lambda | selected epoch | selected validation RPS |
| ---: | ---: | ---: |
| 0.1 | 5 | 0.117412 |
| 0.3 | 3 | 0.118836 |
| 1.0 | 4 | 0.121264 |

The frozen seed-0 prediction files for CE/RPS/weighted CE/SLACE are not
present in this checkout (`outputs/` is ignored), so frozen RPS comparison
values below are the documented Phase 3.1 seed-0 values.  They are not
retrained or regenerated here.

## Overall probability-mode metrics

| Method | Accuracy | MAE | QWK | Severe % | NLL | Brier | RPS | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RPS (frozen documented) | 0.538 | 0.695 | 0.576 | 19.25 | 1.213 | 0.566 | 0.122 | 0.062 |
| lambda 0.1 | 0.510 | 0.788 | 0.555 | 18.50 | 1.399 | 0.592 | 0.126 | 0.071 |
| lambda 0.3 | 0.510 | 0.800 | 0.558 | 19.50 | 1.699 | 0.664 | 0.137 | 0.128 |
| lambda 1.0 | 0.505 | 0.770 | 0.579 | 18.25 | 2.201 | 0.685 | 0.133 | 0.185 |

## Decision-rule metrics

| lambda | Rule | Accuracy | MAE | QWK | Severe count | Severe % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| RPS | L1 | 0.548 | 0.660 | 0.599 | 73 | 18.25 |
| 0.1 | Mode | 0.510 | 0.788 | 0.555 | 74 | 18.50 |
| 0.1 | L1 | 0.525 | 0.683 | 0.594 | 65 | 16.25 |
| 0.1 | L2 | 0.453 | 0.738 | 0.548 | 67 | 16.75 |
| 0.3 | Mode | 0.510 | 0.800 | 0.558 | 78 | 19.50 |
| 0.3 | L1 | 0.505 | 0.780 | 0.570 | 74 | 18.50 |
| 0.3 | L2 | 0.488 | 0.720 | 0.563 | 70 | 17.50 |
| 1.0 | Mode | 0.505 | 0.770 | 0.579 | 73 | 18.25 |
| 1.0 | L1 | 0.490 | 0.760 | 0.561 | 70 | 17.50 |
| 1.0 | L2 | 0.512 | 0.678 | 0.571 | 66 | 16.50 |

## Primary class-4 result: L1 routing and geometry

The frozen RPS L1 route is `1/2/15/2/0` for `4->0/1/2/3/4`.  The candidates
clearly change *decision location*, not merely confidence, but none creates an
exact class-4 decision.

| Method | 4->0 | 4->1 | 4->2 | 4->3 | 4->4 | Accuracy | MAE | Severe % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RPS | 1 | 2 | 15 | 2 | 0 | 0.00 | 2.10 | 90 |
| lambda 0.1 | 2 | 4 | 4 | 10 | 0 | 0.00 | 1.90 | 50 |
| lambda 0.3 | 2 | 4 | 0 | 14 | 0 | 0.00 | 1.70 | 30 |
| lambda 1.0 | 3 | 6 | 0 | 11 | 0 | 0.00 | 2.05 | 45 |

| Method | Mean p4 | Median p4 | Mean p3+p4 | Median p3+p4 | Predictive mean | Inward shrinkage | Mean L1 Bayes risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RPS | 0.0837 | unavailable | 0.3524 | unavailable | 1.9133 | 1.7026 | unavailable |
| lambda 0.1 | 0.1263 | 0.1419 | 0.4514 | 0.4916 | 1.9722 | 1.4986 | 0.9540 |
| lambda 0.3 | 0.1206 | 0.1421 | 0.5226 | 0.5910 | 2.0254 | 1.4220 | 0.8165 |
| lambda 1.0 | 0.0170 | 0.0184 | 0.4745 | 0.5216 | 1.6981 | 1.4263 | 0.7692 |

At lambda 0.1 and 0.3, the near-endpoint increase comes from **both** p3 and
p4: relative to documented RPS, estimated mean p3 changes from `0.2687` to
`0.3251` and `0.4020`, while p4 rises from `0.0837` to `0.1263` and `0.1206`.
At lambda 1.0, the whole near-mass gain is p3 (`0.4575`) while p4 collapses to
`0.0170`; this is adjacent shaping rather than true-endpoint localization.

Thus lambda 0.3 gives the clearest adjacent recovery (15 central routes become
zero, and 14/20 route to class 3), but it remains no exact recovery.  Lambda
0.1 also creates meaningful adjacent movement but routes more cases to 0/1.

## Class-0 control under L1

| lambda | Accuracy | MAE | Severe % | L1 routes 0->0/1/2/3/4 | Mean p0 | Mean p0+p1 | Predictive mean | Inward shrinkage | Mean L1 risk |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.718 | 0.540 | 21.26 | 125/12/29/8/0 | 0.700 | 0.780 | 0.708 | 0.655 | 0.475 |
| 0.3 | 0.701 | 0.684 | 21.26 | 122/15/7/30/0 | 0.689 | 0.808 | 0.714 | 0.596 | 0.403 |
| 1.0 | 0.741 | 0.420 | 8.05 | 129/31/0/14/0 | 0.769 | 0.886 | 0.463 | 0.444 | 0.278 |

Lambda 1.0 sharpens the lower endpoint strongly, while lambda 0.3 introduces
30 class-0-to-3 errors.  These endpoint asymmetries are material controls, not
secondary details.

## L1 risk geometry

| Method | Spearman | Severe AUROC | Severe AUPRC | Mean ordinal-MAE selective risk | Mode->L1 changed % |
| --- | ---: | ---: | ---: | ---: | ---: |
| RPS (frozen documented) | 0.399 | 0.668 | 0.283 | 0.409 | unavailable |
| lambda 0.1 | 0.360 | 0.641 | 0.255 | 0.464 | 41.00 |
| lambda 0.3 | 0.428 | 0.646 | 0.261 | 0.498 | 6.75 |
| lambda 1.0 | 0.478 | 0.661 | 0.319 | 0.465 | 13.00 |

Risk/error Spearman remains useful or increases for lambda 0.3/1.0, and lambda
1.0 improves AUPRC.  However, every candidate has worse mean ordinal-MAE
selective risk than documented RPS; the two lower lambdas also reduce severe
AUROC/AUPRC.  This fails the intended preservation criterion.

## Per-lambda interpretation and decision

* **lambda 0.1:** meaningful p3 and p4 growth plus 10 class-4-to-3 decisions;
  class-4 MAE/severe prevalence improve.  It has the smallest global L1 cost,
  but risk alignment/detection weaken and selective MAE rises from 0.409 to
  0.464.
* **lambda 0.3:** strongest and clearest adjacent class-4 recovery (14/20
  routed to 3; no 4-to-2 routes), with p3 and p4 both higher.  Its global L1
  MAE is much worse (0.780 vs 0.660) and selective MAE deteriorates to 0.498;
  class-0-to-3 errors also increase.
* **lambda 1.0:** removes class-4-to-2 routes and preserves/improves several
  risk rankings, but suppresses p4 and moves mass almost entirely to p3.  It
  loses global L1 accuracy/MAE and does not provide true-endpoint localization.

Overall decision:

\[
\boxed{\text{TRADE-OFF — REVIEW BEFORE MULTI-SEED}}
\]

The experiment supports the original mechanism in the limited sense that a
neighborhood-specific location term can move class-4 probability and L1 actions
out of the central class.  It does not establish an acceptable method: the
auxiliary event rewards class 3 and 4 equally, no lambda achieves exact class-4
recovery, and selective-risk/global controls are not jointly preserved.

No lambda is authorized for seeds 1--4.  If further review is explicitly
approved, lambda 0.1 is the least globally damaging location signal and lambda
0.3 is the clearest adjacent-recovery signal; neither is selected here and no
test-based choice is made.

## Artifacts and verification

Full artifacts are under
`outputs/retinamnist/native28/phase3_2_endpoint_neighborhood/` in isolated
lambda-specific directories, with configurations, histories, validation-RPS
selection summaries, checkpoints, compressed test arrays, probability
evaluation, decision/risk tables, endpoint routing, and risk-coverage curves.

Verification after implementation:

```text
pytest -q                 22 passed
python -m compileall ...  passed
git diff --check          passed
```
