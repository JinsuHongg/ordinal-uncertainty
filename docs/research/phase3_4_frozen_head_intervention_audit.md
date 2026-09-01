# Phase 3.4 — Frozen-Feature Head Intervention Audit

## Motivation and scope

Phase 3.3 found mixed representation/head failure: some true class-4 test
examples were nearest to central class centroids, while others were nearest to
the class-4 training centroid but still received central predictions. This
seed-0 diagnostic asks how much can be recovered by changing only a linear
classifier head. It is not a proposed method.

The experiment used the frozen native-28 RetinaMNIST feature archives from
Phase 3.3: `features.npz` under
`outputs/retinamnist/native28/phase3_3_representation_audit_replay_verified/{ce,rps}/seed_0/`.
They contain train/validation/test features immediately before `model.fc`
(`h` in R^512), with counts 1080/120/400. No backbone was loaded, updated, or
fine-tuned; heads were trained directly on those fixed arrays.

## Controls and protocol

Each frozen feature source (CE and RPS) received exactly four `512 -> 5`
linear Softmax heads. AdamW used learning rate 0.001, weight decay 0.0001,
batch size 64, and a fixed 100-epoch budget. Every condition initialized and
shuffled deterministically with seed 0. Selection used validation data only:
minimum validation RPS for the RPS head and minimum ordinary validation CE for
the three CE-derived heads. The selected epochs were:

| Feature source | CE | RPS | Balanced CE | Logit-adjusted CE |
| --- | ---: | ---: | ---: | ---: |
| CE | 5 | 5 | 22 | 22 |
| RPS | 57 | 9 | 22 | 22 |

Training-class priors were `[0.4500, 0.1185, 0.1907, 0.1796, 0.0611]` from
counts `[486, 128, 206, 194, 66]`. Balanced CE used normalized inverse-prior
weights `[0.2937, 1.1151, 0.6929, 0.7357, 2.1626]`.

The prior-adjusted head follows the training-time logit-adjusted CE convention
of Menon et al.: `CE(z + log(pi_train), y)` with fixed `tau=1`; test
probabilities are `softmax(z)` without a post-hoc correction. This is a
standard long-tail control, not a calibrated or tuned classifier. [Menon et
al., ICLR 2021](https://openreview.net/pdf?id=37nvvqkCo5).

## Global results

The table shows L1 decision metrics. `Original` replays the frozen source
model, while the remaining rows are new heads on unchanged features.

| Feature source / head | Accuracy | MAE | QWK | Severe % |
| --- | ---: | ---: | ---: | ---: |
| CE / Original | 0.535 | 0.665 | 0.619 | 17.0 |
| CE / CE | 0.530 | 0.685 | 0.574 | 18.3 |
| CE / RPS | 0.538 | 0.673 | 0.587 | 17.0 |
| CE / Balanced CE | 0.548 | 0.678 | 0.616 | 18.0 |
| CE / Logit-adjusted | 0.545 | 0.690 | 0.613 | 18.3 |
| RPS / Original | 0.548 | 0.660 | 0.599 | 18.3 |
| RPS / CE | 0.540 | 0.673 | 0.586 | 18.3 |
| RPS / RPS | 0.540 | 0.668 | 0.585 | 17.8 |
| RPS / Balanced CE | 0.523 | 0.700 | 0.613 | 17.0 |
| RPS / Logit-adjusted | 0.523 | 0.700 | 0.617 | 17.0 |

Mode metrics worsened for the balancing/prior controls (CE features: 0.510 and
0.505 accuracy; RPS features: 0.513 and 0.493). L2 reduced severe prevalence
for several heads but retained its known accuracy trade-off. Full mode/L1/L2
metrics, NLL, Brier, RPS, ECE, risk curves, and checkpoint metadata are saved
in `outputs/retinamnist/native28/phase3_4_frozen_head_audit/summary/`.

## Class-4 geometry and routing

The primary result is below, evaluated with the L1 decision. `p3+p4` is the
near-upper-endpoint mass; lower inward shrinkage is better.

| Feature source / head | Class-4 Acc. | MAE | Severe % | Mean p4 | Mean p3+p4 | Mean mu | Shrinkage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE / Original | 0.00 | 1.70 | 55 | 0.105 | 0.410 | 2.045 | 1.537 |
| CE / CE | 0.00 | 2.05 | 70 | 0.103 | 0.333 | 1.787 | 1.536 |
| CE / RPS | 0.00 | 2.05 | 70 | 0.115 | 0.378 | 1.858 | 1.498 |
| CE / Balanced CE | 0.10 | 1.60 | 50 | 0.196 | 0.488 | 2.286 | 1.364 |
| CE / Logit-adjusted | 0.15 | 1.50 | 50 | 0.207 | 0.508 | 2.360 | 1.362 |
| RPS / Original | 0.00 | 2.10 | 90 | 0.084 | 0.352 | 1.913 | 1.703 |
| RPS / CE | 0.00 | 2.15 | 90 | 0.085 | 0.327 | 1.812 | 1.567 |
| RPS / RPS | 0.00 | 2.20 | 90 | 0.083 | 0.311 | 1.725 | 1.595 |
| RPS / Balanced CE | 0.00 | 1.75 | 55 | 0.156 | 0.451 | 2.164 | 1.533 |
| RPS / Logit-adjusted | 0.00 | 1.70 | 55 | 0.169 | 0.457 | 2.206 | 1.526 |

For CE features, L1 class-4 routing changed from `1/1/9/9/0` (predicted
0/1/2/3/4) under the original head to `1/1/8/7/3` with logit adjustment; it
therefore recovered three exact class-4 decisions and moved two additional
examples toward class 3. Balanced CE gave `1/2/7/8/2`. On RPS features, both
balancing controls improved adjacent recovery (`1/1/9/9/0` under logit
adjustment), but no exact class-4 L1 decision occurred.

## Feature-nearest stratification

This audit used the Phase 3.3 raw-Euclidean training-centroid assignment.

- CE features: 9/20 true class-4 samples were feature-nearest to class 4.
  The CE-feature logit-adjusted head exactly recovered 3/9 and recovered
  class 3 or 4 for 9/9; balanced CE exactly recovered 2/9 and recovered
  class 3 or 4 for 9/9. Of the 11 feature-nearest-other cases, 10/11 were not
  class-3-or-4 under any of the four simple heads.
- RPS features: 6/20 were feature-nearest to class 4. The balancing controls
  recovered class 3 or 4 for 6/6, but none exactly; 11/14
  feature-nearest-other cases were not class-3-or-4 under any head.

Thus the descriptive head-recoverable fractions are 9/9 for the best CE
feature heads (including exact recovery 3/9 for logit adjustment) and 6/6
adjacent-or-exact for the best RPS-feature heads, with a large remaining
representation-limited subset. This quantity is diagnostic only, not a new
metric.

## Class-0 control and risk quality

The head corrections did not cause class-0 collapse, but they did introduce a
global trade-off. For CE features, logit adjustment retained 70.1% class-0 L1
accuracy (vs 70.1% original) while class-0 MAE rose from 0.598 to 0.667. For
RPS features it retained 71.8% class-0 accuracy (vs 74.1%) and class-0 MAE
rose from 0.523 to 0.609.

Original RPS features retained the strongest L1 risk quality: Spearman 0.399,
severe AUROC 0.668, severe AUPRC 0.283, and mean selective MAE 0.409. The
RPS-head control on RPS features gave Spearman 0.411 but weaker AUROC/AUPRC
(0.653/0.256) and selective MAE (0.431). Balancing/prior heads recovered upper
mass but weakened risk geometry (for RPS features, logit-adjusted: 0.367,
0.622, 0.227, and 0.465 respectively). Full curves are retained in the
summary artifacts.

## Interpretation

**Outcome: MIXED BUT DECOMPOSABLE FAILURE.** A standard CE or RPS head alone
does not repair the endpoint collapse. Generic balancing and standard
logit-adjusted CE can act on the feature-nearest-to-class-4 subset, including
some exact recovery on CE features, but leave most feature-nearest-central
examples unrecovered and trade off global/risk quality. The audit therefore
supports treating representation and head/prior mechanisms separately in any
future design; it does not select a new method.

Limitations: this is one frozen seed with 20 class-4 test examples and a
single fixed tau=1 diagnostic; no method ranking or multi-seed claim is made.
