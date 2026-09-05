# Phase 3.9 — Solar Rare-Extreme Mechanism Audit

## Decision

\[\boxed{\text{MIXED BUT DECOMPOSABLE FAILURE}}\]

Phase 3.8 established rare-X inward localization. This frozen-model audit shows
two distinct contributors: a nontrivial X representation-collapsed subset and
a much larger X-like subset that the original learned head nevertheless routes
inward. No new loss, backbone training, CE/RPS retraining, seed expansion,
channel variant, or method was introduced.

## Frozen provenance and protocol

The exact Phase 3.8 selected seed-0 checkpoints were reused:
`outputs/solar/phase3_8_shrinkage_confirmation/{ce,rps}/seed_0/selected_checkpoint.pt`.
Feature extraction captured the input to `model.fc`, after ResNet18 average
pooling and flattening: **512 dimensions**. It replayed Phase 3.8 test logits
within `8.58e-6` (CE) and `9.54e-6` (RPS), preserving all aligned identities.
The frozen aligned train/validation/test counts were 45,047 / 2,431 / 28,006,
with 562 / 24 / 921 X examples. Dataset, 3-channel order `[hmi_m,aia1600,aia131]`,
source indices `[8,7,1]`, normalization, timestamp alignment, class mapping,
and splits were unchanged.

Centroids use **training features only**. Raw geometry uses Euclidean distance.
Normalized geometry L2-normalizes every feature, averages training unit
features per class, L2-normalizes those centroids, and uses cosine distance
`1-u·v`. Positive margin `Δ4,j=d(z,μj)-d(z,μ4)` favors X. The original head
and all decision rules remain primary references.

SLURM jobs: initial feature attempts `4296142`/`4296143` failed before artifact
creation because `src/` was absent from batch `PYTHONPATH`; their dependent
geometry job `4296145` was cancelled. Corrected CE/RPS extraction `4296154` /
`4296155` and geometry `4296156` completed successfully. The conditionally
justified frozen-head control `4296780` completed successfully. Retry `4298329`
correctly refused to overwrite that completed output.

## Geometry results

| Model / geometry | X routing [→0,→1,→2,→3,→4] | X nearest X | Collapsed X | X-like head failure mode/L1/L2 |
|---|---|---:|---:|---|
| CE raw | [0,31,40,126,724] | 78.6% | 197 | 100% / 100% / 100% |
| CE cosine | [0,20,40,100,761] | 82.6% | 160 | 100% / 100% / 100% |
| RPS raw | [0,26,68,124,703] | 76.3% | 218 | 100% / 100% / 100% |
| RPS cosine | [0,22,50,120,729] | 79.2% | 192 | 100% / 100% / 100% |

Thus 17.4–23.7% of true X samples are representation-collapsed, mainly toward
M/C; 76.3–82.6% are already nearest to X, yet every one receives an inward
original-head decision. For CE raw / cosine, `Δ4,3` mean/median is 3.938/7.012
and .0123/.0168 (positive fractions 78.6%/82.6%); `Δ4,2` is 11.047/15.868 and
.0631/.0750 (86.6%/92.7%). For RPS raw / cosine, corresponding M margins are
3.098/5.752 and .00746/.0112 (76.3%/79.2%), and C margins are
10.069/14.937 and .0509/.0625 (83.9%/90.6%).

Raw centroid separations D(3,4)/D(2,4) are 7.911/17.244 (CE) and 6.982/16.885
(RPS); normalized cosine separations are .00736/.05289 (CE) and .00517/.04682
(RPS). X is not the most dispersed training class: raw X mean/median centroid
distance is 9.314/7.649 CE and 9.164/7.385 RPS, below class 3 in both models.

## Decomposition and lower-endpoint control

Under primary raw geometry, CE collapsed/X-like groups are 197/724 and RPS are
218/703. Collapsed CE/RPS X has mean p4 .0217/.0350, predictive mean 2.464/2.336,
L1 MAE 1.492/1.665, and severe 37.6%/56.4%. X-like CE/RPS still has mean p4
.0450/.0329, predictive mean 2.973/2.907, L1 MAE 1.012/1.051, and severe
1.24%/5.12%: the head fails even when geometry is X-like.

Class-0 is a strong asymmetry control. Raw nearest-centroid class-0 recovery is
94.1% CE and 94.6% RPS (normalized 94.6%/94.7%), versus 76–83% for X. Raw
class-0-vs-1 margins are positive for 94.1%/94.6%, with mean 7.079/6.416.
D(0,1) is 9.058/8.106 (raw) and .1119/.1015 (cosine) for CE/RPS. The lower
endpoint is therefore better localized in representation space, not merely by
head output.

## Frozen-head control

Because the X-like/head-failed subset was large, two fixed established linear
controls were fit only on frozen training features, selected by validation CE
(epoch 1), and evaluated once. They are diagnostics, not proposed methods.

| Source / head | L1 global acc / MAE | X exact L1 | X MAE | X p4 | X mean | X shrinkage | L1 Spearman |
|---|---|---:|---:|---:|---:|---:|---:|
| CE original | .5767 / .4565 | 0 | 1.115 | .040 | 2.864 | 1.136 | .254 |
| CE balanced | .5059 / .5653 | 520 | .562 | .542 | 3.315 | .685 | .150 |
| CE logit-adjusted | .4649 / .6363 | 620 | .441 | .619 | 3.411 | .589 | .123 |
| RPS original | .5849 / .4517 | 0 | 1.197 | .033 | 2.772 | 1.228 | .243 |
| RPS balanced | .5387 / .5146 | 432 | .701 | .476 | 3.220 | .780 | .157 |
| RPS logit-adjusted | .4902 / .5887 | 583 | .524 | .582 | 3.337 | .663 | .146 |

The controls demonstrate that simple head/prior correction can recover many
X outputs from frozen solar features, while worsening global MAE and risk/error
quality. Class-0 remains high-accuracy (.932–.943) but receives modestly
higher inward displacement. This is informative evidence of head-level bias,
not an authorization to adopt or tune a head control.

## RetinaMNIST comparison and implication

| Finding | RetinaMNIST | Solar |
|---|---|---|
| Substantial upper-extreme representation collapse | Yes | Yes (17–24%) |
| X-like samples fail at original head | Yes | Yes (100% of X-like) |
| Simple frozen head is informative | Yes, limited subset | Yes, many recoveries with trade-offs |
| Endpoint representation asymmetry | Yes | Yes |

Solar qualitatively replicates the RetinaMNIST mixed/decomposable mechanism,
although solar has a much larger centroid-X-like subset. The cross-dataset
shrinkage phenomenon therefore has both geometry and head/prior components in
this audit. This does **not** design a method or authorize one; any future
method decision must separately decide whether and how to address both factors.

Limitations: one seed, centroid geometry is diagnostic rather than a learned
metric, retained solar alignment subset, and frozen-head controls are fixed
standard diagnostics. UTKFace was not audited.
