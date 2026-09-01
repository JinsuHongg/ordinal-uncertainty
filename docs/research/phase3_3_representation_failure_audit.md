# Phase 3.3 — Frozen CE/RPS Representation-Level Failure Audit

## Scope and provenance

This is an inference-only seed-0 diagnostic. No CE or RPS model was trained,
fine-tuned, or otherwise updated. The final audited artifacts are under
`outputs/retinamnist/native28/phase3_3_representation_audit_replay_verified/`.
An earlier partial directory is preserved after its preprocessing replay gate
correctly stopped before RPS analysis; it is not used for results.

| Model | Selected checkpoint | Saved test predictions | Replay preprocessing |
| --- | --- | --- | --- |
| CE | `outputs/retinamnist/resolution_sanity_check/seed_0/size_28/best_checkpoint.pt` | `.../size_28/predictions.csv` | Resize(28), ToTensor, Normalize(0.5, 0.5) |
| RPS | `outputs/retinamnist/native28/phase2_model_comparison/rps/seed_0_artifact_complete/best_checkpoint.pt` | `.../evaluation/predictions.csv` | Resize(28), ToTensor, Normalize(0.5, 0.5) |

The CE checkpoint is the frozen native-28 seed-0 run reused by the resolution
sanity check. The RPS checkpoint is its selected Phase-2 seed-0 checkpoint.
Both replay their stored test logits exactly (maximum absolute error `0.0`). The
minimal RPS configuration did not record preprocessing; `ToTensor()` alone
failed replay (maximum logit error `10.0692`), whereas normalized input replayed
exactly. This is an artifact-provenance check, not a new preprocessing choice.

For every official split, the audit extracted the input to `model.fc` using a
forward pre-hook: the 512-dimensional ResNet18 vector after average pooling and
flattening, immediately before the final linear classifier. Train/validation/test
counts are 1080/120/400. Training features alone define all centroids and
dispersion references; validation and test features are evaluation-only.

The raw analysis uses Euclidean distance. The normalized analysis L2-normalizes
each sample before averaging to form a centroid, then uses cosine distance to the
L2-normalized centroid. No favorable feature convention was selected post hoc.

## Train-centroid geometry

Raw Euclidean centroid-distance matrices (rows/columns classes 0--4):

| CE | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000 | 13.173 | 13.981 | 14.641 | 15.284 |
| 1 | 13.173 | 0.000 | 2.251 | 3.378 | 4.670 |
| 2 | 13.981 | 2.251 | 0.000 | 1.411 | 2.641 |
| 3 | 14.641 | 3.378 | 1.411 | 0.000 | 1.698 |
| 4 | 15.284 | 4.670 | 2.641 | 1.698 | 0.000 |

| RPS | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000 | 13.703 | 14.975 | 15.672 | 16.067 |
| 1 | 13.703 | 0.000 | 1.703 | 2.593 | 3.414 |
| 2 | 14.975 | 1.703 | 0.000 | 0.932 | 1.786 |
| 3 | 15.672 | 2.593 | 0.932 | 0.000 | 0.963 |
| 4 | 16.067 | 3.414 | 1.786 | 0.963 | 0.000 |

Both matrices have the expected endpoint ordering: from centroid 4, class 3 is
closest, followed by 2, 1, and 0; the analogous ordering holds from centroid 0.
The same radial ordinal ordering holds under normalized cosine geometry. However,
the normalized class-3/class-4 cosine separation is smaller for RPS (0.00309)
than CE (0.00650); raw separation is likewise smaller (0.963 versus 1.698).
This is descriptive because feature scales differ across independently trained
models.

## Class-4 test distance and routing

Each value below is over 20 true class-4 test examples.

| Model / space | d0 | d1 | d2 | d3 | d4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CE raw mean | 14.692 | 5.780 | 4.988 | 4.819 | 4.920 |
| CE raw median | 15.329 | 5.253 | 4.174 | 3.807 | 3.489 |
| RPS raw mean | 15.239 | 3.969 | 3.232 | 3.134 | 3.359 |
| RPS raw median | 16.085 | 3.274 | 2.210 | 2.088 | 2.456 |
| CE cosine mean | 0.194 | 0.083 | 0.067 | 0.066 | 0.073 |
| RPS cosine mean | 0.245 | 0.056 | 0.051 | 0.054 | 0.061 |

For \(\Delta_{4,j}=d_j-d_4\), positive values favor class 4. The test margins
show frequent central proximity:

| Model / space | \(\Delta_{4,3}\) mean / median | < 0 | \(\Delta_{4,2}\) mean / median | < 0 |
| --- | ---: | ---: | ---: | ---: |
| CE raw | -0.102 / -0.491 | 55% | 0.068 / -0.546 | 55% |
| CE cosine | -0.0066 / -0.0089 | 55% | -0.0052 / -0.0096 | 55% |
| RPS raw | -0.224 / -0.468 | 70% | -0.127 / -0.529 | 60% |
| RPS cosine | -0.0074 / -0.0084 | 70% | -0.0101 / -0.0095 | 60% |

Nearest-centroid routing provides the complementary result:

| Model / space | 4→0 | 4→1 | 4→2 | 4→3 | 4→4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CE raw | 2 | 5 | 3 | 1 | 9 |
| CE cosine | 2 | 6 | 2 | 1 | 9 |
| RPS raw | 1 | 3 | 7 | 3 | 6 |
| RPS cosine | 2 | 5 | 4 | 3 | 6 |

Thus class-4 feature collapse exists for a substantial subset—11/20 CE and
14/20 RPS raw features are not closest to the class-4 centroid—but it is not
universal. Validation routing is more favorable (CE 5/6 and RPS 4/6 route to
class 4), so the test result should not be generalized from one 20-case endpoint
sample.

## Representation versus classifier head

The head does not simply preserve nearest-centroid class-4 information. In raw
space, CE has 9/20 true class-4 samples nearest to centroid 4, but its mode
predicts class 3 for all nine and its L1 decision also predicts class 3 for all
nine. RPS has 6/20 nearest to centroid 4; its mode routes four of those to class
2 and two to class 3, while L1 routes four to class 2 and two to class 3.

Across all true class-4 cases, raw nearest-centroid/head agreement is 5/20 mode
and 4/20 L1 for CE, versus 10/20 mode and 10/20 L1 for RPS. Therefore the
classifier head/output distribution is an important additional bottleneck, even
though the feature space itself is central for many examples.

## Dispersion and class-0 control

Class 4 is not unusually dispersed in the training representation. Raw
within-class mean distance for class 4 is 2.640 CE and 1.950 RPS, lower than
classes 0--3; normalized mean cosine dispersion is 0.187 CE and 0.162 RPS,
also the smallest class value. This favors neither a high-variance rare-class
explanation nor a simple poorly estimated class-4 centroid explanation.

The lower endpoint is materially different. Under normalized nearest-centroid
routing, 118/174 CE and 122/174 RPS class-0 test examples route to class 0
(67.8% and 70.1%); raw routing is 92/174 and 98/174 (52.9% and 56.3%). Its
margin against class 1 is positive for 67.8% CE and 70.1% RPS normalized cases.
This supports an imbalance-and-upper-endpoint asymmetry rather than a universal
endpoint representation failure.

Global test nearest-centroid performance is only diagnostic, not a competing
classifier: CE raw/cosine accuracy is 0.408/0.465 with MAE 0.928/0.853; RPS is
0.395/0.435 with MAE 0.970/0.933. These weaker figures are expected because the
linear head is trained for the task and nearest-centroid classification is used
only to probe the representation.

## CE versus RPS and decision

RPS does **not** improve class-4 representation separation in this seed. Relative
to CE, it has more negative class-4 margins, fewer class-4 nearest-centroid
assignments (6/20 versus 9/20), and smaller class-3/class-4 centroid separation
in both raw and normalized analyses. RPS can nevertheless retain stronger
probability-risk behavior because representation separation and output-risk
quality are distinct factors.

**Failure classification: MIXED REPRESENTATION / HEAD FAILURE.** The evidence
does not support a head/probability-only explanation: many true class-4 features
are closer to central centroids. It also does not support strong universal
representation collapse: a meaningful subset is already closest to the class-4
training centroid and is then mapped centrally by the head/decision pipeline.

The next method family, if separately authorized, should therefore consider both
representation and probabilistic-head mechanisms. This audit does not select,
design, or implement a representation-learning method. Its limitations are one
seed, 20 class-4 test examples, centroid-based geometry rather than a learned
metric, and independently trained feature scales.
