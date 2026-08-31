# Phase 3.1 — Existing Imbalance-Aware Ordinal Baselines

Weighted CE used training-only normalized inverse-frequency weights `[0.2937,
1.1151, 0.6929, 0.7357, 2.1626]` from counts `[486,128,206,194,66]`.

SLACE follows Nachmani et al. (AAAI 2025), Eq. 7--11: a Softmax probability
vector, training-count proximity/distance, soft labels, and accumulating cross
entropy. The paper and official repository were verified; alpha was fixed at 1.

All 17 tests passed. Two-epoch smoke runs were finite and reduced validation loss:
WCE 1.9267 to 1.3711; SLACE 0.9482 to 0.7204.

Weighted CE full seed 0 selected epoch 11 (validation loss 1.2298). Its L1
accuracy/MAE/severe rate were 0.498/0.768/19.5%. Class 4 remained 0% accurate,
with MAE 2.20, 75% severe errors, p4 0.120, p3+p4 0.360, predictive mean 1.873,
inward shrinkage 1.473, and L1 risk 0.779. It increases p4 but does not recover
class-4 decisions and worsens global performance: **STOP — SCIENTIFICALLY
NONCOMPETITIVE**.

SLACE was initially marked **IMPLEMENTATION CONCERN** because immediate directory
inspection appeared to show no finalized artifacts. The diagnosis below supersedes
that infrastructure observation. No seeds 1--4 or Phase 3.2 method were started.

## SLACE evaluation infrastructure diagnosis

The reported persistence failure was a false observation, not an exception in the
SLACE evaluation path. All three historical full-run directories subsequently
contained complete, byte-identical histories and metric artifacts. An
evaluation-only reproduction loaded the frozen selected checkpoint and reached
every staged operation through artifact finalization with exit code 0 and empty
stderr. SLACE and Weighted CE both supplied finite CPU logits/probabilities of
shape `(400, 5)`, integer labels of shape `(400,)`, and probability sums within
machine precision. The earlier check observed the directory before its completed
contents were visible.

A shared staged finalizer and synthetic regression test now verify Softmax
construction, mode/L1/L2 decisions, decision risks, shrinkage diagnostics, and
the required predictions/metrics/classwise/risk-coverage files. The scientific
analysis below uses those completed artifacts; no model was retrained.

## Seed-0 scientific comparison

This is a single-seed diagnostic, not a multi-seed estimate. CE, RPS, weighted
CE, and SLACE use the same 400 native-28x28 test examples in the same order. The
canonical SLACE run is
`outputs/retinamnist/native28/phase3_1_existing_baselines/slace/seed_0`.
It selected epoch 5 by minimum validation SLACE (0.6717907488). The
`seed_0`, `seed_0_complete`, and `seed_0_diagnostic` histories have the same
SHA-256 (`1a3350b2...c3cd8fd`), and their evaluation metrics have the same
SHA-256 (`65305359...44d99`); they are duplicate executions, not independent
replicates.

Raw probability-mode results were:

| Method | Accuracy | MAE | QWK | Severe % | NLL | Brier | RPS | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE | 0.560 | 0.685 | 0.594 | 18.75 | 1.186 | 0.577 | 0.124 | 0.064 |
| RPS | 0.538 | 0.695 | 0.576 | 19.25 | 1.213 | 0.566 | 0.122 | 0.062 |
| Weighted CE | 0.440 | 0.907 | 0.497 | 24.00 | 1.495 | 0.632 | 0.137 | 0.164 |
| SLACE | 0.510 | 0.730 | 0.537 | 20.50 | 1.234 | 0.590 | 0.126 | 0.089 |

For L1 decisions, CE/RPS/weighted CE/SLACE respectively achieved
accuracy 0.535/0.548/0.498/0.508, MAE 0.665/0.660/0.768/0.700, QWK
0.619/0.599/0.530/0.551, and severe prevalence 17.0/18.25/19.5/18.0%.
SLACE L2 improved its own global MAE and severe prevalence to 0.675 and 17.0%,
but did not resolve the rare upper extreme.

## Extreme-class geometry

The primary class-4 result is negative. Under the L1 decision, none of the four
methods predicted class 4 for any of its 20 test examples. The class-4 predicted
counts for classes 0/1/2/3/4 were:

| Method | 0 | 1 | 2 | 3 | 4 | MAE | Severe % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE | 1 | 1 | 9 | 9 | 0 | 1.70 | 55 |
| RPS | 1 | 2 | 15 | 2 | 0 | 2.10 | 90 |
| Weighted CE | 2 | 5 | 8 | 5 | 0 | 2.20 | 75 |
| SLACE | 2 | 3 | 13 | 2 | 0 | 2.25 | 90 |

SLACE therefore provides neither exact recovery nor meaningful adjacent
recovery: only 2/20 examples moved to class 3, while 18/20 remained severe.
Its class-4 mean p4 was 0.0697, mean p3+p4 was 0.3153, and predictive mean was
1.7765. These are all less favorable than CE (0.1048, 0.4101, 2.0448), RPS
(0.0837, 0.3524, 1.9133), and weighted CE (0.1197, 0.3596, 1.8734).
SLACE inward shrinkage was 1.5589: +0.0223 versus CE, -0.1437 versus RPS, and
+0.0860 versus weighted CE. Its high class-4 L1 Bayes risk (mean 0.8445,
median 0.8759) shows that it often recognizes risk without locating the upper
extreme.

The class-0 control reveals an asymmetric trade-off. SLACE L1 accuracy was
0.741, MAE 0.414, severe prevalence 15.5%, mean p0 0.745, p0+p1 0.832,
predictive mean 0.507, inward shrinkage 0.501, and L1 risk 0.303. It improves
the majority lower extreme while class 4 remains collapsed, so the result is
not a balanced recovery of both extremes.

## Decision-risk behavior

SLACE's L1-risk Spearman correlation was 0.440, higher than CE (0.323), RPS
(0.399), and weighted CE (0.361) in seed 0. However, its severe-error AUROC/AUPRC
(0.629/0.242) were below RPS (0.668/0.283), and its mean ordinal-MAE selective
risk (0.446) was worse than RPS (0.409), though better than CE (0.474) and
weighted CE (0.527). Mode-to-L1 correction occurred for 15.75% of SLACE
predictions versus 12.25% CE, 7.25% RPS, and 32.0% weighted CE. Thus SLACE
retains some useful risk ranking, but does not preserve RPS's full detection and
selective-prediction advantage while fixing the target failure.

## Research questions and decision

1. Generic weighted CE did not solve class-4 shrinkage: it raised p4 but left
   L1 accuracy at zero, class-4 MAE at 2.20, and substantially hurt global
   performance.
2. SLACE did not reduce shrinkage relative to CE, although it was less shrunken
   than RPS in this seed.
3. SLACE produced neither exact nor adjacent class-4 recovery.
4. SLACE decreased, rather than increased, class-4 p4 and p3+p4 versus every
   comparator.
5. SLACE improved majority class 0 but degraded global accuracy, MAE, QWK, and
   severe prevalence relative to CE.
6. SLACE had strong Spearman alignment, but weaker severe detection and selective
   MAE than RPS.
7. The unresolved failure is rare upper-extreme location collapse: existing
   methods can assign high risk without moving enough probability or decisions
   toward class 4.

SLACE status is **STOP — SCIENTIFICALLY NONCOMPETITIVE** and seeds 1--4 should
not run. The final Phase 3.1 interpretation is **SHRINKAGE PERSISTS AFTER STRONG
BASELINES**. The Phase 3.2 gate is **PHASE 3.2 JUSTIFIED**, narrowly, with the
target: reduce rare upper-extreme inward shrinkage while preserving RPS-like
ordinal risk alignment, severe-error detection, selective prediction, and global
predictive quality. This document does not propose or implement such a loss.
