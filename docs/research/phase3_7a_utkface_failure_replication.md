# Phase 3.7A-UTKFace — Ordinal Failure Replication

## Question

This seed-0 replication asked whether the two linked RetinaMNIST findings
transfer to an independent ordinal image dataset: (1) RPS gives more useful
ordinal decision-risk quality than CE, and (2) a rare upper extreme remains
inward-shrunk and poorly localized.

This is a baseline replication only. No new method, representation intervention,
head intervention, calibration refit, or multi-seed run was performed.

## Dataset contract and frozen split

The local corpus is `/mnt/storage/data/utkface/UTKFace`, the provider-distributed
UTKFace cropped/aligned filename corpus. All 23,708 files were valid JPEG images;
none was unparseable or undecodable. The first underscore-delimited filename
field was parsed as age; the observed range was 1–116 years. Filename metadata
also encodes sex and race, but those fields were not used as inputs or targets.

The archived OCQR manifest
`ordinal-cqr/data/manifests/conference_v0_3/utkface/manifest.jsonl` was reused
unchanged. It is a frozen image-level, stratified seed-0 split. Its calibration
partition was preserved but unused. The ordinal target was frozen before training:

| Class | Age interval |
| --- | --- |
| 0 | age < 20 |
| 1 | 20 <= age < 40 |
| 2 | 40 <= age < 60 |
| 3 | 60 <= age < 80 |
| 4 | age >= 80 |

| Split | Total | C0 | C1 | C2 | C3 | C4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 14,224 | 2,756 | 7,128 | 2,726 | 1,210 | 404 |
| Validation | 2,371 | 459 | 1,188 | 455 | 202 | 67 |
| Test | 2,371 | 459 | 1,189 | 454 | 202 | 67 |

Class 4 is genuinely rare (2.8% of each evaluation split); class 1 is the
majority class (about 50.1%). The train majority/minority ratio is 17.6:1.
The test upper endpoint has 67 examples, adequate for a descriptive seed-0
audit but not for formal inference.

## Controlled protocol

Both methods used RGB input, `Resize(128, 128)`, ImageNet normalization, and
training-only random horizontal flip; there was no new face alignment/crop and
no pretrained weight. The backbone was an unpretrained small-stem ResNet18
(3x3 stride-1 first convolution; no max-pool), AdamW (`lr=1e-4`,
`weight_decay=0.01`), seed 0, and a ten-epoch budget. The available 5.6-GiB
local GPU could not fit the historical batch size of 128, so a fixed batch size
of 32 was used for both CE and RPS; this was a capacity constraint, not tuning.

CE used categorical cross-entropy and selected minimum validation CE. RPS used
the verified normalized differentiable RPS and selected minimum validation RPS.
The selected epoch was 6 for both methods: CE validation CE 0.6095 and RPS
validation RPS 0.04984. Smoke forward/backward checks were finite for both
methods; CE loss was 1.3737 and RPS loss was 0.1461 on their first training
batch. Local GPU execution was used because no usable SLURM submit command was
available in this environment.

All test outputs contain 2,371 exactly aligned sample IDs and labels. Both
probability arrays are finite, nonnegative, and normalized to numerical
tolerance (maximum sum error: CE `2.05e-7`, RPS `2.08e-7`).

## Seed-0 global results

| Method | Decision | Accuracy | MAE | QWK | Severe count | Severe % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CE | Mode | 0.7596 | 0.2518 | 0.8386 | 23 | 0.970% |
| CE | L1 | 0.7562 | 0.2526 | 0.8399 | 18 | 0.759% |
| CE | L2 | 0.7571 | 0.2518 | 0.8397 | 20 | 0.844% |
| RPS | Mode | 0.7368 | 0.2779 | 0.8152 | 30 | 1.265% |
| RPS | L1 | 0.7318 | 0.2805 | 0.8160 | 26 | 1.097% |
| RPS | L2 | 0.7296 | 0.2801 | 0.8187 | 22 | 0.928% |

Probability metrics for CE versus RPS were respectively: NLL 0.5882 vs 0.6328,
Brier 0.3284 vs 0.3546, RPS 0.04563 vs 0.04933, and ECE 0.01458 vs 0.04229.
Thus RPS did not improve global predictive or probability quality in this
single UTKFace run.

For matched L1 risk and L1 decisions, CE versus RPS was: Spearman 0.4312 vs
0.4265; severe AUROC 0.8467 vs 0.8804; severe AUPRC 0.1473 vs 0.1242; and mean
ordinal-MAE selective risk 0.1092 vs 0.1301 (lower is better). Severe prevalence
was 0.759% (18/2371) for CE and 1.097% (26/2371) for RPS. Therefore the higher
RPS AUROC alone is not a replicated broad risk-quality advantage, especially
because its AUPRC, alignment, and selective MAE are worse and severe prevalence
is higher. Mode-to-L1 corrections were small: 2.02% CE and 2.70% RPS.

## Extreme-class audit

The oldest bin remains difficult and high risk, but the mechanism is milder
than in RetinaMNIST and RPS partially improves it. Under L1 decisions:

| Method | Class-4 acc. | Class-4 MAE | Severe % | Mean / median p4 | Mean / median p3 | Mean / median p3+p4 | Mean predictive mean | Inward shrinkage | Mean / median L1 risk |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- |
| CE | 0.3582 | 0.7313 | 5.97% | 0.4376 / 0.4332 | 0.4291 / 0.4578 | 0.8667 / 0.9531 | 3.2472 | 0.7109 | 0.4352 / 0.4075 |
| RPS | 0.5224 | 0.5821 | 5.97% | 0.4982 / 0.5296 | 0.3429 / 0.3706 | 0.8411 / 0.9362 | 3.2718 | 0.6863 | 0.4949 / 0.5074 |

The class-4 L1 routing counts `[4→0, 4→1, 4→2, 4→3, 4→4]` were CE
`[1, 0, 3, 39, 24]` and RPS `[1, 1, 2, 28, 35]`. RPS therefore delivered
meaningful exact and adjacent recovery relative to CE, while leaving 32/67
upper-extreme examples inward-routed. It did not eliminate inward predictive
location: both means remain about 0.7 ordinal classes below the true endpoint.

The lower endpoint is easier and serves as the asymmetry control. CE versus RPS
class-0 L1 accuracy was 0.7843 vs 0.6754; MAE 0.2222 vs 0.3312; severe rate
0.65% for both; mean p0 0.7577 vs 0.6569; and inward displacement 0.2791 vs
0.3890. Hence RPS's better class-4 recovery is accompanied by lower-endpoint
and global degradation, rather than a uniform endpoint improvement.

## RetinaMNIST replication assessment

| Finding | UTKFace status | Interpretation |
| --- | --- | --- |
| RPS improves risk/error association | NOT REPLICATED | L1 Spearman is slightly lower for RPS. |
| RPS improves severe-error detection | PARTIALLY REPLICATED | AUROC is higher, but AUPRC is lower at higher RPS severe prevalence. |
| RPS improves ordinal selective prediction | NOT REPLICATED | RPS mean selective MAE is higher. |
| Upper extreme has elevated risk | REPLICATED | Class-4 L1 risk is well above class-0 risk. |
| Upper extreme is inward-shrunk | REPLICATED | Predictive means are 3.25/3.27, below 4. |
| Upper extreme remains poorly localized | PARTIALLY REPLICATED | CE fails strongly; RPS recovers 35/67 exact L1 cases. |
| L1/L2 correction is insufficient | PARTIALLY REPLICATED | Corrections are small and leave substantial class-4 inward routing. |
| Lower endpoint is easier | REPLICATED | Class-0 MAE remains much lower than class-4 MAE. |

## Decision and limitations

\[
\boxed{\text{PARTIAL REPLICATION}}
\]

UTKFace reproduces rare upper-endpoint imbalance, elevated decision risk, and
inward predictive displacement. It does **not** reproduce the central
RetinaMNIST claim that RPS has a broad L1 risk-quality advantage over CE; RPS
only improves severe AUROC, while degrading the other primary matched L1
risk-quality measures and global prediction. It also partly recovers class 4
instead of failing to recover it under all decisions.

Accordingly, a UTKFace representation/head audit is **not automatically
justified** from this single seed-0 baseline gate. It would require a separate
authorization and must not be used to tune bins, objectives, or architecture to
force replication. The solar branch remains **PAUSED BEFORE TRAINING**: its
three-channel implementation and normalization retry existed, but no CE/RPS
full training or scientific solar conclusion occurred.

Artifacts are under `outputs/utkface/phase3_7a_failure_replication/`; summary
tables are in its `summary/` directory.
