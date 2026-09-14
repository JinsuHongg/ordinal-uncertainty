# Stage 1 Mechanism Provenance Audit

**Status:** A — STAGE 1 COMPLETE; RETINA RPS SEEDS 1--4 PROVENANCE VERIFIED
**Project:** Ordinal Uncertainty Quantification for Imbalanced Ordinal Classification
**Last updated:** 2026-09-13

This is a provenance and compatibility audit only. It did not train a backbone,
fit a new scientific head, evaluate new scientific results, or alter historical
results. Seed 0 remains hypothesis-forming; seeds 1--4 remain the only
confirmatory replication set.

## Decision

The historical CE and RPS direction-only head procedures are mechanically the
same intervention for the A/C comparison: frozen 512-D features; A initialized
from the original linear head; C initialized from A while fixing its row norms
and biases; balanced replacement sampling; CE head-adaptation loss; AdamW;
learning rate `1e-3`; batch size `64`; `100` fixed epochs; and zero weight
decay on C's direction parameter. Future CE/RPS A/C runs can therefore use one
standardized implementation and one strict integrity-check suite.

The proposed confirmatory H2 centroid rule is implementable and frozen below.
The RetinaMNIST RPS seed-1--4 replay has now resolved the last Stage 1 reuse
ambiguity. Solar seed-1--4 backbones do not exist locally and are planned future
runs, not reusable artifacts. This audit does not itself authorize training.

## 1. Canonical A/C implementation

`DirectionOnlyLinear` stores fixed norms and fixed biases as buffers and makes
only `direction` trainable. Its effective row is

\[
w_k^C=\|w_k^A\|_2\,\frac{v_k}{\|v_k\|_2},
\]

so no hidden layer, scale parameter, or bias parameter is optimized. All audited
heads are `Linear(512, 5)` and operate on frozen penultimate features.

| Property | Retina CE/RPS | Solar CE/RPS |
| --- | --- | --- |
| A condition | Original checkpoint head; no retraining | Original checkpoint head; no retraining |
| C objective / sampler | CE; balanced replacement on fitting folds | CE; balanced replacement on train |
| Optimizer / LR / batch / epochs | AdamW / `1e-3` / `64` / `100` | AdamW / `1e-3` / `64` / `100` |
| C weight decay | `0.0` | `0.0` |
| B-only weight decay | `1e-4` | `1e-4` |

The CE runners explicitly fail on cached-A or C-initialization replay error
above `2e-5`, and on C norm or bias error above `1e-6`. The historical Retina
RPS and Solar RPS scripts enforce the norm bound; their stored/documented
initial replay and fixed-bias checks are not uniformly runtime assertions.
This is an implementation-audit difference, not a different C parameterization
or objective. The future standardized runner must enforce all four checks:

- cached/original-head replay `<= 2e-5`;
- C-initialization replay `<= 2e-5`;
- fixed-norm error `<= 1e-6`;
- fixed-bias error `<= 1e-6`.

## 2. Retina OOF semantics

The historical Retina runner first uses one fixed, previously trained backbone
to obtain all 1,080 training features. It then applies deterministic five-fold
assignments only to linear-head fitting: a fold's held observations are not used
to fit that fold's B/C head, and each training observation is emitted once as a
held-out head prediction. Validation and test arrays are not loaded by the OOF
head runner.

The correct wording is:

> training-only out-of-fold head evaluation on a fixed learned representation

It is not backbone-level OOF training, fully cross-fitted representation
learning, or an independent backbone replication.

## 3. Historical centroid semantics

The historical Retina subgroup is computed from raw Euclidean centroids of the
full cached training feature archive. A true endpoint sample is `rare-end-like`
exactly when the endpoint centroid is nearest among **all five** centroids;
otherwise it is `representation-inward`. Its own feature can enter the
historical centroid. This is retained unchanged as a descriptive continuity
analysis only.

The historical Solar subgroup uses centroids from aligned training features and
evaluates them on the disjoint archived readout. Neither historical subgroup is
the new continuous H2 endpoint-vs-adjacent margin.

## 4. Solar provenance

The canonical Phase 3.8 CE/RPS artifacts use three channels
`[hmi_m, aia1600, aia131]` at `224x224`, a 5-class unpretrained torchvision
ResNet18 with a `512 -> 5` head, train-only normalization, AdamW (`5e-5`,
weight decay `0.01`), batch size `16`, up to `300` epochs, patience `3`, and
minimum validation CE/RPS checkpoint selection as appropriate.

The aligned split counts are 45,047 train, 2,431 validation, and 28,006 test;
the cached IDs are unique and pairwise disjoint. The test population is roughly
2020--2024 while train/validation are primarily 2010--2019, but the split is not
claimed to be fully chronological. The archived readout has been reused and is
not a new independent confirmation or a model-selection split.

## 5. RPS A/C implementation audit

Retina RPS Phase 3.10C and Solar RPS Phase 3.15 both use the same C training
mechanics as the CE robustness studies: original A weight/bias initialization,
`DirectionOnlyLinear`, frozen features/backbone, balanced replacement batches,
cross-entropy adaptation, AdamW `1e-3`, batch size `64`, 100 epochs, and C
direction weight decay `0.0`. Thus there is no objective/sampler discrepancy
that requires separate CE and RPS head protocols.

Future CE/RPS A/C protocols can be identical. The only required change is to
standardize the stricter replay and fixed-bias runtime assertions in Section 1;
this does not alter the scientific intervention.

## 6. Checkpoint compatibility inventory

### Provenance profiles

| Profile | Architecture / input | Preprocessing and split | Backbone optimizer / selection | Representation / head |
| --- | --- | --- | --- | --- |
| R-CE | Unpretrained small-image ResNet18; RGB `28x28`; 3x3 stride-1 stem; no max-pool | Official Retina train/val/test `1080/120/400`; Resize(28), ToTensor, Normalize(.5,.5,.5); train horizontal flip | AdamW `1e-3`, WD `1e-4`, batch `64`, 20 epochs; minimum validation NLL | `512`; `fc.weight (5,512)`, `fc.bias (5,)` |
| R-RPS-0 | Same small-image ResNet18; `fc (5,512)` verified | Official `1080/120/400`; historical replay uniquely verifies Normalize(.5,.5,.5) | AdamW `1e-3`, WD `1e-4`, batch `64`, 20 epochs; minimum validation RPS | `512`; `fc.weight (5,512)`, `fc.bias (5,)` |
| R-RPS-1--4 | Unpretrained small-image ResNet18; native RGB `28x28`; 3x3 stride-1 stem; no max-pool; `fc (5,512)` | Official local RetinaMNIST NPZ test labels and ordered IDs replay exactly; Normalize(.5,.5,.5) is uniquely verified against saved logits | AdamW `1e-3`, WD `1e-4`, batch `64`, 20 epochs; validation RPS | `512`; `fc.weight (5,512)`, `fc.bias (5,)` |
| S-CE / S-RPS | Unpretrained torchvision ResNet18; 3-channel `224x224`; `fc (5,512)` verified | Aligned manifests `45047/2431/28006`; train-only log1p normalization; channels `[hmi_m,aia1600,aia131]` | AdamW `5e-5`, WD `.01`, batch `16`, max 300, patience 3; minimum validation CE/RPS | `512`; `fc.weight (5,512)`, `fc.bias (5,)` |

`COMPATIBLE` means sufficient local provenance to reuse the checkpoint in the
frozen protocol. `AMBIGUOUS` means a checkpoint exists but cannot count until
the missing field is proven. `MISSING` means no candidate local backbone exists.

| Dataset | Objective | Seed | Checkpoint path | Exists | Profile | Selected epoch / rule | Reuse status | Exact reason |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| RetinaMNIST | CE | 0 | `outputs/retinamnist/resolution_sanity_check/seed_0/size_28/best_checkpoint.pt` | Yes | R-CE | minimum validation NLL; epoch not persisted | COMPATIBLE | Historical CE source; technical all-seed-descriptive compatibility only; excluded from confirmatory success criteria. |
| RetinaMNIST | CE | 1 | `outputs/retinamnist/native28/single_model_baseline/seed_1/best_checkpoint.pt` | Yes | R-CE | minimum validation NLL; epoch not persisted | COMPATIBLE | Complete native-28 config and `5x512` head; fresh feature extraction required. |
| RetinaMNIST | CE | 2 | `outputs/retinamnist/native28/single_model_baseline/seed_2/best_checkpoint.pt` | Yes | R-CE | minimum validation NLL; epoch not persisted | COMPATIBLE | Complete native-28 config and `5x512` head; fresh feature extraction required. |
| RetinaMNIST | CE | 3 | `outputs/retinamnist/native28/single_model_baseline/seed_3/best_checkpoint.pt` | Yes | R-CE | minimum validation NLL; epoch not persisted | COMPATIBLE | Complete native-28 config and `5x512` head; fresh feature extraction required. |
| RetinaMNIST | CE | 4 | `outputs/retinamnist/native28/single_model_baseline/seed_4/best_checkpoint.pt` | Yes | R-CE | minimum validation NLL; epoch not persisted | COMPATIBLE | Complete native-28 config and `5x512` head; fresh feature extraction required. |
| RetinaMNIST | RPS | 0 | `outputs/retinamnist/native28/phase2_model_comparison/rps/seed_0_artifact_complete/best_checkpoint.pt` | Yes | R-RPS-0 | epoch 8; minimum validation RPS | COMPATIBLE | Historical RPS source with saved-logit replay and 512-D feature archive; excluded from confirmatory criteria. |
| RetinaMNIST | RPS | 1 | `outputs/retinamnist/native28/phase2_model_comparison/rps/seed_1_artifact_complete/best_checkpoint.pt` | Yes | R-RPS-1--4 | epoch 2; validation RPS | COMPATIBLE | Unique `Normalize(.5,.5,.5)` replay; ordered IDs and official test labels exactly match; max logit error `1.526e-05`. |
| RetinaMNIST | RPS | 2 | `outputs/retinamnist/native28/phase2_model_comparison/rps/seed_2_artifact_complete/best_checkpoint.pt` | Yes | R-RPS-1--4 | epoch 6; validation RPS | COMPATIBLE | Unique `Normalize(.5,.5,.5)` replay; ordered IDs and official test labels exactly match; max logit error `9.537e-06`. |
| RetinaMNIST | RPS | 3 | `outputs/retinamnist/native28/phase2_model_comparison/rps/seed_3_artifact_complete/best_checkpoint.pt` | Yes | R-RPS-1--4 | epoch 8; validation RPS | COMPATIBLE | Unique `Normalize(.5,.5,.5)` replay; ordered IDs and official test labels exactly match; max logit error `3.052e-05`. |
| RetinaMNIST | RPS | 4 | `outputs/retinamnist/native28/phase2_model_comparison/rps/seed_4_artifact_complete/best_checkpoint.pt` | Yes | R-RPS-1--4 | epoch 5; validation RPS | COMPATIBLE | Unique `Normalize(.5,.5,.5)` replay; ordered IDs and official test labels exactly match; max logit error `2.098e-05`. |
| Solar | CE | 0 | `outputs/solar/phase3_8_shrinkage_confirmation/ce/seed_0/selected_checkpoint.pt` | Yes | S-CE / S-RPS | epoch 1; minimum validation CE | COMPATIBLE | Historical archived-readout source; technical descriptive compatibility only; excluded from confirmatory criteria. |
| Solar | CE | 1 | — | No | S-CE / S-RPS | — | MISSING | No local seed-1 backbone checkpoint. |
| Solar | CE | 2 | — | No | S-CE / S-RPS | — | MISSING | No local seed-2 backbone checkpoint. |
| Solar | CE | 3 | — | No | S-CE / S-RPS | — | MISSING | No local seed-3 backbone checkpoint. |
| Solar | CE | 4 | — | No | S-CE / S-RPS | — | MISSING | No local seed-4 backbone checkpoint. |
| Solar | RPS | 0 | `outputs/solar/phase3_8_shrinkage_confirmation/rps/seed_0/selected_checkpoint.pt` | Yes | S-CE / S-RPS | epoch 1; minimum validation RPS | COMPATIBLE | Historical archived-readout source; technical descriptive compatibility only; excluded from confirmatory criteria. |
| Solar | RPS | 1 | — | No | S-CE / S-RPS | — | MISSING | No local seed-1 backbone checkpoint. |
| Solar | RPS | 2 | — | No | S-CE / S-RPS | — | MISSING | No local seed-2 backbone checkpoint. |
| Solar | RPS | 3 | — | No | S-CE / S-RPS | — | MISSING | No local seed-3 backbone checkpoint. |
| Solar | RPS | 4 | — | No | S-CE / S-RPS | — | MISSING | No local seed-4 backbone checkpoint. |

Twelve of 20 planned conditions have compatible existing backbones: Retina CE
seeds 0--4, Retina RPS seeds 0--4, Solar CE seed 0, and Solar RPS seed 0.
The local, inference-only replay report is
`outputs/retinamnist/mechanism_replication_provenance/rps_replay_audit.json`.

### RetinaMNIST RPS seed-1--4 deterministic replay

The audit script reconstructs the unpretrained small-image ResNet18 from each
checkpoint state dictionary, reads the official local NPZ test split, and tests
only the two historically evidenced candidates: ToTensor-only and
Resize(28)+ToTensor+Normalize(.5,.5,.5). Native images are already `28x28`, so
the latter's resize is an identity. The stored `sample_id` ordering is exactly
`0..399`, and stored labels exactly equal official `test_labels` for every
seed. This is an artifact-identity check only; it calculates no performance
metric or scientific outcome.

| Seed | Checkpoint | Verified transform | ID match | Label match | Max logit error | Status |
| ---: | --- | --- | --- | --- | ---: | --- |
| 1 | `outputs/retinamnist/native28/phase2_model_comparison/rps/seed_1_artifact_complete/best_checkpoint.pt` | Normalize(.5,.5,.5) | Yes | Yes | `1.526e-05` | COMPATIBLE |
| 2 | `outputs/retinamnist/native28/phase2_model_comparison/rps/seed_2_artifact_complete/best_checkpoint.pt` | Normalize(.5,.5,.5) | Yes | Yes | `9.537e-06` | COMPATIBLE |
| 3 | `outputs/retinamnist/native28/phase2_model_comparison/rps/seed_3_artifact_complete/best_checkpoint.pt` | Normalize(.5,.5,.5) | Yes | Yes | `3.052e-05` | COMPATIBLE |
| 4 | `outputs/retinamnist/native28/phase2_model_comparison/rps/seed_4_artifact_complete/best_checkpoint.pt` | Normalize(.5,.5,.5) | Yes | Yes | `2.098e-05` | COMPATIBLE |

ToTensor-only fails decisively for every seed (maximum absolute errors:
`24.0373`, `11.2541`, `44.0253`, and `23.0523`, respectively). Thus exactly
one canonical preprocessing choice passes per seed under the predeclared
`1e-4` tolerance.

## 7. H2 centroid rule

The following rule is frozen for the new confirmatory H2 analysis.

- **RetinaMNIST:** for OOF fold `f`, compute all five raw Euclidean centroids
  only from fixed-backbone training features whose fold is not `f`. Compute
  endpoint-vs-adjacent margin, all-centroid nearest class, and generic margin
  only for held fold `f`; concatenate held-fold rows after all folds. A held
  example never contributes to the centroid used for its H2 quantities.
- **Solar:** compute all five raw Euclidean centroids from aligned training
  features only. Compute all geometry quantities on the fixed archived
  evaluation population. Validation and test/readout features never contribute
  to a centroid.
- **Historical Figure 2 subgroup:** retain its original full-training-centroid
  definition as descriptive continuity only; do not overwrite or substitute it.

Existing feature archives contain the required Retina fold labels/features and
Solar training/readout features, so this rule is implementable without a new
method or result-driven choice.

## 8. H2 generic difficulty control

For the same training-derived centroids used by H2, sort all five distances for
evaluation sample `i` as \(d_{i,(1)}\le d_{i,(2)}\le\cdots\). Freeze

\[
g_i^{\mathrm{centroid}}=d_{i,(2)}-d_{i,(1)}.
\]

It is class-order-agnostic and therefore does not privilege endpoint or adjacent
classes. The endpoint-vs-adjacent variable remains separately defined as
\(d(h_i,\mu_{K-2})-d(h_i,\mu_{K-1})\).

## 9. Required future per-sample artifacts

Each new backbone/head condition must save: immutable checkpoint/config/hash;
split and sample IDs; backbone seed/objective; A/C head parameters; replay,
norm, bias, and backbone-freeze checks; features or immutable feature references;
fold IDs for Retina; training-derived centroid provenance; all-centroid nearest
class; endpoint-vs-adjacent and generic centroid margins; labels; A/C logits and
probabilities; A/C predictive means and shrinkages; rare-end probabilities;
adjacent margins; exact L1 decisions; and all predeclared global, endpoint, and
collateral metrics. These fields permit the fixed H1/H2 analysis without
retraining or hand-entered tables.

## 10. Remaining blockers

1. Solar CE/RPS seed-1--4 are the only missing planned backbone conditions;
   eight new backbone trainings are required before the complete 20-condition
   inventory exists. No existing artifact can replace those planned runs.
2. Add the standardized integrity assertions and required per-sample artifact
   contract to the future replication runner before any training authorization.

## 11. Training authorization status

**Stage 1 is complete.** This audit neither starts nor independently authorizes
backbone training, and it does not change the seed-0 historical or seeds-1--4
confirmatory roles.
