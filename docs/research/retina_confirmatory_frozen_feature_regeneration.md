# RetinaMNIST Confirmatory Frozen Feature Regeneration

**Verdict:** **A — RETINA CONFIRMATORY FEATURES REGENERATED; N FOLLOW-UP READY**

## 1. Motivation and prior blocker

The RetinaMNIST confirmatory A/C analysis retained original-head states, C
states, OOF metadata, and per-sample A/C outputs, but not the 512-D frozen
feature matrices required for a later natural-sampling direction-only head
comparison. Historical seed-0 features were deliberately not substituted.

## 2. Scope

Only deterministic inference was run for RetinaMNIST CE and RPS backbones,
seeds 1–4. No backbone was trained, no classifier head was fit, no A/C or
H1/H2 result was recomputed, and Solar was not read or modified.

## 3. Feature, architecture, and data identity

Each archive contains the 512-D input to `model.fc` after average pooling and
flattening from the existing unpretrained small-image ResNet18: native 28×28
RGB input, 3×3 stride-1 stem, and `Identity` max-pool. The official local
RetinaMNIST train split was loaded in its original index order with
`ToTensor()` and per-channel `Normalize(.5, .5)`. No augmentation was applied.

The CE source checkpoints are the existing native-28 single-model baselines;
the RPS checkpoints are the existing Phase-2 RPS artifacts. Every archive is
bound to the checkpoint path and SHA256 recorded in its local manifest.

## 4. OOF preservation and counts

Each `features.npz` contains `features` (`float32`, 1080×512), `labels`,
`sample_ids`, original `indices`, `folds`, and `split_roles`. IDs retain the
natural official-train order 0–1079. The stored folds exactly reproduce the
archived deterministic five-fold assignment (`seed=0`), with 216 rows per
fold; the same frozen backbone is shared by all folds within a seed.

The class counts are `[486, 128, 206, 194, 66]` for every archive.

## 5. Extraction and identity results

| Objective | Seeds | Selected epochs | Determinism | A-output reproduction | H1 A identity |
|---|---|---|---|---|---|
| CE | 1, 2, 3, 4 | 7, 6, 5, 11 | PASS; 64-row repeat max error 0 | PASS; logits/probabilities/L1 exact | PASS |
| RPS | 1, 2, 3, 4 | 2, 6, 8, 5 | PASS; 64-row repeat max error 0 | PASS; logits/probabilities/L1 exact | PASS |

For every setting, regenerated features applied to the archived A head yielded
zero maximum absolute error against archived A logits and probabilities, and
the exact-L1 decisions matched exactly. The optional C-state replay was not
performed: it is not needed for the required A identity gate and no C head was
retrained.

## 6. Artifacts and provenance

The eight feature archives and checksummed manifests are under:

`outputs/mechanism_replication/features/retina/{ce,rps}/seed_{1,2,3,4}/`

Each directory contains `features.npz` and `feature_manifest.json`, including
checkpoint/archive SHA256 values, source A/C artifact paths, feature contract,
counts, fold provenance, deterministic-repeat result, runtime, and A-output
reproduction result. Extraction ran on local CPU; per-setting full-export
runtime was 3.38–3.73 seconds.

## 7. Phase-B readiness and remaining blockers

All eight Retina archives satisfy the technical prerequisites for the later N
head-only comparison: matching checkpoint identity, 512-D features, labels and
sample IDs, OOF folds, checksums, and A reproduction PASS. **Retina Phase B is
technically ready, but N was not fit in this task.**

Solar remains out of scope and separate cluster work. No Solar feature archive
or scientific manuscript claim was changed.
