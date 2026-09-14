# RetinaMNIST Confirmatory A/C Replication Execution Log

**Status:** READY FOR PREDECLARED H1/H2 ANALYSIS
**Date:** 2026-09-14
**Scope:** execution integrity only; no aggregate H1/H2 analysis or scientific
interpretation is recorded here.

## Frozen scope

This log covers only the eight confirmatory RetinaMNIST backbone conditions:
CE and RPS objectives for seeds 1--4. Seed 0 was not run. No Solar or UTKFace
data, code path, or output was used. Each condition uses training-only
five-fold OOF head evaluation on one fixed learned representation; folds are
not independent backbone replications.

## Checkpoint inventory and preprocessing

| Objective | Seed | Checkpoint | Provenance status | Transform |
| --- | ---: | --- | --- | --- |
| CE | 1 | `outputs/retinamnist/native28/single_model_baseline/seed_1/best_checkpoint.pt` | COMPATIBLE | ToTensor + Normalize(.5,.5,.5) |
| CE | 2 | `outputs/retinamnist/native28/single_model_baseline/seed_2/best_checkpoint.pt` | COMPATIBLE | ToTensor + Normalize(.5,.5,.5) |
| CE | 3 | `outputs/retinamnist/native28/single_model_baseline/seed_3/best_checkpoint.pt` | COMPATIBLE | ToTensor + Normalize(.5,.5,.5) |
| CE | 4 | `outputs/retinamnist/native28/single_model_baseline/seed_4/best_checkpoint.pt` | COMPATIBLE | ToTensor + Normalize(.5,.5,.5) |
| RPS | 1 | `outputs/retinamnist/native28/phase2_model_comparison/rps/seed_1_artifact_complete/best_checkpoint.pt` | COMPATIBLE; deterministic replay verified | ToTensor + Normalize(.5,.5,.5) |
| RPS | 2 | `outputs/retinamnist/native28/phase2_model_comparison/rps/seed_2_artifact_complete/best_checkpoint.pt` | COMPATIBLE; deterministic replay verified | ToTensor + Normalize(.5,.5,.5) |
| RPS | 3 | `outputs/retinamnist/native28/phase2_model_comparison/rps/seed_3_artifact_complete/best_checkpoint.pt` | COMPATIBLE; deterministic replay verified | ToTensor + Normalize(.5,.5,.5) |
| RPS | 4 | `outputs/retinamnist/native28/phase2_model_comparison/rps/seed_4_artifact_complete/best_checkpoint.pt` | COMPATIBLE; deterministic replay verified | ToTensor + Normalize(.5,.5,.5) |

The official local `data/medmnist/retinamnist.npz` training split was read
directly because this workstation's execution environment lacks `medmnist`.
The loader reproduces native `28x28` RGB ToTensor and Normalize behavior; it
does not load validation or test data. The local CUDA driver was unavailable,
so frozen inference and head fitting ran on CPU without changing any model
parameter outside C's direction vectors.

## Fixed OOF and head protocol

- 1,080 training examples with class counts `[486, 128, 206, 194, 66]`.
- Deterministic stratified five-fold assignments (`seed=0`); every sample is
  held out exactly once.
- Per fold, C fits only the four fitting folds using replacement class-balanced
  batches, CE loss, AdamW, learning rate `1e-3`, batch size `64`, 100 epochs,
  and direction weight decay `0`.
- Original row norms and biases are fixed. Per-fold centroids use fitting-fold
  features only; held-fold geometry records `d_3-d_4` and `d_(2)-d_(1)`.

## Post-run execution-integrity audit

Each READY row has: `manifest.json`, `per_sample_arrays.npz`, `per_sample.csv`,
one explicit A head, five C fold heads, five fold histories, finite arrays,
complete OOF sample coverage, 66 class-4 observations, and fitting-fold-only
centroid provenance.

| Objective | Seed | A replay | C init | Norm err | Bias err | N | C4 N | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CE | 1 | `0.0` | `4.768e-07` | `5.960e-08` | `0.0` | 1080 | 66 | READY |
| CE | 2 | `0.0` | `9.537e-07` | `5.960e-08` | `0.0` | 1080 | 66 | READY |
| CE | 3 | `0.0` | `9.537e-07` | `5.960e-08` | `0.0` | 1080 | 66 | READY |
| CE | 4 | `0.0` | `3.815e-06` | `5.960e-08` | `0.0` | 1080 | 66 | READY |
| RPS | 1 | `0.0` | `1.907e-06` | `5.960e-08` | `0.0` | 1080 | 66 | READY |
| RPS | 2 | `0.0` | `9.537e-07` | `5.960e-08` | `0.0` | 1080 | 66 | READY |
| RPS | 3 | `0.0` | `3.815e-06` | `5.960e-08` | `0.0` | 1080 | 66 | READY |
| RPS | 4 | `0.0` | `1.907e-06` | `5.960e-08` | `0.0` | 1080 | 66 | READY |

All values meet the frozen limits: A/C initialization replay `<=2e-5`, norm
and bias error `<=1e-6`.

## Retry record

An initial concurrent CPU attempt for CE seeds 2 and 3 was interrupted by the
local 30-second execution-session limit before completion. The incomplete,
newly created condition directories were removed and rerun sequentially. A
NumPy BLAS accumulation difference then caused CE seed 4's A replay to exceed
the frozen tolerance despite identical checkpoint/features. The runner was
corrected to use PyTorch float32 linear arithmetic matching the model head;
all four CE conditions were regenerated under that same corrected runner.
This was an implementation/infrastructure retry only. No criterion, dataset,
checkpoint, head protocol, or scientific result-selection rule changed.

## Output paths

- `outputs/mechanism_replication/ac/retina/ce/seed_{1,2,3,4}/`
- `outputs/mechanism_replication/ac/retina/rps/seed_{1,2,3,4}/`

The conditions are ready for the separately authorized, predeclared aggregate
H1/H2 analysis. This log does not perform that analysis.
