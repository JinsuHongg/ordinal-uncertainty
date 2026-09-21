# UTKFace prospective cross-domain A/C/N replication protocol

**Status:** STOPPED AT GATE 2 — the data/provenance gate passed and CE/RPS
one-batch smoke losses were finite, but CUDA allocator OOM events occurred on
the current host.  Under the predeclared stop rule, no backbone run, feature
export, or A/C/N result was generated.  This document contains no new
experimental result.

## Scope and scientific status

This is a prospective, supporting cross-domain replication of the **balanced
versus natural sampling response** for direction-only frozen heads.  It is not
a retroactive confirmation of the original H1/H2a block, does not change that
block's status, and must not be described as preregistered original evidence.
The question is whether the new UTKFace seed-level pattern resembles the
already completed descriptive summaries: Retina CE 2/4, Retina RPS 2/4, Solar
CE 2/4, Solar RPS 3/4 improvements for N over A, with C better than N in all
16 completed units.

No manuscript source, original confirmatory result, historical output, or
existing checkpoint may be overwritten.  This protocol authorizes only the
eight specified new backbones and their paired A/C/N analyses if every gate
below passes.

## Frozen data contract

| Field | Frozen value |
| --- | --- |
| Corpus | `/mnt/storage/data/utkface/UTKFace` provider-distributed cropped/aligned filename corpus |
| Corpus audit | 23,708 files; 23,708 valid JPEGs; 0 parse errors; 0 decode errors; age range 1–116 |
| Age parsing | first underscore-delimited filename field; finite non-negative chronological age |
| Ordinal bins | `[20, 40, 60, 80]`: `<20`, `[20,40)`, `[40,60)`, `[60,80)`, `>=80` |
| Manifest | `ordinal-cqr/data/manifests/conference_v0_3/utkface/manifest.jsonl` |
| Manifest SHA-256 | `3ba4118683ff2031df19ae63651ba3a7718e883dc268d1b8bc06a74e79064c83` |
| Split policy | archived `sorted_filename_stratified_60_10_20_10_v1`, seed 0; manifest is reused unchanged |
| Train / validation / calibration / test | 14,224 / 2,371 / 4,742 / 2,371 |
| Train counts C0–C4 | `[2756, 7128, 2726, 1210, 404]` |
| Validation counts C0–C4 | `[459, 1188, 455, 202, 67]` |
| Calibration counts C0–C4 | `[919, 2376, 909, 403, 135]` |
| Test counts C0–C4 | `[459, 1189, 454, 202, 67]` |
| Evaluation split | **archived validation** only; it is disjoint from the C/N fitting population and avoids further iterative use of the historically inspected test split |
| Preprocessing | RGB; `Resize(128,128)`; ImageNet mean/std; train-only random horizontal flip; no extra crop or alignment |

The corpus and manifest were re-audited before this freeze: their filename sets
are identical, manifest IDs are unique, and every pair of train/validation/
calibration/test splits has zero ID overlap.

## Frozen backbone protocol

Exactly eight independent new backbone runs are authorized:

| Objective | Seeds |
| --- | --- |
| CE | 1, 2, 3, 4 |
| RPS | 1, 2, 3, 4 |

Each run uses the historical Phase 3.7A recipe: unpretrained small-image
ResNet18 (3x3, stride-1 stem; no max-pool; five-way linear head), AdamW
(`lr=1e-4`, `weight_decay=.01`), batch size 32, 10 epochs, and the frozen data
contract above.  CE selects the minimum validation CE checkpoint; RPS selects
the minimum validation RPS checkpoint.  The random seed is the listed logical
seed and must be applied before model, sampler, and transform construction.
No seed 0 rerun, pretrained initialization, architecture change, loss change,
class weighting, resampling, calibration, tuning, or test evaluation is
allowed.

Backbone outputs must be new non-overwriting directories:
`outputs/utkface/prospective_replication/backbones/<objective>/seed_<seed>/`.
They must contain the selected checkpoint, full train/validation history,
config, source-manifest checksum, selected epoch and validation score, and
finite normalized validation probabilities keyed by sample ID.

## Frozen A/C/N head protocol

For every selected backbone, freeze the representation and extract full-train
and validation features once.  Fit all heads from the same A state on full
frozen training features and evaluate once on the archived validation split.

| Condition | Initialization and trainable parameters | Sampling | Objective |
| --- | --- | --- | --- |
| A | untouched selected backbone head; no fitting | none | none |
| C | original head directions; original per-class norms and biases fixed | class-balanced replacement batches | CE |
| N | same as C | one shuffled empirical full-train pass per epoch; no weighting/rebalancing | CE |

C and N use `DirectionOnlyLinear`, AdamW (`lr=.001`, direction weight decay
0), batch size 64, and exactly 100 terminal epochs.  Original norms and biases
are immutable; maximum norm error must be `<=1e-6`, maximum bias error must be
`<=1e-6`, and A replay from the extracted feature cache must be numerically
identical to the selected backbone logits to a predeclared maximum absolute
tolerance of `1e-4`.  No B/D, bias-only, prior-only, ROP, scale sweep,
alternative optimizer, validation selection for C/N, or head/backbone refit is
permitted.

All prospective artifacts must be written below
`outputs/utkface/prospective_replication/` with separate atomic paths for
feature caches, A/C/N states, per-sample validation arrays, histories,
integrity metadata, and analysis tables.  Existing `outputs/utkface/phase3_*`
trees are read-only historical evidence.

## Required validation readouts

For each objective × seed × A/C/N unit, preserve sample-level logits,
probabilities, labels, IDs, mode/L1/L2 decisions, and L1 risks.  The aggregate
analysis reports all A/C/N values and deltas for:

- class-4 L1 MAE, exact recovery/routing, predictive mean, inward shrinkage,
  `p4`, `p3`, `p3+p4`, severe prevalence, and endpoint L1 risk;
- class-0 routing/MAE/severe prevalence/predictive location as the
  opposite-endpoint control;
- global accuracy, MAE, QWK, severe prevalence, NLL, Brier, RPS, ECE;
- risk/error Spearman, severe AUROC/AUPRC, risk-coverage, and selective MAE;
- class-4 versus class-3 direction margin, train-derived centroid routing, and
  class-4 geometry using the frozen training features only;
- fixed-norm/fixed-bias checks, A identity replay, and observed per-epoch C/N
  draw counts.

The primary outcome is the number of seeds with negative exact class-4 L1-MAE
delta N−A, reported separately for CE and RPS.  C−A and C−N are secondary
descriptive comparisons.  There is no success threshold and no averaging away
of seed-level contradictions.

## Execution and stop gates

### Gate 1 — provenance and implementation freeze

Before any job, verify the corpus audit, manifest checksum, exact split counts,
split disjointness, checkpoint/head shape `(5,512)`, data transform, historical
backbone recipe, and A/C/N invariants above.  Add focused tests for the UTKFace
routing only after this document is frozen.  Stop if a source detail differs
from this document.

### Gate 2 — smoke and resource preflight

On the intended CUDA execution host, record hostname, GPU model, driver,
PyTorch/CUDA versions, allocated memory, and job command.  Require
`torch.cuda.is_available() == True`, one usable CUDA device, and a finite
one-batch CE and RPS forward/backward smoke run with the frozen 128-pixel
pipeline.  The historical runner's full command pattern is retained as the
reference; the prospective job command may be finalized only after a real
CUDA scheduler/host is available.  A job wall-time limit must cover the
10-epoch backbone plus feature extraction and paired 100-epoch C/N fitting;
timeout, OOM, non-finite loss, or unavailable GPU stops the study cleanly.

**Observed 2026-09-20 on this host:** `torch.cuda.is_available()` was true for
one NVIDIA GeForce GTX 1660 Ti (6,144 MiB; PyTorch `2.13.0+cu130`; driver
`595.91.07`).  Both one-batch historical smoke checks wrote finite losses (CE
`1.3736953`, RPS `.1461430`) with `(32,3,128,128)` inputs.  However, each
smoke emitted CUDA allocator OOM events while trying to allocate 5,112,856,576
bytes with only about 2.1–2.7 GiB free.  This satisfies the protocol's OOM
stop condition.  The exact eight backbone jobs were not launched.

### Gate 3 — paired artifact integrity

Before aggregation, require complete eight backbone cells; A/C/N for every
cell; matching IDs and labels; finite normalized probabilities; A identity;
fixed norm/bias tolerances; and no missing required metric.  Any missing cell
or invariant failure yields an incomplete replication rather than substituted
or repeated runs.

## Prohibited work

Do not run H2a, support/imbalance sweeps, extra seeds, alternative datasets,
alternative resampling or objectives, hyperparameter search, manuscript edits,
commit, or push.  Do not call an incomplete or contradictory result a
replication success.  A clean resource- or integrity-stopped record is the
required outcome when a gate fails.
