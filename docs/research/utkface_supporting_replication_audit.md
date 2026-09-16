# UTKFace Supporting-Replication Provenance and Inventory Audit

**Scope:** provenance/inventory audit only.  No UTKFace backbone was trained,
no A/C replication or H1/H2 analysis was run, and no frozen RetinaMNIST/Solar
protocol or manuscript artifact was changed.

**Decision:** **C — UTKFace HIGH-COST; NOT CURRENT PRIORITY.**  The historical
seed-0 CE and RPS backbones are provenance-clean descriptive artifacts, but
there are no canonical CE or RPS backbone checkpoints for seeds 1–4.  A
supporting multi-seed A/C replication would therefore require eight new
backbone-training runs before any frozen-head analysis.  It is not a low-cost
checkpoint-reuse extension.

UTKFace remains **supporting replication / robustness evidence only**.  This
audit does not add it to the main frozen RetinaMNIST/Solar confirmatory block.

## Evidence reviewed

- `docs/research/phase3_7a_utkface_failure_replication.md`
- `docs/research/phase3_13_utkface_direction_scale_mechanism_confirmation.md`
- `scripts/run_phase3_7a_utkface_replication.py`
- `scripts/run_phase3_13_utkface_direction_scale.py`
- `src/ordinal_uncertainty/data/utkface.py`
- `scripts/run_ac_mechanism_replication.py`
- `outputs/utkface/phase3_7a_failure_replication/`
- `outputs/utkface/phase3_13_direction_scale_mechanism_confirmation/`

The historical Phase 3.7A dataset audit identifies the source manifest by
SHA-256 `3ba4118683ff2031df19ae63651ba3a7718e883dc268d1b8bc06a74e79064c83`.
Phase 3.13 refers to an archived manifest at a different repository path; its
checksum must be checked against that historical manifest before any reuse,
but this does not weaken the fully documented Phase 3.7A seed-0 provenance.

## Historical UTKFace protocol

| Item | Historical evidence |
| --- | --- |
| Dataset source | Provider-distributed cropped/aligned UTKFace JPEG filename corpus at `/mnt/storage/data/utkface/UTKFace`; the historical audit found 23,708 valid JPEGs, ages 1–116, with no parsing/decode errors. |
| Age parsing and bins | The first underscore-delimited filename field is age. `AGE_THRESHOLDS=(20, 40, 60, 80)`: `<20`, `[20,40)`, `[40,60)`, `[60,80)`, and `>=80`. |
| Ordered classes | `K=5`; upper endpoint is class `4`, age `>=80`. |
| Split | Frozen sorted-filename stratified `60/10/20/10` split, policy `sorted_filename_stratified_60_10_20_10_v1`, seed 0. The calibration split was preserved but unused. |
| Split support | Train: 14,224 `[2756, 7128, 2726, 1210, 404]`; validation: 2,371 `[459, 1188, 455, 202, 67]`; calibration: 4,742 `[919, 2376, 909, 403, 135]`; test: 2,371 `[459, 1189, 454, 202, 67]`. |
| Preprocessing | RGB; resize to `128x128`; ImageNet normalization (mean `.485,.456,.406`, std `.229,.224,.225`); train-only random horizontal flip; no additional crop/alignment in project code. |
| Architecture | Unpretrained small-image ResNet18: 3x3 stride-1 stem, no max-pool, five-way final linear head. Frozen representation dimension is 512; `fc.weight` is `(5,512)` and `fc.bias` is `(5,)`. |
| CE/RPS training | Batch size 32; AdamW, learning rate `1e-4`, weight decay `.01`, 10 epochs; only seed 0. CE selected minimum validation CE; RPS selected minimum validation RPS. Both historical runs selected epoch 6. |
| Selection discipline | Validation-only checkpoint selection; the archived test outputs were not used to choose checkpoints. |

The endpoint has 404 training and 67 validation/test observations.  That is
adequate for a bounded descriptive supporting endpoint readout, but is small
for strong between-seed inference and should remain secondary evidence.

## Canonical backbone checkpoint inventory

Only Phase 3.7A `best_checkpoint.pt` files count as candidate trained
backbones.  Phase 3.13 head states and feature archives are frozen-head
intervention artifacts, not independently trained CE/RPS backbones.

| Objective | Seed | Checkpoint | Exists | Architecture / head | Config | Predictions/logits | Provenance status |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| CE | 0 | `outputs/utkface/phase3_7a_failure_replication/ce/seed_0/best_checkpoint.pt` | Yes | Small-image unpretrained ResNet18; `(5,512)` | Yes | `test_arrays.npz`, `predictions.csv` | **COMPATIBLE (historical seed-0 only)** |
| CE | 1 | — | No | — | — | — | **MISSING** |
| CE | 2 | — | No | — | — | — | **MISSING** |
| CE | 3 | — | No | — | — | — | **MISSING** |
| CE | 4 | — | No | — | — | — | **MISSING** |
| RPS | 0 | `outputs/utkface/phase3_7a_failure_replication/rps/seed_0/best_checkpoint.pt` | Yes | Small-image unpretrained ResNet18; `(5,512)` | Yes | `test_arrays.npz`, `predictions.csv` | **COMPATIBLE (historical seed-0 only)** |
| RPS | 1 | — | No | — | — | — | **MISSING** |
| RPS | 2 | — | No | — | — | — | **MISSING** |
| RPS | 3 | — | No | — | — | — | **MISSING** |
| RPS | 4 | — | No | — | — | — | **MISSING** |

Thus, there are two of ten historical objective/seed cells represented, and
zero of the eight seed-1–4 cells needed for a prospective supporting
multi-seed replication.  Neither CE nor RPS has a reusable seed-1–4 backbone.

## Seed-0 replay and artifact feasibility

For each seed-0 objective, the saved `test_arrays.npz` contains authoritative
test-order `logits`, `probabilities`, `labels`, `sample_ids`, mode/L1/L2
decisions, and the corresponding risks (2,371 records; logits/probabilities
are `2371x5`).  `config.json`, `predictions.csv`, selection metadata, and
training/validation histories also exist.  Phase 3.13 additionally preserves
RPS seed-0 frozen train (`14224x512`) and validation (`2371x512`) feature
archives plus original head parameters.

A deterministic **seed-0 provenance replay** is technically feasible without
training: instantiate the documented small-image ResNet18, use RGB
resize-to-128/ImageNet normalization with no evaluation-time flip, load the
Phase 3.7A checkpoint, and reproduce the frozen manifest test ordering.  The
authoritative comparison is `test_arrays.npz`, keyed by `sample_ids`; a
predeclared maximum absolute numerical tolerance of `1e-4` for logits and
probabilities is appropriate pending the actual replay audit.  This audit did
not execute that replay.

This capability confirms historical seed-0 traceability; it does not supply
the missing independent seeds or turn seed 0 into confirmatory evidence.

## A/C runner compatibility assessment

The current A/C runner's frozen-head assumptions are compatible with UTKFace:
it expects a five-class `(5,512)` final layer and its A/C parameterization
preserves original row norms and biases while adapting direction.  The stated
C optimization design (balanced replacement sampling, CE head adaptation,
AdamW `1e-3`, batch 64, 100 epochs, zero direction weight decay) is therefore
architecturally compatible.

It is not plug-and-play: `scripts/run_ac_mechanism_replication.py` currently
routes only `retina` and `solar`.  UTKFace needs a routing branch that uses
`make_resnet18`, `make_utkface_data`, the historical frozen manifest, and the
128-pixel UTKFace transform.  The current Retina training-only OOF path is not
the historical UTKFace analysis semantics.  No dataset-specific feature or
class-count blocker was found; the blockers are absent seeds 1–4 and the
necessary, but modest, dataset-routing implementation.

## Recommended design if later authorized

**OTHER — fixed archived validation evaluation.**  This is the closest
historical UTKFace direction/scale protocol: freeze each independently trained
backbone, fit the C head on the full frozen training split with train-only
centroids, and evaluate once on the fixed archived validation split (2,371
examples, 67 upper-endpoint examples).  It avoids further reuse of the
historically inspected test split and does not falsely present UTKFace as the
Retina-style training-only OOF block.

| Design element | Predeclared supporting-replication population |
| --- | --- |
| C fitting | Full frozen training split (14,224 examples; endpoint support 404). |
| Evaluation | Fixed archived validation split (2,371 examples; endpoint support 67). |
| Centroid fitting | Full frozen training split only. |
| Fold-safe centroids | Not needed: the validation evaluation split is disjoint from the fitting population. |
| Test split | Do not use for iterative supporting-replication evaluation; preserve it as historically inspected evidence. |

For a genuine supporting replication, this design would need predeclared
seed-1–4 CE/RPS backbone training followed by the A/C analyses.  It should
report endpoint support and seed-level uncertainty, retain the supporting-only
label, and make no main-block or universal claim.

## Cost and priority conclusion

The required work is not low-cost reuse: **eight canonical backbones are
missing** (CE seeds 1–4 and RPS seeds 1–4), after which A/C routing and
supporting analyses would still be required.  Seed-0 replay is cheap and
provenance-clean but only audits historical reproducibility.  Therefore UTKFace
is **HIGH-COST / not current priority**, rather than provenance-blocked.

The minimum additional workload is eight 10-epoch backbone-training jobs under
the historical setup, eight frozen-head A/C runs (100 epochs each), the UTK
dataset-routing implementation and provenance replay, then supporting-only
analysis.  No wall-clock estimate is asserted because the archived artifacts do
not provide comparable local GPU timing; relative to reuse-only A/C execution,
this is a substantial training-and-analysis extension.
