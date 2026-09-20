# Solar CE seed-2 replacement protocol

**Status:** frozen before replacement training.
**Unit:** logical seed 2, one end-to-end replacement replication only.
**Reason:** the original seed-2 CE feature identity gate repeatedly failed its
pre-specified probability tolerance before N fitting.

## Scope and provenance

Original seed-2 artifacts remain immutable historical evidence under their
existing backbone, A/C, and failed feature-replay paths. Their follow-up status
is `ORIGINAL_SEED2_ARCHIVED; invalidated_for_n_followup_due_to_feature_identity_gate_failure`.
The new coherent unit is `outputs/solar/mechanism_replication/replacement_seed2/ce/`.
It must never be described as reproducing the historical seed-2 model.

## Environment freeze

| Field | Frozen replacement value | Evidence/status |
| --- | --- | --- |
| account / partition | `csc344r253` / `qGPU24` | historical CE accounting |
| GPU | actual Tesla V100-SXM2-32GB only | runtime fail-closed gate |
| Python | 3.11.15 | current `ocqr` environment |
| PyTorch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 | current `ocqr` environment |
| CUDA / cuDNN | 12.4 / 90100 | current `ocqr` environment |
| driver | runtime-recorded; expected 580.159.04 | node-specific runtime record |
| precision | float32 model/input; no autocast | fixed |
| TF32 | matmul false; cuDNN true; matmul precision `highest` | V100-compatible current runtime |
| backend flags | benchmark false; deterministic false; deterministic algorithms false | recorded per stage |

Every stage logs hostname, job ID, partition, account, CUDA-visible device,
`nvidia-smi -L`, GPU UUID/driver, and PyTorch device properties. A non-V100 or
a reported software-version/GPU-model change fails the stage before work.

## Data and backbone freeze

- Zarr: `/scratch/users/jhong36/data/surya-bench-224.zarr`; index root:
  `/scratch/users/jhong36/data`.
- Channels/order: `[hmi_m, aia1600, aia131]` / `[8, 7, 1]`.
- Transform: float32 `sign(x)*log1p(abs(x))`; no evaluation augmentation.
- Train-only normalization artifact:
  `outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json`,
  SHA256 `1178b05f9fafad22eec58608236ee5ba035a2f4629bfc17c10281797da8ed02b`.
- Mandatory aligned rows: 45,047 train; 2,431 validation; 28,006 readout;
  endpoint-X readout support 921. Counts and disjoint IDs are gates.
- Backbone: torchvision ResNet18, `weights=None`, `Linear(512,5)`, float32
  512-D pre-fc feature export.

## One replacement backbone training run

Use `logical_seed=2`, `run_type=replacement`, and the frozen Python/NumPy/
PyTorch/CUDA/worker seeding of the existing backbone runner. Train CE with
AdamW, LR `5e-5`, weight decay `0.01`, batch size 16, maximum 300 epochs,
patience 3, gradient clip 10, and independent horizontal/vertical train flips.
Select only the minimum validation CE checkpoint. The readout is not loaded for
training or selection. There is exactly one replacement run and no reroll.

## Frozen A/C/N unit

Immediately after validation selection, export train/validation/readout features
in the same environment using batch 128, four workers, pinned sequential
loaders, `eval()`, and `torch.no_grad()`. A is the replacement checkpoint fc
head, with weights, biases, norms, directions, logits, probabilities,
predictive means, modes, and exact-L1 decisions saved as the new reference.

C and N both use only those frozen features, CE adaptation, AdamW LR `0.001`,
zero direction weight decay, batch 64, 100 epochs, original-A direction
initialization, fixed A biases, and fixed A norms. C uses predeclared
replacement class-balanced sampling; N uses one shuffled empirical pass per
epoch. No weighting, logit/bias adjustment, or evaluation/validation selection
is permitted. The maximum fixed norm/bias error is `1e-6`.

## Frozen evaluation and analysis

All A/C/N outputs use the same replacement 28,006-row readout once. Exact L1
defines endpoint MAE and H1: `endpoint_MAE_C - endpoint_MAE_A`. H2a uses the
frozen true-X formulas, within-seed standardized predictors, HC3 interval, and
M0/M1 leave-one-out MSE comparison in the mechanism protocol §17. Compute
endpoint/global/macro MAE, per-class MAE/recall/severe error, endpoint mass,
and routing for A/C/N.

The future auxiliary CE natural-sampling block may use only 1, replacement-2,
3, 4 and reports 4/4 consistent, 3/4 mixed, or at-most-2/4 not consistent. It
is not the original preregistered H1.

## Stop gates

Stop with no retry, promotion, or downstream stage if GPU/environment, splits,
validation checkpoint, feature dimension/alignment, or A/C/N fixed-parameter
assertions fail. No Solar RPS or CE seed 1, 3, or 4 artifact may change.
