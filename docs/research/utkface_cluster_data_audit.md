# UTKFace Cluster Data and Provenance Audit

**Status:** **D — CODE/TEST/GPU PREFLIGHT BLOCKER; NOT READY**

**Scope:** Read-only readiness audit. No UTKFace data was downloaded, no split
was changed, and no model, head, or data-fitting procedure was run.

## Required-artifact check

The prospective-replication artifacts specified for this audit were not
present in the repository or at the supplied cluster manifest location:

| Required artifact | Requested path | Audit result |
| --- | --- | --- |
| Frozen prospective protocol | `docs/research/utkface_prospective_replication_protocol.md` | Missing |
| Prospective runner | `scripts/run_utkface_prospective_replication.py` | Missing |
| Prospective targeted tests | `tests/test_utkface_prospective_replication.py` | Missing |
| Copied frozen manifest directory | `/scratch/users/jhong36/ordinal-uq/manifest/` | Missing |

The repository contains historical UTKFace materials
(`scripts/run_phase3_7a_utkface_replication.py`,
`scripts/run_phase3_13_utkface_direction_scale.py`, and
`tests/test_utkface_data.py`), but these are not substitutes for the stated
frozen prospective protocol, runner, or tests.

## Audit outcome

The required frozen protocol and manifest cannot be located, so the intended
corpus layout, manifest keys, exact split memberships, frozen ordinal-label
reconstruction, preprocessing path, and expected split hashes cannot be
verified. Under the audit stop condition, UTKFace was not downloaded and no
dataset, loader, targeted-test, or GPU preflight checks were run.

## Blockers

1. Restore or provide the frozen prospective protocol at the requested path.
2. Restore or provide the matching prospective runner and targeted test file.
3. Copy the immutable frozen split manifest to
   `/scratch/users/jhong36/ordinal-uq/manifest/` (or provide its authoritative
   cluster path).

After these artifacts are available, a new data/provenance audit can establish
the required download location and validate the corpus without changing any
scientific setting.

## Cluster execution readiness re-audit

**Status:** **E — GPU/VRAM BLOCKER; NOT READY**

### Runner portability remediation

The prospective runner now requires explicit `--data-root` and `--manifest`
arguments on both existing subcommands. The stale executable dependencies on
`/home/jhong90/...` and `/mnt/storage/...` were removed. The canonical cluster
arguments are:

```text
--data-root /scratch/users/jhong36/data/utkface/extracted/UTKFace
--manifest /scratch/users/jhong36/ordinal-uq/manifest/utkface/manifest.jsonl
```

The runner validates both paths before manifest loading and raises clear
`FileNotFoundError` messages for a missing manifest or data root. The frozen
manifest SHA256 remains
`3ba4118683ff2031df19ae63651ba3a7718e883dc268d1b8bc06a74e79064c83`.

### Focused tests and loader dry run

All six targeted prospective-runner tests passed. They cover explicit CLI path
parsing, path propagation to manifest loading, missing-path failures, absence
of the two workstation paths, the natural-head fixed-parameter invariant, and
the frozen train/validation contract.

Using the actual runner loaders with the canonical cluster arguments produced
finite float32 `(32, 3, 128, 128)` tensors and int64 labels. Train, validation,
and test counts and class counts exactly matched the frozen manifest:

| Split | N | Class counts [0,1,2,3,4] | Class-4 support | Sampler |
| --- | ---: | --- | ---: | --- |
| Train | 14,224 | [2756, 7128, 2726, 1210, 404] | 404 | RandomSampler |
| Validation | 2,371 | [459, 1188, 455, 202, 67] | 67 | SequentialSampler |
| Test | 2,371 | [459, 1189, 454, 202, 67] | 67 | SequentialSampler |

The executable transform remains `Resize(128,128)`, train-only random
horizontal flip, `ToTensor`, and ImageNet normalization with mean
`[.485,.456,.406]` and standard deviation `[.229,.224,.225]`. No train-derived
normalization artifact, crop, resolution, batch size, model, loss, seed, or
other scientific setting was changed.

### GPU preflight and stop

On the current execution host, `CUDA_VISIBLE_DEVICES` was unset,
`nvidia-smi` was unavailable, and PyTorch `2.6.0+cu124` reported
`torch.cuda.is_available() == False` with zero visible devices. No CE or RPS
one-batch smoke was run because the frozen protocol requires a usable CUDA
device before smoke execution. No model training or A/C/N fitting occurred.

**Readiness decision:** The portable runner and cluster loaders are ready, but
this host is **not ready** to launch the frozen prospective backbone jobs. Run
the unchanged execution command only in a scheduler allocation that exposes a
usable CUDA GPU, then repeat the frozen one-batch CE/RPS smoke gate.

## Scheduler GPU smoke re-audit

**Status:** **A — GPU SMOKE PASS; READY FOR THE FROZEN BACKBONE RUNS**

This bounded re-audit was submitted as Slurm job `4453801` to `qGPU24`; it
completed successfully in 1 minute 22 seconds on `acidsgcn011`. It used one
scheduler-visible Tesla V100-SXM2-32GB (`CUDA_VISIBLE_DEVICES=0`; driver
580.159.04), with 32,768 MiB total and 32,495 MiB free at job start. The
runtime was PyTorch `2.6.0+cu124`, CUDA `12.4`.

Using the actual frozen training loader, explicit canonical cluster paths, the
unpretrained five-way small-stem ResNet18, batch size 32, 128-pixel ImageNet
preprocessing, AdamW (`lr=1e-4`, `weight_decay=.01`), and a single logical
seed-1 initialization, each loss completed exactly one forward, backward, and
optimizer step:

| Objective | Loss | Peak allocated | Peak reserved | OOM |
| --- | ---: | ---: | ---: | --- |
| CE | 1.6273032 | 2,606,727,168 bytes | 3,124,756,480 bytes | No |
| RPS | .1673243 | 2,656,374,784 bytes | 2,715,818,240 bytes | No |

No backbone history, checkpoint, feature export, A/C/N state, split, or
scientific setting was created or changed. This supersedes the prior
login-host GPU-visibility blocker for execution readiness only; it does not
alter the frozen protocol or authorize work beyond the user-authorized eight
backbone runs.
