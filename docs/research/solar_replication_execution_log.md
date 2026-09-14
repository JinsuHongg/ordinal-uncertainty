# Solar Backbone Replication Execution Log

**Status:** submitted; execution/provenance record only  
**Frozen protocol:** [Mechanism Replication Protocol](mechanism_replication_protocol.md)  
**Submission time (UTC):** 2026-09-14T03:17:18Z

This record concerns only the eight authorized Solar CE/RPS backbone trainings
for confirmatory seeds 1--4. It contains no A/C adaptation or H1/H2 analysis.

## Execution fix

The pre-submission alignment audit now passes the aligned dataframe
(`dataset.d`) to the canonical Phase 3.8 `alignment_summary` helper. This is
an execution/provenance wiring correction only.

## Frozen resources and invocation

- Canonical normalization artifact:
  `outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json`
- SLURM script: `scripts/slurm_solar_replication_backbone.sbatch`
- Submission command: `sbatch scripts/slurm_solar_replication_backbone.sbatch`
- Parent array job: `4391245`
- Environment: `ocqr`; `qGPU24`; one GPU, 8 CPUs, 64 GB, 24-hour limit.

## Deterministic array mapping

| Array task | Objective | Seed | Output directory |
| ---: | --- | ---: | --- |
| 0 | CE | 1 | `outputs/solar/mechanism_replication/backbones/ce/seed_1/` |
| 1 | CE | 2 | `outputs/solar/mechanism_replication/backbones/ce/seed_2/` |
| 2 | CE | 3 | `outputs/solar/mechanism_replication/backbones/ce/seed_3/` |
| 3 | CE | 4 | `outputs/solar/mechanism_replication/backbones/ce/seed_4/` |
| 4 | RPS | 1 | `outputs/solar/mechanism_replication/backbones/rps/seed_1/` |
| 5 | RPS | 2 | `outputs/solar/mechanism_replication/backbones/rps/seed_2/` |
| 6 | RPS | 3 | `outputs/solar/mechanism_replication/backbones/rps/seed_3/` |
| 7 | RPS | 4 | `outputs/solar/mechanism_replication/backbones/rps/seed_4/` |

## Pre-submit and immediate scheduler checks

- Python syntax and CLI: passed in `ocqr`.
- Imports of `phase3_7a_solar_3ch` and `phase3_8_solar_confirmation`: passed.
- Canonical normalization metadata and finite positive statistics: passed.
- All eight deterministic output targets were absent before submission.
- `tests/test_direction_only.py` and `tests/test_oof.py`: 3 passed.
- Shell syntax, frozen wrapper mapping, CUDA guard, and canonical normalization
  argument: passed.
- `git diff --check`: passed.
- Immediate state at 2026-09-14T03:17:30Z: tasks `4391245_0` through
  `4391245_7` were all `RUNNING` on `qGPU24` / `acidsgcn013` with
  `gres/gpu:a30=1`. No task uses seed 0.

No test/readout model evaluation, A/C adaptation, or H1/H2 analysis has been
run in this stage.
