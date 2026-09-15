# Solar A/C Replication Execution Log

**Status:** submitted; execution/provenance record only  
**Frozen protocol:** [Mechanism Replication Protocol](mechanism_replication_protocol.md)  
**Submission time (UTC):** 2026-09-14T19:09:17Z

This record covers only the eight authorized Solar A/C conditions using the
already integrity-passed CE/RPS backbone checkpoints for confirmatory seeds
1--4. No RetinaMNIST or UTKFace condition is included. No aggregate H1/H2
analysis is authorized or recorded here.

## Frozen resources

- A/C runner: `scripts/run_ac_mechanism_replication.py`
- Array wrapper: `scripts/slurm_solar_ac_mechanism_replication.sbatch`
- Canonical normalization artifact:
  `outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json`
- Backbone integrity inventory:
  `outputs/solar/mechanism_replication/backbone_integrity_audit.json`
- SLURM parent array: `4399479`
- Environment: `ocqr`; `qGPU24`; one GPU, 8 CPUs, 64 GB, 24-hour limit.

## Deterministic array mapping

| Task | Objective | Seed | Backbone checkpoint | A/C output directory |
| ---: | --- | ---: | --- | --- |
| 0 | CE | 1 | `outputs/solar/mechanism_replication/backbones/ce/seed_1/selected_checkpoint.pt` | `outputs/mechanism_replication/ac/solar/ce/seed_1/` |
| 1 | CE | 2 | `outputs/solar/mechanism_replication/backbones/ce/seed_2/selected_checkpoint.pt` | `outputs/mechanism_replication/ac/solar/ce/seed_2/` |
| 2 | CE | 3 | `outputs/solar/mechanism_replication/backbones/ce/seed_3/selected_checkpoint.pt` | `outputs/mechanism_replication/ac/solar/ce/seed_3/` |
| 3 | CE | 4 | `outputs/solar/mechanism_replication/backbones/ce/seed_4/selected_checkpoint.pt` | `outputs/mechanism_replication/ac/solar/ce/seed_4/` |
| 4 | RPS | 1 | `outputs/solar/mechanism_replication/backbones/rps/seed_1/selected_checkpoint.pt` | `outputs/mechanism_replication/ac/solar/rps/seed_1/` |
| 5 | RPS | 2 | `outputs/solar/mechanism_replication/backbones/rps/seed_2/selected_checkpoint.pt` | `outputs/mechanism_replication/ac/solar/rps/seed_2/` |
| 6 | RPS | 3 | `outputs/solar/mechanism_replication/backbones/rps/seed_3/selected_checkpoint.pt` | `outputs/mechanism_replication/ac/solar/rps/seed_3/` |
| 7 | RPS | 4 | `outputs/solar/mechanism_replication/backbones/rps/seed_4/selected_checkpoint.pt` | `outputs/mechanism_replication/ac/solar/rps/seed_4/` |

## Pre-execution and immediate scheduler checks

- Runner syntax and CLI: passed in `ocqr`.
- `tests/test_direction_only.py` and `tests/test_oof.py`: 3 passed.
- All eight integrity-audited backbone checkpoints were present and READY.
- All eight deterministic A/C output targets were absent before submission.
- The canonical normalization artifact passed metadata and finite-statistics
  checks.
- Array tasks `4399479_0` through `4399479_7` entered `RUNNING` state on
  `qGPU24`; tasks 0--3 had V100 allocations and tasks 4--7 had A30
  allocations. All allocations requested one GPU, 8 CPUs, and 64 GB.

No aggregate H1/H2 analysis, cross-seed success count, confidence interval, or
manuscript interpretation has been run in this stage.
