# Solar CE GPU Provenance Audit

**Verdict:** **A — RECENT JOBS WERE A30; HISTORICAL CE V100 VERIFIED**

## Executive summary

The recent strict-FP32 CE regeneration array ran on NVIDIA A30 GPUs, not V100s. This is confirmed independently by SLURM accounting, allocated-node GRES inventory, and each task's runtime `nvidia-smi` output. The historical CE A/C array ran on Tesla V100-SXM2-32GB GPUs, also confirmed independently by accounting and task-local runtime output. There is no job/log crossing or GPU-manifest reporting error.

The hardware differential is verified: historical CE A/C was V100, whereas both failed CE regeneration attempts were A30. It supports the earlier subset observation that A30 default TF32 arithmetic contributes materially to the mismatch. It does **not** prove that the GPU model/TF32 differential is the sole full-readout cause: the later strict-FP32 full run still exceeded frozen logit/probability tolerances despite zero mode and exact-L1 changes. CE N fitting remains blocked.

## Recent job IDs and scheduler allocation evidence

The strict-FP32 submission used array master `4424724` with tasks `1--4`. Its `%A_%a` output template deliberately uses the master ID and array index; runtime `SLURM_JOB_ID` records each task identity.

| Array task / seed | Runtime job ID | Log | Partition / node | Requested GPU | Allocated GPU | Runtime GPU |
| --- | ---: | --- | --- | --- | --- | --- |
| 1 / CE seed 1 | 4424725 | `features_4424724_1.out` | `qGPU24` / `acidsgcn013` | `gpu:1` | `gres/gpu:a30=1` | NVIDIA A30, 24,576 MiB |
| 2 / CE seed 2 | 4424726 | `features_4424724_2.out` | `qGPU24` / `acidsgcn013` | `gpu:1` | `gres/gpu:a30=1` | NVIDIA A30, 24,576 MiB |
| 3 / CE seed 3 | 4424727 | `features_4424724_3.out` | `qGPU24` / `acidsgcn013` | `gpu:1` | `gres/gpu:a30=1` | NVIDIA A30, 24,576 MiB |
| 4 / CE seed 4 | 4424724 | `features_4424724_4.out` | `qGPU24` / `acidsgcn013` | `gpu:1` | `gres/gpu:a30=1` | NVIDIA A30, 24,576 MiB |

`scontrol show node acidsgcn013` reports `Gres=gpu:A30:8`; all tasks were terminal `FAILED` only because their full A-identity numerical gate failed after inference.

## Runtime GPU evidence and job/log mapping

The first two stdout lines of every task record its runtime job ID, seed, and task-local `nvidia-smi --query-gpu=name,memory.total` result. Values match the array-task index in the submission script exactly. The seed-4 mapping is normal Slurm array behaviour: the lowest-numbered executable task can retain the array master numeric ID.

The mapping is **PASS**: no historic A30 log was read under a new job ID, and no V100 task was mislabeled as A30.

## Submission request and manifest/reporting path

`scripts/slurm_solar_ce_strict_fp32_feature_export.sbatch` requested `--partition=qGPU24`, `--nodelist=acidsgcn013`, and generic `--gres=gpu:1`. It did not request `gpu:v100:1` or an equivalent typed GPU. The explicit node pin selected the A30 node, and generic GPU accounting records allocated subtype `a30`.

The current exporter writes a successful feature manifest's `gpu_model` from `torch.cuda.get_device_name(0)`; it is neither hard-coded nor copied metadata. No CE strict-FP32 feature manifest exists because output is promoted only after the full A gate passes. Runtime identity is nevertheless directly recorded by `nvidia-smi` in each stdout. Earlier RPS manifests predate `gpu_model` and contain no GPU name, but their task stdout records `nvidia-smi` directly.

## Historical CE and RPS GPU evidence

Historical A/C array `4399479` used generic `--gres=gpu:1`, but the scheduler allocated CE tasks `0--3` to `acidsgcn011` with `gres/gpu:v100=1`. That node's current inventory is `gpu:V100:8`. More importantly, each historical CE stdout (`ac_4399479_0.out` through `ac_4399479_3.out`) records `Tesla V100-SXM2-32GB` from `nvidia-smi` and the matching PyTorch device name. Historical CE V100 provenance is therefore **VERIFIED V100**, not inferred.

Historical RPS tasks `4--7` of the same array were allocated to `acidsgcn013` with `gres/gpu:a30=1`, and every stdout records `NVIDIA A30` and the matching PyTorch device name. Successful current RPS feature array `4416618`, tasks `4--7`, was likewise allocated to `acidsgcn013` with `gres/gpu:a30=1` and records NVIDIA A30 in every task stdout.

| Objective | Historical GPU | Current regeneration GPU | Evidence strength |
| --- | --- | --- | --- |
| CE | Tesla V100-SXM2-32GB | NVIDIA A30 | Direct `nvidia-smi`, PyTorch (historical), SLURM accounting, and node GRES |
| RPS | NVIDIA A30 | NVIDIA A30 | Direct `nvidia-smi`, PyTorch (historical), SLURM accounting, and node GRES |

## CE/RPS differential and TF32 attribution

The CE-only hardware differential is real and the RPS control supports its relevance: CE moved from V100 to A30, while RPS remained on A30. The fixed 64-row CE audit additionally showed large default-A30 differences and zero exact-L1 differences after strict FP32.

The causal statement is **SUPPORTED BUT NOT FULLY PROVEN**, not a complete root-cause proof. Strict FP32 reduced the subset discrepancy but the full readout still exceeded frozen maximum logit/probability tolerances. The evidence establishes a verified hardware provenance change and a numerically supported A30-TF32 contribution; it does not rule out remaining hardware/library/kernel differences within strict FP32.

## Corrections, status, and recommended next step

Earlier wording that described V100-versus-A30 TF32 as the fully established sole cause has been narrowed to the evidence above. No scientific claim, tolerance, checkpoint, feature archive, or experiment result was changed.

**Recommended next step:** retain the CE N block. If separately authorized, first conduct a narrowly scoped compatibility validation on a historically matching V100 path (or otherwise isolate the residual strict-FP32 full-readout discrepancy) before any full export or N fitting.
