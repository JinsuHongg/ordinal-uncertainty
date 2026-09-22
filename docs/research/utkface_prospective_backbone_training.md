# UTKFace prospective backbone training

## Status

All eight prospectively specified UTKFace backbones completed on 2026-09-21. This note records backbone execution only. No frozen-feature export, A/C/N fitting, test-endpoint evaluation, post-result tuning, reroll, manuscript change, commit, or push was performed.

The runs follow the frozen protocol in `docs/research/utkface_prospective_replication_protocol.md`. The protocol’s historical resource-stop record was superseded for this user-authorized execution by the later successful V100 scheduler smoke audit; no frozen scientific setting was changed.

## Frozen inputs and recipe

- Data root: `/scratch/users/jhong36/data/utkface/extracted/UTKFace`.
- Manifest: `/scratch/users/jhong36/ordinal-uq/manifest/utkface/manifest.jsonl`.
- Manifest SHA-256: `3ba4118683ff2031df19ae63651ba3a7718e883dc268d1b8bc06a74e79064c83`.
- Verified split populations: train 14,224 `[2756, 7128, 2726, 1210, 404]`; validation 2,371 `[459, 1188, 455, 202, 67]`; test 2,371 `[459, 1189, 454, 202, 67]`; test class-4 support 67.
- Architecture: randomly initialized small-image ResNet18 with a 3×3 stride-1 stem, no max-pool, and a five-way linear head.
- Preprocessing: RGB; `Resize(128,128)`; training-only random horizontal flip; `ToTensor`; ImageNet normalization mean `[0.485, 0.456, 0.406]`, standard deviation `[0.229, 0.224, 0.225]`. Evaluation is deterministic and has no flip.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `.01`; batch size 32; 10 epochs; no scheduler or early stopping.
- Objectives: ordinary five-class cross-entropy (CE), or the repository five-class ranked probability score loss (RPS).
- Seeds: logical seeds 1–4 for each objective; Python, NumPy, PyTorch CPU, and PyTorch CUDA RNGs are seeded before construction. No additional deterministic backend setting is specified by the frozen runner.
- Checkpoint rule: minimum objective-matched validation loss. No test split was loaded, evaluated, or used for selection.

## Scheduler execution

Two Slurm arrays (`4453809` for CE and `4453810` for RPS) supplied exactly four seed tasks each. All work ran on `acidsgcn011` using an NVIDIA Tesla V100-SXM2-32GB and completed with scheduler exit code `0:0`. The array wrapper records the exact command, host, GPU, Python/PyTorch/CUDA environment, selected epoch, validation score, and checkpoint SHA-256 in each `run_metadata.json`.

The frozen runner does not retain a separate final-epoch checkpoint: it saves the validation-selected `best_checkpoint.pt`, the complete ten-epoch `training_history.csv`, `validation_backbone_arrays.npz`, and `config.json` for every unit. All training and validation losses were finite; no OOM or CPU fallback occurred. There were no infrastructure retries.

## Completed units

| Loss | Seed | Job ID | GPU | Exit | Selected epoch | Validation metric | Checkpoint SHA-256 |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| CE | 1 | 4453811 | Tesla V100-SXM2-32GB | 0:0 | 9 | 0.656172084625 | `dd018c7746bb2ea6be49250c1c45b44eaf5be835a2722d6ecc68d1402a201dd7` |
| CE | 2 | 4453812 | Tesla V100-SXM2-32GB | 0:0 | 6 | 0.633267953131 | `baf7301b920ad511a605a9eba7066cf4ea1255db3596c7b3317488876ea30b71` |
| CE | 3 | 4453813 | Tesla V100-SXM2-32GB | 0:0 | 9 | 0.627874684573 | `ac482024de197aa9857dd472767be5d9c868fc61dd6791585e6e9b5223a6bb6e` |
| CE | 4 | 4453809 | Tesla V100-SXM2-32GB | 0:0 | 9 | 0.665059160294 | `d71f8f6d10a3e4758bc45af28b0c88b6ef9bc69dd481fd01978b77e108b88ba5` |
| RPS | 1 | 4453814 | Tesla V100-SXM2-32GB | 0:0 | 7 | 0.053219095571 | `5e14963a10105fc40a86c45ff74152d5f7788cd7e9b4f25d7e0b74e594351cee` |
| RPS | 2 | 4453815 | Tesla V100-SXM2-32GB | 0:0 | 5 | 0.049320735132 | `f9198609928a23272951faa38626807f569c67fe0f08cd61b0bafc59a0340bfd` |
| RPS | 3 | 4453816 | Tesla V100-SXM2-32GB | 0:0 | 10 | 0.048619374113 | `8a91996096bf0bf6be05dc181dc8e1a43f9366f6b86c2a8688d8d7e0832ab0ec` |
| RPS | 4 | 4453810 | Tesla V100-SXM2-32GB | 0:0 | 7 | 0.051846813777 | `b069cad2533ba84ec42000c4539bd6103ee53574ac85eb6e19fe493346a1a2bf` |

## Integrity and next state

- Each output is a new, non-overwriting path under `outputs/utkface/prospective_replication/backbones/<objective>/seed_<seed>/`.
- Every selected checkpoint exists, has the recorded SHA-256, and has an objective/seed/configuration matching its assigned unit.
- Each configuration records the frozen manifest hash, batch size 32, ten epochs, validation-only evaluation split, and objective-specific minimum-validation selection.
- No A/C/N head work or frozen-feature export has been started.
- The next authorized operation, if separately requested, is frozen-feature export from these selected checkpoints.

Machine-readable run manifest: `outputs/utkface/prospective_replication/backbone_run_manifest.json`.
