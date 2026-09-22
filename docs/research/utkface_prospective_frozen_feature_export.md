# UTKFace prospective frozen feature export

## Purpose and status

This is the strict frozen-feature validation gate for the eight selected prospective UTKFace backbones. All eight archives passed the frozen original-head replay tolerance of maximum absolute error `1e-4`, with zero mode and exact-L1 decision differences. The result permits the separately authorized frozen A/C/N head-adaptation stage; no C or N fitting occurred here.

## Source checkpoints and feature definition

Features are the 512-dimensional penultimate activations immediately before `model.fc`: `h(x)` in `logits = W_A h(x) + b_A`. Each checkpoint was loaded in `eval()` mode. Export uses RGB, `Resize(128,128)`, `ToTensor`, and ImageNet normalization mean `[0.485, 0.456, 0.406]` / standard deviation `[0.229, 0.224, 0.225]`, without augmentation. The classifier is a five-way small-image ResNet18 head with `fc.weight` shape `(5,512)` and `fc.bias` shape `(5,)`.

The source manifest is `/scratch/users/jhong36/ordinal-uq/manifest/utkface/manifest.jsonl` (SHA-256 `3ba4118683ff2031df19ae63651ba3a7718e883dc268d1b8bc06a74e79064c83`). Exported split populations are train 14,224 `[2756, 7128, 2726, 1210, 404]`, validation 2,371 `[459, 1188, 455, 202, 67]`, and test 2,371 `[459, 1189, 454, 202, 67]`; every test archive has class-4 support 67.

| Loss | Seed | Selected checkpoint | Checkpoint SHA-256 verified | Archive path |
| --- | ---: | --- | --- | --- |
| CE | 1 | `outputs/utkface/prospective_replication/backbones/ce/seed_1/best_checkpoint.pt` | yes | `outputs/utkface/prospective_replication/features/ce/seed_1` |
| CE | 2 | `outputs/utkface/prospective_replication/backbones/ce/seed_2/best_checkpoint.pt` | yes | `outputs/utkface/prospective_replication/features/ce/seed_2` |
| CE | 3 | `outputs/utkface/prospective_replication/backbones/ce/seed_3/best_checkpoint.pt` | yes | `outputs/utkface/prospective_replication/features/ce/seed_3` |
| CE | 4 | `outputs/utkface/prospective_replication/backbones/ce/seed_4/best_checkpoint.pt` | yes | `outputs/utkface/prospective_replication/features/ce/seed_4` |
| RPS | 1 | `outputs/utkface/prospective_replication/backbones/rps/seed_1/best_checkpoint.pt` | yes | `outputs/utkface/prospective_replication/features/rps/seed_1` |
| RPS | 2 | `outputs/utkface/prospective_replication/backbones/rps/seed_2/best_checkpoint.pt` | yes | `outputs/utkface/prospective_replication/features/rps/seed_2` |
| RPS | 3 | `outputs/utkface/prospective_replication/backbones/rps/seed_3/best_checkpoint.pt` | yes | `outputs/utkface/prospective_replication/features/rps/seed_3` |
| RPS | 4 | `outputs/utkface/prospective_replication/backbones/rps/seed_4/best_checkpoint.pt` | yes | `outputs/utkface/prospective_replication/features/rps/seed_4` |

## Archive and alignment integrity

Each archive contains stable sample IDs, filenames, parsed ages, split labels, ordinal labels, 512-D features, original A logits and probabilities, mode/L1/L2 decisions, and L1 risks. Every feature/logit/probability value is finite. IDs are unique within each split, exactly match frozen manifest order, and have zero train/validation/test overlap. `A_original_head.pt` preserves the checkpoint head parameters exactly. Per-file SHA-256 values are in the machine-readable feature manifest.

| Loss | Seed | Train / validation / test N | Feature dim | Finite | Test class-4 | Archive metadata SHA-256 |
| --- | ---: | --- | ---: | --- | ---: | --- |
| CE | 1 | 14224 / 2371 / 2371 | 512 | yes | 67 | `48c258ef3371a89123bd861382a92ce02a17ce6aba5a528be90621864e98486c` |
| CE | 2 | 14224 / 2371 / 2371 | 512 | yes | 67 | `da89ffc1d013aceaf66e07022ea3c73b8977125edf89bdc0ac7622c07df058ab` |
| CE | 3 | 14224 / 2371 / 2371 | 512 | yes | 67 | `e6f27ac2d93cbfce218eeadfcaec6c85a3996f890a0b516404de3292f7adc774` |
| CE | 4 | 14224 / 2371 / 2371 | 512 | yes | 67 | `e0981f256cd2f7f9ba490a4cd5ae3663f2ad1b20c649f74411fde756685f25bc` |
| RPS | 1 | 14224 / 2371 / 2371 | 512 | yes | 67 | `1e03ae814d3e2e18cdd6f32ae30c239df8108b35a452e338650200338e510cc8` |
| RPS | 2 | 14224 / 2371 / 2371 | 512 | yes | 67 | `e1938497ab73fe7c2476930bb5dc29924cfccfa30522882be4de9dad4fb72646` |
| RPS | 3 | 14224 / 2371 / 2371 | 512 | yes | 67 | `e767f60548fa5f28d53f625208f5ed49635668254334bbd1a41c6a7a490502fe` |
| RPS | 4 | 14224 / 2371 / 2371 | 512 | yes | 67 | `dfbcf691a239e7a1c5dd42cef0285684644c11bdf900ae3e055e9ec129c972f3` |

## Original A replay

For every exported split, replay used the frozen original head `W_A h + b_A`, then recomputed softmax, mode, and the repository exact discrete L1 decision. The table reports the maximum difference over train, validation, and test. The accepted-backbone validation comparison separately compares direct V100 export logits/probabilities to the saved selected-checkpoint validation arrays.

| Loss | Seed | Max logit diff | Max probability diff | Mode differences | Exact-L1 differences | Accepted validation max logit/prob diff | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| CE | 1 | 6.67572021e-06 | 8.34465027e-07 | 0 | 0 | 0 / 0 | READY |
| CE | 2 | 7.62939453e-06 | 8.94069672e-07 | 0 | 0 | 0 / 0 | READY |
| CE | 3 | 7.62939453e-06 | 1.04308128e-06 | 0 | 0 | 0 / 0 | READY |
| CE | 4 | 7.62939453e-06 | 9.23871994e-07 | 0 | 0 | 0 / 0 | READY |
| RPS | 1 | 5.7220459e-06 | 8.94069672e-07 | 0 | 0 | 0 / 0 | READY |
| RPS | 2 | 5.7220459e-06 | 8.34465027e-07 | 0 | 0 | 0 / 0 | READY |
| RPS | 3 | 6.67572021e-06 | 9.53674316e-07 | 0 | 0 | 0 / 0 | READY |
| RPS | 4 | 4.76837158e-06 | 9.53674316e-07 | 0 | 0 | 0 / 0 | READY |

## Direct A baseline metrics on test

These are descriptive A-only exact-L1 metrics. They were not used to select checkpoints, modify a protocol setting, or fit a head.

| Loss | Seed | C4 MAE | C4 exact-match | C4 mean p4 | C4 predictive mean | C4 inward shrinkage | C4 routing [0,1,2,3,4] | Global MAE | Macro MAE | Severe error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| CE | 1 | 0.447761 | 0.671642 | 0.602169 | 3.445949 | 0.512900 | [1, 1, 3, 17, 45] | 0.253480 | 0.359907 | 0.012231 |
| CE | 2 | 0.537313 | 0.492537 | 0.436789 | 3.328888 | 0.661194 | [0, 0, 2, 32, 33] | 0.267819 | 0.371977 | 0.016871 |
| CE | 3 | 0.641791 | 0.477612 | 0.507764 | 3.294301 | 0.645397 | [1, 0, 5, 29, 32] | 0.257275 | 0.407099 | 0.012653 |
| CE | 4 | 1.328358 | 0.014925 | 0.069761 | 2.598408 | 1.315860 | [1, 0, 20, 45, 1] | 0.281738 | 0.555430 | 0.020666 |
| RPS | 1 | 1.089552 | 0.134328 | 0.245677 | 2.904302 | 1.033319 | [1, 2, 8, 47, 9] | 0.277520 | 0.541985 | 0.016871 |
| RPS | 2 | 0.522388 | 0.597015 | 0.538126 | 3.282828 | 0.644707 | [1, 1, 3, 22, 40] | 0.269085 | 0.422053 | 0.016449 |
| RPS | 3 | 0.432836 | 0.641791 | 0.607948 | 3.445519 | 0.550029 | [0, 0, 5, 19, 43] | 0.270772 | 0.416111 | 0.015183 |
| RPS | 4 | 0.492537 | 0.567164 | 0.537920 | 3.416954 | 0.538659 | [1, 0, 1, 27, 38] | 0.262337 | 0.357249 | 0.018136 |

## Hardware/replay provenance

CE seeds 1–2 exported successfully on V100 in their first attempts. Initial A30 attempts for CE seeds 3–4 and RPS seeds 1–4 exceeded the already frozen `1e-4` validation-output comparison gate (maximum logit differences 0.002617–0.006772; probability differences 0.000634–0.001433). No archive was promoted from those failed attempts. The identical inference-only exports were rerun on `acidsgcn011` V100, the hardware that trained the selected backbones; all then passed the unchanged gate. This is a recorded hardware/provenance remediation, not a tolerance relaxation or scientific reroll.

## Readiness

- Ready units: CE seeds 1–4; RPS seeds 1–4.
- Failed units: none after the V100 provenance reruns.
- No C/N fitting, backbone retraining, checkpoint alteration, split change, or manuscript change occurred.
- Machine-readable feature manifest: `outputs/utkface/prospective_replication/features/feature_export_manifest.json`.
