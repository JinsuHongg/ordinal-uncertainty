# Solar CE seed-2 replacement run

## 1. Motivation

The original Solar CE seed-2 frozen-feature regeneration repeatedly failed the
pre-specified probability identity gate before natural-direction-only N fitting.
This was a provenance/reproducibility failure, not an N-result-driven decision.

## 2. Original seed-2 invalidation

The original checkpoint and A/C artifacts remain archived under their original
paths. They are invalid only for the CE N follow-up because no regenerated
feature archive met the fixed gate. They were not deleted or overwritten.

## 3. Why replacement was authorized

The authorized replacement is one new logical-seed-2 unit trained end-to-end so
that its A, C, N, and analyses share one checkpoint and frozen feature source.
It is not claimed to recreate the old seed-2 model.

## 4. Frozen replacement protocol

The protocol was saved before training in
`docs/research/solar_ce_seed2_replacement_protocol.md` (SHA256
`5a211d988e77112f745a592a7ae27e3be684f78bf3b7a7b94d6c2f90550e27e2`).
It fixes the V100/qGPU24/`csc344r253` environment, canonical Solar splits and
normalization, CE backbone training, and the frozen C/N head protocols.

## 5. Environment and hardware

Backbone job `4445410`, feature/A job `4449190`, C job `4451373`, and N job
`4451438` ran on V100 `acidsgcn007` under account `csc344r253`. The environment
was Python 3.11.15, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, CUDA 12.4,
cuDNN 90100, driver 580.159.04, float32/no autocast, with TF32 matmul disabled.
Two feature preflight attempts on `acidsgcn001` failed before Python inference
because that node had an NVML driver/library mismatch; their logs are retained
and neither created features or a head fit.

## 6. Training result and selected checkpoint

The single actual replacement training run selected epoch 1 by minimum
validation CE (`1.06026047215748`). Checkpoint SHA256:
`4e25bb8128ab902f323805577ba060dc7d1c3722a445274f6fc1a850ff406e0e`.

## 7. Feature export and A definition

The replacement features have 45,047 train, 2,431 validation, and 28,006
readout rows with 921 X rows; they are float32 512-D pre-fc representations.
The original replacement fc head defines A. Feature-to-head replay maximum
absolute error was `7.62939453125e-06`.

## 8. A/C/N integrity

C used the frozen balanced CE direction-only protocol (100 epochs, 70,400
steps); N used the same configuration with exactly one natural empirical pass
per epoch. Both retained fixed biases and norms with maximum recorded drift
zero. N observed empirical counts across 100 epochs exactly equal to the
expected class counts times 100.

## 9. Replacement H1

Using exact-L1 decisions on true X:

| Head | Endpoint MAE |
| --- | ---: |
| A | 1.6667 |
| C | 0.5733 |
| N | 1.3550 |

Replacement H1 (C minus A) is `-1.0934`.

## 10. Replacement H2a

Replacement seed-2 H2a has `beta_ord=-0.3680`, HC3 95% CI
`[-0.4020, -0.3341]`, and `Delta_LOO=0.0616`. The replacement-aware CE
four-seed A/C summary (1, replacement-2, 3, 4) has four negative H1 deltas,
four negative H2a ordinal coefficients, and four positive Delta_LOO values:
the frozen A/C H1 verdict is REPLICATED and H2a verdict is STRONG.

## 11. Trade-off and sampling result

Replacement global/macro/severe metrics (A/C/N) are respectively:

- global MAE: 0.4826 / 0.5439 / 0.4383;
- macro MAE: 0.7131 / 0.5182 / 0.5994;
- global severe error: 0.0452 / 0.0677 / 0.0394.

N improves the replacement endpoint versus A by `-0.3116`, but remains worse
than C by `+0.7818`. Detailed routing and endpoint-mass outputs are in the
replacement analysis directory.

## 12. Cross-seed limitation

A four-seed CE N consistency label cannot be computed: validated CE N heads
for original seeds 1, 3, and 4 do not exist, and this task explicitly forbids
modifying those seeds. The replacement makes no Solar-CE-wide sampling claim.

## 13. Provenance statement

`replacement_provenance.json` preserves the original/replacement relationship.
No original seed-2 result appears in the replacement-aware A/C summaries; old
seed-2 output remains historical archival evidence only. No manuscript or
appendix text was changed.
