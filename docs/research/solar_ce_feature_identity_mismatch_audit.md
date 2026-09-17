# Solar CE Feature Identity Mismatch Audit

**Verdict:** **B — CE ROOT CAUSE IDENTIFIED; REPAIR NEEDS VALIDATION**

## Executive summary

The CE Phase-A mismatch is not explained by checkpoint, sample-alignment,
preprocessing, model-construction, or state-load failure. Direct later
provenance evidence verifies that archived CE A/C jobs ran on V100 GPUs while
the failed feature export ran on an A30. A fixed 64-row A30 strict-FP32 audit
passes all four CE seeds within frozen tolerances and with exact-L1 agreement.
This supports an A30 default-TF32 contribution, but does not alone prove that
hardware/TF32 is the sole full-readout cause. No full regeneration, N fit,
backbone training, A/C fit, or checkpoint modification occurred.

## Observed mismatch and RPS control

Failed CE full-export maximum logit errors for seeds 1--4 were `.00849`,
`.00620`, `.00794`, and `.00446`; exact-L1 differed. The failed exporter did
not retain full new arrays, so its changed-decision count is unavailable.

RPS is the control: it passes with the same model, loader, signed-log,
normalization, channels, and exporter. Archived RPS A/C and RPS export both
used A30; archived CE A/C used V100 while CE export used A30. This is the
verified CE/RPS hardware differential, although the full strict-FP32 result
means it is not a complete exclusive causal explanation.

| Field | CE | RPS | Relevant? |
| --- | --- | --- | --- |
| Checkpoint wrapper | `state_dict` + metadata | same | no |
| Model | torchvision ResNet18, `fc(512,5)` | same | no |
| Channels / transform | `[8,7,1]`, signed-log | same | no |
| Normalization | shared train-only artifact | same | no |
| Archived A/C hardware | V100 | A30 | yes |
| Current export hardware | A30 | A30 | yes |

## Checkpoint, construction, and state load

CE checkpoint SHA256 values for seeds 1--4 are `6cb2accf…2559b`,
`c746ee84…fd01`, `263800d9…36f7`, and `71bb16b3…c909`; all are complete
122-key states with CE method, matching seed, selected epoch 1, and recorded
validation losses `1.221391`, `1.037971`, `1.096287`, and `1.155523`.
Archived A-head weights exactly match checkpoint `fc` weights. Strict loading
has no missing/unexpected keys, BatchNorm running buffers are present, and
model eval mode is set. The archived/current A/C source paths have no relevant
diff; there is no EMA/alternate state, wrapper, dropout, modified stem, or
partial load.

## Preprocessing, normalization, and sample alignment

Both paths use sorted split manifests, no augmentation, float32 inputs,
channels `hmi_m/aia1600/aia131` at `[8,7,1]`, and
`sign(x)*log1p(abs(x))`. The exact train-only normalization artifact has SHA256
`1178b05f9fafad22eec58608236ee5ba035a2f4629bfc17c10281797da8ed02b`.
All fixed-subset IDs and labels match archived rows. Thus divergence begins
after input construction/state loading, in model arithmetic.

## Precision/eval-mode subset validation

Job `4424705` ran the first 64 archived readout rows per seed on A30. Default
TF32-capable A30 execution was compared with strict FP32
(`matmul.allow_tf32=False`, `cudnn.allow_tf32=False`,
`set_float32_matmul_precision("highest")`).

| Seed | Default max / mean logit error | Strict max / mean logit error | Strict L1 changes |
| ---: | ---: | ---: | ---: |
| 1 | `.004373 / .001837` | `7.39e-06 / 1.53e-06` | 0 |
| 2 | `.001617 / .000412` | `4.65e-06 / 1.04e-06` | 0 |
| 3 | `.003615 / .000690` | `6.20e-06 / 1.20e-06` | 0 |
| 4 | `.002101 / .000488` | `6.68e-06 / 1.19e-06` | 0 |

Strict maximum probability error is `1.23e-06` (frozen limit `2e-06`).
Strict 32-versus-64 batch variation is at most `1.91e-06` logits, excluding
train-mode/BatchNorm batch-stat dependence.

## Root cause, repair, and readiness

The bounded finding is **E — PRECISION / EVAL-MODE ISSUE, SUPPORTED BUT NOT
FULLY PROVEN**: default A30 TF32 arithmetic is a demonstrated numerical
contributor, not acceptable drift and not a tolerance issue. The proposed
minimal repair was deterministic CE inference with strict FP32 flags set
before model construction/forwarding, plus recorded GPU/precision metadata.
The strict-FP32 subset passes ID, logits, probabilities, and exact-L1 checks
for all seeds, so it was technically suitable for a full gate. Subsequent
full-readout strict-FP32 results exceeded frozen numerical tolerances; see
`solar_ce_strict_fp32_feature_regeneration.md`. CE N remains blocked. H1/H2a
and manuscript claims are unchanged.

## Artifacts

- `outputs/mechanism_replication/audits/solar_ce_identity_mismatch/subset_precision_audit.json`
- `scripts/audit_solar_ce_identity_subset.py`
- `scripts/slurm_solar_ce_identity_subset_audit.sbatch`
