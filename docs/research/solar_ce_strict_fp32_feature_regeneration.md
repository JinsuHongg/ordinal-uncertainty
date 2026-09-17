# Solar CE Strict-FP32 Frozen Feature Regeneration

**Verdict:** **C — CE FULL-DATA IDENTITY CHECK STILL FAILS**

## Motivation and execution

This CE-only inference task followed the prior subset audit, which supported
A30 TF32 arithmetic as an initial mismatch contributor. It used only CE seeds 1--4,
the audited epoch-1 checkpoints, strict state loading, the verified signed-log
and train-only normalization path, fixed loader order, no augmentation,
`model.eval()`, and `torch.inference_mode()`. No backbone training, head fit,
A/C refit, checkpoint change, N fit, or RPS access occurred.

Jobs `4424725`, `4424726`, `4424727`, and `4424724` ran on NVIDIA A30 GPUs.
They explicitly set `torch.backends.cuda.matmul.allow_tf32=False`,
`torch.backends.cudnn.allow_tf32=False`, and
`torch.set_float32_matmul_precision("highest")`; no autocast was used and
model/input dtypes were float32.

## Identity inputs

All four checkpoints remained `READY`, hash-verified, and selected at epoch 1.
The normalization artifact remained
`outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json`
with SHA256
`1178b05f9fafad22eec58608236ee5ba035a2f4629bfc17c10281797da8ed02b`.
The attempted train/validation/readout populations were the audited
`45,047 / 2,431 / 28,006` rows with 921 endpoint X rows and 512-D float32
pre-`fc` features.

## Full A-output identity gate

All full jobs reached the evaluation identity comparison but failed the frozen
numerical tolerance. IDs and labels matched; mode differences and exact-L1
differences were zero in all four seeds. However, logits/probabilities did not
remain within the frozen `2e-05 / 2e-06` max-error limits.

| Seed | Max / mean logit error | Max probability error | Mode / L1 differences |
| ---: | ---: | ---: | ---: |
| 1 | `2.289e-05 / 2.957e-06` | `3.657e-06` | 0 / 0 |
| 2 | `1.669e-05 / 1.880e-06` | `3.809e-06` | 0 / 0 |
| 3 | `2.289e-05 / 2.379e-06` | `5.047e-06` | 0 / 0 |
| 4 | `2.384e-05 / 2.398e-06` | `4.647e-06` | 0 / 0 |

The successful 64-row strict-FP32 subset did not bound full-readout numerical
drift tightly enough. The task does not loosen the tolerance post hoc and does
not accept decision agreement alone as identity.

## Artifact disposition and readiness

The exporter creates the canonical output directory only after A identity PASS.
Consequently, no strict-FP32 CE feature archive or manifest was promoted and
no failed archive overwrote historical evidence. The earlier failed default-A30
exports likewise produced only logs because their gates failed before writing.

CE N follow-up remains blocked for all seeds. The later GPU-provenance audit
verified historical CE V100 and current A30 execution, but the residual
strict-FP32 full-readout error means that differential is not yet a complete
causal explanation. A future repair would need a new, separately authorized
compatibility investigation (for example, a historically compatible V100
execution path); this task stops without retrying.

## Artifacts

- CE strict-FP32 array: `4424724` (`4424725--4424727` child jobs)
- Terminal logs: `logs/solar_ce_strict_fp32/`
- Exporter: `scripts/export_solar_confirmatory_frozen_features.py`
