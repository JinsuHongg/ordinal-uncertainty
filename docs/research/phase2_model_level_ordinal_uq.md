# Phase 2 — Model-Level Ordinal Uncertainty Study

## Goal

Phase 2 tested whether established ordinal training objectives improve the
predictive distribution beyond the gains available from decision-rule correction
alone. The canonical setup was native-28×28 RetinaMNIST with the same
small-image ResNet18 and official splits. The critical control was **CE +
L1-optimal decision**, not only CE + mode.

## Methods

- **CE:** Softmax probabilities trained with cross-entropy.
- **CORAL:** cumulative ordinal baseline. It was evaluated at seed 0 and then
  stopped.
- **RPS:** Softmax probabilities trained with normalized differentiable Ranked
  Probability Score; evaluated over seeds 0–4.

All probability-based comparisons used exact discrete mode, L1-optimal, and
L2-optimal decisions, with matched expected decision risks.

## CORAL seed-0 result

CORAL mode prediction achieved accuracy **0.4275**, MAE **1.1900**, QWK
**0.3422**, and severe-error prevalence **39.5%**. This was a valid baseline
result, but scientifically uncompetitive for the project endpoints.

**Decision: STOP — SCIENTIFICALLY NONCOMPETITIVE.** No additional CORAL seeds
were run.

## RPS primary five-seed comparison

Values are mean ± sample standard deviation for the L1-optimal decision.

| Endpoint | CE + L1 | RPS + L1 |
| --- | ---: | ---: |
| Accuracy | 0.5295 ± 0.0116 | 0.5330 ± 0.0108 |
| MAE | 0.6950 ± 0.0352 | 0.6845 ± 0.0327 |
| QWK | 0.5771 ± 0.0294 | 0.5759 ± 0.0290 |
| Severe prevalence | 18.30% ± 1.67% | 18.35% ± 1.33% |
| L1-risk Spearman | 0.3832 ± 0.0403 | 0.4191 ± 0.0181 |
| Severe AUROC | 0.6098 ± 0.0377 | 0.6530 ± 0.0108 |
| Severe AUPRC | 0.2512 ± 0.0538 | 0.2804 ± 0.0278 |
| Mean MAE selective risk | 0.4842 ± 0.0304 | 0.4412 ± 0.0313 |

The available seed-0 artifacts also compare CE/RPS mode, L1, and L2 directly;
the multi-seed table above is the primary decision-controlled result.

## Interpretation

**Phase 2 decision: MIXED MODEL-LEVEL SIGNAL.** RPS improves useful decision-risk
geometry, severe-error detection, and ordinal selective prediction, but does not
uniformly improve actual severe/extreme decisions or QWK. RPS is retained as the
strongest probabilistic ordinal baseline; it is not evidence that ordinal training
solves rare upper-extreme localization.

Phase 2.5 subsequently showed that validation-only temperature scaling does not
remove RPS's risk-quality advantage. See
[`phase2_5_calibration_control_audit.md`](phase2_5_calibration_control_audit.md)
for that control.
