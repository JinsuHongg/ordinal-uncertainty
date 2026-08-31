# Current Research State

## Status
**Active project:** Ordinal Uncertainty Quantification for Imbalanced Ordinal Classification  
**Current stage:** Phase 3.2 — method design justified, not started

The project now focuses on a specific failure in imbalanced ordinal classification: rare upper-extreme samples can receive high predicted decision risk while their predictive probability mass is still pulled strongly toward central classes.

## Canonical RetinaMNIST Setup
- Native 28×28 RGB
- Official train/validation/test splits
- Unpretrained small-image ResNet18
- 3×3 stride-1 stem, no max-pool
- Seeds: `0, 1, 2, 3, 4`
- Historical 64×64 resizing is sensitivity evidence only

## Conceptual Decomposition
The project separates:

\[
\text{Predictive distribution}
\rightarrow
\text{Decision rule}
\rightarrow
\text{Expected decision risk}.
\]

New models must be evaluated beyond argmax accuracy. The learned distribution, decision rule, and expected-risk/realized-loss alignment are separate experimental factors.

## Phase 1 / 1.5 — Uncertainty Diagnostics
Native-28 CE baseline, five seeds:
- Accuracy: **0.5320 ± 0.0169**
- MAE: **0.7430 ± 0.0541**
- QWK: **0.5526 ± 0.0290**
- NLL: **1.3650 ± 0.2566**
- Brier: **0.5905 ± 0.0106**
- RPS: **0.1288 ± 0.0033**
- ECE: **0.0954 ± 0.0408**

The strongest useful signals were decision-risk quantities rather than a new simple ordinal spread measure. The new-simple-metric branch is stopped.

## Phase 1.75 — Decision-Rule Audit
Five-seed means:
- Mode: Accuracy **0.5320**, MAE **0.7430**, QWK **0.5526**, severe **20.3%**
- L1-optimal: Accuracy **0.5295**, MAE **0.6950**, QWK **0.5771**, severe **18.3%**
- L2-optimal: Accuracy **0.4975**, MAE **0.7080**, QWK **0.5600**, severe **17.8%**

Verified:
- `prediction_distance_l1` = mode-centered expected L1 decision risk
- `bayes_risk_l2` = exact discrete L2 Bayes risk

Decision: **decision rule introduces a trade-off**. New ordinal models must be compared against **CE + L1**, not only CE + mode.

## Phase 2 — Existing Ordinal Learning
### CORAL
Seed-0 mode:
- Accuracy **0.4275**
- MAE **1.1900**
- QWK **0.3422**
- Severe **39.5%**

Decision: **STOP — SCIENTIFICALLY NONCOMPETITIVE**.

### RPS
Five-seed `CE + L1` vs `RPS + L1`:
- Accuracy: **0.5295 ± 0.0116** vs **0.5330 ± 0.0108**
- MAE: **0.6950 ± 0.0352** vs **0.6845 ± 0.0327**
- QWK: **0.5771 ± 0.0294** vs **0.5759 ± 0.0290**
- Severe prevalence: **18.30% ± 1.67%** vs **18.35% ± 1.33%**
- L1-risk Spearman: **0.3832 ± 0.0403** vs **0.4191 ± 0.0181**
- Severe AUROC: **0.6098 ± 0.0377** vs **0.6530 ± 0.0108**
- Severe AUPRC: **0.2512 ± 0.0538** vs **0.2804 ± 0.0278**
- Mean MAE selective risk: **0.4842 ± 0.0304** vs **0.4412 ± 0.0313**

Decision: **MIXED MODEL-LEVEL SIGNAL**. RPS is retained as the strongest probabilistic ordinal baseline.

## Phase 2.5 — Calibration Control
Temperature scaling improved calibration but did not remove the RPS risk-quality advantage.

Calibrated five-seed L1:
- Spearman: CE **0.3829 ± 0.0389**, RPS **0.4224 ± 0.0061**
- Severe AUROC: CE **0.6192 ± 0.0327**, RPS **0.6517 ± 0.0241**
- Severe AUPRC: CE **0.2699 ± 0.0558**, RPS **0.2940 ± 0.0418**
- Mean selective MAE risk: CE **0.4825 ± 0.0305**, RPS **0.4479 ± 0.0328**

Decision: **MIXED AFTER CALIBRATION**.

## Phase 3.0 — Extreme-Class Failure Audit
Class counts:
- Train: `[486, 128, 206, 194, 66]`
- Validation: `[54, 12, 28, 20, 6]`
- Test: `[174, 46, 92, 68, 20]`

Class 4 is a rare upper extreme. Main findings:
- class-4 L1 accuracy is 0% across CE/RPS raw and calibrated conditions
- predictive means are pulled toward the center
- true class-4 probability is suppressed
- expected L1 risk is high, but localization is poor
- temperature scaling does not fix the location bias
- L1/L2 decision changes do not recover class 4

Dominant failure: **mixed failure with strong ordinal center shrinkage**.

## Phase 3.1 — Existing Imbalance-Aware Ordinal Baselines
### Weighted CE
Inverse-frequency weights:
`[0.2937, 1.1151, 0.6929, 0.7357, 2.1626]`

It raises class-4 probability somewhat but does not recover class-4 L1 predictions and damages global performance.

Decision: **STOP — SCIENTIFICALLY NONCOMPETITIVE**.

### SLACE
SLACE was implemented from the AAAI 2025 formulation and official implementation reference. The earlier artifact-persistence concern was a false diagnosis caused by premature inspection / delayed workspace visibility.

Seed-0:
- Mode: Accuracy **0.510**, MAE **0.730**, QWK **0.537**, severe **20.5%**
- L1: Accuracy **0.508**, MAE **0.700**, severe **18.0%**
- Class-4 L1 accuracy **0%**
- Class-4 MAE **2.25**
- Class-4 severe **90%**
- Mean `p4` **0.070**
- Mean `p3+p4` **0.315**
- Predictive mean **1.777**
- Inward shrinkage **1.559**

Decision: **STOP — SCIENTIFICALLY NONCOMPETITIVE**.

## Current Research Conclusion
Strong existing controls do not resolve the rare upper-extreme location collapse.

- CE: strong global accuracy, poor rare-upper-extreme localization
- RPS: better risk alignment / severe detection / selective prediction, but class 4 remains collapsed
- Weighted CE: more rare-class mass, but no class-4 recovery and poor global quality
- SLACE: useful risk ranking, but no class-4 recovery and weak near-extreme mass

\[
\boxed{\text{Phase 3.1: SHRINKAGE PERSISTS AFTER STRONG BASELINES}}
\]

\[
\boxed{\text{PHASE 3.2 JUSTIFIED}}
\]

## Phase 3.2 Target
> **Reduce rare upper-extreme inward shrinkage while preserving RPS-like ordinal risk alignment, severe-error detection, selective prediction, and global predictive quality.**

No Phase 3.2 method is frozen yet.

## Guardrails
Do not currently:
- claim a new uncertainty metric
- claim a new ordinal loss
- claim universal superiority of ordinal UQ
- restart CORAL, Weighted CE, or SLACE multi-seed experiments
- return to 64×64 RetinaMNIST
- start ensemble or conformal extensions

## Immediate Next Action
Design a small number of minimal Phase 3.2 candidate objectives that directly target probability-location shrinkage. Validate each with a controlled seed-0 GO/NO-GO experiment before any multi-seed or multi-dataset expansion.
