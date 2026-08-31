# Phase 3.0 — Extreme-Class Failure Audit

## Protocol

This is a frozen-prediction diagnostic: native-28x28 CE and RPS outputs for seeds 0--4 were reused in raw and validation-temperature-scaled forms. No model was trained, calibrated again, or otherwise changed. Source test IDs and labels align exactly across all conditions (400 examples per seed). The requested historical Phase 2 note was absent; complete Phase 2.5 artifacts were the authoritative inputs.

## Dataset imbalance

| split | class 0 | class 1 | class 2 | class 3 | class 4 |
|---|---:|---:|---:|---:|---:|
| train | 486 (45.0%) | 128 (11.9%) | 206 (19.1%) | 194 (18.0%) | 66 (6.1%) |
| validation | 54 (45.0%) | 12 (10.0%) | 28 (23.3%) | 20 (16.7%) | 6 (5.0%) |
| test | 174 (43.5%) | 46 (11.5%) | 92 (23.0%) | 68 (17.0%) | 20 (5.0%) |

Class 4 is rarest (13.6% of the largest training class); class 0 is most common, not underrepresented. Frequency plausibly contributes to class-4 failure but does not prove its cause.

## Extreme-class results

L1-decision five-seed means:

| true class / condition | accuracy | MAE | severe prevalence |
|---|---:|---:|---:|
| class 0, CE raw | 0.740 | 0.518 | 0.224 |
| class 0, CE calibrated | 0.743 | 0.515 | 0.224 |
| class 0, RPS raw | 0.724 | 0.534 | 0.229 |
| class 0, RPS calibrated | 0.717 | 0.529 | 0.233 |
| class 4, CE raw | 0.000 | 2.140 | 0.760 |
| class 4, CE calibrated | 0.000 | 2.150 | 0.770 |
| class 4, RPS raw | 0.000 | 2.120 | 0.820 |
| class 4, RPS calibrated | 0.000 | 2.240 | 0.950 |

For true class 4 under L1, CE raw routes 15%, 8%, 53%, 24%, and 0% to classes 0--4; RPS raw routes 9%, 12%, 61%, 18%, and 0%. RPS reduces the farthest 4-to-0 error (9% vs 15%) but concentrates at 4-to-2. Scaling makes RPS still more central (75% 4-to-2; 5% 4-to-3). No class-4 sample is predicted as class 4 by mode, L1, or L2. This is therefore not corrected by an alternate Bayes decision.

## Predictive-distribution geometry

| condition, true class 4 | [p0, p1, p2, p3, p4] |
|---|---|
| CE raw | [0.269, 0.112, 0.286, 0.258, 0.074] |
| CE calibrated | [0.270, 0.114, 0.277, 0.265, 0.074] |
| RPS raw | [0.255, 0.124, 0.278, 0.307, 0.036] |
| RPS calibrated | [0.257, 0.141, 0.287, 0.281, 0.034] |

For true class 4, CE/RPS raw mean p4 is 0.074/0.036; p3+p4 is 0.332/0.343; and p2+p3+p4 is 0.619/0.620. Some mass is nearby, but the true extreme itself is strongly suppressed. RPS raises p3 relative to CE but reduces p4, explaining improved ordinal-risk diagnostics without recovered class-4 accuracy.

For true class 0, CE/RPS raw mean p0 is 0.729/0.723; p0+p1 is 0.776/0.782; and p0+p1+p2 is 0.898/0.890. Thus this is not a symmetric general inability to represent both extremes.

For class 4, CE/RPS raw predictive means are 1.756/1.744 (truth 4), inward shrinkage is 1.460/1.595, and predictive-mean bias is -2.244/-2.256. L1 decision bias is -2.140/-2.120. For class 0, predictive bias is upward (+0.618 CE, +0.616 RPS), but class-0 mass remains dominant.

## Risk and calibration

Class-4 mean L1 Bayes risk is high compared with class 0: raw CE 0.794 vs 0.362 and raw RPS 0.851 vs 0.369. Hence this is not uniformly confident failure: the models identify class-4 examples as risky on average but cannot locate the extreme. Within class 4, however, risk/error association is weak and unstable (raw Spearman -0.163 CE, 0.106 RPS), unlike class 0 (0.706 CE, 0.730 RPS).

Temperature scaling changes sharpness, not location: CE p4 remains 0.074 and RPS falls from 0.036 to 0.034. It increases RPS shrinkage (1.595 to 1.608) and class-4 severe prevalence (82% to 95%).

## Failure classification and gate

**Dominant failure: MIXED FAILURE.** Evidence supports (1) **PRIOR / IMBALANCE COLLAPSE**—class 4 is rare and its probability is suppressed; (2) **ORDINAL CENTER SHRINKAGE**—means are strongly pulled inward; and (3) **HIGH-RISK BUT UNRESOLVED FAILURE**—risk is high while decisions remain wrong. It is not primarily **DECISION-RULE FAILURE**, because mode/L1/L2 never recover class 4.

**Phase 3.1 gate: TARGET ORDINAL SHRINKAGE.** The next diagnostic target is reducing inward extreme-class probability shrinkage while preserving calibrated risk geometry. Imbalance remains a plausible contributor, but five class frequencies cannot establish it as the sole mechanism. No Phase 3.1 method is proposed here.

Per-seed confusion, error-distance, probability, bias, and risk artifacts are under `outputs/retinamnist/native28/phase3_0_extreme_class_audit_complete/summary/`.
