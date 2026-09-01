# Current Research State

## Status
**Active project:** Ordinal Uncertainty Quantification for Imbalanced Ordinal Classification  
**Current stage:** Pre-Phase 3.5 checkpoint — method-design audit next, no new method frozen

The project now has a decomposed diagnosis of the rare upper-extreme failure on RetinaMNIST. Output-only corrections are stopped. Phase 3.3 and 3.4 show that the remaining failure has both representation-level and classifier/head-level components.

## Canonical RetinaMNIST Setup
- Native 28×28 RGB
- Official train/validation/test splits
- Unpretrained small-image ResNet18
- 3×3 stride-1 stem, no max-pool
- Canonical seeds: `0, 1, 2, 3, 4`
- Historical 64×64 resizing is sensitivity evidence only

## Core Conceptual Decomposition
The project separates:

\[
\text{Predictive distribution}
\rightarrow
\text{Decision rule}
\rightarrow
\text{Expected decision risk}.
\]

After Phase 3.3–3.4, representation and head effects must also be separated:

\[
\text{Representation}
\rightarrow
\text{Probabilistic head}
\rightarrow
\text{Decision rule / risk}.
\]

## Established Baselines
- **CE:** canonical nominal probabilistic baseline.
- **RPS:** retain as the strongest probabilistic ordinal baseline for decision-risk alignment, severe-error detection, and selective ordinal prediction.
- **CORAL:** STOP — scientifically noncompetitive.
- **Weighted CE:** STOP — scientifically noncompetitive.
- **SLACE:** STOP — scientifically noncompetitive.
- **Simple-new-uncertainty-metric branch:** STOP.

## Phase 1–2.5 Summary
- Native-28 CE five-seed baseline established.
- L1-optimal decision reduces MAE/severe burden relative to mode; all new methods must be compared against **CE + L1**, not only CE + mode.
- RPS gives a **mixed model-level signal**: stronger risk alignment/severe detection/selective prediction, but no uniform ordinal-decision improvement.
- Validation-only temperature scaling does not remove the RPS risk-quality advantage.

## Phase 3.0–3.1 — Failure Diagnosis and Strong Baselines
Class 4 is the rare upper extreme (`66` train, `6` validation, `20` test).

Main failure:
- class-4 probability mass and decisions are pulled inward toward classes 2–3;
- expected ordinal risk is often high, but localization remains poor;
- L1/L2 decision correction and temperature scaling do not recover class 4;
- generic inverse-frequency CE and SLACE do not resolve the failure.

Decision:

\[
\boxed{\text{ORDINAL CENTER SHRINKAGE UNDER IMBALANCE}}
\]

remains the actionable phenomenon.

## Phase 3.2 — Output-Level Candidate Branch
### Candidate 1: Endpoint-Neighborhood RPS
Endpoint-neighborhood supervision moved class-4 L1 decisions outward from class 2 toward class 3, confirming that location movement is trainable. However, no exact `4→4` recovery occurred and global/risk-quality trade-offs remained.

Decision:

\[
\boxed{\text{TRADE-OFF — NO MULTI-SEED}}
\]

### Candidate 1b: True-Endpoint Preference
A fixed true-endpoint preference (`rho=0.5`) did not increase class-4 `p4` relative to Candidate 1, weakened adjacent recovery, and still produced no exact `4→4` recovery.

Decision:

\[
\boxed{\text{CANDIDATE 1 BRANCH — NO-GO}}
\]

Do not create Candidate 1c from RetinaMNIST test diagnostics. The output-only correction branch is closed.

## Phase 3.3 — Frozen Representation Failure Audit
Frozen seed-0 CE/RPS checkpoints were audited without training. Penultimate features were the 512-D inputs to `model.fc`; training features alone defined class centroids.

Key findings:
- class-4 representation collapse is real but not universal;
- raw nearest-centroid class-4 routing assigned only `9/20` CE and `6/20` RPS samples to the class-4 centroid;
- RPS had smaller class-3/class-4 centroid separation than CE in both raw and normalized geometry;
- many class-4 samples were closer to class 2/3 than class 4;
- however, some samples already nearest to class 4 were still mapped centrally by the classifier head.

Decision:

\[
\boxed{\text{MIXED REPRESENTATION / HEAD FAILURE}}
\]

RPS improves risk geometry but does not improve class-4 representation separation relative to CE.

## Phase 3.4 — Frozen-Feature Head Intervention Audit
Eight seed-0 linear-head conditions were evaluated on frozen CE/RPS 512-D features. No backbone training occurred.

Key findings:
- balanced/prior-adjusted heads recovered some feature-nearest-to-4 samples;
- on CE features, logit adjustment produced `3/9` exact recoveries among feature-nearest-to-4 class-4 cases;
- most feature-nearest-central class-4 cases remained unrecovered under every simple head: `10/11` for CE and `11/14` for RPS;
- head corrections introduced global and/or risk-quality trade-offs and did not solve the failure overall.

Decision:

\[
\boxed{\text{MIXED BUT DECOMPOSABLE FAILURE}}
\]

Interpretation:
- **head/prior bias is actionable for a subset** where representation already contains class-4 geometry;
- **representation limitation is dominant for the collapsed subset**;
- neither head-only nor representation-only explanations are sufficient.

## Current Scientific Conclusion
The strongest current diagnosis is:

\[
\boxed{\text{Dual-component rare-extreme failure}}
\]

with two components:
1. **representation collapse** for many rare class-4 samples;
2. **head-level inward bias** for some samples whose representation is already class-4-like.

RPS remains valuable because its decision-risk signal identifies difficult samples better than CE, even though it does not localize the rare endpoint well.

## Pre-Phase 3.5 Research Question
The next design question is:

> **Can ordinal decision risk guide representation learning so that rare, high-risk extreme samples become better localized without destroying probabilistic risk quality?**

The most promising design direction is an **ordinally structured representation intervention**, with any prior/logit-adjusted head treated as an established control/secondary component rather than the primary novelty.

## Phase 3.5 Status
**Not started.** The next task is a literature-and-method-design audit only. It should compare at most three candidates, including risk-conditioned ordinal separation, collapse-aware adjacent margin, and risk-weighted prototype/compactness regularization.

No Phase 3.5 method is frozen or implemented.

## Development-Benchmark Guardrail
RetinaMNIST has been inspected extensively during method development and should now be treated as a **development benchmark**. The next seed-0 method experiment should be the last major method-selection step based on RetinaMNIST test diagnostics. If promising, freeze the method before multi-seed and multi-dataset confirmation.

## Guardrails
Do not currently:
- create Candidate 1c;
- restart the output-only correction branch;
- run Candidate 1/1b seeds 1–4;
- claim a new uncertainty metric;
- claim novelty before Phase 3.5 literature verification;
- implement generic SupCon / balanced SupCon / prototype / logit-adjustment combinations as a proposed method without novelty analysis;
- add ensemble, Bayesian, or conformal extensions;
- tune a new method on RetinaMNIST test outcomes repeatedly.

## Next Authorized Work
Run **Phase 3.5 — Risk-Conditioned Ordinal Representation Method Design Audit**:
1. literature overlap audit;
2. exactly three candidate mechanisms;
3. rank novelty/mechanistic fit/stability;
4. specify one primary mathematical objective;
5. predeclare one final RetinaMNIST seed-0 falsification experiment;
6. no implementation or training during the design audit.
