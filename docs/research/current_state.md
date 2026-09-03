# Current Research State

## Status
**Active project:** Ordinal Uncertainty Quantification for Imbalanced Ordinal Classification  
**Current stage:** Phase 3.7A-UTKFace complete — PARTIAL REPLICATION. Next:
**Phase 3.8 Solar Rare-Extreme Shrinkage Confirmation**.

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
- **RPS:** retain as a core probabilistic ordinal baseline. Its RetinaMNIST
  risk-quality results were strong, and it partly improved UTKFace class-4
  localization, but its broad risk-quality advantage did not replicate on
  UTKFace.
- **CORAL:** STOP — scientifically noncompetitive.
- **Weighted CE:** STOP — scientifically noncompetitive.
- **SLACE:** STOP — scientifically noncompetitive.
- **Simple-new-uncertainty-metric branch:** STOP.

## Phase 1–2.5 Summary
- Native-28 CE five-seed baseline established.
- L1-optimal decision reduces MAE/severe burden relative to mode; all new methods must be compared against **CE + L1**, not only CE + mode.
- RPS gives a **mixed RetinaMNIST model-level signal**: stronger risk
  alignment/severe detection/selective prediction, but no uniform
  ordinal-decision improvement.
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

On RetinaMNIST, RPS remains valuable because its decision-risk signal identifies
difficult samples better than CE, even though it does not localize the rare
endpoint well. This is not a universal cross-dataset conclusion.

## Phase 3.5 Research Question
The design question addressed by the completed audit was:

> **Can ordinal decision risk guide representation learning so that rare, high-risk extreme samples become better localized without destroying probabilistic risk quality?**

The most promising design direction is an **ordinally structured representation intervention**, with any prior/logit-adjusted head treated as an established control/secondary component rather than the primary novelty.

## Phase 3.5 Status
**Method-design audit complete; no implementation or training started.** The audit selected risk-gated adjacent-centroid ranking for one predeclared final RetinaMNIST seed-0 falsification experiment, with detached L1 Bayes-risk weighting and an RPS base loss. It is literature-overlapping and empirically unvalidated; it is neither a novelty nor an effectiveness claim. The collapse-aware adaptive margin is backup and risk-weighted prototype compactness is deferred. See [Phase 3.5 design audit](phase3_5_risk_conditioned_representation_design.md).

## Phase 3.6 — RG-ACR Seed-0 Falsification
**NO-GO — RG-ACR branch stopped.** The validation-selected λ=.05 condition did not produce a clear, cross-geometry class-4 representation improvement and violated predeclared class-0 and risk-quality tolerances. Its favorable downstream class-4 output changes do not establish the intended representation mechanism. The unselected λ=.20 geometry observation must not be used for post-hoc redesign. See [Phase 3.6 seed-0 record](phase3_6_rg_acr_seed0.md).

## Phase 3.7A-UTKFace — Ordinal Failure Replication
Using the frozen historical UTKFace manifest, five age bins (`<20`, `20–39`,
`40–59`, `60–79`, `>=80`), and matched CE/RPS seed-0 ResNet18 runs, the
independent replication was **PARTIAL**.

- The rare oldest class remains elevated-risk and inward-shrunk; its predictive
  mean was 3.25 (CE) / 3.27 (RPS), below the true class 4.
- RPS partly improved class-4 L1 recovery (35/67 exact vs CE 24/67), but at a
  global and lower-endpoint cost.
- The broad RetinaMNIST RPS L1 risk-quality advantage did not reproduce: CE had
  higher L1 Spearman, AUPRC, and lower selective MAE; RPS only had higher AUROC.

Decision: no automatic UTKFace representation/head audit. See
[the detailed replication record](phase3_7a_utkface_failure_replication.md).

**Phase 3.7A-Solar remains recorded as PAUSED BEFORE TRAINING.** No solar
CE/RPS full run or scientific conclusion exists; Phase 3.8 has not yet started.

## Updated Cross-Dataset Diagnosis

The common evidence across RetinaMNIST and UTKFace is:

\[
\boxed{\text{Rare upper-extreme inward localization bias under ordinal imbalance}}
\]

RetinaMNIST showed severe upper-extreme collapse and a useful RPS risk-quality
advantage. UTKFace showed a milder but clear inward shift, elevated upper-endpoint
risk, and a lower endpoint that was materially easier; RPS partly improved
class-4 localization but was not globally or broadly risk-quality superior.

The current research question is:

> **Why do imbalanced ordinal classifiers exhibit systematic inward
> localization bias for rare upper-extreme classes, even when uncertainty,
> risk quality, and exact-class performance differ across datasets and
> objectives?**

## Development-Benchmark Guardrail
RetinaMNIST has been inspected extensively during method development and remains
a **development benchmark**. Do not use either RetinaMNIST or UTKFace test
diagnostics for iterative method redesign while Phase 3.8 confirms the
cross-dataset failure pattern.

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
**Phase 3.8 — Solar Rare-Extreme Shrinkage Confirmation** is authorized. Its
primary replication target is upper-extreme elevated risk, inward shrinkage,
endpoint asymmetry, and the insufficiency of L1/L2 decision correction under
matched CE/RPS controls. RPS superiority is secondary, not required.

Do not restart method development, revive RG-ACR, create UTKFace-specific
objectives, perform automatic representation/head audits, or launch multi-seed
expansions before the third-dataset evidence is reviewed.
