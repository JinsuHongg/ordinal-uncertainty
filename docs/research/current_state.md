# Current Research State

## Status
**Active project:** Ordinal Uncertainty Quantification for Imbalanced Ordinal Classification  
**Current stage:** Phase 3.20A RetinaMNIST Controlled Imbalance-Severity Dose
Response complete — **B — PARTIAL / NON-MONOTONIC IMBALANCE EFFECT**.

### Project-level observed phenomenon

In the studied imbalanced ordinal settings, rare upper extremes exhibit
systematic inward localization. This is an observed three-dataset pattern, not
a controlled causal isolation of class imbalance from class difficulty,
representation quality, dataset structure, or label ambiguity.

### Cross-dataset frozen-head mechanism

Across the tested frozen RPS representations on RetinaMNIST, UTKFace, and
Solar, rare upper-end localization consistently improves under classifier
direction adaptation. Phase 3.18A and 3.18B additionally reproduce the A→C
response on frozen CE representations in RetinaMNIST and Solar. Direct
cross-objective direction evidence therefore covers CE and RPS in those two
domains, within evaluated one-backbone-seed protocols; it is not architecture-,
objective-, or population-universal.

### Dataset-specific mechanism findings

Scale, bias, opposite-endpoint, global predictive, and risk/UQ effects vary by
dataset. Controlled scale is a diagnostic probe, not a transferable correction;
bias-only correction remains stopped; ROP is diagnostic/secondary; and the
combined Retina candidate is not freeze-ready.

### RetinaMNIST-specific diagnosis

The historical RetinaMNIST representation/head audits support a mixed,
dual-component rare-extreme failure. That diagnosis remains valid for the
audited RetinaMNIST evidence but is not the project-wide current diagnosis.

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

## Historical RetinaMNIST-Specific Synthesis (Phases 3.3–3.4)
The RetinaMNIST-specific diagnosis was:

\[
\boxed{\text{Dual-component rare-extreme failure}}
\]

with two components:
1. **representation collapse** for many rare class-4 samples;
2. **head-level inward bias** for some samples whose representation is already class-4-like.

On RetinaMNIST, RPS remains valuable because its decision-risk signal identifies
difficult samples better than CE, even though it does not localize the rare
endpoint well. This is not a universal cross-dataset conclusion.

## Historical Phase 3.5 Research Question
The completed design audit asked:

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

**Phase 3.8 Solar — STRONG CONFIRMATION.** On 28,006 aligned future-period test examples, including 921 X-class examples, matched CE/RPS seed-0 models made zero exact X decisions under mode/L1/L2. Their X predictive means were 2.864/2.772 (inward shrinkage 1.136/1.228) while class-0 MAE was .104/.082 versus X MAE 1.115/1.197. RPS did not improve solar risk-quality metrics. See [the Phase 3.8 record](phase3_8_solar_rare_extreme_shrinkage_confirmation.md).

## Historical Two-Dataset Synthesis (superseded by Phases 3.16–3.17)

The common evidence across RetinaMNIST and UTKFace is:

\[
\boxed{\text{Rare upper-extreme inward localization bias under ordinal imbalance}}
\]

RetinaMNIST showed severe upper-extreme collapse and a useful RPS risk-quality
advantage. UTKFace showed a milder but clear inward shift, elevated upper-endpoint
risk, and a lower endpoint that was materially easier; RPS partly improved
class-4 localization but was not globally or broadly risk-quality superior.

Its then-current research question was:

> **Why do imbalanced ordinal classifiers exhibit systematic inward
> localization bias for rare upper-extreme classes, even when uncertainty,
> risk quality, and exact-class performance differ across datasets and
> objectives?**

## Development-Benchmark Guardrail
RetinaMNIST has been inspected extensively during method development and remains
a **development benchmark**. Do not use either RetinaMNIST or UTKFace test
diagnostics for iterative method redesign. Phase 3.8 is complete; any new action requires separate authorization.

## Phase 3.10A — Training-Only ROP Falsification

The authorized 5-fold OOF frozen-RPS-head experiment used only the 1,080
canonical training samples and replay-verified 512-D RPS features. ROP was
active and modestly improved pair concordance and portions of the risk-quality
trade-off versus balanced head, especially around λ=.5. It retained the
balanced-head class-4 localization gain versus original RPS but did not improve
localization over balanced head. Every λ violated the predeclared class-0 MAE
safety tolerance versus original RPS.

Decision: **TRADE-OFF**. No lambda is frozen. Do not run historical test,
ROP-v2, multi-seed work, UTKFace, Solar, or another method automatically. See
[the Phase 3.10A record](phase3_10a_retinamnist_rop_objective_falsification.md).

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

## Phase 3.10B — Head-Bias Localization Audit

The training-only OOF B-versus-original audit found that balanced-head class-4
recovery and class-0 damage are explained by large classifier direction
rotations and norm inflation, not meaningful bias shifts. Original weights plus
balanced biases are effectively original behavior; balanced weights plus
original biases retain the trade-off. A norm diagnostic moderates both effects
but removes most exact class-4 recovery.

Decision: **MIXED PARAMETER MECHANISM**. This does not authorize a bias-only,
norm-only, direction-only, or mixed correction. See
[the Phase 3.10B record](phase3_10b_head_bias_localization_audit.md).

## Phase 3.13 — UTKFace Direction/Scale Confirmation

The frozen seed-0 train/validation protocol produced **PARTIAL CONFIRMATION**.
B/C/D all improved class-4 localization relative to original RPS; controlled
scale D was strongest (`C4 MAE .493` versus A `.612`) and its learned
directions were close to B. Large norm inflation and direction rotation
replicated. However, B did not produce RetinaMNIST-like class-0 MAE damage, D
was globally worse than B/A, and balanced bias shifts were not negligible.
Therefore rare-end direction/scale localization transports, while the safety
trade-off and bias contribution are dataset-dependent. The archived UTKFace
test split was not loaded. See [the Phase 3.13 record](phase3_13_utkface_direction_scale_mechanism_confirmation.md).

That Phase 3.13 conclusion was superseded as a next-step decision by Phase
3.14; it did not itself authorize execution of a follow-up experiment.

## Phase 3.14 — Cross-Dataset Mechanism Consolidation

The evidence-only synthesis concludes that rare upper-end localization responds
consistently to direction adaptation on RetinaMNIST and UTKFace, with added
scale strengthening recovery. Controlled scale is retained as a correction-
strength mechanism probe, not a universal classifier or novelty claim.
Opposite-endpoint/global effects and bias behavior differ by dataset; ROP stays
diagnostic/secondary and the combined Retina candidate is not freeze-ready.

Decision: **A — SOLAR DIRECTION/SCALE CONFIRMATION JUSTIFIED**. The only
authorized next scientific question is whether frozen Solar A/B/C/D controls
reproduce the direction/scale decomposition. See
[the Phase 3.14 record](phase3_14_cross_dataset_mechanism_consolidation.md).

## Phase 3.10C — Direction-Only Causal Falsification

With original class-specific norms and biases fixed, balanced direction-only
training retained partial class-4 localization improvement over original RPS
and improved global/class-0 behavior relative to full balanced head. It lost
most exact class-4 recovery. Decision: **PARTIAL SUPPORT**—direction is useful,
but controlled scale is also implicated. No controlled-scale design, test
evaluation, seed expansion, or dataset follow-up is authorized. See
[the Phase 3.10C record](phase3_10c_direction_only_head_falsification.md).

## Phase 3.10D — Controlled-Scale Causal Falsification

The predeclared `.25/.50/.75` fixed interpolated-scale heads provide a useful
training-only OOF region. α=.50 retains 8 exact C4 decisions versus B's 11 and
improves B's C0/global MAE; α=.50/.75 retain six B-exact cases. Decision:
**GO — CONTROLLED SCALE SUPPORTED**. This is mechanism evidence only, with no
authorization for a final/adaptive scale method, test evaluation, seed
expansion, or new datasets. See
[the Phase 3.10D record](phase3_10d_controlled_scale_head_falsification.md).

## Phase 3.10E — Controlled Scale × ROP Interaction

At the fixed D α=.50 point, the pre-existing λ=1.0 ROP hinge improves
teacher/student pair preservation, L1-risk Spearman, and selective MAE while
retaining controlled-scale localization and class-0 safety. AUROC/AUPRC and
probability quality are mixed but not materially degraded. Decision: **GO —
COMPLEMENTARY MECHANISMS SUPPORTED**. This is a training-only interaction
result, not a final method freeze or authorization for tuning, adaptive
mechanisms, historical-test evaluation, seeds, or new datasets. See
[the Phase 3.10E record](phase3_10e_controlled_scale_rop_interaction.md).

## Phase 3.11 — Frozen Candidate One-Shot Validation

The fixed α=.50 / λ=1.0 candidate was fitted on all training data before a
single historical-validation evaluation. Class-4 outward localization
replicated, but class-0/global safety and the D→E Spearman/selective-MAE
effects reversed relative to OOF. Decision: **MIXED — HOLD FROZEN, NO
REDESIGN**. The candidate is not method-frozen; no test evaluation, tuning, or
follow-up is authorized. See [the Phase 3.11 record](phase3_11_frozen_candidate_validation.md).

## Phase 3.12 — Evidence Consolidation and Candidate Disposition

The completed evidence retains direction adaptation, scale amplification, and
controlled scale as RetinaMNIST mechanism findings; stops bias correction;
demotes ROP to a diagnostic/secondary idea; and holds α=.50/λ=1.0 as not
freeze-ready. Decision: **CROSS-DATASET MECHANISM CONFIRMATION JUSTIFIED**.
See [the Phase 3.12 record](phase3_12_evidence_consolidation_and_candidate_disposition.md).

## Phase 3.16 — Three-Dataset Mechanism Consolidation

**A — MECHANISM EVIDENCE SUFFICIENT; MOVE TO PAPER FRAMING.** Direction
adaptation improves rare-end localization across the three tested frozen RPS
representations. Solar C beats D, superseding any general beneficial-scale
interpretation of Phase 3.14. Scale benefits, bias behavior, and opposite-endpoint
effects are dataset-dependent. Higher global L1 MAE occurs under all tested
adapted heads, but QWK and individual risk/probability measures vary.
Localization does not imply global/UQ improvement.

Controlled scale remains a mechanistic/diagnostic probe, bias-only correction
stays stopped, ROP stays secondary, and the combined candidate stays HOLD —
NOT FREEZE-READY. Historical phase conclusions remain preserved. See
[Phase 3.16](phase3_16_three_dataset_mechanism_consolidation.md).

## Phase 3.17 — Paper Framing and Novelty Boundary

**A — PROCEED AS MECHANISM PAPER.** The paper contribution is a bounded
phenomenon-and-mechanism analysis: rare upper-extreme inward localization bias,
frozen-head actionability, and the consistently observed role of classifier
direction across the evaluated settings.
It is not a new balanced-head, scale-control, bias-correction, ROP, or
conformal method. Scale, bias, lower-endpoint, global, and risk/UQ effects must
be written as dataset-dependent or unsupported where appropriate. See
[Phase 3.17](phase3_17_paper_framing_and_novelty_boundary.md).

**Current paper research question:**

> How do rare upper extremes localize in the studied imbalanced ordinal
> settings, and what frozen classifier-head mechanisms explain the recoverable
> portion of that failure?

This wording describes the studied settings; it does not claim that imbalance
was independently manipulated and established as the causal source.

## Phase 3.18A — RetinaMNIST CE Representation Robustness

**A — RETINAMNIST CROSS-OBJECTIVE DIRECTION ROBUSTNESS CONFIRMED.** In the
same training-only OOF protocol, fixed-norm/fixed-bias CE direction-only C
improves C4 MAE (`1.439→1.030`), exact decisions (`0→11`), and shrinkage
(`1.678→1.244`) relative to the original CE head. Exact recovery is
concentrated among CE feature-nearest-4 cases. This was a RetinaMNIST cross-objective robustness result. Solar CE has now been
confirmed separately in Phase 3.18B. See [Phase 3.18A](phase3_18a_retinamnist_ce_direction_robustness.md).

## Phase 3.18B — Solar CE Representation Robustness

**A — TWO-DOMAIN CROSS-OBJECTIVE DIRECTION ROBUSTNESS CONFIRMED.** Frozen CE
direction-only C improves Solar X MAE `1.115→.586`, exact routing `0→559/921`,
and shrinkage `1.136→.656`; the direction agrees with saved Solar RPS. Direct
direction evidence now covers frozen CE and RPS representations in RetinaMNIST
and Solar, without an objective-independent claim. See [Phase 3.18B](phase3_18b_solar_ce_direction_robustness.md).

## Phase 3.19 — RetinaMNIST Sampling vs Objective Disentanglement

**A — BALANCED SAMPLING IS THE PRIMARY DIRECTION-ADAPTATION DRIVER.** On the
frozen RPS representation and exact five-fold training-only OOF protocol,
balanced C/F both substantially improved C4 MAE, shrinkage, `z4-z3` margin,
and outward routing over natural E/G; CE-versus-RPS differences within either
sampling stratum were small. This is a bounded RetinaMNIST/RPS-representation
training-signal result, not a universal causal claim or a selected correction.
See [Phase 3.19](phase3_19_retinamnist_sampling_objective_direction_disentanglement.md).

## Phase 3.20A — RetinaMNIST Controlled Imbalance-Severity Dose Response

**B — PARTIAL / NON-MONOTONIC IMBALANCE EFFECT.** In the predeclared 25-run
full CE grid, reduced C4 support consistently lowers p4 and the C4-vs-C3
margin, but C4 MAE, shrinkage, severe burden, and train-centroid routing are
seed-variable and non-monotonic. This controlled result does not support a
universal dose-response claim. See [Phase 3.20A](phase3_20a_retinamnist_imbalance_severity_dose_response.md).

## Next Authorized Work

Manuscript planning and writing from completed evidence are authorized. The
[paper story architecture](paper_story_architecture.md) is the manuscript-level
planning source of truth. The dedicated [Figure 1--4 source-artifact and data-availability audit](figure_source_artifact_audit.md) is complete with **A — FIGURE SOURCES READY FOR DATA CONSOLIDATION**; its canonical data-table generation then passed all 55 recorded provenance and aggregate checks. Do not generate final figures without separate authorization.

No new method design, training, evaluation, tuning, test access, ROP/bias
revival, seeds, or datasets are authorized automatically.

Do not restart method development, revive RG-ACR, create dataset-specific objectives, perform automatic representation/head audits, launch multi-seed expansions, or add datasets/channels. Any next action requires separate authorization.
