# Experiment Plan

## Current Stage
Completed through **Phase 3.20A RetinaMNIST controlled imbalance-severity dose
response**. Current status: **B — PARTIAL / NON-MONOTONIC IMBALANCE EFFECT**.

Experiment development is paused. Plan and draft the mechanism paper from
completed evidence only. In the three evaluated frozen-RPS settings,
direction-only head adaptation improved rare-end localization; scale, bias,
global, and opposite-endpoint effects varied by dataset. No method is frozen
and no follow-up is authorized automatically. See
[Phase 3.17](phase3_17_paper_framing_and_novelty_boundary.md).

Phase 3.18A and Phase 3.18B confirm the A→C direction effect on frozen CE
features in RetinaMNIST and Solar. Direct mechanism evidence now covers the
tested frozen CE and RPS representations in those two domains; this does not
claim objective independence or authorize any additional execution.

## Phase 3.10A — Completed Training-Only OOF Gate

The authorized frozen-feature head experiment used canonical seed-0 RPS
features, original RPS-head initialization, deterministic training-only
five-fold OOF, balanced CE batches, and independent natural-population ROP
batches at λ `.1/.5/1.0`. No validation/test labels, backbone update,
representation loss, class-4 weighting, or new dataset was used.

Decision: **TRADE-OFF**. ROP was active and modestly improved several
risk/pair-order measures relative to balanced head, but did not improve
class-4 localization over it and violated the predeclared class-0 MAE safety
tolerance relative to original RPS. No lambda is selected and no objective
freeze, historical-test evaluation, ROP-v2, seed expansion, or cross-dataset
run is authorized. See [Phase 3.10A](phase3_10a_retinamnist_rop_objective_falsification.md).

## Phase 3.10B — Completed Mechanism Audit

The exact same training-only OOF folds and frozen features were used to audit
original RPS versus balanced heads. The parameter/logit and diagnostic-swap
evidence identifies a **MIXED PARAMETER MECHANISM**: class-4 recovery and
class-0 damage come primarily from classifier direction and norm changes, not
bias shifts. This is explanatory evidence only. No correction, objective,
lambda, seed expansion, historical-test evaluation, or dataset follow-up is
authorized. See [Phase 3.10B](phase3_10b_head_bias_localization_audit.md).

## Phase 3.10C — Completed Direction-Only Causal Gate

The fixed-original-norm/bias direction-only head used the same frozen training
features and OOF folds. It retains partial C4 gain versus original RPS and
partly restores B's global/class-0 cost, but loses most exact C4 recovery.
Decision: **PARTIAL SUPPORT**. Direction is useful but scale is implicated; do
not automatically implement controlled-scale adaptation or any other follow-up.
See [Phase 3.10C](phase3_10c_direction_only_head_falsification.md).

## Phase 3.10D — Completed Controlled-Scale Causal Gate

The predeclared `.25/.50/.75` fixed interpolated-scale direction heads used
the same frozen OOF protocol. α=.50/.75 retain substantial B C4 recovery while
improving B class-0/global MAE; α=.50 is the stronger recovery/safety point.
Decision: **GO — CONTROLLED SCALE SUPPORTED**, limited to the training-only
mechanism. No final method, adaptive scale, test evaluation, seed expansion,
or cross-dataset run is authorized. See
[Phase 3.10D](phase3_10d_controlled_scale_head_falsification.md).

## Phase 3.10E — Completed Controlled Scale × ROP Interaction

At the single fixed α=.50 / λ=1.0 interaction point, the exact Phase 3.10A
ROP loss was added to the Phase 3.10D controlled-scale direction-only head,
using separate balanced CE and natural ROP batches. ROP improves direct pair
preservation, Spearman, and selective MAE while retaining class-4 localization
and class-0/global safety; AUROC/AUPRC and probability metrics remain mixed.
Decision: **GO — COMPLEMENTARY MECHANISMS SUPPORTED** as training-only
mechanism evidence. No alpha/lambda tuning, adaptive/joint objective, ROP-v2,
test evaluation, seed expansion, or cross-dataset work is authorized. See
[Phase 3.10E](phase3_10e_controlled_scale_rop_interaction.md).

## Phase 3.11 — Completed Frozen Candidate One-Shot Validation

The frozen α=.50 / λ=1.0 candidate was fit on all 1,080 training samples and
all B/D/E states were saved before its one historical-validation evaluation.
Class-4 localization relative to A replicated, but B→E class-0/global safety
and D→E Spearman/selective-MAE directions reversed. Decision: **MIXED — HOLD
FROZEN, NO REDESIGN**. No method freeze, test evaluation, tuning, seed
expansion, or cross-dataset work is authorized. See
[Phase 3.11](phase3_11_frozen_candidate_validation.md).

## Phase 3.12 — Completed Evidence Consolidation

Direction adaptation, scale amplification, and controlled scale are retained as
RetinaMNIST mechanisms; bias correction is stopped; ROP is diagnostic/secondary
only; and α=.50/λ=1.0 is not freeze-ready. Decision: **CROSS-DATASET
MECHANISM CONFIRMATION JUSTIFIED**. The next frozen question is UTKFace
original/balanced/direction-only/controlled-scale confirmation, excluding ROP,
with no grid, test evaluation, or new method. See
[Phase 3.12](phase3_12_evidence_consolidation_and_candidate_disposition.md).

## Canonical Development Setup
Primary development dataset: **RetinaMNIST**

- native 28×28 RGB
- official train/validation/test splits
- class order `0,1,2,3,4`
- unpretrained small-image ResNet18
- 3×3 stride-1 stem, no max-pool
- canonical seeds `0,1,2,3,4`

Class counts:

| Split | C0 | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|---:|
| Train | 486 | 128 | 206 | 194 | 66 |
| Validation | 54 | 12 | 28 | 20 | 6 |
| Test | 174 | 46 | 92 | 68 | 20 |

Class 4 is the rare upper extreme.

## Historical Experiment Evaluation Decomposition
Completed mechanism experiments separated:

\[
\text{Representation}
\rightarrow
\text{Probabilistic head}
\rightarrow
\text{Decision rule}
\rightarrow
\text{Expected decision risk}.
\]

Primary decision control remains **CE + L1**; RPS remains a core probabilistic
ordinal baseline, with its risk-quality advantage established on RetinaMNIST
but not broadly replicated on UTKFace.

## Completed Evidence
### Phase 1–2.5
- CE native-28 baseline complete.
- Simple new uncertainty-metric branch stopped.
- L1 decision correction is required as a control.
- CORAL stopped.
- RPS retained: mixed predictive result but stronger risk alignment/severe detection/selective prediction.
- Temperature scaling does not explain away the RPS advantage.

### Phase 3.0–3.1
- Rare class 4 shows strong inward shrinkage and high-risk but poor localization.
- Weighted CE and SLACE do not solve the failure.

### Phase 3.2 — Output-only candidates
Candidate 1 moved class-4 decisions from class 2 toward class 3 but introduced global/risk trade-offs and no exact recovery.

Candidate 1b added fixed true-endpoint preference but did not improve p4 or exact recovery and weakened adjacent recovery.

Decision:

\[
\boxed{\text{OUTPUT-ONLY CANDIDATE BRANCH — STOP}}
\]

No Candidate 1c. No Candidate 1/1b multi-seed expansion.

### Phase 3.3 — Representation Audit
Frozen CE/RPS seed-0 features showed:
- representation collapse for many class-4 examples;
- worse class3/class4 separation for RPS than CE;
- but some class-4 samples are correctly nearest to the class-4 centroid and still fail at the head.

Decision:

\[
\boxed{\text{MIXED REPRESENTATION / HEAD FAILURE}}
\]

### Phase 3.4 — Head Intervention Audit
Eight simple linear-head conditions on frozen CE/RPS features showed:
- some feature-nearest-to-4 cases are head-recoverable;
- CE features + logit adjustment recovered 3/9 such cases exactly;
- most feature-nearest-central class-4 cases remained unrecovered: 10/11 CE and 11/14 RPS;
- global/risk-quality trade-offs remain.

Decision:

\[
\boxed{\text{MIXED BUT DECOMPOSABLE FAILURE}}
\]

## Historical RetinaMNIST Diagnosis (Phases 3.3–3.4; dataset-specific)

\[
\boxed{\text{Dual-component rare-extreme failure}}
\]

- representation collapse affects many rare extreme samples;
- head/prior bias affects a recoverable subset;
- neither head-only nor representation-only explanations are sufficient.

## Phase 3.5 — Historical Design Audit (complete; branch closed)
### Historical Research Question
> Can ordinal decision risk guide representation learning so that rare, high-risk extreme samples become better localized without destroying probabilistic risk quality?

The literature audit compared exactly three mechanisms. It selected risk-gated adjacent-centroid ranking (RG-ACR) as one empirically unvalidated primary candidate, retained collapse-aware adaptive adjacent margin as backup, and deferred risk-weighted prototype compactness because of high overlap/risk. RG-ACR uses RPS plus detached L1 Bayes-risk-weighted local adjacent-centroid ranking. Full mathematical specification, literature overlap, and safeguards are in [the Phase 3.5 note](phase3_5_risk_conditioned_representation_design.md).

### Design Requirements
A candidate should:
- preserve RPS or another strong probabilistic ordinal base objective;
- directly target representation geometry;
- use ordinal structure explicitly;
- use established decision-risk quantities rather than inventing a new UQ metric;
- avoid class-4 hard-coding;
- use training-only class counts/statistics if imbalance enters;
- require at most 1–2 new hyperparameters;
- be testable with one seed-0 falsification experiment.

### Literature Guardrail
Reject/high-risk any design that reduces to standard SupCon, balanced SupCon, generic center/prototype loss, generic hard-example mining, simple distance-weighted SupCon, or a known contrastive + logit-adjustment formulation.

No implementation/training occurred during the design audit.

## Historical Final RetinaMNIST Seed-0 Gate (closed)
The one authorized final major RetinaMNIST seed-0 method-selection experiment was RG-ACR. The validation-selected λ=.05 condition failed the representation gate and violated predeclared class-0/risk tolerances. Decision: **NO-GO**. See [Phase 3.6](phase3_6_rg_acr_seed0.md).

A promising method must improve representation-specific quantities such as:
- class-4 nearest-centroid assignment;
- class-4 vs class-3/class-2 margins;
- class3/class4 separation;

and downstream quantities such as:
- p4 / p3+p4;
- predictive mean / inward shrinkage;
- 4→3/4 routing;
- class-4 severe burden;

while preserving acceptable:
- global Accuracy/MAE/QWK;
- NLL/Brier/RPS;
- L1-risk Spearman;
- severe AUROC/AUPRC;
- selective MAE;
- class-0 behavior.

## Historical Method Freeze Rule (closed)
If the final seed-0 candidate is promising:

\[
\boxed{\text{METHOD FREEZE}}
\]

Then:
- no more RetinaMNIST-test-driven objective redesign;
- run seeds 1–4;
- aggregate paired multi-seed results;
- move to new datasets for confirmatory evidence.

If the candidate fails representation geometry:

\[
\boxed{\text{NO-GO}}
\]

Do not iterate repeatedly on RetinaMNIST test results.

## Development-Benchmark Guardrail
RetinaMNIST is now a **development benchmark** because the test set has been
inspected extensively during method development. No additional
RetinaMNIST-test-informed method experiment is authorized. Phase 3.8 is complete and any next action requires separate authorization.

## Phase 3.7A — Independent Replication Gate

UTKFace was evaluated only as a frozen-manifest seed-0 CE/RPS baseline
replication. The age bins were `<20`, `20–39`, `40–59`, `60–79`, and `>=80`;
the oldest class has 67 test examples.

Decision:

\[
\boxed{\text{PARTIAL REPLICATION}}
\]

The upper endpoint remained high-risk and inward-shrunk, and RPS partly
improved its exact recovery. However, the RetinaMNIST RPS advantage in matched
L1 risk/error association, AUPRC, and ordinal selective MAE did not reproduce;
RPS only improved severe AUROC and had worse global/lower-endpoint results.
Therefore no UTKFace representation/head audit, bin redesign, multi-seed run,
or method iteration is authorized automatically. See
[the Phase 3.7A record](phase3_7a_utkface_failure_replication.md).

**Phase 3.8 Solar is complete.** Its valid Phase 3.7A preparation artifacts were preserved; the resumed Phase 3.8 CE/RPS jobs used the frozen three-channel contract and yielded a STRONG CONFIRMATION, not a method-development signal.

## Phase 3.8 — Solar Rare-Extreme Shrinkage Confirmation

### Primary question

> Does rare upper-extreme inward localization bias observed on RetinaMNIST and
> UTKFace also appear in ordinal solar-flare classification?

Use matched CE and RPS controls. The primary cross-dataset phenomena are:

1. the upper extreme has elevated decision risk;
2. the upper extreme is inward-shrunk;
3. the upper endpoint is harder than the lower endpoint;
4. L1/L2 decision correction is insufficient to remove the bias; and
5. localization bias persists across CE/RPS objectives.

Solar produced a STRONG CONFIRMATION: both CE and RPS had zero exact X decisions under mode/L1/L2, X predictive means 2.864/2.772, and much worse X than class-0 MAE/severe burden. RPS did not improve solar risk quality. The third-dataset gate is complete; no new objective, representation/head audit, seed expansion, or dataset is authorized without a separate decision.

## Archived Future-Expansion Ideas (not active)
The following were historical possibilities and are not authorized by Phase 3.17:
- any multi-seed confirmation on RetinaMNIST or UTKFace;
- additional ordinal datasets beyond solar;
- method design, if the cross-dataset phenomenon is sufficiently stable;
- later epistemic-UQ baselines, if the single-model question warrants them.

## Archived Phase 4 — Epistemic UQ Idea (not active)
MC Dropout / Deep Ensemble is not authorized by Phase 3.17.

## Phase 3.13 — UTKFace Direction/Scale Confirmation (complete)

The frozen seed-0 A/B/C/D training/validation experiment concluded **PARTIAL
CONFIRMATION**. Direction adaptation and scale amplification improved rare
class-4 localization across datasets, with α=.50 D strongest on UTKFace, but
the RetinaMNIST lower-endpoint/global safety ordering did not replicate. No ROP,
alpha search, backbone retraining, or test access occurred. See
[the Phase 3.13 record](phase3_13_utkface_direction_scale_mechanism_confirmation.md).

## Phase 3.14 — Cross-Dataset Mechanism Consolidation (complete)

Direction adaptation is cross-dataset supported and additional scale can
strengthen rare-end outward movement; global/opposite-endpoint safety and bias
effects are dataset-dependent. Decision: **A — SOLAR DIRECTION/SCALE
CONFIRMATION JUSTIFIED**. See
[the Phase 3.14 record](phase3_14_cross_dataset_mechanism_consolidation.md).

## Phase 3.15 — Solar Confirmation (complete)

PARTIAL CONFIRMATION: direction-only C recovers 496/921 X decisions versus
A's zero; fixed .50 D recovers 262 and worsens rare MAE relative to C.
Global MAE rises under every adapted head. The direction effect transports;
the beneficial-scale ordering does not. See
[Phase 3.15](phase3_15_solar_direction_scale_mechanism_confirmation.md).

## Phase 3.16 — Three-Dataset Consolidation (complete)

**A — MECHANISM EVIDENCE SUFFICIENT; MOVE TO PAPER FRAMING.** The current
claim is direction responsiveness in tested frozen RPS representations, with
dataset-dependent scale benefits, bias behavior, and collateral effects.
Earlier two-dataset scale claims are historical, superseded by this synthesis.
Controlled scale is diagnostic; ROP is secondary; bias correction stays stopped;
the combined candidate is not freeze-ready. See
[Phase 3.16](phase3_16_three_dataset_mechanism_consolidation.md).

## Phase 3.17 — Paper Framing and Novelty Boundary (complete)

**A — PROCEED AS MECHANISM PAPER.** The final framing is rare upper-extreme
inward localization bias under ordinal imbalance and its bounded frozen-head
direction mechanism. The paper is an empirical mechanism analysis, not a new
generic balancing, scale, bias, ROP, or conformal method. See
[Phase 3.17](phase3_17_paper_framing_and_novelty_boundary.md).

## Phase 3.18A — RetinaMNIST CE Representation Robustness (complete)

**A — RETINAMNIST CROSS-OBJECTIVE DIRECTION ROBUSTNESS CONFIRMED.** With
frozen CE features, direction-only C improves C4 MAE `1.439→1.030`, exact
routing `0→11`, and shrinkage `1.678→1.244` over A in five-fold training-only
OOF. At Phase 3.18A completion, it supported a separately authorized Solar CE
confirmation; Phase 3.18B subsequently completed that confirmation. See
[Phase 3.18A](phase3_18a_retinamnist_ce_direction_robustness.md).

## Phase 3.18B — Solar CE Representation Robustness (complete)

**A — TWO-DOMAIN CROSS-OBJECTIVE DIRECTION ROBUSTNESS CONFIRMED.** Under the
predeclared frozen Solar CE A/B/C protocol, C improves X MAE `1.115→.586`,
exact routing `0→559/921`, and shrinkage `1.136→.656` versus original CE. The
Solar CE response agrees with saved Solar RPS; combined with Phase 3.18A,
direction responsiveness is now directly observed on CE and RPS frozen
representations in RetinaMNIST and Solar. Global and risk/UQ effects remain
adverse or mixed. See [Phase 3.18B](phase3_18b_solar_ce_direction_robustness.md).

## Phase 3.19 — RetinaMNIST Sampling vs Objective Disentanglement (complete)

**A — BALANCED SAMPLING IS THE PRIMARY DIRECTION-ADAPTATION DRIVER.** In the
fixed-norm/fixed-bias frozen-RPS 2×2 training-only OOF grid, C/F balanced
cells improve C4 MAE/shrinkage and flip the mean `z4-z3` margin positive,
whereas natural E/G cells do not. Objective contrasts are small within either
sampling stratum. See [Phase 3.19](phase3_19_retinamnist_sampling_objective_direction_disentanglement.md).

## Phase 3.20A — RetinaMNIST Imbalance-Severity Dose Response (complete)

**B — PARTIAL / NON-MONOTONIC IMBALANCE EFFECT.** The 25 predeclared full CE
runs show consistent p4 and C4-vs-C3 margin deterioration as C4 support falls,
but no fully seed-consistent monotonic MAE/shrinkage/geometry response. The
mechanism paper must retain this boundary. See
[Phase 3.20A](phase3_20a_retinamnist_imbalance_severity_dose_response.md).

## Immediate Next Action

Perform the dedicated Figure 1--4 source-artifact and data-availability audit
specified by the [paper story architecture](paper_story_architecture.md), then
construct reproducible figure-data tables and plotting scripts. This audit
must verify source artifacts, populations, split roles, objectives, seeds,
checkpoints, and cross-phase sample-ID alignment; it must flag unsupported
panels rather than infer values. Do not generate final figures before it.

No method design, training/evaluation, tuning, new dataset, seed expansion, or
test access is authorized. The manuscript must separate the consistently observed
cross-dataset direction effect from
dataset-specific scale, bias, global, and opposite-endpoint outcomes. It may state that direction responsiveness was reproduced on both CE- and
RPS-trained frozen representations in RetinaMNIST and Solar. The three-dataset
mechanism claim remains frozen-RPS scoped because UTKFace CE was not tested. No candidate is
method-frozen.
