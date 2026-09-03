# Experiment Plan

## Current Stage
Completed through **Phase 3.7A-UTKFace CE/RPS seed-0 replication**. Active
stage: **Phase 3.8 Solar Rare-Extreme Shrinkage Confirmation**.

Current gate:

\[
\boxed{\text{Solar confirmation — next authorized cross-dataset gate}}
\]

RG-ACR was implemented and tested only under its predeclared seed-0 protocol,
then stopped. The independent UTKFace CE/RPS seed-0 gate found a partial, not
strong, replication. No method is frozen.

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

## Current Evaluation Decomposition
Future experiments must separate:

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

## Current Diagnosis

\[
\boxed{\text{Dual-component rare-extreme failure}}
\]

- representation collapse affects many rare extreme samples;
- head/prior bias affects a recoverable subset;
- neither head-only nor representation-only explanations are sufficient.

## Phase 3.5 — Design Audit (complete)
### Research Question
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

## Final RetinaMNIST Seed-0 Gate
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

## Method Freeze Rule
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
RetinaMNIST-test-informed method experiment is authorized while Phase 3.8
confirms the cross-dataset failure pattern.

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

**Phase 3.7A-Solar remains recorded as paused before training.** Its
implementation artifacts must be preserved, and no scientific solar outcome
may be assigned. Phase 3.8 has not yet started.

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

Whether RPS improves risk quality or extreme-class localization is secondary;
it is not required for replication. Do not restart method development or tune a
new method from the solar result. Review third-dataset evidence before any new
objective, representation/head audit, or multi-seed expansion is authorized.

## Future Expansion
After Phase 3.8 evidence is reviewed and only with separate authorization:
- any multi-seed confirmation on RetinaMNIST or UTKFace;
- additional ordinal datasets beyond solar;
- method design, if the cross-dataset phenomenon is sufficiently stable;
- later epistemic-UQ baselines, if the single-model question warrants them.

## Phase 4 — Epistemic UQ
Not active. MC Dropout / Deep Ensemble should be considered only after the single-model method is established.

## Immediate Next Action
Run only the predeclared Phase 3.8 solar CE/RPS confirmation protocol. Do not
create RG-ACR-v2, run seeds 1–4, perform an automated UTKFace follow-up, or
restart method development. Any new method family or expansion requires a
separate authorization after the solar evidence is reviewed.
