# Experiment Plan

## Current Stage
Completed through **Phase 3.4**.

Current gate:

\[
\boxed{\text{PHASE 3.5 — METHOD-DESIGN AUDIT NEXT}}
\]

No new proposed method is frozen or implemented.

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

Primary decision control remains **CE + L1**; RPS remains the strongest probabilistic ordinal risk-quality baseline.

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

## Phase 3.5 — Design Audit
### Research Question
> Can ordinal decision risk guide representation learning so that rare, high-risk extreme samples become better localized without destroying probabilistic risk quality?

### Candidate Limit
Compare exactly three candidates at most:
1. risk-conditioned ordinal separation;
2. collapse-aware adjacent margin;
3. risk-weighted prototype/compactness regularization.

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

### Phase 3.5 Deliverable
The design audit should produce:
- literature-overlap table;
- exactly three candidates;
- one primary candidate;
- one backup;
- one defer;
- exact mathematical primary objective;
- gradient-detachment policy for risk weighting;
- minimal hyperparameter policy;
- predeclared seed-0 falsification criteria;
- method-freeze rule.

No implementation/training during the design audit.

## Final RetinaMNIST Seed-0 Gate
After Phase 3.5 review, authorize at most one primary method for a final major RetinaMNIST seed-0 method-selection experiment.

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
RetinaMNIST is now a **development benchmark** because the test set has been inspected extensively during method development. The next seed-0 method experiment should be the last major method-selection step using its test diagnostics.

## Future Confirmatory Expansion
Only after method freeze:
- RetinaMNIST seeds 1–4;
- UTKFace;
- solar flare ordinal classification;
- Amazon Reviews or another appropriate ordinal benchmark;
- additional literature-supported datasets as needed.

## Phase 4 — Epistemic UQ
Not active. MC Dropout / Deep Ensemble should be considered only after the single-model method is established.

## Immediate Next Action
Run the **Phase 3.5 Risk-Conditioned Ordinal Representation Method Design Audit**. Do not implement or train during this task.
