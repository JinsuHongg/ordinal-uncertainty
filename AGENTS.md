# AGENTS.md

## Project

Repository: `ordinal-uncertainty`

Working title: **Ordinal Uncertainty Quantification for Imbalanced Ordinal Classification**

This is the active ML/UQ workspace. The previous `ordinal-aware-conformal`
repository is a separate research archive.

---

## Current Stage

**Phase 3.20A RetinaMNIST Controlled Imbalance-Severity Dose Response complete
— B — PARTIAL / NON-MONOTONIC IMBALANCE EFFECT.**

Completed:

- Experiment 0 / Phase 1
- Phase 1.5 ordinal-UQ baseline audit
- RetinaMNIST resolution sanity check
- Phase 1.75 decision-rule audit
- Phase 2 ordinal-learning baseline study
- Phase 2.5 calibration-control audit
- Phase 3.0 extreme-class failure audit
- Phase 3.1 imbalance-aware baseline audit
- Phase 3.2 output-level Candidate 1 / 1b diagnostics
- Phase 3.3 frozen representation audit
- Phase 3.4 frozen-feature head intervention audit
- Phase 3.5 risk-conditioned representation method-design audit
- Phase 3.6 RG-ACR seed-0 falsification
- Phase 3.7A-UTKFace ordinal-failure replication
- Phase 3.10A RetinaMNIST training-only ROP objective falsification
- Phase 3.10B RetinaMNIST training-only head-bias localization audit
- Phase 3.10C RetinaMNIST direction-only head falsification
- Phase 3.10D RetinaMNIST controlled-scale head falsification
- Phase 3.10E RetinaMNIST controlled-scale × ROP interaction
- Phase 3.11 RetinaMNIST frozen candidate one-shot validation
- Phase 3.12 evidence consolidation and candidate disposition
- Phase 3.13 UTKFace direction/scale mechanism confirmation
- Phase 3.14 cross-dataset mechanism consolidation
- Phase 3.15 Solar direction/scale mechanism confirmation
- Phase 3.16 three-dataset mechanism consolidation
- Phase 3.17 paper framing and novelty boundary
- Phase 3.18A RetinaMNIST CE representation robustness
- Phase 3.18B Solar CE representation robustness
- Phase 3.19 RetinaMNIST sampling vs objective disentanglement
- Phase 3.20A RetinaMNIST controlled imbalance-severity dose response

Canonical RetinaMNIST uses:

- native **28×28 RGB**
- official train/validation/test splits
- unpretrained small-image ResNet18
- 3×3 stride-1 stem
- no max-pool

Historical 64×64 results are sensitivity evidence only.

---

## Established Baselines and Decisions

- **CE:** canonical nominal probabilistic baseline.
- **RPS:** **RETAIN** as a core probabilistic ordinal baseline. It improved
  several risk-quality measures on RetinaMNIST and partly improved
  upper-extreme localization on UTKFace, but its broad risk-quality advantage
  did not replicate on UTKFace.
- **CORAL:** **STOP — SCIENTIFICALLY NONCOMPETITIVE**.
- **Weighted CE:** **STOP — SCIENTIFICALLY NONCOMPETITIVE**.
- **SLACE:** **STOP — SCIENTIFICALLY NONCOMPETITIVE**.
- **Candidate 1 / 1b output-only branch:** **STOP**.
- **Simple-new-uncertainty-metric branch:** **STOP**.

Do not silently revive a stopped branch without an explicit new scientific
rationale and user authorization.

### Cross-dataset replication status

UTKFace seed 0 was a **PARTIAL REPLICATION** only. The rare oldest age bin is
high-risk and inward-shrunk, and the lower endpoint is materially easier. RPS
partly improved upper-extreme localization, but did not reproduce the broad
RetinaMNIST matched-L1 risk-quality advantage: it improved severe AUROC only,
while CE had stronger L1 Spearman, AUPRC, selective MAE, and global/
lower-endpoint quality. Do not infer a universal RPS advantage.

**Phase 3.8 Solar is complete — STRONG CONFIRMATION.** Matched CE/RPS seed-0 controls showed zero exact X-class decisions under mode/L1/L2, pronounced inward shrinkage, and severe endpoint asymmetry. **Phase 3.15 then partially confirmed the frozen head mechanism:** direction-only adaptation recovered rare X strongly, but fixed `.50` controlled scale did not improve on direction-only. No automatic follow-up experiment is authorized.

---

## Current Project-Level Evidence

The project-level observed phenomenon is:

\[
\boxed{\text{Rare upper-extreme inward localization bias under ordinal imbalance}}
\]

In the studied imbalanced ordinal settings, rare upper extremes are
systematically localized inward. This is an observed pattern, not a controlled
causal isolation of class imbalance from dataset structure, class difficulty,
representation quality, or label ambiguity.

Across the tested frozen RPS representations on RetinaMNIST, UTKFace, and
Solar, direction adaptation consistently improves rare-end localization.
Phase 3.18A/3.18B additionally reproduce the A→C response on frozen CE
features in RetinaMNIST and Solar. Thus, in the two directly cross-objective
domains, direction responsiveness is observed on both CE- and RPS-trained
frozen representations. Scale, bias, global, and opposite-endpoint effects are
dataset-dependent.

RetinaMNIST additionally supports the following dataset-specific diagnosis:

\[
\boxed{\text{Dual-component rare-extreme failure}}
\]

### 1. Representation collapse

Many true class-4 samples are closer to class 2/3 than class 4 in frozen feature
space.

### 2. Head-level inward bias

Some samples already feature-nearest to class 4 are still mapped centrally by
the classifier/output pipeline.

Phase 3.4 showed that simple balanced/prior-adjusted heads can recover some
head-recoverable samples, but they cannot recover most representation-collapsed
samples and introduce global/risk-quality trade-offs.

Therefore:

\[
\boxed{\text{Head-only solution: insufficient}}
\]

and:

\[
\boxed{\text{Representation-only explanation: insufficient}}
\]

The RetinaMNIST-specific interpretation is a **mixed but decomposable failure**.

UTKFace confirms elevated risk, inward location bias, and endpoint asymmetry,
but not a broad RPS advantage. RPS-specific superiority is therefore not the
project-level cross-dataset claim.

---

## Core Research Question

> **How do rare upper extremes localize in the studied imbalanced ordinal
> settings, and what frozen classifier-head mechanisms explain the recoverable
> portion of that failure?**

This wording describes the studied imbalanced ordinal settings; it does not
claim that imbalance was independently isolated as the cause.

Phase 3.8 strongly confirmed this signal in ordinal solar-flare classification. CE and RPS were matched controls; neither result establishes universal RPS superiority.

---

## Phase 3.5 Design-Audit Decision

The Phase 3.6 RG-ACR seed-0 falsification is complete: **NO-GO**. The
validation-selected λ=.05 model lacked clear cross-geometry class-4
representation improvement and violated class-0/risk tolerances. RG-ACR is
stopped; see docs/research/phase3_6_rg_acr_seed0.md.

Required behavior:

RG-ACR branch guardrails:

1. Do not create RG-ACR-v2 or tune it from observed test outcomes.
2. Do not run RG-ACR seeds 1–4 or additional datasets.
3. Preserve the valid negative result and its artifacts.

The Phase 3.5 design note remains the rationale for the now-completed test; it
does not authorize a variant.

---

## Development-Benchmark Guardrail

RetinaMNIST is the current **method-development benchmark** and has been
inspected extensively.

Detailed RetinaMNIST development rules are centralized in:

`docs/research/retinamnist_method_development_policy.md`

That policy is authoritative for:

- frozen canonical RetinaMNIST/RPS setup;
- training-only OOF development;
- historical validation/test restrictions;
- common evaluation metrics;
- artifact requirements;
- method-freeze rules;
- confirmatory-dataset progression.

Do not duplicate those rules in phase prompts unless a phase explicitly
overrides them.

The historical RetinaMNIST test set must not be used for iterative method
selection. Test access requires an explicit method-freeze/final-evaluation
authorization.

---

## Shared Development Policies

Phase-specific prompts should contain only the scientific delta whenever a
shared policy already defines the common protocol.

Current shared policy:

- `docs/research/retinamnist_method_development_policy.md`

A phase prompt should normally specify only:

1. scientific question;
2. prior evidence;
3. new intervention;
4. comparison conditions;
5. phase-specific diagnostics;
6. decision gate;
7. stop condition.

Dataset setup, OOF rules, common metrics, repository safety, verification, and
documentation synchronization should be inherited from the shared policy unless
explicitly overridden.

## Core Evaluation Decomposition

Keep these components separate:

\[
\text{Representation}
\rightarrow
\text{Probabilistic head}
\rightarrow
\text{Decision rule}
\rightarrow
\text{Expected decision risk}.
\]

Do not attribute an improvement to representation learning if the head changed
simultaneously without a control.

Do not attribute an improvement to a head intervention if the representation was
also retrained without a matched comparison.

### Decision rules

Evaluate:

- mode;
- exact discrete L1-optimal decision;
- exact discrete L2-optimal decision.

The established L1 decision control remains mandatory.

### Global metrics

Report:

- Accuracy
- MAE
- QWK
- severe-error prevalence
- NLL
- Brier score
- RPS
- ECE

### Rare-extreme diagnostics

For true class 4, report:

- `4→0`
- `4→1`
- `4→2`
- `4→3`
- `4→4`
- mean/median `p4`
- mean/median `p3`
- mean/median `p3+p4`
- predictive mean
- inward shrinkage
- L1 Bayes risk
- class-4 MAE
- class-4 severe prevalence

### Class-0 control

For true class 0, report:

- routing
- `p0`
- `p1`
- `p0+p1`
- predictive mean
- inward shrinkage
- MAE
- severe prevalence

### Representation diagnostics

When relevant, report:

- train-derived class centroids;
- class-4 nearest-centroid routing;
- class-4 vs class-3/class-2 distance margins;
- class-3/class-4 centroid separation;
- within-class dispersion;
- raw and normalized feature geometry where appropriate.

### Risk-quality metrics

Using established ordinal decision risk, report:

- risk/error Spearman;
- severe-error AUROC;
- severe-error AUPRC;
- ordinal-MAE risk-coverage;
- mean selective MAE.

Do not invent a new UQ metric unless a separate task explicitly justifies it.

---

## Experimental Discipline

1. Ask one narrow scientific question per experiment.
2. Freeze data split, preprocessing, validation role, and test role before
   evaluation.
3. Use validation-only checkpoint selection and calibration fitting.
4. Save complete configuration and random seed.
5. Save training and validation histories for trained models.
6. Save selected checkpoint metadata.
7. Save sample-level logits/probabilities, labels, sample IDs, decisions, risks,
   and diagnostics.
8. Use multiple seeds only after a valid seed-0 signal and explicit approval.
9. Preserve valid negative results.
10. Distinguish development evidence from confirmatory evidence.
11. Do not broaden an experiment automatically because an intermediate result is
    interesting.
12. Stop when the predeclared GO/NO-GO question has been answered.

---

## Guardrails

Do not currently:

- create Candidate 1c;
- restart the output-only correction branch;
- run Candidate 1/1b seeds 1–4;
- restart CORAL, Weighted CE, or SLACE multi-seed branches;
- return to 64×64 RetinaMNIST as the canonical setup;
- add a generic contrastive/prototype method and call it novel;
- treat logit adjustment as the primary novelty;
- use historical RetinaMNIST test data for iterative method development;
- launch all datasets before method freeze;
- add ensembles, Bayesian methods, or conformal prediction as a substitute for
  the current single-model question;
- claim universal superiority of ordinal UQ;
- claim novelty before literature verification and empirical support.

---

## Phase 3.17 Decision

**A — PROCEED AS MECHANISM PAPER.**

Phenomenon: rare upper-extreme inward localization bias under ordinal imbalance.
Across the tested frozen RPS representations, direction-only adaptation improves
rare-end localization on RetinaMNIST, UTKFace, and Solar. Scale benefits, bias
behavior, and global/opposite-endpoint effects depend on the dataset. Solar C
outperforms D, superseding any universal scale-benefit interpretation.

The paper's bounded contribution is the phenomenon and frozen-head mechanism,
not a universal classifier correction. Controlled scale remains a
diagnostic/mechanistic probe only. Bias-only correction stays stopped; ROP is
diagnostic/secondary; the combined candidate is HOLD — NOT FREEZE-READY.
Localization does not imply global or UQ gains. See
docs/research/phase3_17_paper_framing_and_novelty_boundary.md.

## Phase 3.18A Decision

**A — RETINAMNIST CROSS-OBJECTIVE DIRECTION ROBUSTNESS CONFIRMED.** Frozen CE
direction-only C improves C4 localization over the original CE head in the
same training-only OOF protocol. This permits a separately authorized Solar CE
robustness check, but does not authorize it, a method redesign, or an
objective-wide claim. See
docs/research/phase3_18a_retinamnist_ce_direction_robustness.md.

## Phase 3.18B Decision

**A — TWO-DOMAIN CROSS-OBJECTIVE DIRECTION ROBUSTNESS CONFIRMED.** Frozen Solar CE direction-only C improves X MAE 1.115→.586, exact X routing 0→559/921, and shrinkage 1.136→.656 on the one predeclared archived readout. Direction responsiveness is therefore directly observed on both CE- and RPS-trained frozen representations in RetinaMNIST and Solar. This is not an objective-independent or universal claim. See docs/research/phase3_18b_solar_ce_direction_robustness.md.

## Phase 3.19 Decision

**A — BALANCED SAMPLING IS THE PRIMARY DIRECTION-ADAPTATION DRIVER.** On the
frozen RetinaMNIST RPS representation, balanced C/F cells improve C4 direction
margins and localization relative to natural E/G under both CE and RPS;
within-stratum objective differences are small. This is a single-dataset,
training-only factorial result, not a universal causal statement, selected
method, or authorization for a follow-up run. See
`docs/research/phase3_19_retinamnist_sampling_objective_direction_disentanglement.md`.

## Phase 3.20A Decision

**B — PARTIAL / NON-MONOTONIC IMBALANCE EFFECT.** In the fixed 25-run CE
grid, lower C4 support consistently decreases p4 and the C4-vs-C3 head margin,
but C4 MAE, shrinkage, severe burden, and centroid routing are seed-variable
and non-monotonic. This does not justify a general controlled dose-response
claim. See `docs/research/phase3_20a_retinamnist_imbalance_severity_dose_response.md`.

## Next Authorized Work

The next stage is manuscript planning and writing from completed evidence. No
method design, training, evaluation, tuning, ROP/bias revival, extra seed, or
dataset run is authorized automatically. The framing does not
authorize method freeze or final test evaluation.

All common RetinaMNIST development rules are inherited from:

`docs/research/retinamnist_method_development_policy.md`

unless a phase-specific task explicitly overrides them.

Phase 3.20A does **not** authorize:

- ROP, alpha/lambda tuning, ROP-v2, adaptive/joint objectives, or further test evaluation;
- adaptive or class-specific norm scaling;
- representation retraining;
- historical validation/test evaluation;
- seeds 1–4;
- UTKFace CE, other datasets, or any further Solar execution;
- commit or push.

Do not proceed from this mechanism result without separate authorization.

---

## Research Documentation Policy

Research documentation is part of the experiment, not optional cleanup.

**A research task is not considered complete until its scientific result,
decision, and next-step implications are reflected in the relevant ecosystem
documentation.**

For every completed research phase, audit, candidate experiment, or major
baseline study, create or update a dedicated phase-specific note under:

`docs/research/`

The phase-specific note is the detailed scientific source of truth for that task.

Examples:

- `phase3_3_representation_failure_audit.md`
- `phase3_4_frozen_head_intervention_audit.md`
- `phase3_5_risk_conditioned_representation_design.md`

Do not replace detailed phase notes with only a short ecosystem summary.

---

## Documentation Synchronization Policy

After every completed research phase, audit, baseline study, candidate
experiment, or method-design task, review the repository documentation and keep
the ecosystem internally consistent.

At minimum, review:

- `docs/research/current_state.md`
- `docs/research/decision_log.md`
- `docs/research/experiment_plan.md`
- `AGENTS.md`
- `README.md`

Update only files affected by the task, but do not leave:

- stale phase numbers;
- outdated current-stage text;
- obsolete next-step instructions;
- invalidated candidate plans;
- contradictory method-status statements;
- stale guardrails.

### `current_state.md`

- Record the completed phase and scientific conclusion.
- Update the active stage.
- Update the active research question.
- Update the current diagnosis when changed.
- Remove or replace stale `Next Authorized Work` instructions.
- Link the relevant phase-specific note.

### `decision_log.md`

Record every important:

- GO
- NO-GO
- STOP
- RETAIN
- TRADE-OFF
- method-freeze

decision.

Preserve negative results rather than deleting or rewriting history.

### `experiment_plan.md`

- Mark completed phases as complete.
- Update the active phase.
- Update the next experiment.
- Update evaluation criteria and guardrails.
- Remove or clearly archive plans that were executed or invalidated.
- Do not leave an older phase labeled as the current stage.

### `AGENTS.md`

Keep synchronized:

- current stage;
- active research question;
- established decisions;
- guardrails;
- next authorized work;
- development-vs-confirmatory policy.

`AGENTS.md` must never describe an earlier phase as the current phase.

### `README.md`

Update `README.md` only when the high-level public-facing state changes, such as:

- project stage;
- main research question;
- canonical setup;
- major method status;
- overall project direction.

Do not copy detailed experiment logs into the README.

---

## Documentation Consistency Check

Before finishing any research task:

1. Review the phase-specific note.
2. Review `decision_log.md`.
3. Review `current_state.md`.
4. Review `experiment_plan.md`.
5. Review `AGENTS.md`.
6. Review `README.md` for high-level consistency.
7. Fix stale phase numbers, current-stage statements, method statuses,
   guardrails, and next-step instructions affected by the task.
8. Report which ecosystem files were updated.

If a task changes the scientific state of the project but none of the ecosystem
documents require modification, explicitly state why.

---

## Research Documentation Hierarchy

When documents disagree, use the following priority:

1. Latest **completed phase-specific research note relevant to the disputed
   scientific state**
2. `docs/research/decision_log.md`
3. `docs/research/current_state.md`
4. `docs/research/experiment_plan.md`
5. `AGENTS.md`
6. `README.md`

Only completed and relevant phase-specific notes take precedence over ecosystem
summaries.

After resolving a disagreement:

1. identify the stale document;
2. update it before finishing the task;
3. do not merely note the inconsistency and leave it unresolved.

---

## Repository Safety

Do not:

- delete datasets or checkpoints automatically;
- overwrite historical outputs;
- rewrite valid negative-result artifacts;
- remove unrelated working-tree changes;
- commit or push unless explicitly requested.

Preserve unrelated working-tree changes.

If a required action would overwrite or remove historical research evidence,
stop and report the conflict instead of proceeding automatically.

---

## Task Completion Rule

Before declaring a research task complete, verify all of the following:

- the scientific question was answered;
- the GO/NO-GO/TRADE-OFF decision was recorded when applicable;
- a phase-specific note exists or was updated;
- ecosystem documentation is synchronized;
- stale next-step instructions were removed;
- no unauthorized multi-seed or multi-dataset work was launched;
- no commit or push occurred unless explicitly requested.

A task that completes the experiment but leaves the documented research state
stale is **not complete**.
