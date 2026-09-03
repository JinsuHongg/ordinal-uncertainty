# AGENTS.md

## Project

Repository: `ordinal-uncertainty`

Working title: **Ordinal Uncertainty Quantification for Imbalanced Ordinal Classification**

This is the active ML/UQ workspace. The previous `ordinal-aware-conformal`
repository is a separate research archive.

---

## Current Stage

**Phase 3.7A-UTKFace complete — PARTIAL REPLICATION. Phase 3.8 Solar
Rare-Extreme Shrinkage Confirmation is next.**

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

**Phase 3.7A-Solar was PAUSED BEFORE TRAINING.** It is now the authorized
Phase 3.8 confirmation task; do not perform implementation changes, tuning, or
method development as part of it.

---

## Current Failure Diagnosis

The cross-dataset common signal is:

\[
\boxed{\text{Rare upper-extreme inward localization bias under ordinal imbalance}}
\]

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

> **Why do imbalanced ordinal classifiers exhibit systematic inward
> localization bias for rare upper-extreme classes, even when uncertainty,
> risk quality, and exact-class performance differ across datasets and
> objectives?**

Phase 3.8 asks whether this common signal also appears in ordinal solar-flare
classification. CE and RPS are matched controls; RPS superiority is secondary,
not a replication requirement.

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

RetinaMNIST has been inspected extensively during method development.

Treat RetinaMNIST as a **development benchmark** from now on. Method
development remains paused while the cross-dataset confirmation question is
resolved.

RetinaMNIST test results may be used for:

- retrospective diagnosis;
- the one predeclared final seed-0 falsification experiment.

They must not be used to:

- tune new hyperparameters after observing test outcomes;
- repeatedly redesign objectives;
- create Candidate A1/A2/A3-style iterative test-informed variants;
- select among many post-hoc methods.

If the final seed-0 candidate is promising:

\[
\boxed{\text{METHOD FREEZE}}
\]

Then move to:

- RetinaMNIST seeds 1–4;
- additional ordinal datasets;
- confirmatory evidence.

---

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
- tune on test data;
- launch all datasets before method freeze;
- add ensembles, Bayesian methods, or conformal prediction as a substitute for
  the current single-model question;
- claim universal superiority of ordinal UQ;
- claim novelty before literature verification and empirical support.

---

## Next Authorized Work

**Phase 3.8 — Solar Rare-Extreme Shrinkage Confirmation** is authorized.
Evaluate matched CE and RPS controls for upper-extreme risk, inward shrinkage,
endpoint asymmetry, and the limits of L1/L2 correction. Do not require RPS to
win for replication, and do not restart method development, revive RG-ACR,
create UTKFace-specific objectives, or launch multi-seed expansions.

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
