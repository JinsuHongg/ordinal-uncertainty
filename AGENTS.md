# AGENTS.md

## Project

Repository: `ordinal-uncertainty`
Working title: **Ordinal Uncertainty Quantification for Imbalanced Ordinal Classification**

This is the active ML/UQ workspace. The previous `ordinal-aware-conformal`
repository is a separate research archive.

## Current Stage

**Phase 3.2 — candidate-method design justified, not started.**

The following are complete:

- Experiment 0 / Phase 1
- Phase 1.5 ordinal-UQ baseline audit
- RetinaMNIST resolution sanity check
- Phase 1.75 decision-rule audit
- Phase 2 ordinal-learning baseline study
- Phase 2.5 calibration-control audit
- Phase 3.0 extreme-class failure audit
- Phase 3.1 imbalance-aware baseline audit

Canonical RetinaMNIST uses the official split at native **28×28 RGB** with the
unpretrained small-image ResNet18 (3×3 stride-1 stem, no max-pool). Historical
64×64 results are sensitivity evidence only.

## Established Baselines and Decisions

- **CE:** canonical nominal probabilistic baseline.
- **RPS:** **RETAIN**; strongest probabilistic ordinal baseline for risk
  alignment, severe-error detection, and ordinal-MAE selective prediction.
- **CORAL:** **STOP — SCIENTIFICALLY NONCOMPETITIVE**.
- **Weighted CE:** **STOP — SCIENTIFICALLY NONCOMPETITIVE**.
- **SLACE:** **STOP — SCIENTIFICALLY NONCOMPETITIVE**.

The simple-new-uncertainty-metric branch is also stopped: established
decision-risk quantities explain the strongest original signal.

## Current Unresolved Target

> Reduce rare upper-extreme inward shrinkage while preserving RPS-like ordinal
> risk alignment, severe-error detection, selective prediction, and global
> predictive quality.

Class 4 is the rare upper extreme in RetinaMNIST. Existing methods often assign
it high expected risk while still placing its probability mass and decisions too
far toward central classes. This is not primarily an argmax-only problem: mode,
L1, and L2 decisions have all been audited.

## Phase 3.2 Execution Policy

1. No proposed Phase 3.2 objective is frozen.
2. Check literature overlap before implementing a candidate or making a novelty
   claim.
3. Test only a small number of minimal candidate objectives.
4. For every candidate: add unit tests; run a short seed-0 smoke run; run one
   full seed-0 GO/NO-GO evaluation; do not run seeds 1–4 without explicit user
   approval.
5. Compare against frozen CE and RPS baselines.
6. Evaluate predictive quality; mode/L1/L2 decisions; class-4 geometry; class-0
   control; risk alignment; severe-error detection; and ordinal-MAE selective
   prediction.
7. Do not claim novelty before literature verification and empirical evidence.

Phase 3.2 should prefer a shared training/evaluation pipeline with
objective/config selection rather than a separate full trainer for every
candidate. Do not implement that refactor unless a Phase 3.2 task explicitly
requires it.

## Core Evaluation

For a class-probability vector \(p\), keep these factors separate:

\[
\text{Predictive distribution}
\rightarrow
\text{Decision rule}
\rightarrow
\text{Expected decision risk}.
\]

Evaluate mode, exact discrete L1-optimal, and exact discrete L2-optimal decisions
with the established smallest-class tie-break. Report prediction metrics
(accuracy, MAE, QWK, severe-error prevalence, and probability metrics where
appropriate), expected-risk/realized-error alignment, severe-error AUROC/AUPRC,
selective prediction, and class-wise/extreme-class diagnostics.

## Experimental Discipline

1. Ask one narrow question per experiment.
2. Freeze split, preprocessing, and validation/test roles before test evaluation.
3. Use validation-only checkpoint selection and calibration fitting.
4. Save sample-level logits/probabilities, IDs, labels, decisions, and risks.
5. Save complete config, training/validation history, selected checkpoint, and
   evaluation artifacts for every full run.
6. Use multiple seeds only after a valid seed-0 run and explicit approval.
7. Report aggregate and class-wise metrics; keep rare-class denominators visible.
8. Preserve valid negative results and distinguish exploratory from final results.

## Guardrails

Do not currently:

- restart CORAL, Weighted CE, or SLACE multi-seed branches;
- return to 64×64 RetinaMNIST as the canonical configuration;
- add ensembles, Bayesian methods, or conformal prediction as a substitute for
  the current single-model question;
- tune on test data;
- claim a universal ordinal-UQ advantage or a novel loss without verified support.

## Documentation and Safety

Maintain:

- `docs/research/current_state.md`
- `docs/research/experiment_plan.md`
- `docs/research/decision_log.md`

Do not delete datasets/checkpoints automatically, overwrite historical outputs,
or commit/push unless explicitly requested. Preserve unrelated working-tree
changes.
