# AGENTS.md

## Project
Repository: `ordinal-uncertainty`

Working title: **Ordinal Uncertainty Quantification for Imbalanced Ordinal Classification**

This is the active ML/UQ workspace. The previous `ordinal-aware-conformal` repository is a separate research archive.

## Primary Research Goal
Study whether standard uncertainty measures adequately represent predictive uncertainty when labels have ordinal structure.

Initial question:

> Do ordinal-aware uncertainty measures better identify severe ordinal prediction errors than standard categorical uncertainty measures?

Only design a new ordinal uncertainty-aware loss after the baseline study shows a clear gap.

## Current Stage
**Phase 1 — single-model uncertainty analysis**

Start with RetinaMNIST and one deterministic classifier.

Do not begin with:
- Deep Ensembles as the primary method
- full Bayesian neural networks
- evidential learning
- conformal prediction
- a new custom loss

## Initial Model
Primary baseline:
- lightweight image backbone, preferably ResNet18
- Softmax output
- cross-entropy loss

Later baselines:
- distance-aware ordinal losses
- CORAL/CORN-style cumulative ordinal regression
- other strong ordinal methods from the literature audit

## Initial Uncertainty Measures
For probabilities \(p_k(x)\), compare:

Standard:
- predictive entropy
- confidence uncertainty
- probability-margin uncertainty

Ordinal-aware diagnostics:

\[
\mu(x)=\sum_{k=0}^{K-1} k p_k(x)
\]

\[
U_{\mathrm{ord-var}}(x)
=
\sum_k p_k(x)(k-\mu(x))^2
\]

\[
U_{\mathrm{ord-abs}}(x)
=
\sum_k p_k(x)|k-\mu(x)|
\]

These are diagnostic baselines, not claimed contributions.

## Primary Error Severity
For \(\hat Y=\arg\max_k p_k(X)\),

\[
E=|Y-\hat Y|.
\]

Analyze:
- \(E=0\): correct
- \(E=1\): adjacent error
- \(E\ge2\): severe ordinal error

Central question:

\[
E\uparrow \Rightarrow U(X)\uparrow?
\]

## Evaluation
Prediction:
- Accuracy
- MAE
- QWK
- NLL
- Brier score
- RPS where appropriate

Uncertainty:
- uncertainty by error severity
- AUROC/AUPRC for any-error detection
- AUROC/AUPRC for severe-error detection
- risk–coverage/selective-prediction curves
- class-wise and rare/extreme-class diagnostics
- calibration diagnostics

## Experimental Discipline
1. Ask one narrow question per experiment.
2. Freeze split/preprocessing before test evaluation.
3. Use multiple seeds after smoke testing.
4. Save sample-level logits/probabilities.
5. Save config and seed.
6. Report aggregate and class-wise metrics.
7. Separate exploratory and final-paper results.

## Dataset Roadmap
Phase 1:
- RetinaMNIST

Later candidates:
- UTKFace
- Solar flare ordinal classification
- Amazon Reviews
- additional ordinal benchmarks from literature

## Compute Policy
Escalate gradually:
1. single deterministic model
2. ordinal learning baselines
3. MC Dropout
4. Deep Ensemble with \(M=3\)
5. larger ensemble only as a limited ablation

The ensemble is a UQ backend, not automatically the proposed method.

## Relation to Previous CP Work
Do not restart:
- DKW/KS certification
- local-path borrowing
- cross-class exact-recovery calibration

Conformal prediction may later return as a baseline or post-hoc comparison method.

## Documentation
Maintain:
- `docs/research/current_state.md`
- `docs/research/experiment_plan.md`

Add later:
- `docs/research/decision_log.md`
- `docs/research/related_work.md`

Update `current_state.md` frequently.

## Repository Safety
Do not delete datasets/checkpoints automatically, overwrite outputs, or commit/push unless explicitly requested.
