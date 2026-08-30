# Current Research State

## Status
**Active project:** Ordinal Uncertainty Quantification for Imbalanced Ordinal Classification  
**Stage:** Phase 1 — single-model uncertainty analysis

The project has pivoted from theory-heavy cross-class conformal borrowing to a machine-learning-focused uncertainty quantification problem for ordinal classification.

The previous conformal borrowing branch is stopped and preserved separately in the `ordinal-aware-conformal` research archive.

## Core Research Question
> Do standard uncertainty measures adequately represent predictive uncertainty in ordinal classification, especially when prediction errors differ greatly in ordinal severity?

Immediate hypothesis: entropy, confidence, and probability margin may not distinguish uncertainty between adjacent ordinal classes from uncertainty spanning distant classes.

The first objective is to test this empirically before designing a new loss.

## Current Methodological Position
No proposed method is frozen.

Do not currently claim:
- a new ordinal loss
- a new Bayesian model
- a new uncertainty decomposition
- a new calibration theorem

This is a diagnostic baseline stage.

## Initial Dataset
**RetinaMNIST**

Reasons:
- naturally ordinal
- computationally lightweight
- suitable for multi-seed experiments
- appropriate for uncertainty diagnostics

## Initial Model
Start with:
- one lightweight classifier
- Softmax output
- cross-entropy loss

Preferred first backbone: ResNet18 if compatible with the existing pipeline.

Do not use Deep Ensembles in Experiment 0.

## Initial Measures
Standard:
- entropy
- maximum confidence
- probability margin

Ordinal-aware diagnostics:

\[
\mu(x)=\sum_k k p_k(x)
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

These are baselines only and require literature review before any novelty claim.

## Primary Diagnostic
Define:

\[
E=|Y-\hat Y|.
\]

Analyze:
- \(E=0\)
- \(E=1\)
- \(E\ge2\)

Primary question:

\[
E\uparrow \Rightarrow U(X)\uparrow?
\]

## Initial Evaluation Targets
Prediction:
- Accuracy
- MAE
- QWK
- NLL
- Brier score
- RPS if appropriate

Uncertainty:
- uncertainty by error severity
- AUROC/AUPRC for any-error detection
- AUROC/AUPRC for severe-error detection
- selective prediction / risk–coverage
- class-wise uncertainty
- rare/extreme-class diagnostics

## Roadmap
### Phase 1 — Single-model diagnostic
Test whether ordinal-aware uncertainty measures provide information standard categorical measures miss.

### Phase 2 — Existing ordinal learning baselines
Compare:
- Softmax + CE
- distance-aware ordinal losses
- CORAL/CORN-style cumulative ordinal regression
- other strong baselines from literature

### Phase 3 — Ordinal uncertainty-aware learning
Proceed only if a clear gap remains.

Possible generic form:

\[
\mathcal L
=
\mathcal L_{\mathrm{predictive}}
+
\lambda\mathcal L_{\mathrm{ordinal\ uncertainty}}.
\]

No exact loss is selected yet.

### Phase 4 — Epistemic UQ
Only after the single-model framework is justified:
1. MC Dropout
2. Deep Ensemble with \(M=3\)
3. optional larger/snapshot ensemble ablation

## Main Risks
1. Ordinal variance or related measures may already be established.
2. Existing ordinal classifiers may already solve much of the issue.
3. Severe ordinal errors may be rare.
4. Improvements may reflect probability calibration rather than ordinal structure.
5. Better uncertainty may simply come from a more accurate model.

## Immediate Next Action
Run Experiment 0 from `docs/research/experiment_plan.md`.

Do not design a new loss before reviewing the baseline uncertainty landscape.
