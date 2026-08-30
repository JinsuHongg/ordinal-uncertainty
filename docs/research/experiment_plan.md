# Experiment Plan

## Experiment 0 — Single-Model Ordinal Uncertainty Baseline

### Goal
Determine whether ordinal-aware uncertainty measures contain useful information about **ordinal error severity** beyond standard categorical uncertainty measures.

This is a diagnostic experiment, not a proposed-method benchmark.

## Research Questions
### RQ1
Do standard uncertainty measures increase as ordinal prediction error becomes more severe?

### RQ2
Do ordinal-aware uncertainty measures distinguish severe ordinal errors better than entropy, confidence, and margin?

### RQ3
Are the findings stable across seeds and rare/extreme classes?

A positive result is required before designing a new ordinal uncertainty-aware loss.

## Dataset
Primary dataset: **RetinaMNIST**

Use a fixed official split if supported.

Record:
- dataset version
- train/validation/test counts
- class counts
- preprocessing
- augmentation

Do not change the split after inspecting test results.

## Model
Start with one deterministic classifier:
- ResNet18 preferred
- Softmax output
- cross-entropy loss

Do not perform architecture search.

## Seeds
Smoke test: 1 seed

First interpretable result: 5 seeds

Suggested:
`0, 1, 2, 3, 4`

## Save Per-Sample Outputs
For every test sample save:
- sample ID/index
- true class
- predicted class
- logits
- probability vector
- correctness
- absolute ordinal error
- entropy
- confidence uncertainty
- margin uncertainty
- ordinal predictive mean
- ordinal variance
- ordinal expected absolute deviation

Suggested output:
`outputs/retinamnist/single_model_baseline/<seed>/`

Save:
- `config.json`
- `metrics.json`
- `predictions.csv` or `.parquet`
- checkpoint reference
- training log

## Uncertainty Measures
Entropy:

\[
H(p)=-\sum_k p_k\log p_k.
\]

Confidence uncertainty:

\[
U_{\mathrm{conf}}=1-\max_k p_k.
\]

Margin uncertainty, with \(p_{(1)}\ge p_{(2)}\):

\[
U_{\mathrm{margin}}
=
1-(p_{(1)}-p_{(2)}).
\]

Ordinal predictive mean:

\[
\mu=\sum_k k p_k.
\]

Ordinal variance:

\[
U_{\mathrm{ord-var}}
=
\sum_k p_k(k-\mu)^2.
\]

Ordinal expected absolute deviation:

\[
U_{\mathrm{ord-abs}}
=
\sum_k p_k|k-\mu|.
\]

Use the same orientation for every uncertainty measure: higher = more uncertain.

## Error Severity
Define:

\[
E=|Y-\hat Y|.
\]

Primary bins:
- \(E=0\)
- \(E=1\)
- \(E\ge2\)

Retain exact integer error as well.

## Prediction Metrics
Report:
- Accuracy
- MAE
- QWK
- NLL
- Brier score
- RPS where appropriate

Report mean and standard deviation across seeds.

## Analysis A — Uncertainty by Error Severity
For each measure report uncertainty for:
- correct predictions
- adjacent errors
- severe errors

Primary expected ordering:

\[
U(E=0)<U(E=1)<U(E\ge2).
\]

Failure of this ordering is a valid research result.

## Analysis B — Association with Error Magnitude
Measure association between uncertainty and:

\[
|Y-\hat Y|.
\]

Use Spearman correlation as the primary statistic.

## Analysis C — Error Detection
Any-error target:

\[
Z_{\mathrm{err}}
=
\mathbf 1\{Y\ne\hat Y\}.
\]

Severe-error target:

\[
Z_{\mathrm{severe}}
=
\mathbf 1\{|Y-\hat Y|\ge2\}.
\]

Compute:
- AUROC
- AUPRC

Severe-error detection is the more important task for the project hypothesis.

## Analysis D — Selective Prediction
Sort examples from lowest to highest uncertainty.

Evaluate risk as uncertain samples are rejected.

Use:
- classification error risk
- ordinal MAE risk
- risk–coverage curves

Question:
> Does ordinal-aware uncertainty preferentially reject samples that would cause large ordinal errors?

## Analysis E — Class-Wise Behavior
For each true class report:
- sample count
- accuracy / MAE
- mean uncertainty
- severe-error rate

Pay attention to rare and extreme classes.

## Calibration Diagnostic
Track:
- NLL
- Brier score
- ECE
- reliability-diagram data

All Experiment 0 uncertainty measures should come from the same predictive probabilities, helping isolate the value of the metric itself.

## Success Criteria
### STRONG POSITIVE SIGNAL
Continue toward ordinal uncertainty-aware learning if:
1. ordinal-aware measures correlate more strongly with ordinal error severity;
2. severe-error detection improves meaningfully;
3. effects are stable across seeds;
4. gains are not driven by one class.

### PARTIAL SIGNAL
If differences are small or inconsistent, test existing ordinal models before designing a new loss.

### NO-GO
Do not design a new loss if:
- ordinal-aware measures do not materially outperform standard measures;
- results are unstable;
- gains are explained by calibration artifacts.

## Phase 2 Trigger
Only after Experiment 0 review, compare:
1. Softmax + CE
2. distance-aware ordinal loss
3. CORAL/CORN-style cumulative ordinal regression

Then repeat the same uncertainty evaluation.

## Compute Budget
Experiment 0 uses one model per seed.

No ensemble training yet.

If epistemic uncertainty is needed later:
1. MC Dropout
2. Deep Ensemble with \(M=3\)

## Checklist
- [ ] RetinaMNIST loader verified
- [ ] fixed split recorded
- [ ] one-model training pipeline verified
- [ ] probabilities saved
- [ ] uncertainty metrics implemented/tested
- [ ] ordinal error severity saved
- [ ] prediction metrics computed
- [ ] any-error AUROC/AUPRC computed
- [ ] severe-error AUROC/AUPRC computed
- [ ] risk–coverage data saved
- [ ] class-wise diagnostics saved
- [ ] 5-seed summary produced
- [ ] result documented in `current_state.md`

## Do Not Do Yet
Do not yet:
- design a custom loss
- train an ensemble
- implement a full Bayesian neural network
- implement evidential learning
- add conformal calibration
- run all candidate datasets
- tune metrics after seeing the result
