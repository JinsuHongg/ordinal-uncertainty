# Phase 1.5 — Existing Ordinal-UQ Baseline Audit

## Scope and provenance

This audit reuses, without retraining, the saved test probability vectors from
Experiment 0 (`outputs/retinamnist/single_model_baseline/seed_{0..4}`). Each seed
contains 400 test cases; severe-error counts are 75, 87, 91, 99, and 83 (prevalence
18.75%, 21.75%, 22.75%, 24.75%, and 20.75%). The phase evaluates the same fixed
argmax predictions and error labels as Experiment 0, so all comparisons are paired.

## Verified literature measures

The source is Haas and Hüllermeier, *Uncertainty quantification in ordinal
classification: A comparison of measures*, IJAR 186 (2025), 109479,
doi:10.1016/j.ijar.2025.109479. Equation (10) defines the unnormalised OCS sum
over the K-1 ordinal binary splits. For CDF values `F_k=sum_{j<=k}p_j`, this audit
implements `ocs_entropy=sum_k H(F_k,1-F_k)` and
`ocs_variance=sum_k F_k(1-F_k)`. Equation (5) supplies the C2 consensus measure;
we use its complementary dispersion `1-C2`. Equation (6) supplies Tastle--Wierman
consensus; we use complementary dissention `1-Cns`. Equations for `R_l2` define the
Bayes risk `sum_y (y-hat_y)^2 p_y`.

The 2026 Haas--Hüllermeier paper (Machine Learning 115, 63,
doi:10.1007/s10994-025-06960-5) verifies that OCS total uncertainty is defined on
first-order predictive probabilities. Its aleatoric/epistemic decomposition requires
multiple predictors and is intentionally not implemented in this single-model audit.

`prediction_distance_l1=sum_y |y-hat_y|p_y` is the required exploratory,
decision-centered simple ordinal baseline; it is also the paper's `R_l1` formula but
is labelled exploratory here because Experiment 0 fixes argmax predictions rather
than changing decisions by loss.

## Results

The full per-seed tables are under `outputs/retinamnist/phase1_5_ordinal_uq_audit/`.
The five-seed primary comparison is in `summary/baseline_comparison.csv`; rankings
include the mean ordinal-MAE risk over the fixed coverage grid.

`bayes_risk_l2` was best for severe-error AUROC (0.6620 +/- 0.0358) and AUPRC
(0.3777 +/- 0.0742), exceeding ordinal absolute deviation (0.6376 +/- 0.0148;
0.3271 +/- 0.0375). The exploratory decision-centered L1 measure had the strongest
Spearman association (0.3911 +/- 0.0560) and the lowest mean ordinal-MAE
risk-coverage value (0.5080). Nominal margin remained competitive for Spearman
(0.3770 +/- 0.0340). OCS and consensus measures did not improve the severe-error
endpoint on these frozen RetinaMNIST distributions.

## Decision and limitations

**Decision: STRONG EXISTING-BASELINE EXPLANATION.** A published ordinal Bayes-risk
baseline exceeds the simple ordinal absolute-deviation severe-error signal. This is
one imbalanced 400-case test split per seed; high AUPRC standard deviations and
identical-model dependence mean the result is a baseline-audit finding, not a broad
claim about all ordinal tasks. Do not design a new uncertainty metric from the
current idea; Phase 2, if pursued later, should examine established model/output
formulations rather than a new UQ metric.
