# Native 28x28 Experiment 0 and Phase 1.5 Rerun

The resolution sanity check found seed-level uncertainty rankings sensitive to the
historical 28-to-64 resize. Native 28x28 is now the official RetinaMNIST input.
The small-image ResNet18 stem, CE/AdamW protocol, fixed splits, 20 epochs, batch
size 64, validation-NLL selection, and all uncertainty definitions were unchanged.
Seed 0 was reused from the fully compatible resolution check; seeds 1--4 were run
fresh. Historical 64x64 outputs remain untouched.

Native 28x28 prediction means +/- sample standard deviations are: accuracy 0.5320
+/- 0.0169, MAE 0.7430 +/- 0.0541, QWK 0.5526 +/- 0.0290, NLL 1.3650 +/- 0.2566,
Brier 0.5905 +/- 0.0106, RPS 0.1288 +/- 0.0033, and ECE 0.0954 +/- 0.0408.
Severe-error counts for seeds 0--4 are 75, 82, 88, 84, and 77 (18.75%--22.00%).

Native Phase 1.5 ranks `prediction_distance_l1` first for Spearman (0.3962 +/-
0.0561), severe AUROC (0.6529 +/- 0.0246), severe AUPRC (0.3316 +/- 0.0318), and
mean ordinal-MAE risk coverage (0.4937). `bayes_risk_l2` is second for severe AUROC
(0.6480 +/- 0.0358) and AUPRC (0.3288 +/- 0.0336), and remains ahead of ordinal
absolute deviation (0.6428 +/- 0.0385; 0.3152 +/- 0.0560). OCS/consensus methods
remain weaker at the severe-error endpoint.

Compared with historical 64x64, native performance has slightly higher accuracy
(0.5320 vs 0.5150) and lower MAE (0.7430 vs 0.7745), but the leading severe-error
measure changes from published L2 Bayes risk to exploratory decision-centered L1.

**Decision: Phase 1.5 interpretation revised.** Existing ordinal risk baselines
remain strong, but the native five-seed result does not support the earlier stronger
claim that the published L2 baseline uniquely explains the signal. The empirical
question should focus on calibrated, decision-centered ordinal risk baselines versus
other established ordinal-UQ measures; do not propose a new metric from this result.
