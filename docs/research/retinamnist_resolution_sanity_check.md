# RetinaMNIST Resolution Sanity Check

RetinaMNIST native images are 28x28 RGB. The historical Experiment 0 loader
explicitly applied `Resize((64,64))`; the ResNet18 is already a small-image model
(unpretrained 3x3, stride-1 first convolution and no max-pool). Thus only input
resolution changed here, not the stem or architecture.

Seed 0 used the official fixed split, CE, AdamW, learning rate 1e-3, weight decay
1e-4, batch size 64, 20 epochs, random horizontal flip, and minimum validation NLL
selection in both conditions. The 64x64 historical seed-0 output was reused because
its saved configuration confirms this exact setup; native 28x28 was newly trained.

| Metric | 28x28 | 64x64 | 28 - 64 |
|---|---:|---:|---:|
| Accuracy | 0.5600 | 0.5325 | +0.0275 |
| MAE | 0.6850 | 0.6950 | -0.0100 |
| QWK | 0.5938 | 0.5483 | +0.0456 |
| NLL | 1.1863 | 1.3645 | -0.1782 |
| ECE | 0.0637 | 0.0680 | -0.0043 |
| Severe errors | 75 | 75 | 0 |

For severe errors, `bayes_risk_l2` remains above ordinal absolute deviation at both
resolutions (AUROC 0.5950/0.2887 AUPRC at 28 versus 0.5885/0.2442; 0.6274/0.2821
at 64 versus 0.6289/0.2662). However, ranking and signal magnitude differ: native
28 has decision-centered L1 as best severe AUPRC (0.3240) and margin as best
Spearman (0.3356), whereas the historical 64 run has L1 as best Spearman (0.3992)
and strong OCS/ordinal signals. L1 also has the best ordinal-MAE risk summary at
both sizes (0.4719 vs 0.4465).

**Decision: RESOLUTION SENSITIVE.** Predictive metrics improve modestly at native
resolution and severe prevalence is identical, but uncertainty association and the
ranking of literature measures change in this single-seed check. Future experiments
should use native 28x28, and Experiment 0/Phase 1.5 should be rerun at 28x28 before
making a Phase 2 decision.
