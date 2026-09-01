# Research Decision Log

| Stage | Decision | Status | Reason |
| --- | --- | --- | --- |
| Resolution | Native 28×28 canonical; 64×64 resize historical only | Complete | Predictive metrics and uncertainty rankings are resolution-sensitive. |
| Simple uncertainty metrics | Stop as primary contribution | Complete | Established decision-risk quantities explain the strongest original signal. |
| Decision rule | L1 decision control required | Complete | L1 materially improves MAE and severe-error burden relative to mode. |
| CORAL | Stop — scientifically noncompetitive | Complete | Seed-0 predictive performance and severe-error burden were poor. |
| RPS | Retain | Complete | Strongest probabilistic baseline for risk alignment, severe detection, and ordinal selective prediction. |
| Temperature scaling | Insufficient explanation | Complete | Calibration improved probabilities but did not remove RPS's risk-quality advantage. |
| Phase 3.0 | Mixed failure; target ordinal center shrinkage | Complete | Rare class 4 is high-risk yet its mass and decisions remain pulled inward. |
| Weighted CE | Stop — scientifically noncompetitive | Complete | Increased rare-class mass without recovering class-4 decisions; global quality degraded. |
| SLACE | Stop — scientifically noncompetitive | Complete | Did not recover class-4 geometry despite useful seed-0 risk ranking. |
| Phase 3.2 gate | Justified, narrowly | Pending candidate design | Reduce rare upper-extreme inward shrinkage while preserving RPS-like ordinal risk alignment, severe detection, selective prediction, and global predictive quality. |
| Phase 3.2 Candidate 1 | Branch closed — NO-GO | Complete seed-0 diagnostics | Neighborhood correction trained adjacent class-4 recovery but did not jointly preserve global/selective-risk controls. Fixed-rho true-endpoint preference did not improve p4 or yield exact recovery. Do not create Candidate 1c from RetinaMNIST test diagnostics. |
| Phase 3.3 representation audit | Mixed representation / head failure | Complete frozen seed-0 audit | Some true class-4 features are nearest to central training centroids, while others are nearest to class 4 but still receive central head decisions. No representation method selected. |
| Phase 3.4 frozen-head audit | Mixed but decomposable failure | Complete frozen seed-0 audit | Balanced/prior-adjusted linear heads recover part of the feature-nearest-to-4 subset, but leave the feature-nearest-central subset largely collapsed and weaken global/risk quality. No new method selected. |

## SLACE infrastructure correction

The prior SLACE persistence issue was a **false diagnosis** caused by premature
artifact inspection / delayed workspace visibility. The scientific artifacts are
valid; evaluation-only reproduction completed successfully. This is an
infrastructure correction, not a revision of the negative SLACE scientific result.
