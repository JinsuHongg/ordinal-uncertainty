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
| Phase 3.2 Candidate 1 | Trade-off; no multi-seed | Complete | Endpoint-neighborhood correction converted many class-4 central errors to adjacent errors but did not preserve global/risk quality or recover class 4 exactly. |
| Phase 3.2 Candidate 1b | Branch closed — NO-GO | Complete | Fixed true-endpoint preference did not increase p4 relative to Candidate 1, weakened adjacent recovery, and produced no exact recovery. No Candidate 1c. |
| Output-only correction | Stop | Complete | Probability-level shaping can move location but does not jointly solve true-endpoint localization and risk/global quality. |
| Phase 3.3 | Mixed representation / head failure | Complete | Many class-4 samples are representation-collapsed, while a subset is feature-nearest to class 4 but still mapped centrally by the head. |
| Phase 3.4 | Mixed but decomposable failure | Complete | Simple head correction recovers some feature-nearest-to-4 samples, but most feature-nearest-central samples remain unrecovered. |
| Head-only solution | Insufficient | Complete | Head/prior correction helps a subset but introduces global/risk trade-offs and cannot recover representation-collapsed cases. |
| Representation-only explanation | Insufficient | Complete | Some class-4 representations are already correctly localized but still fail at the head. |
| Current diagnosis | Dual-component rare-extreme failure | Active | Representation collapse and head-level inward bias are both supported and affect different subsets. |
| Phase 3.5 gate | Ready for design audit | Next | Evaluate risk-conditioned ordinal representation candidates before any new implementation/training. |

## SLACE infrastructure correction
The prior SLACE persistence issue was a **false diagnosis** caused by premature artifact inspection / delayed workspace visibility. The scientific artifacts are valid; evaluation-only reproduction completed successfully. This is an infrastructure correction, not a revision of the negative SLACE scientific result.

## Development-benchmark guardrail
RetinaMNIST is now a development benchmark. The next seed-0 method experiment should be the last major method-selection experiment based on RetinaMNIST test diagnostics; a promising method must be frozen before multi-seed and multi-dataset confirmation.
