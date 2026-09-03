# Research Decision Log

| Stage | Decision | Status | Reason |
| --- | --- | --- | --- |
| Resolution | Native 28×28 canonical; 64×64 resize historical only | Complete | Predictive metrics and uncertainty rankings are resolution-sensitive. |
| Simple uncertainty metrics | Stop as primary contribution | Complete | Established decision-risk quantities explain the strongest original signal. |
| Decision rule | L1 decision control required | Complete | L1 materially improves MAE and severe-error burden relative to mode. |
| CORAL | Stop — scientifically noncompetitive | Complete | Seed-0 predictive performance and severe-error burden were poor. |
| RPS | Retain | Complete | Strong RetinaMNIST probabilistic baseline for risk alignment, severe detection, and ordinal selective prediction; not a universal cross-dataset superiority claim. |
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
| Phase 3.5 design audit | One predeclared seed-0 falsification candidate selected; no implementation | Complete | RG-ACR uses detached L1 Bayes-risk weighting with local adjacent-centroid ranking on top of RPS. Literature overlap is material, so this is not a novelty claim or method freeze. |
| Phase 3.5 backup/defer | Adaptive margin backup; risk-weighted prototype compactness deferred | Complete | Adaptive margins add moving-geometry instability; prototype compactness has high overlap with center/prototype methods and risks erasing useful uncertainty. |
| Phase 3.5 execution gate | One final RetinaMNIST seed-0 falsification experiment executed as Phase 3.6 | Complete | Validation-only selection was used; the result was NO-GO rather than method freeze. |
| Phase 3.6 RG-ACR | NO-GO — branch stopped | Complete | Validation-selected λ=.05 lacked clear class-4 representation improvement across raw/normalized geometry and violated class-0 MAE and risk-Spearman tolerances. Downstream gains cannot rescue the mechanism failure. No RG-ACR-v2. |
| Phase 3.7A-UTKFace | PARTIAL REPLICATION | Complete | Rare upper-extreme inward displacement and elevated risk reproduced, but the broad RetinaMNIST RPS matched-L1 risk-quality advantage did not; RPS partly improved class-4 recovery with global/lower-endpoint trade-offs. No automatic representation/head audit. |
| Phase 3.7A-Solar | Paused before training | Active pause | Three-channel setup and normalization retry exist, but no CE/RPS full training or scientific conclusion occurred. |
| Rare upper-extreme inward shrinkage | Replicated across RetinaMNIST and UTKFace | Complete | Both datasets show elevated upper-endpoint risk, predictive means pulled inward, and a lower endpoint that is materially easier. |
| Broad RPS matched-L1 risk-quality advantage | Not replicated on UTKFace | Complete | RPS improved UTKFace severe AUROC only; CE had stronger Spearman, severe AUPRC, selective MAE, global prediction, and probability quality. |
| UTKFace representation/head audit | Not automatically justified | Complete | UTKFace has only a partial baseline replication; no test-informed follow-up is authorized. |
| Phase 3.8 direction | Solar rare-extreme shrinkage confirmation | Next | Confirm cross-dataset inward localization bias with matched CE/RPS controls; RPS superiority is secondary. |

## SLACE infrastructure correction
The prior SLACE persistence issue was a **false diagnosis** caused by premature artifact inspection / delayed workspace visibility. The scientific artifacts are valid; evaluation-only reproduction completed successfully. This is an infrastructure correction, not a revision of the negative SLACE scientific result.

## Development-benchmark guardrail
RetinaMNIST is now a development benchmark. No additional
RetinaMNIST-test-informed method experiment is authorized while Phase 3.8
confirms the cross-dataset failure pattern.
