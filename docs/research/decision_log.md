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
| RetinaMNIST diagnosis (Phases 3.3–3.4) | Dual-component rare-extreme failure | Historical, dataset-specific | Representation collapse and head-level inward bias are both supported for distinct audited RetinaMNIST subsets; this is not the active project-wide diagnosis. |
| Phase 3.5 design audit | One predeclared seed-0 falsification candidate selected; no implementation | Complete | RG-ACR uses detached L1 Bayes-risk weighting with local adjacent-centroid ranking on top of RPS. Literature overlap is material, so this is not a novelty claim or method freeze. |
| Phase 3.5 backup/defer | Adaptive margin backup; risk-weighted prototype compactness deferred | Complete | Adaptive margins add moving-geometry instability; prototype compactness has high overlap with center/prototype methods and risks erasing useful uncertainty. |
| Phase 3.5 execution gate | One final RetinaMNIST seed-0 falsification experiment executed as Phase 3.6 | Complete | Validation-only selection was used; the result was NO-GO rather than method freeze. |
| Phase 3.6 RG-ACR | NO-GO — branch stopped | Complete | Validation-selected λ=.05 lacked clear class-4 representation improvement across raw/normalized geometry and violated class-0 MAE and risk-Spearman tolerances. Downstream gains cannot rescue the mechanism failure. No RG-ACR-v2. |
| Phase 3.7A-UTKFace | PARTIAL REPLICATION | Complete | Rare upper-extreme inward displacement and elevated risk reproduced, but the broad RetinaMNIST RPS matched-L1 risk-quality advantage did not; RPS partly improved class-4 recovery with global/lower-endpoint trade-offs. No automatic representation/head audit. |
| Phase 3.7A-Solar | Paused before training | Historical | Three-channel setup and normalization retry were preserved; no Phase 3.7A scientific conclusion was made. |
| Phase 3.8 Solar | STRONG CONFIRMATION | Complete | Matched CE/RPS solar controls both showed zero exact X decisions under mode/L1/L2, >1-class inward predictive shrinkage, and a much harder upper than lower endpoint. RPS did not improve solar risk-quality metrics. |
| Rare upper-extreme inward shrinkage | Replicated across RetinaMNIST and UTKFace | Complete | Both datasets show elevated upper-endpoint risk, predictive means pulled inward, and a lower endpoint that is materially easier. |
| Broad RPS matched-L1 risk-quality advantage | Not replicated on UTKFace | Complete | RPS improved UTKFace severe AUROC only; CE had stronger Spearman, severe AUPRC, selective MAE, global prediction, and probability quality. |
| UTKFace representation/head audit | Not automatically justified | Complete | UTKFace has only a partial baseline replication; no test-informed follow-up is authorized. |
| Phase 3.8 direction | Solar rare-extreme shrinkage confirmation | Complete | The predeclared third-dataset gate was executed under frozen CE/RPS controls; no automatic follow-up is authorized. |
| Phase 3.9 Solar | MIXED BUT DECOMPOSABLE FAILURE | Complete | Both X representation collapse (17–24%) and complete original-head failure among X-like features were observed; fixed frozen-head controls recover X only with global/risk trade-offs. |
| Phase 3.10A RetinaMNIST ROP | TRADE-OFF | Complete | Training-only 5-fold OOF ROP was active and modestly improved portions of the balanced-head risk trade-off, but did not improve class-4 recovery over balanced head and violated the predeclared class-0 MAE safety tolerance versus original RPS. No objective freeze, test evaluation, ROP-v2, seed expansion, or dataset follow-up. |
| Phase 3.10B RetinaMNIST head audit | MIXED PARAMETER MECHANISM | Complete | Exact OOF parameter/logit decomposition finds that balanced-head class-4 recovery and class-0 damage are almost entirely feature-projection changes from large classifier direction rotations and norm inflation; bias swaps are inert. This is an audit, not authorization for a correction. |
| Phase 3.10C RetinaMNIST direction-only | PARTIAL SUPPORT | Complete | Fixed-original-norm/bias direction-only OOF head retains partial C4 localization gain and improves global/class-0 behavior versus full balanced head, but loses most exact C4 recovery. No controlled-scale method is authorized automatically. |
| Phase 3.10D RetinaMNIST controlled scale | GO — CONTROLLED SCALE SUPPORTED | Complete | Predeclared intermediate fixed scales, especially α=.50, retain substantial balanced C4 recovery while improving C0/global MAE and probability quality versus B. Training-only mechanism support only; no adaptive scales or test evaluation. |
| Phase 3.10E RetinaMNIST controlled scale × ROP | GO — COMPLEMENTARY MECHANISMS SUPPORTED | Complete | At fixed α=.50 and pre-existing λ=1.0, ROP improves pair preservation, L1-risk Spearman, and selective MAE while retaining controlled-scale C4 localization and C0/global behavior. AUROC/AUPRC and probability quality are mixed; no tuning, ROP-v2, test evaluation, seed expansion, or new dataset is authorized. |
| Phase 3.11 RetinaMNIST frozen candidate validation | MIXED — HOLD FROZEN, NO REDESIGN | Complete | The one-shot historical validation reproduces E's outward C4 movement, but reverses the OOF B→E C0/global safety and D→E Spearman/selective-MAE effects. No method freeze or historical-test evaluation is authorized. |
| Phase 3.12 candidate disposition | CROSS-DATASET MECHANISM CONFIRMATION JUSTIFIED | Complete | Direction adaptation, scale amplification, and controlled scale remain RetinaMNIST mechanisms; bias correction is stopped, ROP is diagnostic/secondary only, and the combined candidate is not freeze-ready. A frozen UTKFace direction/scale confirmation is justified; ROP is excluded. |
| Phase 3.13 UTKFace direction/scale confirmation | PARTIAL CONFIRMATION | Complete | Balanced direction/scale adaptation transports as a rare-upper-end localization mechanism, and fixed α=.50 D is strongest on C4; however, UTKFace does not reproduce RetinaMNIST's B-like class-0 damage or controlled-scale global safety ordering. Bias shifts are also non-negligible. No test evaluation or UTKFace-specific redesign. |
| Phase 3.14 cross-dataset mechanism consolidation | SOLAR DIRECTION/SCALE CONFIRMATION JUSTIFIED | Complete | Direction adaptation transports across RetinaMNIST/UTKFace and added scale can strengthen outward rare-end movement. Opposite-endpoint/global trade-offs and bias behavior are dataset-dependent; controlled scale is a mechanism probe, not a universal method, and ROP remains secondary. A minimal frozen Solar A/B/C/D decomposition is justified. |
| Phase 3.15 Solar direction/scale confirmation | PARTIAL CONFIRMATION | Complete | Original RPS X recovery was 0/921; balanced B recovered 334 and direction-only C recovered 496 while lowering X MAE `1.197 -> .675`. Fixed `.50` scale D recovered only 262 and was worse than C. Direction transports; fixed scale is not a universal increment. No tuning or method development is authorized. |
| Project-level phenomenon | Rare upper-extreme inward localization observed across three imbalanced ordinal settings | Active | Observed on RetinaMNIST, UTKFace, and Solar; class imbalance was not independently manipulated, so this is not a causal isolation claim. |
| Cross-dataset frozen-head mechanism | Direction adaptation is the strongest replicated head finding | Active | In the three tested frozen-RPS, one-backbone-seed protocols, direction-only adaptation improved rare-end localization; scope is not architecture-, objective-, or population-universal. |
| Classifier scale | Dataset-specific correction-strength modulator | Active | C→D helps RetinaMNIST and UTKFace but harms Solar; `.50` is not a transferable prescription. |

## Phase 3.16 — Current disposition

**A — MECHANISM EVIDENCE SUFFICIENT; MOVE TO PAPER FRAMING.** Direction-only
adaptation improves rare-end localization across the three tested frozen RPS
representations. Solar rejects universal benefit of the .50 scale increment.
Scale benefits, bias behavior, and collateral effects are dataset-dependent.
Controlled scale is diagnostic, ROP secondary, bias-only correction stopped,
and the combined candidate HOLD — NOT FREEZE-READY. No method-design or
experimental work is authorized. See
[Phase 3.16](phase3_16_three_dataset_mechanism_consolidation.md).

This interpretation supersedes the historical Phase 3.14 scale claim, while
preserving that phase's conclusion at the time. Localization gains do not
establish global probabilistic or risk-quality gains; nonzero bias shifts do
not establish a causal bias-only intervention.

## Phase 3.17 — Paper framing

**A — PROCEED AS MECHANISM PAPER.** The bounded contribution is the
three-dataset rare-upper-end localization phenomenon and frozen-head direction
mechanism. Generic head rebalancing, scale control, bias correction, ROP, and
conformal prediction are excluded as claimed novel methods.

## SLACE infrastructure correction
The prior SLACE persistence issue was a **false diagnosis** caused by premature artifact inspection / delayed workspace visibility. The scientific artifacts are valid; evaluation-only reproduction completed successfully. This is an infrastructure correction, not a revision of the negative SLACE scientific result.

## Development-benchmark guardrail
RetinaMNIST is now a development benchmark. No additional
RetinaMNIST-test-informed method experiment is authorized. Phase 3.8 is complete and any next action requires separate authorization.
