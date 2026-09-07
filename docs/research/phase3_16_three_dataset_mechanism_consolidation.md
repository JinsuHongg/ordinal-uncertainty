# Phase 3.16 — Three-Dataset Mechanism Consolidation

## Scope, provenance, and comparison limits

Analysis only: this note consolidates completed evidence without training,
feature extraction, split evaluation, or access to test data. Historical phase
notes and scientific artifacts are preserved. Values are rounded from the
saved records, not newly evaluated predictions.

| Evidence | Role and support | Source |
|---|---|---|
| RetinaMNIST 3.10B–D | Seed-0 frozen RPS; pooled five-fold training-only head OOF, N=1080, C4=66, C0=486 | [3.10B](phase3_10b_head_bias_localization_audit.md), [3.10C](phase3_10c_direction_only_head_falsification.md), [3.10D](phase3_10d_controlled_scale_head_falsification.md) |
| RetinaMNIST 3.11 | Full-training heads; one-shot historical validation, N=120, C4=6, C0=54 | [3.11](phase3_11_frozen_candidate_validation.md) |
| Candidate disposition | ROP demotion, candidate hold, two-dataset consolidation | [3.12](phase3_12_evidence_consolidation_and_candidate_disposition.md), [3.14](phase3_14_cross_dataset_mechanism_consolidation.md) |
| UTKFace 3.13 | Seed-0 frozen RPS; archived validation, N=2371, C4=67, C0=459 | [3.13](phase3_13_utkface_direction_scale_mechanism_confirmation.md) |
| Solar 3.8–3.9 | Baseline phenomenon and centroid/head decomposition; previously evaluated aligned future-period subset | [3.8](phase3_8_solar_rare_extreme_shrinkage_confirmation.md), [3.9](phase3_9_solar_mechanism_audit.md) |
| Solar 3.15 | Seed-0 frozen RPS; explicitly authorized archived confirmatory readout, N=28006, X=921, C0=5284 | [3.15](phase3_15_solar_direction_scale_mechanism_confirmation.md) |

Retina D continuous diagnostics below additionally reuse the saved
`outputs/retinamnist/phase3_10d_controlled_scale_head/summary/results.json`
entry `0.5`. Other values are transcribed from the cited completed notes.
A/B/C/D mean original RPS / unrestricted balanced / fixed-original-norm,
fixed-bias direction-only / fixed interpolated norms at alpha=.50 with original
biases and directions retrained. Phase 3.9 Solar heads had different selection
rules and must not be substituted for Phase 3.15 B.

Compare within-dataset effects, not raw performance rankings between datasets.
Retina and UTKFace have different class distributions and difficulty; Solar
has temporal shift and substantial alignment missingness (about 34–40% by
split). Head OOF does not make Retina's canonical backbone OOF: it was trained
on the full training set. Five folds are not five independent seeds. All
mechanism results use one backbone seed per dataset and the tested RPS
representations; they do not establish architecture- or objective-universal
causality. Solar's previously inspected test readout is not pristine final
method validation.

## Three-dataset evidence matrix

### Rare upper endpoint

Exact means total exact L1 decisions, not newly recovered sample overlap.
In particular, UTKFace A already has 35 exact cases. All errors are measured
using the exact discrete L1 decision; shrinkage is 4 minus predictive mean.

| Dataset | Head | Exact / support | Rare MAE | Predictive mean | Shrinkage | Mean p4 | Severe % |
|---|---|---:|---:|---:|---:|---:|---:|
| Retina OOF | A | 0/66 | 1.697 | 2.203 | 1.797 | .116 | 65.2 |
| Retina OOF | B | 11/66 | 1.121 | 2.727 | 1.273 | .338 | 25.8 |
| Retina OOF | C | 2/66 | 1.348 | 2.616 | 1.384 | .306 | 33.3 |
| Retina OOF | D | 8/66 | 1.258 | 2.640 | 1.360 | .302 | 33.3 |
| UTKFace validation | A | 35/67 | .6119 | 3.2454 | .7546 | .4988 | 10.4 |
| UTKFace validation | B | 38/67 | .5224 | 3.4080 | .5920 | .5389 | 6.0 |
| UTKFace validation | C | 39/67 | .5075 | 3.4181 | .5819 | .5766 | 6.0 |
| UTKFace validation | D | 40/67 | .4925 | 3.4497 | .5503 | .5914 | 6.0 |
| Solar archived readout | A | 0/921 | 1.197 | 2.772 | 1.228 | .033 | 17.3 |
| Solar archived readout | B | 334/921 | 1.049 | 2.941 | 1.059 | .385 | 38.7 |
| Solar archived readout | C | 496/921 | .675 | 3.227 | .773 | .513 | 19.0 |
| Solar archived readout | D | 262/921 | 1.024 | 2.952 | 1.048 | .326 | 28.4 |

A→C improves rare MAE, predictive mean, shrinkage, p4, and total exact decisions
on all three datasets with original norms and biases fixed. This makes
direction adaptation the strongest transported intervention axis. It does not
improve every endpoint diagnostic: Solar C severe prevalence remains slightly
above A (19.0% versus 17.3%). Nor does it solve every failure.

C→D improves Retina MAE/exact recovery and modestly improves UTKFace, but
substantially harms Solar: 234 fewer exact X decisions, MAE +.349, shrinkage
+.275, and mean pX -.187 at displayed precision. Retina D also has slightly
lower p4 than C despite improved routing. The Phase 3.14 general statement
that added scale enables stronger movement must therefore be narrowed.
Scale is a **dataset-dependent modulation of correction strength**.

C and D retrain directions under different fixed norms; their difference
measures the effect of the scale constraint on the whole adaptation procedure,
including learned directions and optimization. It is not a pure post-hoc
scale change at identical directions. Solar rejects universal benefit of the
tested .50 rule, not every possible increase in scale. No new scale is selected.

### Lower endpoint, global probability, and ordinal risk

| Dataset | Head | C0 MAE / severe % | Global L1 MAE / QWK | NLL / Brier / RPS | Spearman / selective MAE |
|---|---|---|---|---|---|
| Retina OOF | A | .570 / 25.1 | .696 / .604 | 1.141 / .554 / .122 | .467 / .393 |
| Retina OOF | B | .733 / 24.3 | .744 / .612 | 1.236 / .603 / .136 | .471 / .427 |
| Retina OOF | C | .691 / 22.0 | .721 / .621 | 1.229 / .602 / .133 | .447 / .422 |
| Retina OOF | D | .698 / 22.4 | .731 / .619 | 1.221 / .599 / .133 | .460 / .427 |
| UTKFace validation | A | .3268 / .65 | .2691 / .8191 | .6583 / .3563 / .04984 | .3822 / .1381 |
| UTKFace validation | B | .1808 / 1.53 | .2737 / .8362 | .6360 / .3552 / .05035 | .3987 / .1279 |
| UTKFace validation | C | .1895 / 2.83 | .3138 / .8209 | .6899 / .3908 / .05544 | .4001 / .1588 |
| UTKFace validation | D | .1656 / 2.18 | .3096 / .8235 | .6838 / .3881 / .05489 | .3956 / .1565 |
| Solar archived readout | A | .082 / 1.9 | .452 / .808 | 1.012 / .576 / .081 | .243 / .325 |
| Solar archived readout | B | .095 / 1.6 | .585 / .740 | 1.431 / .727 / .108 | .239 / .446 |
| Solar archived readout | C | .086 / 1.9 | .552 / .786 | 1.110 / .643 / .095 | .148 / .449 |
| Solar archived readout | D | .074 / 1.5 | .538 / .764 | 1.181 / .653 / .097 | .220 / .409 |

Every tested B/C/D has higher global L1 MAE than A across these three readouts.
That is a consistent empirical cost in this protocol, not a theorem that
rare-end correction must degrade global prediction. QWK increases on Retina
and UTKFace but decreases on Solar. UTKFace B improves NLL/Brier and all four
reported risk metrics; its selective MAE falls .1381→.1279. Thus a universal
global/probability/risk degradation claim would be inaccurate.

Conversely, localization improvement does not imply global predictive or UQ
improvement. Solar C has the strongest X localization but worse selective MAE
(.325→.449), Spearman (.243→.148), and AUROC (.628→.495). Solar AUPRC actually
increases under B/C/D (.047→.123/.055/.091), alongside higher severe-error
prevalence. Accordingly, the historical 3.15 phrase about no risk-metric
improvement means no broad risk-quality improvement, not that every individual
metric worsened. Its original note is preserved.

Class-0 effects depend on metric as well as dataset: Retina MAE worsens while
severe prevalence falls; UTKFace MAE improves while severe prevalence rises;
Solar MAE changes mildly in both directions. Universal opposite-endpoint
damage or universal safety restoration is not supported.

### Parameter mechanism and representation

| Evidence | RetinaMNIST | UTKFace | Solar |
|---|---|---|---|
| B rare-class norm A→B | .575→3.285 (fold mean) | .688→6.634 | .594→17.266 |
| B rare-class direction cosine vs A | .314 (fold mean) | .250 | .143 |
| C rare-end result | Partial recovery; only 1/11 B-exact cases retained | Already slightly better than B | Strongest rare-X head |
| D vs C | More exact recovery, lower MAE | Small localization gain | Large localization loss |
| B bias movement | Approximately .002–.008 in magnitude; swaps effectively inert | C0 -.061, C4 +.108 | C0 -.219, C3 +.228, X +.006 |
| Bias causality | Negligible in tested swaps | Not isolated causally | Not isolated causally |

The dimensionally correct decomposition is
\(w_k=s_k\hat v_k\), \(\|\hat v_k\|_2=1\), and
\(z_k(h)=s_k\hat v_k^\top h+b_k\). A scalar bias is added to the logit,
not to the classifier weight vector. This clarifies the conceptual shorthand
in Phase 3.14 without changing its historical conclusion.

Retina representation/head audits and Solar 3.9 support mixed failure.
Solar centroid diagnostics identify 17–24% of X as collapsed and 76–83% as
X-like, all inward-routed by the original heads. These are geometry-dependent
diagnostics, not proof of irrecoverability. Frozen-head recovery demonstrates
that useful endpoint information remains accessible for many samples.
UTKFace lacks an equivalent centroid-collapse audit; do not impute the same
subset proportions there. Head geometry contributes to a substantial subset
of failures, while representation limitations remain plausible and supported
in the audited datasets. Generic balanced CE is actionable, not universally
optimal: C exceeds B on UTKFace and Solar endpoint results.

The project retains the decomposition:

Representation → head geometry → ordinal localization → decision/risk behavior.

Its strongest experimentally controlled component is the A→C direction
intervention on frozen features, norms, and biases, within the tested training
protocols. This does not identify direction as the exclusive cause of failure.

## Candidate disposition and historical validation

Retina 3.11 prevents method freeze: E preserves outward localization but
reverses the OOF safety and ROP interaction. Validation B→E C0/global MAE
worsens .630→.722 / .642→.667; D→E Spearman decreases .450→.438 and selective
MAE increases .359→.366. The six rare-end validation samples limit endpoint
certainty. No subsequent phase tested ROP on another dataset.

Controlled scale is retained solely as a mechanistic/diagnostic probe of
scale constraints. Its generally superior operating-point claim is rejected;
alpha=.50 is not a transferable prescription or a freeze-ready component.
Bias-only correction remains stopped: observed shifts on UTKFace/Solar do not
establish a useful standalone intervention. ROP remains diagnostic/secondary;
the unchanged combined alpha=.50, lambda=1 candidate remains held, not
freeze-ready. Generic balancing, direction constraints, or scale control are
not claimed as novel algorithms; the contribution is the ordinal localization
diagnosis and its separation from decision-risk behavior.

## Final phenomenon and mechanism statements

Phenomenon: **Rare upper-extreme inward localization bias under ordinal
imbalance.** Severity varies: RetinaMNIST/Solar exhibit pronounced inward
failure; UTKFace is milder but clearly shrunk. This describes the observed
imbalanced settings, not an isolated causal experiment on imbalance itself.

Mechanism: **Across the tested frozen RPS representations on RetinaMNIST,
UTKFace, and Solar, rare upper-end localization consistently improves under
classifier direction adaptation, while the benefits of scale changes, bias
behavior, and global and opposite-endpoint effects are dataset-dependent.**

## Final claim and component disposition

Statuses below classify the stated claim; operational component decisions are
given in the interpretation column. Strong transport denotes consistency in
these saved experiments, not multi-seed or universal validation.

| Claim / component | Status | Final interpretation |
|---|---|---|
| A. Rare upper-end inward bias | STRONGLY TRANSPORTED | Present across all three datasets; severity differs. |
| Representation contribution | PARTIALLY TRANSPORTED | Mixed representation/head evidence on Retina and Solar; no corresponding UTKFace collapse audit. |
| B. Frozen-head actionability | STRONGLY TRANSPORTED | Frozen heads materially change rare-end localization; generic balanced CE is not established as the correct solution. |
| C. Direction adaptation | TRANSPORTED | Fixed-original-scale/bias C improves rare MAE and outward location on all three tested representations. |
| D. Additional scale improves recovery | DATASET-SPECIFIC | C→D helps Retina, modestly helps UTKFace, and substantially harms Solar. Scale is a modulator, not a required universal increment. |
| E. Controlled scale is generally better | NOT SUPPORTED | Retain as mechanistic/diagnostic probe only; no transferable .50 prescription or general candidate claim. |
| F. Opposite-endpoint damage is general | DATASET-SPECIFIC | MAE and severe-rate directions differ across datasets and within endpoints. |
| G. Global predictive degradation is general | PARTIALLY TRANSPORTED | Higher global L1 MAE in every tested adapted head; QWK/probability results vary, so no inevitability claim. |
| H. Bias is a key cross-dataset mechanism | NOT SUPPORTED | Bias movement is dataset/class-dependent; causal evidence outside Retina is absent. Bias-only correction stays stopped. |
| Risk/UQ improvement follows localization | NOT SUPPORTED | Selective MAE, Spearman, AUROC/AUPRC, and probability quality can disagree with localization and each other. |
| I. ROP is robust | NOT SUPPORTED | Diagnostic/secondary only; Retina validation did not reproduce the OOF interaction; no transport evidence. |
| J. Combined candidate is freeze-ready | NOT SUPPORTED | HOLD — NOT FREEZE-READY; no alpha/lambda change or method freeze. |

## Next-stage decision

**A — MECHANISM EVIDENCE SUFFICIENT; MOVE TO PAPER FRAMING**

Three domains establish a bounded, coherent story: inward localization,
accessible frozen-head recovery, a transported direction intervention, and
non-universal scale/safety behavior. Solar's negative scale result strengthens
the boundary of that story. More generic confirmation is unlikely to change
its central framing. A new effective method is not required to report these
mechanistic and negative findings honestly.

Paper framing should organize the existing evidence around those claims and
their limits, keeping development OOF, historical validation, and archived
Solar readout roles explicit. The priority is an evidence-led mechanism paper,
not a universal head-correction algorithm. This phase does not write the paper,
design a method, authorize new runs, revive ROP/bias correction, or authorize
test access. Future paper work is the next stage; experiments require separate
authorization. Verification is documentation review and the requested Git
checks only.
