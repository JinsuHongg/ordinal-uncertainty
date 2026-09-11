# Paper Story Architecture

## Purpose and status

This is the manuscript-level scientific source of truth after Phases 3.18A,
3.18B, 3.19, and 3.20A. It guides manuscript structure, claim strength,
results ordering, figure design, main-text versus appendix allocation, and
future figure-data consolidation. It is **not** an experiment log: historical
phase notes remain authoritative for exact protocols and results.

The Figure 1--4 source-artifact audit and deterministic canonical data-table
consolidation are complete; see [the source-artifact audit](figure_source_artifact_audit.md).
Its 55 provenance and aggregate checks passed. Do not start experiments, extend
Phase 3.20, run an RPS severity replication, add datasets/seeds, design a
correction, or generate final figures without a separate authorization.

## Direction and thesis

This is a **mechanism / empirical analysis paper**, not a new-method paper.
The central problem is recurring ordinal localization failure at a rare upper
endpoint: in the studied imbalanced settings, rare upper extremes are often
predicted inward toward more central classes.

> Across three studied imbalanced ordinal classification settings, rare upper
> extremes exhibit systematic inward localization. Representation and
> classifier-head analyses show that this failure is mixed: some rare-end
> samples are already displaced inward in feature space, while others retain
> rare-end geometry but are mapped inward by the classifier head.
> Classifier-direction adaptation is the most consistent actionable head
> mechanism and reproduces on both CE- and RPS-trained frozen representations
> in RetinaMNIST and Solar. In a controlled RetinaMNIST factorial study,
> balanced sampling is the dominant training factor associated with the
> beneficial direction change. However, controlled reductions in rare-class
> support do not produce a clean monotonic localization dose response, so
> imbalance severity is not treated as an isolated causal explanation.

The paper asks whether the phenomenon recurs across datasets; whether it
arises in representation, classifier head, or both; which head component is
most consistently actionable; whether direction responsiveness depends on the
original probability-training objective; which training signal induces it; and
whether increasing rare-class severity produces a clean dose response.

## Claim hierarchy and contributions

### C1 — Cross-dataset phenomenon

Rare upper extremes exhibit systematic inward localization across the studied
imbalanced ordinal settings (RetinaMNIST, UTKFace, Solar). The direction
recurs, the probability distribution itself can be displaced inward, and
severity is dataset-dependent. Do not claim imbalance universally causes the
pattern, that it occurs in every ordinal task, or that severity is directly
comparable across datasets.

### C2 — Mixed representation/head failure

Some rare-end samples are closer to interior centroids in frozen feature space;
others remain nearest the rare-end centroid but the original classifier maps
them inward. The allowed interpretation is **mixed but decomposable**. This is
supported by RetinaMNIST Phases 3.3/3.4, Solar Phase 3.9, and aligned subgroup
recovery where available. Nearest-centroid routing is diagnostic, not causal
ground truth; it neither proves representation collapse nor unrecoverability.

### C3 — Direction is the most consistent actionable head component

Classifier-direction adaptation consistently improves rare-end localization on
the tested frozen representations: RetinaMNIST RPS/CE, Solar RPS/CE, plus
UTKFace RPS. Direction is the most consistent actionable head axis across
these settings, while scale, bias, and collateral effects are dataset-dependent.
It is not universal, necessarily causal, the best classifier, the only relevant
head component, or objective-independent.

### C4 — Balanced sampling is the dominant adaptation factor

On frozen RetinaMNIST RPS features, the direction-only 2x2 study gives C4 MAE
`C=1.348`, `F=1.318`, versus natural-sampling `E=1.697`, `G=1.712`:

| Sampling | CE objective | RPS objective |
| --- | ---: | ---: |
| Natural | E | G |
| Balanced | C | F |

The same grouping appears in shrinkage, outward routing, and `z4-z3`. The
bounded claim is that balanced sampling is the dominant training factor
associated with beneficial classifier-direction adaptation in this frozen-RPS
RetinaMNIST study; CE-versus-RPS differences are comparatively small. It is
neither a universal cause nor a novel method.

### C5 — Boundary: no clean imbalance-severity localization law

The Phase 3.20A full-CE grid (`N4 in {66,50,33,16,8}` x five seeds) shows
consistent deterioration in `p4` and `z4-z3` as support falls. C4 MAE,
shrinkage, severe-error prevalence, and feature-nearest-4 fraction are
non-monotonic and seed-sensitive. Thus lower support weakens direct rare-class
evidence, but does not establish a simple monotonic localization dose-response
law or an isolated causal explanation.

Use three contributions: (1) characterize the phenomenon across three settings
with ordinal routing, predictive location, rare-end error, and geometry;
(2) show mixed representation/head failure and identify direction as the most
consistent actionable head mechanism, including CE/RPS robustness in Retina and
Solar; (3) establish the bounded sampling-source result and severity boundary.

## Introduction argument

1. Ordinal prediction depends on error location: adjacent and distant errors
   differ, so probability location matters beyond exact accuracy.
2. Rare extremes create an under-characterized regime at the intersection of
   long-tail classification and ordinal prediction.
3. Introduce **rare-upper-extreme inward localization** and state that it is
   studied in imbalanced settings without presuming imbalance severity is its
   cause.
4. Pose `x -> h(x) -> Wh+b -> p(y|x)` and ask whether failure is in the
   representation, head, or both.
5. For `z_k=w_k^T h+b_k`, write `w_k=s_kv_k`, where `s_k=||w_k||_2` and
   `||v_k||_2=1`; direction, scale, and bias interventions are diagnostic tools,
   not proposed prediction methods.
6. Summarize C1--C5 and the three contributions above.

## Results architecture

Organize results by scientific question, not phase chronology.

### 4.1 Rare upper extremes exhibit inward localization

Use RetinaMNIST, UTKFace, and Solar rare-end routing, MAE, predictive mean,
and shrinkage. Conclude that rare upper extremes are shifted inward, with
dataset-dependent severity.

### 4.2 The failure is mixed between representation and classifier head

Use training-derived centroid routing, representation-inward rare samples, and
rare-end-like samples that the original head still maps inward. Conclude the
failure is mixed but decomposable.

### 4.3 Classifier direction is the most consistent actionable head component

Compare original A, unrestricted balanced B, direction-only C, and controlled
scale D where relevant. Conclude that direction is most consistent; scale,
bias, and collateral effects are dataset-dependent. Do not present C as a
universally better classifier.

### 4.4 Direction responsiveness across CE and RPS

Show A->C rare-end MAE: Retina RPS `1.697->1.348`, Retina CE
`1.439->1.030`, Solar RPS `1.197->0.675`, Solar CE `1.115->0.586`. Exact
recovery is Retina RPS `0->2`, Retina CE `0->11`, Solar RPS `0->496/921`, and
Solar CE `0->559/921`. Conclude only that the response is not restricted to
RPS-trained representations in these two evaluated domains.

### 4.5 Balanced sampling is the dominant adaptation factor in RetinaMNIST

Show the 2x2 factorial pattern `C approximately F < E approximately G` for C4
MAE, alongside shrinkage, routing, and `z4-z3`. Conclude balanced sampling is
the dominant factor in this controlled frozen-RPS mechanism study.

### 4.6 Reduced rare-class support does not yield a simple localization dose response

Show the 25-model CE study: support reduction consistently lowers `p4` and
`z4-z3`, but not C4 MAE, shrinkage, severe error, or feature-nearest-4
fraction. Do not smooth away non-monotonicity.

## Main figure architecture

Use about four figures.

### Figure 1 — Phenomenon: rare upper extremes exhibit inward localization

Freeze this figure to the original **CE** baseline for RetinaMNIST, UTKFace, and Solar: the original, pre-intervention seed-0 model and exact discrete L1 Bayes decision. CE is the standard baseline for the phenomenon figure; CE/RPS robustness belongs in Figure 3.

- **A:** Three small-multiple L1-Bayes routing bars for `Y=K-1` in RetinaMNIST,
  UTKFace, and Solar.
- **B:** Predictive location `mu_p=sum_k k p_k` and shrinkage
  `S=(K-1)-mu_p`, shown as ordinal-axis dots/segments to the true endpoint.
- **C:** Paired lower-versus-upper endpoint L1 MAE control.

This is phenomenon-only: no mechanism intervention and no causal imbalance or
cross-task-severity inference.

### Figure 2 — Failure is mixed between representation and head

Freeze this figure to the CE mechanism pathway where valid aligned artifacts exist:
`CE representation -> original CE head -> CE direction-only adaptation`; Figure 3 separately
shows CE/RPS robustness.

- **A:** RetinaMNIST and Solar stacked bars of train-centroid states:
  representation-inward versus rare-end-like.
- **B:** Original A-head L1 routing conditioned on those states.
- **C:** A->C exact recovery by state. Retina CE has 11 recoveries, all among
  feature-nearest-4 samples. Solar CE is confirmed: 724 X-like cases change `0/0/9/715/0` to
  `0/0/49/116/559` (0→559 exact), while 197 representation-inward cases
  change `0/23/51/123/0` to `0/23/83/91/0` (0→0 exact). All exact recoveries
  are X-like; this remains descriptive.
- **D:** `Delta shrinkage = S_C-S_A` (or `Delta mu_p`) by state.

Never mix incompatible populations, particularly Phase 3.3 Retina test C4
with Phase 3.18A OOF C4, in one sample-level analysis.

### Figure 3 — Direction adapts across objectives and domains

For Retina RPS/CE and Solar RPS/CE, show paired A->C plots for (A) rare-end
MAE, (B) **exact rare-end fraction** with useful raw-count annotations only, (C)
shrinkage, and (D) rare-versus-adjacent margin (`z4-z3` or `zX-zM`) centred at
zero. Verify the Retina rare-end denominator from artifacts before creating any figure table.

RetinaMNIST direction results are training-only OOF head evaluations, whereas Solar results are the predeclared archived confirmatory readout. Treat the scientific comparison as sign and qualitative consistency of A→C localization response, not as four equivalent independent test-set replications. Known shrinkage values are Retina RPS `1.797->1.384`, Retina CE
`1.678->1.244`, Solar RPS `1.228->0.773`, Solar CE `1.136->0.656`; retrieve
exact saved RPS margins rather than hard-coding them. Known CE margin changes
are Retina `-0.903->+0.261` and Solar `-3.109->+1.190`. Do not claim mediation.

### Figure 4 — Dominant sampling signal and severity boundary

Clearly label two distinct experimental regimes; do not draw them as one continuous causal chain. **Mechanism intervention study (Phase 3.19):** frozen RetinaMNIST RPS representation, direction-only heads, training-only OOF, one backbone seed, natural/balanced × CE/RPS factorial. **Controlled severity study (Phase 3.20A):** full CE backbone-plus-head retraining, five support levels × five seeds, validation checkpoint selection, and final predeclared test evaluation.

Do not numerically connect the Phase 3.18A Retina CE A baseline (C4 MAE approximately 1.439 in frozen-head OOF) with the separate Phase 3.20A full-model `N4=66` CE baseline (approximately 2.330 ± .239): protocols and populations differ. Severity panels must identify the Phase 3.20A full-model protocol.

- **A:** Factorial C4 MAE grouped plot using the C/F/E/G values above.
- **B:** Factorial `z4-z3`; retrieve exact Phase 3.19 artifact values.
- **C:** Severity sweep mean `p4` plus all five seed traces.
- **D:** Same for `z4-z3`.
- **E:** Same for shrinkage, preserving non-monotonicity.

The Figure 4 message is: balanced sampling is the dominant adaptation factor providing the beneficial training signal for direction
adaptation in the Retina frozen-RPS factorial study, while lower support
consistently weakens direct endpoint evidence but does not yield a clean
localization dose response. Do not interpret sample-level uncertainty as
model-level uncertainty.

## Figure-data audit and schemas

Use `saved raw artifacts -> figure-specific data table -> plot`; do not
hard-code manuscript numbers where artifacts exist. The audit must locate exact
machine-readable sources for every panel; verify sample population, split role,
objective, seed, and checkpoint; identify note-only panels; check cross-phase
sample-ID alignment; produce long-form CSV/Parquet tables; and flag unsupported
panels rather than reconstructing values. It must not run new experiments or
select metrics post hoc.

Required Figure 1 sample fields: `dataset`, `model_objective`, `seed`,
`sample_id`, `true_label`, `l1_prediction`, `predictive_mean`,
`inward_shrinkage`, and per-class probabilities.

Required Figure 2 fields: `dataset`, `representation_objective`, `split_role`,
`sample_id`, `true_label`, `nearest_centroid`, `representation_group`, A/C L1
predictions, predictive means, shrinkages, rare probabilities, and exact flags;
derive representation group, exact recovery, delta predictive mean, and delta
shrinkage.

Required Figure 3 summary fields: `dataset`, `representation_objective`,
`condition`, `rare_class`, `rare_n`, rare MAE/exact count/exact fraction,
predictive mean, shrinkage, mean rare probability, severe fraction,
adjacent-margin mean/median/positive fraction, and global L1 MAE.

Required Figure 4 factorial fields: `condition`, `sampling`, `objective`,
`representation_objective`, C4 MAE/shrinkage/exact fraction, `p4_mean`,
`z4_z3_mean`, `z4_z3_positive_fraction`, and class-4 direction cosine.
Required severity fields: `seed`, `n4_support`, `retained_fraction`, C4
MAE/shrinkage/predictive mean, `p4` mean/median, `z4_z3` mean/median/positive
fraction, C4 severe fraction, feature-nearest-4 fraction, C0 MAE, and global
L1 MAE.

## Tables and allocation

Keep two main tables: (1) dataset/protocol summary (dataset, ordered classes,
split counts, rare endpoint/support, backbone, objectives); (2) cross-domain
direction summary (representation objective, A/C rare MAE, exact recovery,
shrinkage change, global MAE change).

Main text contains the phenomenon, decomposition, direction mechanism,
CE/RPS robustness, Phase 3.19 factorial, Phase 3.20A boundary, and at most one
concise controlled-scale falsification. Put full CE/RPS global metrics,
class-0 controls, B-head details, scale/bias sweeps, ROP, stopped baselines,
temperature, complete UQ metrics, full 25-run tables, and extra geometry in
the appendix. Do not write a chronological method-search history.

## Reviewer-facing boundaries and terminology

Tail underperformance alone is insufficient: routing, predictive mean,
shrinkage, and probability mass establish ordinal interior displacement. The
CE/RPS findings answer the RPS-artifact concern without claiming objective
independence. Sampling is the training signal; direction-only parameterization
is the fixed-norm/fixed-bias head pathway. Localization can improve while
global/UQ performance worsens, which is part of the mechanism finding.

Prefer: rare upper extreme/endpoint, inward localization, rare-end-like
representation, representation-inward, head-actionable component, classifier
direction, direction-only adaptation, controlled mechanism study, and
dataset-dependent effect. Avoid unqualified “collapse.”

Never claim: imbalance causes the phenomenon; rarity monotonically worsens
localization; direction is universally causal or corrective; balanced sampling
is novel or universal; scale is generally beneficial; bias is the main cause;
nearest-centroid routing proves collapse; representation-inward samples are
unrecoverable; CE/RPS are equivalent; or localization gains imply global/UQ
gains.

Suggested figure captions are the claims stated in the Figure 1--4 sections:
phenomenon; mixed representation/head components; CE/RPS and Retina/Solar
direction response with dataset-dependent magnitude; and the paired sampling
driver/severity-boundary result.
